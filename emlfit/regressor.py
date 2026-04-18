"""scikit-learn-style regressor wrapper around EMLTree."""

from __future__ import annotations

import numpy as np
import torch

from .core import EMLTree
from .refine import refine


class EMLRegressor:
    """Fit a mixed-operator tree (default: eml-only) to data.

    Pass ops=("eml", "mul", "add") to enable practical formula discovery like
    pi*r^2. Pure EML (ops=("eml",)) preserves the paper's single-operator
    property but restricts what's reachable at shallow depth.
    """

    def __init__(
        self,
        depth: int = 3,
        ops: tuple[str, ...] = ("eml",),
        epochs: int = 2000,
        lr: float = 0.05,
        n_restarts: int = 4,
        beam_width: int = 32,
        beam_iterations: int = 30,
        max_bf_combinations: int = 10_000_000,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.depth = depth
        self.ops = ops
        self.epochs = epochs
        self.lr = lr
        self.n_restarts = n_restarts
        self.beam_width = beam_width
        self.beam_iterations = beam_iterations
        self.max_bf_combinations = max_bf_combinations
        self.device = device or _default_device()
        self.verbose = verbose

        self.model_: EMLTree | None = None
        self.leaves_: list[dict] | None = None
        self.ops_: list[str] | None = None
        self.loss_: float | None = None

    def _as_tensor(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xt = torch.from_numpy(X).to(self.device)
        if y is None:
            return Xt
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1)).to(self.device)
        return Xt, yt

    def fit(self, X, y):
        Xt, yt = self._as_tensor(X, y)
        n_inputs = Xt.size(1)

        best_model, best_leaves, best_ops, best_loss = None, None, None, float("inf")
        for restart in range(self.n_restarts):
            torch.manual_seed(restart)
            model = EMLTree(
                depth=self.depth, n_inputs=n_inputs, ops=self.ops
            ).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)

            for epoch in range(self.epochs):
                model.temperature = 1.0 - 0.5 * (epoch / max(1, self.epochs - 1))
                opt.zero_grad()
                pred = model(Xt)
                loss = torch.mean((pred - yt) ** 2)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            if not torch.isfinite(loss):
                if self.verbose:
                    print(f"restart {restart}: diverged")
                continue

            leaves, ops_, refined_loss = refine(
                model, Xt, yt,
                max_bf_combinations=self.max_bf_combinations,
                beam_width=self.beam_width,
                beam_iterations=self.beam_iterations,
            )
            if self.verbose:
                print(
                    f"restart {restart}: soft={float(loss.item()):.4g} "
                    f"refined={refined_loss:.4g}"
                )
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_model = model
                best_leaves = leaves
                best_ops = ops_

        if best_model is None:
            raise RuntimeError("All restarts diverged; try lowering lr or depth.")

        self.model_ = best_model
        self.leaves_ = best_leaves
        self.ops_ = best_ops
        self.loss_ = best_loss
        return self

    def predict(self, X):
        self._check_fitted()
        Xt = self._as_tensor(X)
        with torch.no_grad():
            return self.model_.evaluate(Xt, self.leaves_, self.ops_).cpu().numpy()

    def score(self, X, y) -> float:
        yhat = self.predict(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    def formula(self, input_names=None, snap: bool = True):
        from .formula import extract_formula

        self._check_fitted()
        return extract_formula(
            self.model_,
            leaves=self.leaves_,
            ops=self.ops_,
            input_names=input_names,
            snap=snap,
        )

    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("EMLRegressor not fitted yet. Call .fit(X, y).")


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
