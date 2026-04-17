"""scikit-learn-style regressor wrapper around EMLTree."""

from __future__ import annotations

import numpy as np
import torch

from .core import EMLTree
from .refine import refine


class EMLRegressor:
    """Fit an EML tree to numeric data, then predict / extract a formula.

    Training pipeline per restart:
      1. Gradient training of soft-selection EML tree (Adam).
      2. Discrete refinement: greedy search over clean leaf assignments.
    Best restart (lowest refined MSE) wins.
    """

    def __init__(
        self,
        depth: int = 3,
        epochs: int = 2000,
        lr: float = 0.05,
        n_restarts: int = 4,
        refine_passes: int = 3,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.n_restarts = n_restarts
        self.refine_passes = refine_passes
        self.device = device or _default_device()
        self.verbose = verbose

        self.model_: EMLTree | None = None
        self.leaves_: list[dict] | None = None
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

        best_model, best_leaves, best_loss = None, None, float("inf")
        for restart in range(self.n_restarts):
            torch.manual_seed(restart)
            model = EMLTree(depth=self.depth, n_inputs=n_inputs).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Mild temperature anneal (1 -> 0.5) — refinement does the final commit.
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

            leaves, refined_loss = refine(model, Xt, yt, n_passes=self.refine_passes)
            if self.verbose:
                print(
                    f"restart {restart}: soft_loss={float(loss.item()):.4g} "
                    f"refined_loss={refined_loss:.4g}"
                )
            if refined_loss < best_loss:
                best_loss = refined_loss
                best_model = model
                best_leaves = leaves

        if best_model is None:
            raise RuntimeError("All restarts diverged; try lowering lr or depth.")

        self.model_ = best_model
        self.leaves_ = best_leaves
        self.loss_ = best_loss
        return self

    def predict(self, X):
        self._check_fitted()
        Xt = self._as_tensor(X)
        with torch.no_grad():
            return self.model_.evaluate_leaves(Xt, self.leaves_).cpu().numpy()

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
            self.model_, leaves=self.leaves_, input_names=input_names, snap=snap
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
