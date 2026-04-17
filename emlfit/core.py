"""EML operator and trainable binary tree.

The EML operator is eml(x, y) = exp(x) - ln(y). A full binary tree of depth d
has 2^d leaves and 2^d - 1 internal nodes, all identical EML operators. Any
elementary function is reachable by choosing appropriate leaf values from
{inputs, learnable constants, 1}.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EXP_CLIP = 20.0
LOG_EPS = 1e-4


def safe_eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable EML: exp(clip(x)) - log(|y| + eps).

    The paper uses complex numbers for negative log arguments; for real-valued
    regression we take |y| and add a floor epsilon to keep gradients finite.
    Symbolic extraction mirrors this with log(Abs(y)).
    """
    x_c = torch.clamp(x, -EXP_CLIP, EXP_CLIP)
    y_abs = torch.abs(y) + LOG_EPS
    return torch.exp(x_c) - torch.log(y_abs)


class EMLTree(nn.Module):
    """Full binary tree of EML operators with soft-selection leaves.

    Each leaf is a convex combination (softmax) over:
      - the n_inputs feature columns of X, and
      - one learnable scalar constant.

    After training, leaves can be "snapped" to a discrete choice for symbolic
    extraction. See EMLTree.evaluate_with_leaves.
    """

    def __init__(self, depth: int, n_inputs: int = 1, temperature: float = 1.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.n_inputs = n_inputs
        self.n_leaves = 2**depth
        self.temperature = temperature

        self.leaf_selector = nn.Parameter(torch.randn(self.n_leaves, n_inputs + 1) * 0.1)
        self.leaf_const = nn.Parameter(torch.randn(self.n_leaves))

    def leaf_values(self, X: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.leaf_selector / self.temperature, dim=-1)
        input_w = weights[:, :-1]
        const_w = weights[:, -1]
        input_part = X @ input_w.T
        const_part = (const_w * self.leaf_const).unsqueeze(0)
        return input_part + const_part

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return _fold_tree(self.leaf_values(X))

    @torch.no_grad()
    def discrete_leaves(self) -> list[dict]:
        """Snap each leaf to its argmax slot."""
        choices = self.leaf_selector.argmax(dim=-1).tolist()
        consts = self.leaf_const.tolist()
        out = []
        for k, c in enumerate(choices):
            if c < self.n_inputs:
                out.append({"kind": "input", "index": c})
            else:
                out.append({"kind": "const", "value": float(consts[k])})
        return out

    @torch.no_grad()
    def evaluate_leaves(self, X: torch.Tensor, leaves: list[dict]) -> torch.Tensor:
        """Evaluate the tree for an explicit leaf assignment (no soft mixing)."""
        cols = []
        B = X.size(0)
        for leaf in leaves:
            if leaf["kind"] == "input":
                cols.append(X[:, leaf["index"]])
            else:
                cols.append(torch.full((B,), leaf["value"], dtype=X.dtype, device=X.device))
        values = torch.stack(cols, dim=1)
        return _fold_tree(values)


def _fold_tree(values: torch.Tensor) -> torch.Tensor:
    """Pairwise fold the leaf tensor (B, L) down to (B,) with safe_eml."""
    while values.size(1) > 1:
        left = values[:, 0::2]
        right = values[:, 1::2]
        values = safe_eml(left, right)
    return values.squeeze(-1)
