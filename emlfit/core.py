"""EML operator and trainable binary tree.

A depth-d full binary tree has 2^d leaves and 2^d - 1 internal nodes. Each
internal node picks an operator from a configurable set (default: eml only).
Enabling additional operators like *, +, - makes shallow trees much more
expressive for practical formula discovery at the cost of the paper's "single
operator" elegance.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

EXP_CLIP = 20.0
LOG_EPS = 1e-4


def safe_eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable EML: exp(clip(x)) - log(|y| + eps)."""
    x_c = torch.clamp(x, -EXP_CLIP, EXP_CLIP)
    y_abs = torch.abs(y) + LOG_EPS
    return torch.exp(x_c) - torch.log(y_abs)


OPS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "eml": safe_eml,
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
}


class EMLTree(nn.Module):
    """Full binary tree with soft-mix leaves and soft-mix operator per node."""

    def __init__(
        self,
        depth: int,
        n_inputs: int = 1,
        ops: tuple[str, ...] = ("eml",),
        temperature: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for op in ops:
            if op not in OPS:
                raise ValueError(f"unknown op: {op}")
        self.depth = depth
        self.n_inputs = n_inputs
        self.ops = tuple(ops)
        self.n_leaves = 2**depth
        self.n_internal = self.n_leaves - 1
        self.temperature = temperature

        self.leaf_selector = nn.Parameter(torch.randn(self.n_leaves, n_inputs + 1) * 0.1)
        self.leaf_const = nn.Parameter(torch.randn(self.n_leaves))
        self.op_logits = nn.Parameter(torch.randn(self.n_internal, len(self.ops)) * 0.1)

    def leaf_values(self, X: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.leaf_selector / self.temperature, dim=-1)
        input_w = weights[:, :-1]
        const_w = weights[:, -1]
        input_part = X @ input_w.T
        const_part = (const_w * self.leaf_const).unsqueeze(0)
        return input_part + const_part

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = self.leaf_values(X)
        op_w = torch.softmax(self.op_logits / self.temperature, dim=-1)
        idx = 0
        for _ in range(self.depth):
            left = values[:, 0::2]
            right = values[:, 1::2]
            level_size = left.size(1)
            out = torch.zeros_like(left)
            for oi, op_name in enumerate(self.ops):
                w = op_w[idx : idx + level_size, oi].unsqueeze(0)
                out = out + w * OPS[op_name](left, right)
            values = out
            idx += level_size
        return values.squeeze(-1)

    @torch.no_grad()
    def discrete_leaves(self) -> list[dict]:
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
    def discrete_ops(self) -> list[str]:
        return [self.ops[i] for i in self.op_logits.argmax(dim=-1).tolist()]

    @torch.no_grad()
    def evaluate(
        self,
        X: torch.Tensor,
        leaves: list[dict],
        ops: list[str] | None = None,
    ) -> torch.Tensor:
        """Discrete evaluation with explicit leaf assignment and op choice."""
        if ops is None:
            ops = [self.ops[0]] * self.n_internal
        cols = []
        B = X.size(0)
        for leaf in leaves:
            if leaf["kind"] == "input":
                cols.append(X[:, leaf["index"]])
            else:
                cols.append(torch.full((B,), leaf["value"], dtype=X.dtype, device=X.device))
        values = torch.stack(cols, dim=1)

        idx = 0
        for _ in range(self.depth):
            left = values[:, 0::2]
            right = values[:, 1::2]
            level_size = left.size(1)
            cols = [OPS[ops[idx + k]](left[:, k], right[:, k]) for k in range(level_size)]
            values = torch.stack(cols, dim=1)
            idx += level_size
        return values.squeeze(-1)
