"""Discrete refinement of a trained EML tree.

Two strategies, picked by tree size:
  - brute_force: exhaust all (op^n_internal) * (candidates^n_leaves) configs.
    Feasible for depth <= 2 with a small candidate set.
  - greedy: per-leaf local search, seeded by argmax snap. Used for deeper trees
    where brute force is infeasible.
"""

from __future__ import annotations

import itertools
import math

import torch

from .core import EMLTree

_CONST_CANDIDATES: list[float] = [
    0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
    0.5, -0.5,
    math.e, 1.0 / math.e,
    math.pi, math.pi / 2, 2 * math.pi,
]


def _loss(tree: EMLTree, leaves, ops, Xt, yt) -> float:
    pred = tree.evaluate(Xt, leaves, ops)
    return float(torch.mean((pred - yt) ** 2).item())


def _shared_candidates(tree: EMLTree) -> list[dict]:
    cands: list[dict] = [{"kind": "input", "index": i} for i in range(tree.n_inputs)]
    cands.extend({"kind": "const", "value": v} for v in _CONST_CANDIDATES)
    return cands


def brute_force(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    max_combinations: int = 10_000_000,
) -> tuple[list[dict], list[str], float] | None:
    """Exhaust (ops × leaves) combinations. Returns None if above budget."""
    cands = _shared_candidates(tree)
    n_ops = len(tree.ops)
    total = (n_ops**tree.n_internal) * (len(cands) ** tree.n_leaves)
    if total > max_combinations:
        return None

    best_loss = float("inf")
    best_leaves = None
    best_ops = None

    op_names = list(tree.ops)
    for op_combo in itertools.product(op_names, repeat=tree.n_internal):
        op_list = list(op_combo)
        for leaf_combo in itertools.product(cands, repeat=tree.n_leaves):
            leaves = list(leaf_combo)
            loss = _loss(tree, leaves, op_list, Xt, yt)
            if loss < best_loss:
                best_loss = loss
                best_leaves = leaves
                best_ops = op_list
    return best_leaves, best_ops, best_loss


def greedy(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    n_passes: int = 3,
) -> tuple[list[dict], list[str], float]:
    """Greedy per-leaf and per-op search seeded by argmax snap."""
    leaves = tree.discrete_leaves()
    ops = tree.discrete_ops()
    best_loss = _loss(tree, leaves, ops, Xt, yt)

    learned_const = tree.leaf_const.detach().cpu().tolist()
    shared = _shared_candidates(tree)
    per_leaf = [
        [{"kind": "const", "value": float(learned_const[k])}] + shared
        for k in range(tree.n_leaves)
    ]

    for _ in range(n_passes):
        improved = False
        for k in range(tree.n_leaves):
            original = leaves[k]
            for cand in per_leaf[k]:
                leaves[k] = cand
                loss = _loss(tree, leaves, ops, Xt, yt)
                if loss + 1e-12 < best_loss:
                    best_loss = loss
                    original = cand
                    improved = True
            leaves[k] = original
        for k in range(tree.n_internal):
            original_op = ops[k]
            for op_name in tree.ops:
                ops[k] = op_name
                loss = _loss(tree, leaves, ops, Xt, yt)
                if loss + 1e-12 < best_loss:
                    best_loss = loss
                    original_op = op_name
                    improved = True
            ops[k] = original_op
        if not improved:
            break
    return leaves, ops, best_loss


def refine(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    n_passes: int = 3,
    max_bf_combinations: int = 10_000_000,
) -> tuple[list[dict], list[str], float]:
    """Find the best discrete assignment. Uses brute force when feasible."""
    bf = brute_force(tree, Xt, yt, max_combinations=max_bf_combinations)
    if bf is not None:
        return bf
    return greedy(tree, Xt, yt, n_passes=n_passes)
