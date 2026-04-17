"""Discrete refinement of a trained EML tree.

Two strategies, picked by tree size:
  - brute_force: exhaust all (candidates^n_leaves) configurations. Fine for
    depth <= 2 with a small candidate set (~17^4 = 83k). Pike rule #3 — at
    small n, simple is best.
  - greedy: per-leaf local search, seeded by argmax snap. Used for deeper
    trees where brute force is infeasible.
"""

from __future__ import annotations

import itertools
import math

import torch

from .core import EMLTree

# Candidate constants tried at every leaf during refinement.
_CONST_CANDIDATES: list[float] = [
    0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
    0.5, -0.5,
    math.e, -math.e, 1.0 / math.e,
    math.pi, -math.pi, math.pi / 2, 2 * math.pi,
]


def _loss(tree: EMLTree, leaves: list[dict], Xt: torch.Tensor, yt: torch.Tensor) -> float:
    pred = tree.evaluate_leaves(Xt, leaves)
    return float(torch.mean((pred - yt) ** 2).item())


def _shared_candidates(tree: EMLTree) -> list[dict]:
    """Candidate set shared across all leaves."""
    cands: list[dict] = [{"kind": "input", "index": i} for i in range(tree.n_inputs)]
    cands.extend({"kind": "const", "value": v} for v in _CONST_CANDIDATES)
    return cands


def brute_force(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    max_combinations: int = 200_000,
) -> tuple[list[dict], float] | None:
    """Exhaust all candidate^n_leaves assignments. Returns None if too large."""
    cands = _shared_candidates(tree)
    total = len(cands) ** tree.n_leaves
    if total > max_combinations:
        return None

    best_loss = float("inf")
    best_leaves = None
    for combo in itertools.product(cands, repeat=tree.n_leaves):
        leaves = list(combo)
        loss = _loss(tree, leaves, Xt, yt)
        if loss < best_loss:
            best_loss = loss
            best_leaves = leaves
    return best_leaves, best_loss


def greedy(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    n_passes: int = 3,
) -> tuple[list[dict], float]:
    """Greedy leaf-wise search seeded by argmax snap, plus learned scalar."""
    leaves = tree.discrete_leaves()
    best_loss = _loss(tree, leaves, Xt, yt)

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
                loss = _loss(tree, leaves, Xt, yt)
                if loss + 1e-12 < best_loss:
                    best_loss = loss
                    original = cand
                    improved = True
            leaves[k] = original
        if not improved:
            break
    return leaves, best_loss


def refine(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    n_passes: int = 3,
    max_bf_combinations: int = 200_000,
) -> tuple[list[dict], float]:
    """Find the best discrete leaf assignment. Uses brute force when feasible."""
    bf = brute_force(tree, Xt, yt, max_combinations=max_bf_combinations)
    if bf is not None:
        return bf
    return greedy(tree, Xt, yt, n_passes=n_passes)
