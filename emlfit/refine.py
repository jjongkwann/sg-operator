"""Discrete refinement of a trained tree.

Three strategies:
  - brute_force: exhaust (ops^n_internal) * (candidates^n_leaves) configs.
  - beam_search: maintain top-K candidates, expand by changing one slot. Uses
    a batched evaluator to amortize Python overhead across many variants.
  - greedy: final polish around the best state.
"""

from __future__ import annotations

import itertools
import math
import random

import torch

from .core import OPS, EMLTree

_CONST_CANDIDATES: list[float] = [
    0.0, 1.0, -1.0, 2.0, 0.5,
    math.e, math.pi, 2 * math.pi,
]


def _loss(tree: EMLTree, leaves, ops, Xt, yt) -> float:
    pred = tree.evaluate(Xt, leaves, ops)
    return float(torch.mean((pred - yt) ** 2).item())


def _shared_candidates(tree: EMLTree) -> list[dict]:
    cands: list[dict] = [{"kind": "input", "index": i} for i in range(tree.n_inputs)]
    cands.extend({"kind": "const", "value": v} for v in _CONST_CANDIDATES)
    return cands


def _leaf_key(leaf: dict):
    if leaf["kind"] == "input":
        return ("input", leaf["index"])
    return ("const", round(leaf["value"], 6))


def _state_key(leaves, ops):
    return (tuple(_leaf_key(l) for l in leaves), tuple(ops))


def _leaf_column(leaf: dict, Xt: torch.Tensor) -> torch.Tensor:
    """Compute a (B,) column for a single leaf assignment."""
    B = Xt.size(0)
    if leaf["kind"] == "input":
        return Xt[:, leaf["index"]]
    return torch.full((B,), leaf["value"], dtype=Xt.dtype, device=Xt.device)


def _batch_losses(
    tree: EMLTree,
    states: list[tuple[list[dict], list[str]]],
    Xt: torch.Tensor,
    yt: torch.Tensor,
) -> torch.Tensor:
    """Evaluate K (leaves, ops) states at once. Returns (K,) MSE losses."""
    if not states:
        return torch.empty(0)

    K = len(states)
    B = Xt.size(0)
    L = tree.n_leaves
    op_names = list(OPS.keys())
    op_index = {name: i for i, name in enumerate(op_names)}

    # Build (K, B, L) leaf tensor.
    leaf_stack = torch.empty((K, B, L), dtype=Xt.dtype, device=Xt.device)
    for k, (leaves, _ops) in enumerate(states):
        for i, leaf in enumerate(leaves):
            leaf_stack[k, :, i] = _leaf_column(leaf, Xt)

    # Build (K, n_internal) op-index tensor.
    op_idx = torch.empty((K, tree.n_internal), dtype=torch.long, device=Xt.device)
    for k, (_leaves, ops) in enumerate(states):
        for i, op_name in enumerate(ops):
            op_idx[k, i] = op_index[op_name]

    values = leaf_stack  # (K, B, L)
    node_cursor = 0
    for _ in range(tree.depth):
        left = values[:, :, 0::2]
        right = values[:, :, 1::2]
        level_size = left.size(2)

        # Compute every op on the full batch: (n_ops, K, B, level_size)
        all_ops = torch.stack([OPS[name](left, right) for name in op_names])

        # Gather the chosen op per (k, i).
        level_idx = op_idx[:, node_cursor : node_cursor + level_size]  # (K, level_size)
        gather_idx = level_idx[None, :, None, :].expand(1, K, B, level_size)
        values = torch.gather(all_ops, 0, gather_idx).squeeze(0)  # (K, B, level_size)
        node_cursor += level_size

    preds = values.squeeze(-1)  # (K, B)
    return torch.mean((preds - yt.unsqueeze(0)) ** 2, dim=1)


def brute_force(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    max_combinations: int = 10_000_000,
    batch_size: int = 4096,
) -> tuple[list[dict], list[str], float] | None:
    cands = _shared_candidates(tree)
    n_ops = len(tree.ops)
    total = (n_ops**tree.n_internal) * (len(cands) ** tree.n_leaves)
    if total > max_combinations:
        return None

    op_names = list(tree.ops)
    best_loss = float("inf")
    best_leaves = None
    best_ops = None

    buffer: list[tuple[list[dict], list[str]]] = []

    def flush():
        nonlocal best_loss, best_leaves, best_ops
        if not buffer:
            return
        losses = _batch_losses(tree, buffer, Xt, yt).tolist()
        for (l, o), lv in zip(buffer, losses):
            if lv < best_loss:
                best_loss = lv
                best_leaves = l
                best_ops = o
        buffer.clear()

    for op_combo in itertools.product(op_names, repeat=tree.n_internal):
        op_list = list(op_combo)
        for leaf_combo in itertools.product(cands, repeat=tree.n_leaves):
            buffer.append((list(leaf_combo), op_list))
            if len(buffer) >= batch_size:
                flush()
    flush()
    return best_leaves, best_ops, best_loss


def greedy(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    leaves: list[dict] | None = None,
    ops: list[str] | None = None,
    n_passes: int = 3,
) -> tuple[list[dict], list[str], float]:
    if leaves is None:
        leaves = tree.discrete_leaves()
    if ops is None:
        ops = tree.discrete_ops()
    leaves = list(leaves)
    ops = list(ops)
    best_loss = _loss(tree, leaves, ops, Xt, yt)

    shared = _shared_candidates(tree)

    for _ in range(n_passes):
        improved = False
        for k in range(tree.n_leaves):
            variants = [(list(leaves[:k]) + [cand] + list(leaves[k + 1 :]), ops)
                        for cand in shared]
            losses = _batch_losses(tree, variants, Xt, yt).tolist()
            best_idx = int(min(range(len(losses)), key=losses.__getitem__))
            if losses[best_idx] + 1e-12 < best_loss:
                best_loss = losses[best_idx]
                leaves = variants[best_idx][0]
                improved = True
        for k in range(tree.n_internal):
            variants = [(leaves, list(ops[:k]) + [name] + list(ops[k + 1 :]))
                        for name in tree.ops]
            losses = _batch_losses(tree, variants, Xt, yt).tolist()
            best_idx = int(min(range(len(losses)), key=losses.__getitem__))
            if losses[best_idx] + 1e-12 < best_loss:
                best_loss = losses[best_idx]
                ops = variants[best_idx][1]
                improved = True
        if not improved:
            break
    return leaves, ops, best_loss


def beam_search(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    beam_width: int = 16,
    n_iterations: int = 20,
    seed_count: int = 32,
    seed: int = 0,
) -> tuple[list[dict], list[str], float]:
    rng = random.Random(seed)
    shared = _shared_candidates(tree)
    op_names = list(tree.ops)

    # Initial beam: argmax snap + random.
    initial: list[tuple[list[dict], list[str]]] = []
    leaves0 = tree.discrete_leaves()
    ops0 = tree.discrete_ops()
    initial.append((leaves0, ops0))
    for _ in range(seed_count - 1):
        leaves = [dict(rng.choice(shared)) for _ in range(tree.n_leaves)]
        ops = [rng.choice(op_names) for _ in range(tree.n_internal)]
        initial.append((leaves, ops))
    losses = _batch_losses(tree, initial, Xt, yt).tolist()
    beam = sorted(
        [(s[0], s[1], lv) for s, lv in zip(initial, losses)], key=lambda x: x[2]
    )[:beam_width]

    for _ in range(n_iterations):
        best_before = beam[0][2]
        variants: list[tuple[list[dict], list[str]]] = []
        for leaves, ops, _lv in beam:
            cur_keys = [_leaf_key(l) for l in leaves]
            for k in range(tree.n_leaves):
                for cand in shared:
                    if _leaf_key(cand) == cur_keys[k]:
                        continue
                    new_leaves = list(leaves)
                    new_leaves[k] = cand
                    variants.append((new_leaves, list(ops)))
            for k in range(tree.n_internal):
                for op_name in op_names:
                    if op_name == ops[k]:
                        continue
                    new_ops = list(ops)
                    new_ops[k] = op_name
                    variants.append((list(leaves), new_ops))

        if not variants:
            break
        var_losses = _batch_losses(tree, variants, Xt, yt).tolist()
        combined = beam + [
            (s[0], s[1], lv) for s, lv in zip(variants, var_losses)
        ]
        combined.sort(key=lambda x: x[2])

        seen = set()
        new_beam = []
        for entry in combined:
            key = _state_key(entry[0], entry[1])
            if key in seen:
                continue
            seen.add(key)
            new_beam.append(entry)
            if len(new_beam) >= beam_width:
                break

        if new_beam[0][2] >= best_before - 1e-12:
            beam = new_beam
            break
        beam = new_beam

    best = beam[0]
    return best[0], best[1], best[2]


def refine(
    tree: EMLTree,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    max_bf_combinations: int = 10_000_000,
    beam_width: int = 16,
    beam_iterations: int = 20,
) -> tuple[list[dict], list[str], float]:
    bf = brute_force(tree, Xt, yt, max_combinations=max_bf_combinations)
    if bf is not None:
        return bf
    leaves, ops, _loss_v = beam_search(
        tree, Xt, yt,
        beam_width=beam_width,
        n_iterations=beam_iterations,
    )
    return greedy(tree, Xt, yt, leaves=leaves, ops=ops, n_passes=2)
