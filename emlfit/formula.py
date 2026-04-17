"""Symbolic extraction from a trained tree (mixed-op supported)."""

from __future__ import annotations

import math

import sympy as sp

from .core import EMLTree

_FAMILIAR = [
    (0.0, sp.Integer(0)),
    (1.0, sp.Integer(1)),
    (-1.0, sp.Integer(-1)),
    (2.0, sp.Integer(2)),
    (-2.0, sp.Integer(-2)),
    (3.0, sp.Integer(3)),
    (-3.0, sp.Integer(-3)),
    (0.5, sp.Rational(1, 2)),
    (-0.5, sp.Rational(-1, 2)),
    (math.e, sp.E),
    (-math.e, -sp.E),
    (math.pi, sp.pi),
    (-math.pi, -sp.pi),
    (math.pi / 2, sp.pi / 2),
    (2 * math.pi, 2 * sp.pi),
    (1 / math.e, 1 / sp.E),
]


def _snap_constant(value: float, tol: float = 0.02) -> sp.Expr:
    for target, sym in _FAMILIAR:
        if abs(value - target) < tol * max(1.0, abs(target)):
            return sym
    if abs(value - round(value)) < tol:
        return sp.Integer(round(value))
    return sp.Float(round(value, 4))


def _apply_op(op: str, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    if op == "eml":
        return sp.exp(left) - sp.log(sp.Abs(right))
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    raise ValueError(f"unknown op: {op}")


def extract_formula(
    tree: EMLTree,
    leaves: list[dict] | None = None,
    ops: list[str] | None = None,
    input_names=None,
    snap: bool = True,
) -> sp.Expr:
    n_inputs = tree.n_inputs
    if input_names is None:
        input_names = [f"x{i}" if n_inputs > 1 else "x" for i in range(n_inputs)]
    symbols = [sp.Symbol(name, real=True) for name in input_names]

    if leaves is None:
        leaves = tree.discrete_leaves()
    if ops is None:
        ops = tree.discrete_ops()

    expr_leaves: list[sp.Expr] = []
    for leaf in leaves:
        if leaf["kind"] == "input":
            expr_leaves.append(symbols[leaf["index"]])
        else:
            v = leaf["value"]
            expr_leaves.append(_snap_constant(v) if snap else sp.Float(round(v, 4)))

    idx = 0
    while len(expr_leaves) > 1:
        nxt = []
        for i in range(0, len(expr_leaves), 2):
            nxt.append(_apply_op(ops[idx], expr_leaves[i], expr_leaves[i + 1]))
            idx += 1
        expr_leaves = nxt

    return sp.simplify(expr_leaves[0])
