"""Test beam search on depth 3 targets.

Brute force is infeasible at depth 3 (17^8 leaf combos alone ~7B), so refine
falls back to beam search. Targets here require at least depth 3 with mixed
operators.
"""

from __future__ import annotations

import time

import numpy as np

from emlfit import EMLRegressor


def task(name, X, y, depth, ops, input_names, epochs=1500, restarts=2,
         beam_width=32, beam_iter=30):
    print(f"\n=== {name} (depth={depth}, ops={ops}) ===")
    t0 = time.time()
    model = EMLRegressor(
        depth=depth, ops=ops, epochs=epochs, n_restarts=restarts,
        beam_width=beam_width, beam_iterations=beam_iter,
    ).fit(X, y)
    dt = time.time() - t0
    print(f"  time = {dt:.1f}s")
    print(f"  loss = {model.loss_:.4g}")
    print(f"  R^2  = {model.score(X, y):.6f}")
    print(f"  formula ≈ {model.formula(input_names=input_names)}")


def main():
    rng = np.random.default_rng(0)

    # y = a^2 + b^2 + c^2  (3 inputs, depth 3) — needs wider beam
    a = rng.uniform(-1, 1, 200)
    b = rng.uniform(-1, 1, 200)
    c = rng.uniform(-1, 1, 200)
    X = np.stack([a, b, c], axis=1)
    task("y = a^2 + b^2 + c^2", X, a**2 + b**2 + c**2,
         depth=3, ops=("mul", "add"), input_names=["a", "b", "c"],
         beam_width=64, beam_iter=40, restarts=4)

    # y = 2*pi*r  (depth 2 should suffice with brute force)
    r = np.linspace(0.3, 3.0, 200).reshape(-1, 1)
    task("y = 2*pi*r", r, 2 * np.pi * r[:, 0],
         depth=2, ops=("eml", "mul"), input_names=["r"])

    # y = (a+b)*(a-b) = a^2 - b^2
    a = rng.uniform(-1, 1, 200)
    b = rng.uniform(-1, 1, 200)
    X = np.stack([a, b], axis=1)
    task("y = a^2 - b^2", X, a**2 - b**2,
         depth=2, ops=("mul", "add", "sub"), input_names=["a", "b"])


if __name__ == "__main__":
    main()
