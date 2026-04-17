"""Discover closed-form formulas from numeric data using EML trees.

eml(x, y) = exp(x) - ln(y). At depth d there are 2^d leaves and 2^d - 1 eml
nodes; simple targets live at shallow depths. For depth <= 2 with ~17
candidate values per leaf, brute-force enumeration is exhaustive (<100k
configs) and finds the globally best clean assignment.
"""

from __future__ import annotations

import numpy as np

from emlfit import EMLRegressor


def task(name: str, X: np.ndarray, y: np.ndarray, depth: int, epochs: int = 2000):
    print(f"\n=== {name} (depth={depth}) ===")
    model = EMLRegressor(depth=depth, epochs=epochs, n_restarts=4).fit(X, y)
    print(f"  loss = {model.loss_:.4g}")
    print(f"  R^2  = {model.score(X, y):.6f}")
    try:
        print(f"  formula ≈ {model.formula()}")
    except Exception as e:
        print(f"  formula extraction failed: {e}")


def main():
    rng = np.random.default_rng(0)

    # depth 1: 2 leaves, 1 eml node.
    x = np.linspace(-1.0, 1.5, 200).reshape(-1, 1)
    task("y = exp(x)", x, np.exp(x[:, 0]), depth=1)  # eml(x, 1)

    x = np.linspace(0.2, 3.0, 200).reshape(-1, 1)
    task("y = 1 - log(x)", x, 1.0 - np.log(x[:, 0]), depth=1)  # eml(0, x)

    x = np.linspace(0.2, 2.0, 200).reshape(-1, 1)
    task("y = exp(x) - log(x)", x, np.exp(x[:, 0]) - np.log(x[:, 0]), depth=1)  # eml(x, x)

    # depth 1 with two inputs.
    x1 = rng.uniform(-0.5, 1.0, 200)
    x2 = rng.uniform(0.2, 2.0, 200)
    X = np.stack([x1, x2], axis=1)
    task("y = exp(x1) - log(x2)", X, np.exp(x1) - np.log(x2), depth=1)  # eml(x1, x2)

    # depth 2: 4 leaves. exp(exp(x)) = eml(eml(x,1), 1).
    x = np.linspace(-0.5, 1.0, 200).reshape(-1, 1)
    task("y = exp(exp(x))", x, np.exp(np.exp(x[:, 0])), depth=2)

    # depth 2 constant discovery: eml(1, 1) = e - 0 = e.
    x = np.linspace(0.1, 2.0, 200).reshape(-1, 1)
    task("y = e  (constant)", x, np.full(200, np.e), depth=1)

    # depth 3 — not cleanly expressible; included to show honest failure.
    r = np.linspace(0.2, 3.0, 200).reshape(-1, 1)
    task("y = 2*pi*r  [not expressible at this depth]", r, 2 * np.pi * r[:, 0], depth=3)


if __name__ == "__main__":
    main()
