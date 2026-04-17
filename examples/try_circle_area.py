"""Discover A = pi * r^2 using mixed operators (eml + mul)."""

from __future__ import annotations

import numpy as np

from emlfit import EMLRegressor


def main():
    r = np.linspace(0.3, 2.5, 200).reshape(-1, 1)
    y = np.pi * r[:, 0] ** 2

    print("--- pi * r^2 with ops=(eml, mul) at depth=2 ---")
    model = EMLRegressor(
        depth=2, ops=("eml", "mul"), epochs=500, n_restarts=2
    ).fit(r, y)
    print(f"  loss = {model.loss_:.4g}")
    print(f"  R^2  = {model.score(r, y):.6f}")
    print(f"  ops  = {model.ops_}")
    print(f"  leaves = {model.leaves_}")
    print(f"  formula ≈ {model.formula(input_names=['r'])}")


if __name__ == "__main__":
    main()
