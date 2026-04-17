# emlfit

Symbolic regression built on a single operator:

```
eml(x, y) = exp(x) − ln(y)
```

Based on Odrzywołek (2026), *All elementary functions from a single operator* (arXiv:2603.21852). Every EML expression is a uniform binary tree, so leaf assignments fully determine the formula. At shallow depths the search space is small enough for **brute-force exhaustive search** over a clean candidate set — gradient descent is only needed as a fallback for deeper trees.

## Install (dev)

```bash
uv venv
uv pip install -e .
```

## Usage

```python
import numpy as np
from emlfit import EMLRegressor

x = np.linspace(-1, 1.5, 200).reshape(-1, 1)
y = np.exp(x[:, 0])

model = EMLRegressor(depth=1).fit(x, y)
print(model.formula())   # exp(x)
print(model.score(x, y)) # 1.000000
```

## Demo output

```
$ python examples/demo.py

=== y = exp(x) (depth=1) ===            formula ≈ exp(x)                   R² = 1.000000
=== y = 1 - log(x) (depth=1) ===        formula ≈ 1 - log(Abs(x))          R² = 1.000000
=== y = exp(x) - log(x) (depth=1) ===   formula ≈ exp(x) - log(Abs(x))     R² = 1.000000
=== y = exp(x1) - log(x2) (depth=1) === formula ≈ exp(x0) - log(Abs(x1))   R² = 1.000000
=== y = exp(exp(x)) (depth=2) ===       formula ≈ exp(exp(x))              R² = 1.000000
=== y = e  (constant) (depth=1) ===     formula ≈ E                        loss = 1e-8
```

## How it works

1. **Gradient warm-up** (Adam): trains a soft EML tree where each leaf is a softmax mixture over inputs and a learnable constant.
2. **Discrete refinement**: at depth ≤ 2 (≤ ~100k configurations), enumerate every assignment from a fixed candidate pool: `{inputs} ∪ {0, ±1, ±2, ±3, ±½, ±e, ±π, π/2, 2π, 1/e}`. At deeper depths, greedy per-leaf search seeded by argmax snap.
3. **Symbolic extraction**: fold the discrete leaves through `sp.exp(x) - sp.log(Abs(y))`; mirrors training-time numeric behavior.

## Limitations

- Only elementary functions expressible within the chosen depth are recoverable. `2πr` needs deeper trees than depth 3.
- Log arguments use `|y|` (matches `safe_eml`); the paper's full form uses complex numbers.
- Depth ≥ 3 relies on greedy refinement — quality varies; restart count helps.
