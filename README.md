# emlfit

Symbolic regression on binary trees. Internal nodes pick an operator from a
configurable set (`eml`, `add`, `sub`, `mul`); leaves pick an input feature or
a clean constant. Inspired by the EML operator of Odrzywołek (2026), *All
elementary functions from a single operator* (arXiv:2603.21852):

```
eml(x, y) = exp(x) − ln(y)
```

Shallow search spaces are enumerated exhaustively with batched GPU-friendly
evaluation; deeper trees fall back to beam search seeded by a gradient
warm-up.

## Install

```bash
uv venv
uv pip install -e .
```

## Usage

```python
import numpy as np
from emlfit import EMLRegressor

r = np.linspace(0.3, 2.5, 200).reshape(-1, 1)
y = np.pi * r[:, 0] ** 2

model = EMLRegressor(depth=2, ops=("eml", "mul")).fit(r, y)
print(model.formula(input_names=["r"]))  # pi*r**2
print(model.score(r, y))                 # 1.000000
```

## What it finds

| Target              | Depth | Ops                   | Result             | Time |
| ------------------- | ----- | --------------------- | ------------------ | ---- |
| `exp(x)`            | 1     | `(eml,)`              | `exp(x)`           | <1s  |
| `exp(x) - log(x)`   | 1     | `(eml,)`              | `exp(x) - log|x|`  | <1s  |
| `exp(exp(x))`       | 2     | `(eml,)`              | `exp(exp(x))`      | ~1s  |
| `π*r²`              | 2     | `(eml, mul)`          | `pi*r**2`          | ~30s |
| `2π*r`              | 2     | `(eml, mul)`          | `2*pi*r`           | ~30s |
| `a² - b²`           | 2     | `(mul, add, sub)`     | `(a-b)*(a+b)`      | ~2m  |
| `a² + b² + c²`      | 3     | `(mul, add)`          | partial (R²≈0.73)  | ~1m  |
| `2πr` at depth 3    | 3     | eml-only              | — not reachable    |      |

## How it works

1. **Gradient warm-up** (Adam): trains a soft tree where each leaf is a softmax mixture of inputs and a learnable constant, and each internal node is a softmax mixture of the allowed operators.
2. **Discrete refinement**: 
   - `brute_force`: exhausts (ops^n_internal) × (candidates^n_leaves) configs when below the budget. Batched in groups of ~4k and evaluated in a single tensor op.
   - `beam_search`: maintains top-K candidates, expands by single-slot changes (leaf or op). Seeded from argmax snap + random restarts. Used when brute force is infeasible (depth ≥ 3 with mixed ops).
   - `greedy` polish follows beam search.
3. **Symbolic extraction** (`sympy`): folds discrete leaves through `exp(x) - log(|y|)` for EML nodes, or the direct arithmetic op. Constants snap to `{0, ±1, ±2, ½, e, π, π/2, 2π, 1/e}` when close.

## Candidate pool

Default constants tried at every leaf: `{0, 1, -1, 2, ½, e, π, 2π}`. Kept tight so brute force at depth 2 stays under 5M configurations.

## Limitations

- **Depth 3+ with complex targets** (e.g. `a²+b²+c²`) — beam search's single-slot neighborhood gets stuck in local minima. A full evolutionary search (PySR-style) would do better; planned.
- **Log arguments use `|y|`**, mirroring `safe_eml`. The paper's full construction uses complex numbers; we stay real for regression use-cases.
- **High dimensions** — candidate pool grows linearly with input count; depth ≤ 2 brute force still tractable up to ~8 inputs.
- **No noise handling yet** — constants snap by absolute tolerance, which is brittle near zero in noisy data.

## Roadmap

- [ ] Evolutionary search for depth ≥ 3 (crossover between beam members)
- [ ] Bootstrap: treat discovered sub-trees (`e`, `ln(x)`, `x²`) as leaves for the next search
- [ ] Noise-aware constant snapping
- [ ] Feynman Symbolic Regression Benchmark comparison vs PySR
