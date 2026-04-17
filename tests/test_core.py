import numpy as np
import torch

from emlfit import EMLRegressor, EMLTree, safe_eml


def test_safe_eml_matches_definition():
    from emlfit.core import LOG_EPS
    x = torch.tensor([0.0, 1.0, -0.5])
    y = torch.tensor([1.0, 2.0, 0.5])
    out = safe_eml(x, y).numpy()
    expected = np.exp(x.numpy()) - np.log(np.abs(y.numpy()) + LOG_EPS)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_tree_shapes():
    tree = EMLTree(depth=3, n_inputs=2)
    X = torch.randn(16, 2)
    y = tree(X)
    assert y.shape == (16,)


def test_regressor_fits_linear_exp():
    # eml(x, 1) = exp(x): should be discoverable at depth 1
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, 128).reshape(-1, 1).astype(np.float32)
    y = np.exp(x[:, 0])
    model = EMLRegressor(depth=1, epochs=1500, n_restarts=3).fit(x, y)
    assert model.score(x, y) > 0.95


def test_brute_force_recovers_clean_exp_formula():
    # Brute force at depth 1 should find leaves = [x, 1] exactly.
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 0.5, 64).reshape(-1, 1).astype(np.float32)
    y = np.exp(x[:, 0]).astype(np.float32)
    model = EMLRegressor(depth=1, epochs=200, n_restarts=1).fit(x, y)
    assert model.score(x, y) > 0.999
    leaves = model.leaves_
    kinds = {leaf["kind"] for leaf in leaves}
    assert "input" in kinds  # one leaf is x
    # The constant leaf should snap to 1 (via brute force candidate set).
    const_leaf = next(leaf for leaf in leaves if leaf["kind"] == "const")
    assert abs(const_leaf["value"] - 1.0) < 0.01


def test_mixed_ops_recover_pi_r_squared():
    # With mul enabled, pi*r^2 = mul(mul(pi, 1), mul(r, r)) at depth 2.
    r = np.linspace(0.3, 2.5, 128).reshape(-1, 1).astype(np.float32)
    y = (np.pi * r[:, 0] ** 2).astype(np.float32)
    model = EMLRegressor(
        depth=2, ops=("eml", "mul"), epochs=300, n_restarts=1
    ).fit(r, y)
    assert model.score(r, y) > 0.999
