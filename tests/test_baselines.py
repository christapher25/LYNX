"""
Stage 14 baselines: tests that need the real model libraries or inspect the implementation source.
catboost and xgboost are NOT installed (adding a dependency needs the user), so the library tests skip with a reason; the
model-agnostic logic is covered with toy models in tests/invariants/vanta/test_baselines_invariants.py.
Interface: docs/vanta/INTERFACES.md section 7.
"""

import re

import numpy as np
import pytest

SEEDS = (0, 1, 2, 3, 4)
ZERO_FILL = re.compile(r"fillna\(\s*0|nan_to_num|np\.where\([^)]*isnan[^)]*,\s*0|SimpleImputer|\.interpolate\(")


def check_source_has_no_zero_fill(src):
    """No-zero-fill (CLAUDE.md): the baselines source never imputes a missing feature or label (fillna(0), nan_to_num, an imputer)."""
    hits = ZERO_FILL.findall(src)
    assert not hits, f"zero-fill / imputation found: {hits}"


def test_source_scan_accepts_clean_and_rejects_zero_fill():
    check_source_has_no_zero_fill("X = frame[cols].to_numpy(float)\ny = y[np.isfinite(y)]\n")
    for bad in ("X = frame.fillna(0)", "X = np.nan_to_num(X)", "imp = SimpleImputer()", "X = df.fillna( 0.0)"):
        with pytest.raises(AssertionError):
            check_source_has_no_zero_fill(bad)


def test_real_baselines_source_has_no_zero_fill():
    import inspect
    mod = pytest.importorskip("vanta.baselines")
    check_source_has_no_zero_fill(inspect.getsource(mod))


def test_b1_params_are_a_frozen_module_constant_without_a_seed():
    """M2_PREREG section 3: B1 hyperparameters are recorded in code before the first OOS run and the seed list is not tuned: the
    parameter dict exists at module level, holds no seed (the seed comes only from B1_SEEDS) and B1_SEEDS is (0,1,2,3,4)."""
    mod = pytest.importorskip("vanta.baselines")
    assert tuple(mod.B1_SEEDS) == SEEDS
    assert isinstance(mod.B1_PARAMS, dict) and mod.B1_PARAMS
    assert not {"random_seed", "random_state", "seed"} & set(mod.B1_PARAMS)


def test_catboost_factory_uses_given_seed_handles_nan_and_is_deterministic():
    """B1: factory(seed) builds a CatBoost model with random_seed == seed; fitting with NaN features does not raise (CatBoost
    handles NaN natively, nothing imputes); same seed twice gives identical predictions; different seeds differ."""
    pytest.importorskip("catboost", reason="catboost is not installed; adding it needs the user")
    mod = pytest.importorskip("vanta.baselines")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] * 0.1 + rng.normal(size=300)
    X[rng.random(X.shape) < 0.2] = np.nan
    f = mod.b1_model_factory(iterations=20)
    a, b, c = f(3), f(3), f(4)
    for m in (a, b, c):
        m.fit(X, y)
    pa, pb, pc = (np.asarray(m.predict(X), float) for m in (a, b, c))
    assert np.isfinite(pa).all()
    np.testing.assert_array_equal(pa, pb)
    assert not np.array_equal(pa, pc)


def test_xgboost_a1_factory_predicts_win_probability_with_nan_features():
    """A1: the factory's model is a binary classifier whose predict() returns P(win) in [0, 1]; NaN features are accepted."""
    pytest.importorskip("xgboost", reason="xgboost is not installed; adding it needs the user")
    mod = pytest.importorskip("vanta.baselines")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(float)
    X[rng.random(X.shape) < 0.2] = np.nan
    m = mod.a1_model_factory(n_estimators=20)(0)
    m.fit(X, y)
    p = np.asarray(m.predict(X), float)
    assert p.shape == (300,) and ((p >= 0) & (p <= 1)).all()
