"""
tests/invariants/vanta/test_vanta_regime.py

VANTA Stage 13B invariant N7 (docs/vanta/VANTA_SPEC.md):

  N7  Regime weights use filtered posteriors   the posterior at t computed on data up to t equals the posterior
                                               computed on the full series. Appending the future changes nothing.

Real module (vanta.regime, exists), a causality checker, and deliberately broken variants that the checker must
reject: a smoothed (forward-backward) posterior, and a filter that standardises with full-series statistics.
Also: a NaN observation is never zero-filled, and the fit uses rows <= train_end only.
"""

import numpy as np
import pandas as pd
import pytest

from vanta import regime as R

FEATS = list(R.FEATURES)
SAMPLE_T = (30, 77, 150, 233, 299)


def _series(n=420, seed=7):
    """Synthetic 3-regime features with persistent regimes, so smoothing visibly differs from filtering."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.5], [1.5, -3.0, -1.5, 1.0]])
    states = np.zeros(n, int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.96 else rng.integers(0, 3)
    X = centres[states] + rng.normal(0, 0.8, size=(n, 4))
    return pd.DataFrame(X, index=pd.bdate_range("2015-01-01", periods=n), columns=FEATS)


@pytest.fixture(scope="module")
def data():
    df = _series()
    params = R.fit_hmm(df, train_end=df.index[199])
    return df, params


def smoothed_BUGGY(params, X):
    """WRONG: forward-backward posterior, conditions on the whole sequence."""
    Z = R._standardise(params, R._as_array(params, X))
    logb = R._log_emissions(Z, params.means, params.variances)
    gamma, _, _ = R._e_step(params.startprob, params.transmat, logb)
    return gamma


def full_series_standardised_BUGGY(params, X):
    """WRONG: standardises with statistics of the series passed in (look-ahead via the mean/std)."""
    A = R._as_array(params, X)
    leaky = R.RegimeHMM(**{**params.__dict__, "mu": np.nanmean(A, axis=0), "sigma": np.nanstd(A, axis=0)})
    return R.filter_posterior(leaky, X)


def causality_gap(fn, params, df, ts=SAMPLE_T):
    """Max |posterior_t(full series) - posterior_t(series truncated at t)|."""
    full = fn(params, df)
    return max(float(np.abs(full[t] - fn(params, df.iloc[: t + 1])[t]).max()) for t in ts)


def test_n7_filtered_posterior_is_causal(data):
    df, params = data
    assert causality_gap(R.filter_posterior, params, df) == 0.0


def test_n7_appending_the_future_changes_nothing(data):
    df, params = data
    short = R.filter_posterior(params, df.iloc[:250])
    longer = R.filter_posterior(params, df)
    assert np.array_equal(short, longer[:250])


def test_n7_checker_rejects_smoothed_posterior(data):
    df, params = data
    assert causality_gap(smoothed_BUGGY, params, df) > 1e-3


def test_n7_checker_rejects_full_series_standardisation(data):
    df, params = data
    assert causality_gap(full_series_standardised_BUGGY, params, df) > 1e-6


def test_filtered_differs_from_smoothed_so_the_guard_is_meaningful(data):
    df, params = data
    diff = np.abs(R.filter_posterior(params, df) - smoothed_BUGGY(params, df)).max()
    assert diff > 0.05


def test_posterior_is_a_probability_vector(data):
    df, params = data
    p = R.filter_posterior(params, df)
    assert p.shape == (len(df), 3)
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.0)


def test_nan_observation_is_not_zero_filled(data):
    df, params = data
    t = 120
    nan_df = df.copy()
    nan_df.iloc[t] = np.nan
    zero_df = df.copy()
    zero_df.iloc[t] = 0.0
    p_nan = R.filter_posterior(params, nan_df)
    p_zero = R.filter_posterior(params, zero_df)
    # all-NaN row: transition step only
    assert np.allclose(p_nan[t], p_nan[t - 1] @ params.transmat)
    assert not np.allclose(p_nan[t], p_zero[t])
    assert np.isfinite(p_nan).all()


def test_partially_missing_feature_is_marginalised_not_zero_filled(data):
    df, params = data
    t = 120
    a = df.copy()
    a.iloc[t, 1] = np.nan
    b = df.copy()
    b.iloc[t, 1] = 0.0  # a real, in-range-looking value that would be a lie if substituted
    assert not np.allclose(R.filter_posterior(params, a)[t], R.filter_posterior(params, b)[t])


def test_fit_uses_training_rows_only(data):
    df, params = data
    tampered = df.copy()
    tampered.iloc[200:] = tampered.iloc[200:] * 50 + 1000
    p2 = R.fit_hmm(tampered, train_end=df.index[199])
    for name in ("startprob", "transmat", "means", "variances", "mu", "sigma", "lo", "hi"):
        assert np.array_equal(getattr(params, name), getattr(p2, name)), name


_FIT_FIELDS = ("startprob", "transmat", "means", "variances", "mu", "sigma", "lo", "hi")


def _same_fit(a, b):
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in _FIT_FIELDS)


def test_fit_boundary_train_end_is_inclusive_and_next_row_excluded(data):
    df, params = data
    end = df.index[199]
    at_end = df.copy()
    at_end.iloc[199] = at_end.iloc[199] + 5.0
    after_end = df.copy()
    after_end.iloc[200] = after_end.iloc[200] + 5.0
    assert not _same_fit(params, R.fit_hmm(at_end, train_end=end))  # row AT train_end is used
    assert _same_fit(params, R.fit_hmm(after_end, train_end=end))  # row train_end+1 is not


def test_fit_boundary_mutants_are_detected(data):
    """A fit using `< train_end` (one row short) or `<= train_end + 1 row` must differ from the real fit."""
    df, params = data
    assert not _same_fit(params, R.fit_hmm(df, train_end=df.index[198]))
    assert not _same_fit(params, R.fit_hmm(df, train_end=df.index[200]))


def test_fold_train_end_respects_embargo_in_sessions():
    sessions = pd.bdate_range("2020-01-01", periods=300)
    test_start = sessions[200]
    end = R.fold_train_end(sessions, test_start, embargo_sessions=20)
    gap = sessions.searchsorted(test_start) - sessions.searchsorted(end) - 1
    assert gap == 20
    with pytest.raises(ValueError):
        R.fold_train_end(sessions, test_start, embargo_sessions=0)


def test_states_ordered_calm_to_stressed(data):
    _, params = data
    assert np.all(np.diff(params.means[:, 0]) > 0)
