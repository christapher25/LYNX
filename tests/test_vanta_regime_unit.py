"""Unit tests for vanta.regime: availability rule, fit determinism, and the real-data feature loader's shape."""

import numpy as np
import pandas as pd
import pytest

from vanta import regime as R


def _df(n=60, vix_from=10):
    idx = pd.bdate_range("2014-05-14", periods=n)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 4)), index=idx, columns=list(R.FEATURES))
    df.iloc[:vix_from, 0] = np.nan
    return df


def test_first_available_is_warmup_after_first_vix():
    df = _df()
    assert R.first_available(df, warmup=20) == df.index[30]
    assert R.first_available(df.iloc[:25], warmup=20) is None


def test_frame_is_nan_before_availability_and_finite_after():
    df = _df(n=200)
    params = R.fit_hmm(df, train_end=df.index[149])
    f = R.filtered_posterior_frame(params, df, warmup=20)
    start = R.first_available(df, 20)
    assert f.loc[f.index < start].isna().all().all()
    assert f.loc[f.index >= start].notna().all().all()


def test_fit_is_deterministic():
    df = _df(n=200, vix_from=0)
    a = R.fit_hmm(df, train_end=df.index[149])
    b = R.fit_hmm(df, train_end=df.index[149])
    assert np.array_equal(a.transmat, b.transmat) and np.array_equal(a.means, b.means)


def test_too_few_rows_raises():
    df = _df(n=60, vix_from=0)
    with pytest.raises(ValueError):
        R.fit_hmm(df, train_end=df.index[10])
