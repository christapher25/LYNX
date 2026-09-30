"""
tests/test_indicators.py

Invariant tests for signals/indicators.py — Stage 9. Covers the two failure
modes that matter for a technical-indicator engine: silently wrong values
(bounds violations) and silently missing-vs-zero confusion (an indicator
reading 0 when it should read "not yet computable" is a lie a downstream
model will happily learn from).
"""

import numpy as np
import pandas as pd
import pytest

from signals.indicators import (
    INDICATOR_COLUMNS,
    MIN_ROWS_REQUIRED,
    compute_indicators,
    prepare_ohlcv,
)


def _synthetic_ohlcv(n: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestInputValidation:
    def test_missing_column_raises(self):
        df = _synthetic_ohlcv().drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_indicators(df)

    def test_unsorted_dates_raise(self):
        """
        An input silently sorted wrong corrupts every rolling/ewm computation
        in this module without raising on its own — this is the specific
        failure mode this check exists to catch before it happens.
        """
        df = _synthetic_ohlcv().iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="sorted ascending"):
            compute_indicators(df)


class TestNoZeroFilling:
    """
    Every indicator must be NaN before its lookback window has filled, never
    zero — a zero-filled RSI reads as "maximally oversold" and a zero-filled
    EMA(200) reads as "price is infinitely above its 200-day average," both
    of which a downstream model would learn from as if they were real.
    """

    def test_ema_200_is_nan_before_200_rows_populated_after(self):
        out = compute_indicators(_synthetic_ohlcv(n=260))
        assert out["ema_200"].iloc[:199].isna().all()
        assert out["ema_200"].iloc[199:].notna().all()

    def test_rsi_14_is_nan_at_series_start(self):
        out = compute_indicators(_synthetic_ohlcv(n=260))
        assert out["rsi_14"].iloc[0:5].isna().all()
        assert out["rsi_14"].iloc[-1:].notna().all()

    def test_short_history_below_min_rows_still_computes_without_error(self):
        """
        MIN_ROWS_REQUIRED is a pipeline-level gate (the driver skips symbols
        below it), not something compute_indicators itself enforces — it
        must still run cleanly on short input and simply return more NaNs,
        since the pure function has no business knowing about the pipeline's
        skip policy.
        """
        short = _synthetic_ohlcv(n=50)
        assert len(short) < MIN_ROWS_REQUIRED
        out = compute_indicators(short)  # must not raise
        assert len(out) == len(short)
        assert out["ema_200"].isna().all()  # correctly never populated


def _monotone_ohlcv(step: float, n: int = 60) -> pd.DataFrame:
    """Close moves by exactly `step` every day: step > 0 -> zero losses, step == 0 -> flat."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + step * np.arange(n)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.full(n, 100_000.0),
    })


class TestRsiZeroLossWindow:
    """
    avg_loss == 0 is not "undefined". With any gain it means RS -> infinity, so RSI is 100
    (maximally overbought). Emitting NULL there hides the most extreme overbought readings
    from the scorer's `rsi > 70` rule. NULL is correct only when avg_gain and avg_loss are
    BOTH 0 (a flat window), where RSI genuinely has no value.
    """

    def test_rsi_is_100_when_window_has_gains_and_no_losses(self):
        out = compute_indicators(_monotone_ohlcv(step=0.5))
        assert out["rsi_14"].iloc[:14].isna().all()  # warm-up stays NULL, not 100
        assert (out["rsi_14"].iloc[14:] == 100.0).all()

    def test_rsi_is_null_not_a_number_when_window_is_flat(self):
        out = compute_indicators(_monotone_ohlcv(step=0.0))
        assert out["rsi_14"].isna().all()  # 0/0 is undefined; never 0, 50 or 100


class TestCausality:
    """
    An indicator at day t may depend only on days <= t. Adding future rows must not change any
    past value: a centred window, a backward fill or a full-history normalisation would all
    change past rows once later data arrives, and would make a backtest look better than live.
    """

    @pytest.mark.parametrize("cut", [120, 230, 300, 399])
    def test_past_values_unchanged_when_future_rows_are_added(self, cut):
        df = _synthetic_ohlcv(n=400)  # long enough that ema_200 and the 52w window are live
        full = compute_indicators(df)
        truncated = compute_indicators(df.iloc[:cut])
        # assert_frame_equal also requires the NULL pattern to match, not just the numbers.
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True), full.iloc[:cut].reset_index(drop=True),
            check_exact=False, rtol=1e-9, atol=1e-9,
        )


class TestValueBounds:
    def test_rsi_bounded_0_100(self):
        out = compute_indicators(_synthetic_ohlcv())
        rsi = out["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_position_clipped_0_1(self):
        out = compute_indicators(_synthetic_ohlcv())
        bb = out["bb_position"].dropna()
        assert (bb >= 0).all() and (bb <= 1).all()

    def test_all_declared_indicator_columns_present(self):
        out = compute_indicators(_synthetic_ohlcv())
        for col in INDICATOR_COLUMNS:
            assert col in out.columns, f"{col} declared in INDICATOR_COLUMNS but not produced"


def _stage4_frame() -> pd.DataFrame:
    """Shaped like a data/processed/adjusted/{SYMBOL}.parquet file, raw != adjusted."""
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    return pd.DataFrame({
        "SYMBOL": "TEST",
        "DATE": dates,
        "OPEN": [200.0, 202.0, 204.0, 206.0], "ADJ_OPEN": [100.0, 101.0, 102.0, 103.0],
        "HIGH": [210.0, 212.0, 214.0, 216.0], "ADJ_HIGH": [105.0, 106.0, 107.0, 108.0],
        "LOW": [190.0, 192.0, 194.0, 196.0], "ADJ_LOW": [95.0, 96.0, 97.0, 98.0],
        "CLOSE": [205.0, 207.0, 209.0, 211.0], "ADJ_CLOSE": [102.0, 103.0, 104.0, 105.0],
        "VOLUME": [1_000, 1_000, 1_000, 1_000], "ADJ_VOLUME": [2_000.0, 2_000.0, 2_000.0, 2_000.0],
    })


class TestPrepareOhlcv:
    def test_uses_adjusted_columns_never_raw(self):
        """
        Raw VOLUME/prices aren't split/bonus-adjusted. Feeding them in would make
        volume_ratio, OBV and every price indicator jump at each corporate action.
        """
        out = prepare_ohlcv(_stage4_frame())
        assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert out["close"].tolist() == [102.0, 103.0, 104.0, 105.0]
        assert out["volume"].tolist() == [2_000.0] * 4

    def test_non_positive_prices_become_nan_not_zero(self):
        """
        The raw archive records a missing open/high/low as 0. A zero LOW would pin
        the 52-week low at 0; it must reach compute_indicators as NaN.
        """
        raw = _stage4_frame()
        raw.loc[1, "ADJ_LOW"] = 0.0
        raw.loc[2, "ADJ_OPEN"] = 0.0
        out = prepare_ohlcv(raw)
        assert np.isnan(out.loc[1, "low"]) and np.isnan(out.loc[2, "open"])
        assert not (out[["open", "high", "low", "close"]] <= 0).any().any()
        assert out.loc[0, "low"] == 95.0  # untouched neighbours

    def test_sorts_ascending_by_date(self):
        out = prepare_ohlcv(_stage4_frame().iloc[::-1])
        assert out["date"].is_monotonic_increasing


class TestRowCountPreserved:
    def test_output_row_count_matches_input(self):
        df = _synthetic_ohlcv(n=260)
        out = compute_indicators(df)
        assert len(out) == len(df)
        assert (out["date"] == df["date"]).all()