"""
Invariant: every normalized feature must be z-scored CROSS-SECTIONALLY
— against other stocks on the same day — never TIME-SERIES — against
its own full history, which would include dates after the one being
scored.

Why this matters more than the other three: this bug produces a
plausible-looking number, not an obviously broken one. A model
trained on a leaky feature doesn't crash and doesn't look wrong in
a spot check — its rank IC is simply about double what it should be,
and that's exactly the kind of number a hopeful eye rationalizes
instead of investigates. The only defense is a test that checks the
NORMALIZATION METHOD itself, not the model's eventual output.
"""

import pytest
import pandas as pd
import numpy as np


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
) -> pd.Series:
    """
    CORRECT: normalizes each date's cross-section against itself only.
    A stock's z-score on 2020-01-01 depends only on every stock's
    value on 2020-01-01 — never on any other date.
    """
    return df.groupby(date_col)[value_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )


def time_series_zscore_BUGGY(
    df: pd.DataFrame,
    value_col: str,
    symbol_col: str = "symbol",
) -> pd.Series:
    """
    WRONG — kept here only so the test suite has a known-bad example
    to detect. Normalizes each stock against its OWN full history,
    which on any date before the series ends necessarily includes
    future values. Never call this in real feature code.
    """
    return df.groupby(symbol_col)[value_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )


def assert_normalization_is_cross_sectional(
    df: pd.DataFrame,
    value_col: str,
    zscore_col: str,
    date_col: str = "date",
    tolerance: float = 1e-6,
) -> None:
    """
    Given a dataframe with both a raw value column and its claimed
    z-score column, verify the z-scores are consistent with
    cross-sectional (same-date) normalization — not time-series.

    This works by recomputing the correct cross-sectional z-score
    independently and checking it matches what was supplied.
    """
    expected = cross_sectional_zscore(df, value_col, date_col)
    actual = df[zscore_col]

    mismatch = (expected - actual).abs() > tolerance
    assert not mismatch.any(), (
        f"{mismatch.sum()} rows have z-scores inconsistent with "
        f"cross-sectional normalization. This is bug #3 from the "
        f"invariant list — normalization leakage. The scorer is "
        f"very likely normalizing against a stock's own history "
        f"(time-series) instead of against the same-day cross-section."
    )


def _sample_panel() -> pd.DataFrame:
    """3 stocks x 5 days, values chosen so cross-sectional and
    time-series normalization give visibly different results."""
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    data = {
        "TATASTEEL": [10, 12, 50, 11, 13],   # day 3 is a huge outlier
        "RELIANCE":  [20, 21, 22, 23, 24],
        "INFY":      [30, 29, 31, 30, 32],
    }
    rows = []
    for symbol, values in data.items():
        for date, value in zip(dates, values):
            rows.append({"symbol": symbol, "date": date, "raw_value": value})
    return pd.DataFrame(rows)


def test_buggy_time_series_normalization_is_caught():
    """Feed the WRONG normalization in as if it were the real
    pipeline's output — the test must catch it."""
    df = _sample_panel()
    df["zscore"] = time_series_zscore_BUGGY(df, "raw_value")

    with pytest.raises(AssertionError, match="normalization leakage"):
        assert_normalization_is_cross_sectional(df, "raw_value", "zscore")


def test_correct_cross_sectional_normalization_passes():
    """The correct method, used correctly, must pass cleanly."""
    df = _sample_panel()
    df["zscore"] = cross_sectional_zscore(df, "raw_value")

    assert_normalization_is_cross_sectional(df, "raw_value", "zscore")  # should not raise


def test_single_day_outlier_does_not_contaminate_other_days():
    """The defining property of cross-sectional normalization: a
    huge outlier on day 3 should NOT change day 1's z-scores at all.
    Time-series normalization fails this — the outlier shifts every
    date's mean/std for that stock, including days before it happened."""
    df = _sample_panel()
    df["zscore"] = cross_sectional_zscore(df, "raw_value")

    day1 = df[df["date"] == df["date"].min()]
    day1_tatasteel_z = day1[day1["symbol"] == "TATASTEEL"]["zscore"].iloc[0]

    # Recompute day 1 in isolation, as if day 3's outlier never existed
    df_no_outlier = df[df["date"] != df["date"].unique()[2]].copy()
    df_no_outlier["zscore_isolated"] = cross_sectional_zscore(df_no_outlier, "raw_value")
    day1_isolated = df_no_outlier[df_no_outlier["date"] == df_no_outlier["date"].min()]
    day1_tatasteel_isolated = day1_isolated[day1_isolated["symbol"] == "TATASTEEL"]["zscore_isolated"].iloc[0]

    assert abs(day1_tatasteel_z - day1_tatasteel_isolated) < 1e-9, (
        "Day 1's z-score changed depending on whether a LATER day's "
        "outlier was present in the data. Cross-sectional normalization "
        "must be independent of any other date's values — this is "
        "exactly the leakage pathway bug #3 exists to close."
    )