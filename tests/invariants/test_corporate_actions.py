"""
Invariant: no adjusted daily return may exceed the price band in
force on that date without a matching corporate-action record.

Why this matters: an unadjusted bonus/split shows up as a fake ±50%
"return." The ranker will happily learn to predict bonus calendars
instead of anything real, and nothing will look wrong until someone
notices the model is suspiciously good at picking stocks right before
splits.

This test does not check real data yet — Stage 4 (the corporate
action engine) doesn't exist. It defines the CONTRACT that engine
must satisfy, written first, so the engine has something to fail
against the moment it's built.
"""

import pytest
import pandas as pd
import numpy as np


def assert_no_unexplained_jumps(
    adjusted_prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    max_band_pct: float = 0.20,
) -> None:
    """
    adjusted_prices: columns [symbol, date, close]
    corporate_actions: columns [symbol, date, action_type, ratio]

    Raises AssertionError on the first unexplained jump found.
    """
    for symbol, group in adjusted_prices.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        daily_return = group["close"].pct_change()

        suspicious = group[daily_return.abs() > max_band_pct]

        for _, row in suspicious.iterrows():
            has_action = (
                (corporate_actions["symbol"] == symbol)
                & (corporate_actions["date"] == row["date"])
            ).any()
            assert has_action, (
                f"{symbol} on {row['date']}: return "
                f"{daily_return.loc[row.name]:.1%} exceeds {max_band_pct:.0%} "
                f"band with no matching corporate action record. "
                f"This is bug #1 from the invariant list — unadjusted "
                f"corporate action."
            )


def test_unadjusted_bonus_is_caught():
    """A 1:1 bonus with no adjustment must fail this test.
    If this test passes when it shouldn't, the detector is broken."""
    prices = pd.DataFrame({
        "symbol": ["TATASTEEL"] * 3,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "close": [150.0, 150.0, 75.0],  # unadjusted 1:1 bonus on day 3
    })
    empty_actions = pd.DataFrame(columns=["symbol", "date", "action_type", "ratio"])

    with pytest.raises(AssertionError, match="unadjusted corporate action"):
        assert_no_unexplained_jumps(prices, empty_actions)


def test_adjusted_bonus_passes():
    """The same bonus, properly adjusted, must pass cleanly."""
    prices = pd.DataFrame({
        "symbol": ["TATASTEEL"] * 3,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "close": [150.0, 150.0, 148.0],  # adjusted — no discontinuity
    })
    empty_actions = pd.DataFrame(columns=["symbol", "date", "action_type", "ratio"])

    assert_no_unexplained_jumps(prices, empty_actions)  # should not raise


def test_documented_action_is_allowed():
    """A real jump WITH a matching action record must not be flagged —
    the test shouldn't cry wolf on legitimate large moves."""
    prices = pd.DataFrame({
        "symbol": ["TATASTEEL"] * 2,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "close": [150.0, 75.0],
    })
    actions = pd.DataFrame({
        "symbol": ["TATASTEEL"],
        "date": pd.to_datetime(["2024-01-02"]),
        "action_type": ["bonus"],
        "ratio": ["1:1"],
    })
    assert_no_unexplained_jumps(prices, actions)  # should not raise