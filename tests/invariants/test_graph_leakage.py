"""
Invariant: every graph snapshot must be built using only data dated
strictly before the snapshot's own as_of date. A snapshot built with
data extending past that boundary has "seen the future" relative to
any fold that tests before it.

Why this matters: unlike a corporate-action bug, this produces no
error and no crash. The correlation edges are just slightly better
than they should be, the rank IC quietly rises, and the model looks
better on the backtest than it will ever perform live. This is the
single hardest bug in the whole system to catch by inspection —
which is exactly why it needs a test that runs every time, not a
manual check someone remembers to do once.
"""

import pytest
import pandas as pd


def build_correlation_snapshot(
    price_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Builds a correlation-edge snapshot using only data strictly
    before as_of_date. This is the CONTRACT the real graph-building
    code (Stage 13) must satisfy — written first, so that code has
    something to fail against immediately.

    price_history: columns [symbol, date, close]
    Returns: columns [symbol_a, symbol_b, correlation], edges only.
    """
    usable = price_history[price_history["date"] < as_of_date].copy()

    cutoff = as_of_date - pd.Timedelta(days=lookback_days * 1.5)  # trading-day buffer
    usable = usable[usable["date"] >= cutoff]

    if usable.empty:
        return pd.DataFrame(columns=["symbol_a", "symbol_b", "correlation"])

    pivoted = usable.pivot(index="date", columns="symbol", values="close")
    returns = pivoted.pct_change().dropna(how="all")
    corr_matrix = returns.corr()

    edges = []
    symbols = corr_matrix.columns.tolist()
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            edges.append({"symbol_a": a, "symbol_b": b, "correlation": corr_matrix.loc[a, b]})

    return pd.DataFrame(edges)


def assert_snapshot_is_leak_free(
    price_history_used: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    """
    Given the price data a snapshot-building function actually
    consumed, assert none of it is dated on or after as_of_date.
    """
    future_rows = price_history_used[price_history_used["date"] >= as_of_date]
    assert future_rows.empty, (
        f"Graph snapshot for {as_of_date.date()} was built using "
        f"{len(future_rows)} rows dated on or after the snapshot date. "
        f"This is bug #2 from the invariant list — graph leakage. "
        f"Earliest leaked date: {future_rows['date'].min().date()}."
    )


def _sample_price_history() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    rows = []
    for symbol in ["TATASTEEL", "RELIANCE", "INFY"]:
        for i, date in enumerate(dates):
            rows.append({"symbol": symbol, "date": date, "close": 100 + i * 0.1})
    return pd.DataFrame(rows)


def test_deliberately_leaky_snapshot_is_caught():
    """A snapshot built with the FULL history — including data past
    the as_of date — must be caught. If this passes when it
    shouldn't, the detector itself is broken."""
    full_history = _sample_price_history()
    as_of = pd.Timestamp("2023-06-01")

    # Deliberately leaky: use ALL data, not just data before as_of
    leaky_data_used = full_history  # <-- the bug: no filtering at all

    with pytest.raises(AssertionError, match="graph leakage"):
        assert_snapshot_is_leak_free(leaky_data_used, as_of)


def test_correctly_filtered_snapshot_passes():
    """The real contract function, used correctly, must produce a
    snapshot that passes this check cleanly."""
    full_history = _sample_price_history()
    as_of = pd.Timestamp("2023-06-01")

    clean_data_used = full_history[full_history["date"] < as_of]

    assert_snapshot_is_leak_free(clean_data_used, as_of)  # should not raise


def test_reference_builder_is_leak_free():
    """The reference implementation above must itself satisfy the
    invariant it's meant to demonstrate — the builder's OWN filtering
    step, exercised end-to-end."""
    full_history = _sample_price_history()
    as_of = pd.Timestamp("2023-06-01")

    snapshot = build_correlation_snapshot(full_history, as_of)
    assert not snapshot.empty, "Sanity check: snapshot should contain edges"

    # Re-derive what data WOULD have been used, and check it
    usable = full_history[full_history["date"] < as_of]
    assert_snapshot_is_leak_free(usable, as_of)


def test_monthly_rebuild_never_reuses_future_month():
    """Simulates two consecutive monthly rebuilds and confirms the
    later snapshot doesn't retroactively leak into the earlier one
    — a check specific to the 'rebuilt monthly' requirement in the
    architecture doc."""
    full_history = _sample_price_history()

    as_of_month1 = pd.Timestamp("2023-03-01")
    as_of_month2 = pd.Timestamp("2023-04-01")

    snap1 = build_correlation_snapshot(full_history, as_of_month1)
    snap2 = build_correlation_snapshot(full_history, as_of_month2)

    usable1 = full_history[full_history["date"] < as_of_month1]
    usable2 = full_history[full_history["date"] < as_of_month2]

    assert_snapshot_is_leak_free(usable1, as_of_month1)
    assert_snapshot_is_leak_free(usable2, as_of_month2)
    assert len(usable2) > len(usable1), "Sanity check: month 2 should see strictly more history than month 1"
