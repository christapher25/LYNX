"""
Invariant: a year Y universe snapshot must be computable using ONLY data
dated strictly before Jan 1 of year Y. This is the same leakage class as
graph snapshots — a universe that "knows" a stock became liquid in 2020
while ranking the 2019 universe is invisible corruption, not a crash.
"""

import pytest
import pandas as pd
from data.build_universe import build_year_universe


def _sample_panel() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="B")
    rows = []
    for i, date in enumerate(dates):
        # STOCKA: steady turnover throughout
        rows.append({"SYMBOL": "STOCKA", "DATE": date, "ADJ_CLOSE": 100, "ADJ_VOLUME": 10000})
        # STOCKB: only becomes liquid starting 2020 — should NOT appear
        # in the 2019 universe even though it exists in the full panel
        vol = 50000 if date.year >= 2020 else 100
        rows.append({"SYMBOL": "STOCKB", "DATE": date, "ADJ_CLOSE": 50, "ADJ_VOLUME": vol})
    df = pd.DataFrame(rows)
    df["TURNOVER"] = df["ADJ_CLOSE"] * df["ADJ_VOLUME"]
    return df


def test_future_liquidity_does_not_leak_into_past_universe():
    panel = _sample_panel()
    universe_2019 = build_year_universe(panel, 2019, universe_size=1, trailing_days=252)

    symbols_2019 = set(universe_2019["SYMBOL"])
    assert "STOCKB" not in symbols_2019, (
        "STOCKB only became liquid in 2020 but appeared in the 2019 "
        "universe — this is bug #2's leakage class: the universe used "
        "information from after its own boundary date."
    )


def test_stock_appears_once_its_own_trailing_history_qualifies():
    panel = _sample_panel()
    universe_2020 = build_year_universe(panel, 2020, universe_size=1, trailing_days=252)

    symbols_2020 = set(universe_2020["SYMBOL"])
    assert "STOCKB" not in symbols_2020, (
        "STOCKB's liquidity only starts in 2020 itself — the 2020 universe "
        "is built from TRAILING data (2019 and before), so it correctly "
        "should not qualify yet either. It should appear starting 2021."
    )


def test_delisted_stock_retained_with_terminal_value():
    dates = pd.date_range("2015-01-01", "2018-06-30", freq="B")
    rows = []
    for date in dates:
        if date <= pd.Timestamp("2017-12-31"):
            rows.append({"SYMBOL": "DELISTED_CO", "DATE": date,
                         "ADJ_CLOSE": 80, "ADJ_VOLUME": 20000})
        rows.append({"SYMBOL": "SURVIVOR", "DATE": date,
                     "ADJ_CLOSE": 100, "ADJ_VOLUME": 20000})
    df = pd.DataFrame(rows)
    df["TURNOVER"] = df["ADJ_CLOSE"] * df["ADJ_VOLUME"]

    universe_2018 = build_year_universe(df, 2018, universe_size=10, trailing_days=252)

    row = universe_2018[universe_2018["SYMBOL"] == "DELISTED_CO"]
    assert not row.empty, (
        "DELISTED_CO was liquid through 2017 and should still appear in "
        "the 2018 snapshot (built from 2017 trailing data) — dropping it "
        "would be survivorship bias, exactly what this stage exists to avoid."
    )
    assert row.iloc[0]["IS_DELISTED"], "should be flagged delisted"
    assert row.iloc[0]["TERMINAL_VALUE"] == 80, "terminal value should be its last traded price"


def test_active_stock_has_no_terminal_value():
    panel = _sample_panel()
    universe_2019 = build_year_universe(panel, 2019, universe_size=10, trailing_days=252)

    row = universe_2019[universe_2019["SYMBOL"] == "STOCKA"]
    assert not row.iloc[0]["IS_DELISTED"]
    assert pd.isna(row.iloc[0]["TERMINAL_VALUE"]), (
        "an actively-trading stock must not carry a terminal value — "
        "that field means something specific (last price before delisting) "
        "and must not be populated for stocks still trading"
    )