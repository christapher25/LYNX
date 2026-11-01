"""
Stage 7 invariant: each rebalance month is traded with ITS OWN calendar year's point-in-time universe, once.

The legacy main() ran every year's universe over every month of that universe's history, producing ~21 overlapping
copies of ~369 months (7,749 rebalances instead of ~252). These tests record what backtest_month is actually called
with, so they test behaviour, not the return value. Set LYNX_STAGE7_MODULE_PATH to run them against another copy of the
module (used to prove the tests fail on the old code).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = Path(os.environ.get("LYNX_STAGE7_MODULE_PATH", ROOT / "eval" / "momentum_baseline.py"))

UNIVERSES = {
    2005: {f"A{i}" for i in range(12)},
    2006: {f"B{i}" for i in range(12)} | {f"A{i}" for i in range(3)},
    2007: {f"C{i}" for i in range(12)},
}


def _load():
    spec = importlib.util.spec_from_file_location("stage7_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panel(symbols: set[str]) -> pd.DataFrame:
    months = pd.date_range("2003-01-31", "2008-03-31", freq="ME")
    rng = np.random.default_rng(0)
    rows = []
    for s in sorted(symbols):
        px = 100 * np.cumprod(1 + rng.normal(0.01, 0.05, len(months)))
        rows.append(pd.DataFrame({"SYMBOL": s, "DATE": months, "ADJ_CLOSE": px}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def recorded(monkeypatch):
    mod = _load()
    calls = []

    def fake_backtest(panel_month, next_month_prices, as_of, exit_date):
        calls.append({"as_of": as_of, "exit": exit_date, "picked_from": set(panel_month["SYMBOL"]),
                      "next": set(next_month_prices["SYMBOL"])})
        return {"date": as_of, "n_long": 1, "n_short": 1, "gross_spread": 0.01,
                "avg_round_trip_cost": 0.002, "net_spread": 0.008}

    monkeypatch.setattr(mod, "load_universe_year", lambda y: set(UNIVERSES.get(y, set())))
    monkeypatch.setattr(mod, "load_monthly_panel", _panel)
    monkeypatch.setattr(mod, "backtest_month", fake_backtest)
    mod.main(2005, 2007)
    return calls


def test_no_month_is_rebalanced_twice(recorded):
    dates = [c["as_of"] for c in recorded]
    assert len(dates) == len(set(dates)), f"{len(dates) - len(set(dates))} duplicated rebalance months"


def test_twelve_rebalances_per_year_and_no_months_outside_range(recorded):
    years = [c["as_of"].year for c in recorded]
    assert sorted(set(years)) == [2005, 2006, 2007]
    for y in (2005, 2006, 2007):
        assert years.count(y) == 12, f"{y}: {years.count(y)} rebalances"
    assert len(recorded) == 36


def test_each_month_uses_its_own_years_universe(recorded):
    for c in recorded:
        allowed = UNIVERSES[c["as_of"].year]
        assert c["picked_from"] <= allowed, (c["as_of"], sorted(c["picked_from"] - allowed))
        assert c["picked_from"], c["as_of"]


def test_exit_is_the_next_month_end(recorded):
    for c in recorded:
        assert (pd.Timestamp(c["exit"]) - pd.Timestamp(c["as_of"])).days in range(28, 32)
