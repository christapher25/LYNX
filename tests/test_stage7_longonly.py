"""Tests for scripts/stage7_longonly.py (queue row 15a): long-only top decile next to the Stage 7 long-short."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import stage7_longonly as lo  # noqa: E402


def _month_panel(n=20, nan_symbol=None):
    """n symbols S00..S{n-1}; momentum rises with the index, so S{n-1} is the top."""
    syms = [f"S{i:02d}" for i in range(n)]
    return pd.DataFrame({
        "SYMBOL": syms,
        "DATE": pd.Timestamp("2020-01-31"),
        "ADJ_CLOSE": 100.0,
        "MOMENTUM_1M_LAG": np.arange(n, dtype=float),
    })


def _next(panel, rets):
    return pd.DataFrame({"SYMBOL": panel["SYMBOL"], "ADJ_CLOSE": 100.0 * (1 + np.asarray(rets, dtype=float))})


def test_top_decile_membership():
    p = _month_panel(20)
    assert lo.top_decile(p) == ["S19", "S18"]
    # shuffled input order must not matter
    assert lo.top_decile(p.sample(frac=1, random_state=1)) == ["S19", "S18"]


def test_top_decile_matches_stage7_longs():
    p = _month_panel(37)
    ranked = p.sort_values("MOMENTUM_1M_LAG", ascending=False)
    assert lo.top_decile(p) == ranked.head(max(1, int(37 * 0.10)))["SYMBOL"].tolist()


def test_long_only_month_gross_and_cost_applied():
    p = _month_panel(20)
    rets = np.zeros(20)
    rets[19], rets[18] = 0.10, 0.02
    r = lo.long_only_month(p, _next(p, rets), date(2020, 1, 31), date(2020, 2, 29), prev_longs=None)
    assert r["gross"] == pytest.approx(0.06)
    assert r["rt_cost"] > 0
    assert r["net_roundtrip"] == pytest.approx(r["gross"] - r["rt_cost"])


def test_cost_scales_with_turnover():
    # full overlap: nothing traded; half overlap: half; no overlap/first month: all.
    assert lo.turnover_fraction(["A", "B"], ["A", "B"]) == 0.0
    assert lo.turnover_fraction(["A", "B"], ["A", "C"]) == 0.5
    assert lo.turnover_fraction(["A", "B"], ["C", "D"]) == 1.0
    assert lo.turnover_fraction(None, ["C", "D"]) == 1.0
    p = _month_panel(20)
    nxt = _next(p, np.zeros(20))
    full = lo.long_only_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29), prev_longs=None)
    held = lo.long_only_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29), prev_longs=["S19", "S18"])
    assert held["turnover"] == 0.0 and held["net_turnover"] == pytest.approx(held["gross"])
    assert full["net_turnover"] == pytest.approx(full["gross"] - full["rt_cost"])


def test_missing_exit_price_is_dropped_not_zero_filled():
    p = _month_panel(20)
    nxt = _next(p, np.zeros(20))
    nxt.loc[nxt["SYMBOL"] == "S19", "ADJ_CLOSE"] = np.nan
    nxt.loc[nxt["SYMBOL"] == "S18", "ADJ_CLOSE"] = 110.0
    r = lo.long_only_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29), prev_longs=None)
    assert r["n"] == 1 and r["n_dropped"] == 1
    assert r["gross"] == pytest.approx(0.10)       # a zero-fill would give 0.05


def test_all_missing_returns_none():
    p = _month_panel(20)
    nxt = _next(p, np.zeros(20))
    nxt["ADJ_CLOSE"] = np.nan
    assert lo.long_only_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29), prev_longs=None) is None


def test_long_short_reproduces_stage7_backtest_month():
    """Our both-legs function must equal eval.momentum_baseline.backtest_month exactly."""
    rng = np.random.default_rng(0)
    p = _month_panel(40)
    nxt = _next(p, rng.normal(0.01, 0.08, 40))
    mine = lo.long_short_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29))
    ref = lo.stage7.backtest_month(p, nxt, date(2020, 1, 31), date(2020, 2, 29))
    assert mine["net_spread"] == pytest.approx(ref["net_spread"], abs=1e-15)
    assert mine["gross_spread"] == pytest.approx(ref["gross_spread"], abs=1e-15)


def test_no_lookahead_signal_uses_only_data_up_to_t():
    """Changing prices after month t must not change the month-t selection."""
    idx = pd.date_range("2018-01-31", periods=30, freq="ME")
    rows = []
    rng = np.random.default_rng(3)
    for s in range(15):
        px = 100 * np.cumprod(1 + rng.normal(0.01, 0.05, len(idx)))
        rows += [(f"S{s}", d, p) for d, p in zip(idx, px)]
    panel = pd.DataFrame(rows, columns=["SYMBOL", "DATE", "ADJ_CLOSE"])
    t = idx[20]
    base = lo.stage7.compute_momentum_signal(panel.copy())
    mutated = panel.copy()
    mutated.loc[mutated["DATE"] > t, "ADJ_CLOSE"] *= rng.uniform(0.5, 2, (mutated["DATE"] > t).sum())
    sig2 = lo.stage7.compute_momentum_signal(mutated)
    a = lo.top_decile(base[base["DATE"] == t])
    b = lo.top_decile(sig2[sig2["DATE"] == t])
    assert a == b


def test_causality_by_truncation_of_series():
    """Series for months up to T computed on truncated vs full data must agree."""
    rng = np.random.default_rng(5)
    months = pd.date_range("2015-01-31", periods=60, freq="ME")
    rows = []
    for s in range(30):
        px = 100 * np.cumprod(1 + rng.normal(0.01, 0.06, len(months)))
        rows += [(f"S{s:02d}", d, p) for d, p in zip(months, px)]
    panel = pd.DataFrame(rows, columns=["SYMBOL", "DATE", "ADJ_CLOSE"])
    full = lo.run_panel(panel)
    cut_date = months[40]
    part = lo.run_panel(panel[panel["DATE"] <= cut_date])
    common = part.index
    pd.testing.assert_frame_equal(full.loc[common].iloc[:-1], part.iloc[:-1], check_exact=False)


def test_stats_basic():
    x = np.array([0.01, -0.02, 0.03, 0.0, 0.02, 0.01])
    s = lo.series_stats(x, n_boot=200, seed=1)
    assert s["ann_return"] == pytest.approx(x.mean() * 12)
    assert s["hit_rate"] == pytest.approx(4 / 6)
    assert s["vol"] == pytest.approx(x.std(ddof=1) * np.sqrt(12))
    assert s["max_dd"] <= 0
    assert s["nw_ci95"][0] < s["ann_return"] < s["nw_ci95"][1]


def test_max_drawdown_known():
    assert lo.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert lo.max_drawdown(np.array([0.1, 0.1])) == 0.0


# ---- review fixes (15a findings) ------------------------------------------------------------------------------------
def _fake_res(year, months, longs_fn=lambda i: ["A", "B"]):
    idx = pd.to_datetime([f"{year}-{m:02d}-28" for m in months]) + pd.offsets.MonthEnd(0)
    return pd.DataFrame({"lo_gross": 0.02, "lo_rt_cost": 0.01, "lo_turnover": 99.0, "lo_net_turn": 99.0,
                         "longs": [longs_fn(i) for i in range(len(idx))]}, index=idx)


def test_benchmark_is_forward_month_return():
    # Index closes at end of Jan/Feb/Mar: 100, 110, 121. Strategy row indexed Jan earns Jan-end -> Feb-end = +10%.
    nif = pd.Series([100.0, 110.0, 99.0], index=pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M"))
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    out = lo.align_benchmark(nif, idx)
    assert out.iloc[0] == pytest.approx(0.10)          # misaligned pct_change()[Jan] would be NaN
    assert out.iloc[1] == pytest.approx(99 / 110 - 1)  # misaligned would be +10%
    assert list(out.index) == list(idx)


def test_benchmark_missing_month_is_nan_not_zero():
    nif = pd.Series([100.0, 110.0], index=pd.PeriodIndex(["2020-01", "2020-02"], freq="M"))
    out = lo.align_benchmark(nif, pd.to_datetime(["2020-02-29", "2020-06-30"]))
    assert out.isna().all()


def test_finalize_turnover_year_boundary():
    a = _fake_res(2020, [11, 12], lambda i: ["A", "B"])
    b = _fake_res(2021, [1, 2], lambda i: ["A", "C"] if i == 0 else ["A", "C"])
    df = lo.finalize_turnover(pd.concat([b, a]))              # unsorted input
    assert list(df["lo_turnover"]) == [1.0, 0.0, 0.5, 0.0]    # first month 1; Dec->Jan changes half the book
    assert df["lo_net_turn"].iloc[0] == pytest.approx(0.02 - 0.01)
    assert df["lo_net_turn"].iloc[1] == pytest.approx(0.02)
    assert df["lo_net_turn"].iloc[2] == pytest.approx(0.02 - 0.5 * 0.01)


def test_clean_series_unique_months_each_from_own_year():
    # Legacy: every year's universe covers all months of both years. Clean must take month m only from year(m)'s run.
    parts = {}
    for y in (2005, 2006):
        r = pd.concat([_fake_res(2005, range(1, 13)), _fake_res(2006, range(1, 13))])
        r["tag"] = y
        parts[y] = r
    clean = lo.clean_series(parts)
    assert clean.index.is_unique and len(clean) == 24
    assert (clean.loc[clean.index.year == 2005, "tag"] == 2005).all()
    assert (clean.loc[clean.index.year == 2006, "tag"] == 2006).all()
    legacy = pd.concat(parts.values())
    assert not legacy.index.is_unique            # the legacy duplication this guards against


def test_clean_series_rejects_duplicate_months():
    r = _fake_res(2005, [1, 2])
    with pytest.raises(ValueError):
        lo.clean_series({2005: pd.concat([r, r])})


def test_ew_universe_month_skips_invalid_prices():
    p = _month_panel(4)
    nxt = _next(p, [0.10, 0.20, 0.0, 0.0])
    nxt.loc[nxt["SYMBOL"] == "S02", "ADJ_CLOSE"] = np.nan
    p.loc[p["SYMBOL"] == "S03", "ADJ_CLOSE"] = np.nan
    assert lo.ew_universe_month(p, nxt) == pytest.approx(0.15)   # zero-fill would give 0.075
    nxt["ADJ_CLOSE"] = np.nan
    assert lo.ew_universe_month(p, nxt) is None


def test_nifty50_monthly_takes_last_file_of_month_and_nifty_row(tmp_path):
    def write(day, nifty, other=1.0, name="Nifty 50"):
        pd.DataFrame({"INDEX_NAME": ["CNX 100", f" {name} "], "DATE": [day, day], "CLOSE": [other, nifty]}
                     ).to_parquet(tmp_path / f"{day}.parquet")
    write("2020-01-02", 10.0)
    write("2020-01-31", 12.0)
    write("2020-02-03", 13.0)
    write("2020-02-28", 15.0)
    write("2020-03-02", 99.0, name="Other")        # month with no Nifty 50 row is absent, not zero
    s = lo.nifty50_monthly(tmp_path)
    assert list(s.index) == [pd.Period("2020-01", "M"), pd.Period("2020-02", "M")]
    assert list(s.values) == [12.0, 15.0]
