"""
M1b diagnostics (13b), reported NOT JUDGED. Failing-first draft: skips until vanta.m1b has the functions.

Planned interface (vanta.m1b, INVENTED by quant-13a, to be added after 13a review):
  weekly_fdp(log) -> Series indexed by decision_date: losers / filled-with-known-outcome among PUBLISHED trades (loser = y_net <= 0);
                     NaN for a week with no published filled trade (never 0).
  fdp_vs_market(fdp, market_ret) -> dict(corr, n_weeks, fdr_up, fdr_down, n_up, n_down): Pearson correlation of the weekly FDP with the
                     market's weekly return over weeks where both are finite; fdr_up / fdr_down = simple mean of weekly FDP over weeks
                     with market_ret > 0 / <= 0.
  nifty_relative_y(log, index_close) -> Series: y_net minus the index return over the same trade window (entry_date to label_end_date),
                     NaN when either end of the index window is unavailable.
  realised_fdr_vs_index(log, index_close) -> dict: realised_fdr semantics with loser = (y_net - index return) <= 0.
"""

import numpy as np
import pandas as pd
import pytest

TS = pd.Timestamp


def _m():
    m = pytest.importorskip("vanta.m1b")
    if not hasattr(m, "weekly_fdp"):
        pytest.skip("13b diagnostics not implemented yet")
    return m


def _log():
    d = pd.to_datetime(["2013-01-04"] * 4 + ["2013-01-11"] * 4)
    return pd.DataFrame({"decision_date": d, "published": [True, True, True, False, True, True, False, False],
                         "fill_status": ["FILLED"] * 8,
                         "y_net": [0.01, -0.01, -0.02, -0.5, np.nan, 0.03, 0.9, 0.9]})


def test_weekly_fdp_counts_published_filled_known_only():
    m = _m()
    s = m.weekly_fdp(_log())
    assert s.loc[TS("2013-01-04")] == pytest.approx(2 / 3)
    assert s.loc[TS("2013-01-11")] == pytest.approx(0.0)       # the NaN outcome is not a winner and not counted


def test_weekly_fdp_week_without_published_is_nan_not_zero():
    m = _m()
    log = _log()
    log.loc[log["decision_date"] == TS("2013-01-11"), "published"] = False
    assert np.isnan(m.weekly_fdp(log).loc[TS("2013-01-11")])


def test_fdp_vs_market_correlation_and_up_down_split():
    m = _m()
    idx = pd.to_datetime([f"2013-01-{d:02d}" for d in (4, 11, 18, 25)])
    fdp = pd.Series([0.2, 0.4, 0.6, 0.8], index=idx)
    mkt = pd.Series([0.03, 0.01, -0.01, -0.03], index=idx)
    r = m.fdp_vs_market(fdp, mkt)
    assert r["corr"] == pytest.approx(-1.0) and r["n_weeks"] == 4
    assert r["fdr_up"] == pytest.approx(0.3) and r["fdr_down"] == pytest.approx(0.7) and (r["n_up"], r["n_down"]) == (2, 2)


def test_fdp_vs_market_missing_market_week_is_excluded_not_zero():
    m = _m()
    idx = pd.to_datetime(["2013-01-04", "2013-01-11", "2013-01-18"])
    r = m.fdp_vs_market(pd.Series([0.2, 0.4, 0.9], index=idx), pd.Series([0.01, -0.01, np.nan], index=idx))
    assert r["n_weeks"] == 2 and r["fdr_up"] == pytest.approx(0.2) and r["fdr_down"] == pytest.approx(0.4)


def test_nifty_relative_loser_uses_the_trade_window_and_nan_without_index():
    m = _m()
    px = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2013-01-04", "2013-01-11", "2013-01-18"]))
    log = pd.DataFrame({"decision_date": pd.to_datetime(["2013-01-04"] * 3), "published": True, "fill_status": "FILLED",
                        "y_net": [0.05, 0.15, 0.05],
                        "entry_date": pd.to_datetime(["2013-01-04", "2013-01-04", "2013-01-04"]),
                        "label_end_date": pd.to_datetime(["2013-01-11", "2013-01-11", "2013-02-01"])})
    rel = m.nifty_relative_y(log, px)
    assert rel.iloc[:2].to_numpy() == pytest.approx([0.05 - 0.10, 0.15 - 0.10])
    assert np.isnan(rel.iloc[2])                                # window end after the index series: missing, not 0
    s = m.realised_fdr_vs_index(log, px)
    assert s["n_filled"] == 2 and s["n_losers"] == 1
