"""
M1 driver invariants (vanta.m1 driver layer) on synthetic inputs: momentum is causal and NaN-honest, maturity lag, the two impact
modes, no calibration leak through run_arms, the ungated arm publishes everything, appending the future changes nothing.
"""

import numpy as np
import pandas as pd
import pytest

from vanta import m1

TS = pd.Timestamp


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_momentum_is_12_1_by_rows_and_nan_before_lookback():
    c = _closes()
    d = c.index[[100, 260, 300]]
    s = m1.momentum_12_1(c, d)
    assert np.isnan(s.iloc[0])
    assert s.iloc[1] == pytest.approx(c.iloc[260 - 21] / c.iloc[260 - 252] - 1)
    assert s.iloc[2] == pytest.approx(c.iloc[300 - 21] / c.iloc[300 - 252] - 1)


def test_momentum_is_causal_future_prices_do_not_change_it():
    c = _closes()
    t = c.index[300]
    a = m1.momentum_12_1(c, pd.DatetimeIndex([t]))
    c2 = c.copy()
    c2.iloc[301:] *= 7.0
    assert m1.momentum_12_1(c2, pd.DatetimeIndex([t])).iloc[0] == a.iloc[0]


def test_momentum_missing_price_is_nan_not_zero():
    c = _closes()
    gap = c.index[300 - 21]
    c.loc[gap] = np.nan
    t = c.index[300]
    got = m1.momentum_12_1(c, pd.DatetimeIndex([t])).iloc[0]
    clean = c.drop(gap)                                   # the NaN bar is dropped, never filled: rows shift by one
    pos = clean.index.get_loc(t)
    assert got == pytest.approx(clean.iloc[pos - 21] / clean.iloc[pos - 252] - 1)
    assert got != 0.0


def test_momentum_long_suspension_gives_nan():
    c = _closes(n=400)
    shifted = c.copy()
    shifted.index = shifted.index.where(np.arange(len(c)) < 150, shifted.index + pd.Timedelta(days=200))
    s = m1.momentum_12_1(shifted, pd.DatetimeIndex([shifted.index[300]]))
    assert np.isnan(s.iloc[0])


def test_matured_is_next_session_after_label_end():
    sessions = pd.bdate_range("2013-01-01", periods=30)
    end = pd.Series(pd.to_datetime([sessions[5], pd.NaT, sessions[-1]]))
    r = m1.matured_at(end, sessions)
    assert r.iloc[0] == sessions[6]
    assert pd.isna(r.iloc[1])
    assert pd.isna(r.iloc[2])        # no later session known: not matured, never guessed


def _labels(ranks, impact=0.0):
    return pd.DataFrame({"fill_status": ["FILLED"] * len(ranks), "y_net": [0.01] * len(ranks), "cost_impact": [impact] * len(ranks),
                         "univ_rank": ranks})


def test_impact_zero_leaves_y_net():
    d = _labels([1, 300])
    out = m1.apply_impact(d, d["univ_rank"], "zero")
    assert list(out["y_net"]) == [0.01, 0.01]


def test_impact_floor_is_round_trip_by_rank_and_unknown_rank_is_nan():
    d = _labels([1, 200, 201, 700, np.nan])
    out = m1.apply_impact(d, d["univ_rank"], "floor")
    np.testing.assert_allclose(out["y_net"].iloc[:4], [0.01 - 0.001, 0.01 - 0.001, 0.01 - 0.004, 0.01 - 0.004])
    assert np.isnan(out["y_net"].iloc[4])


def test_impact_floor_refuses_double_count_and_bad_mode():
    d = _labels([1], impact=0.001)
    with pytest.raises(ValueError):
        m1.apply_impact(d, d["univ_rank"], "floor")
    with pytest.raises(ValueError):
        m1.apply_impact(_labels([1]), pd.Series([1]), "both")


def _cands(n_weeks=80, per=10, seed=1, poison_after=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-06", periods=n_weeks, freq="7D")
    sessions = pd.bdate_range("2011-12-01", periods=n_weeks * 6)
    rows = []
    for d in dates:
        for i in range(per):
            s = rng.uniform(0, 1)
            y = 0.05 * (s - 0.4) + rng.normal(0, 0.05)
            rows.append(dict(decision_date=d, symbol=f"S{i}", horizon=5, score=s, y_net=y, fill_status="FILLED",
                             label_end_date=d + pd.Timedelta(days=7), regime_posterior=[1 / 3] * 3, hmm_version="h"))
    c = pd.DataFrame(rows)
    c["matured_at"] = m1.matured_at(c["label_end_date"], pd.DatetimeIndex(pd.date_range("2011-12-01", periods=n_weeks * 8, freq="D")))
    if poison_after is not None:
        c.loc[c["decision_date"] > poison_after, "y_net"] = -9.0
    return c, dates


def test_ungated_arm_publishes_every_candidate():
    c, dates = _cands()
    logs = m1.run_arms(c, dates)
    assert logs["ungated"]["published"].all() and len(logs["ungated"]) == len(c)


def _signature(logs, T):
    """Everything about the gates at T that depends on which calibration rows were visible."""
    cols = {"v4": ["bound"], "gate_age": ["p_value", "n_calibration", "effective_n"],
            "gate_age_regime": ["p_value", "n_calibration", "effective_n"]}
    return {arm: logs[arm][logs[arm]["decision_date"] == T][c].reset_index(drop=True) for arm, c in cols.items()}


def _poisoned_vs_clean():
    c, dates = _cands()
    T = dates[60]
    # rows decided in the week before T (and later) have matured_at > T: their outcomes are unknown at T
    c2 = c.copy()
    late = (c2["decision_date"] >= T - pd.Timedelta(days=7)) & (c2["decision_date"] < T)    # not T: its candidates are inputs
    c2.loc[late, "y_net"] = -9.0
    c2.loc[late, "score"] = 0.99                                  # and a score that would dominate if visible
    assert late.any() and (c2.loc[late, "matured_at"] > T).all()
    return _signature(m1.run_arms(c, dates[:61]), T), _signature(m1.run_arms(c2, dates[:61]), T)


def test_arms_cannot_see_unmatured_outcomes():
    """Rewriting outcomes (and scores) of rows unmatured at T must not change p-values, calibration counts or the v4 bound at T."""
    clean, dirty = _poisoned_vs_clean()
    for arm in clean:
        pd.testing.assert_frame_equal(clean[arm].reset_index(drop=True), dirty[arm].reset_index(drop=True), obj=arm)


def test_unmatured_leak_test_can_fail_v4_without_maturity_filter(monkeypatch):
    real = m1.v4_gate_reconstruction

    def leaky(cands, cal, T, **kw):
        return real(cands, cal.assign(matured_at=cal["decision_date"]), T, **kw)    # every row counts as matured

    monkeypatch.setattr(m1, "v4_gate_reconstruction", leaky)
    clean, dirty = _poisoned_vs_clean()
    assert not clean["v4"].equals(dirty["v4"])


def test_unmatured_leak_test_can_fail_rw_gate_without_maturity_filter(monkeypatch):
    import vanta.gate as G
    real = G._usable_mask

    def leaky(cal, T, mv, cf, w):
        return real(cal.assign(matured_at=cal["decision_date"]), T, mv, cf, w)

    monkeypatch.setattr(G, "_usable_mask", leaky)
    clean, dirty = _poisoned_vs_clean()
    assert not clean["gate_age"].equals(dirty["gate_age"])


def test_appending_future_weeks_changes_nothing_in_the_past():
    c, dates = _cands()
    short = m1.run_arms(c[c["decision_date"] <= dates[50]], dates[:51])
    full = m1.run_arms(c, dates)
    for arm in short:
        a = short[arm].reset_index(drop=True)
        b = full[arm][full[arm]["decision_date"] <= dates[50]].reset_index(drop=True)
        assert list(a["published"]) == list(b["published"]), arm


def test_gates_respect_the_cap():
    c, dates = _cands(per=30)
    logs = m1.run_arms(c, dates, cap=10)
    for arm in ("gate_age", "gate_age_regime", "v4"):
        assert logs[arm].groupby("decision_date")["published"].sum().max() <= 10


def test_nan_posterior_forces_age_only_not_a_crash():
    c, dates = _cands()
    c["regime_posterior"] = [[np.nan] * 3] * len(c)
    logs = m1.run_arms(c, dates)
    r = logs["gate_age_regime"]
    assert r.loc[r["n_calibration"] > 0, "age_only_fallback"].all()
    a = logs["gate_age"]
    assert list(r["published"]) == list(a["published"])


def test_summary_nw_lag_is_ceil_h_over_5_and_unmatured_not_counted():
    c, dates = _cands(n_weeks=120)
    logs = m1.run_arms(c, dates)
    for h, lag in ((5, 1), (20, 4)):
        s = m1.summarise_arm(logs["gate_age"], logs["ungated"], dates, h, n_boot=50)
        assert s["nw_lag_weeks"] == lag
    bad = logs["ungated"].copy()
    bad.loc[bad.index[:5], "y_net"] = np.nan
    r = m1.realised_fdr(bad)
    assert r["n_unmatured"] == 5 and r["n_filled"] == len(bad) - 5


# ---------------------------------------------------------------- review follow-ups ---------------------------------
def test_asof_values_never_fill_across_a_nan_row_and_respect_max_age():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    post = pd.DataFrame({"p0": [0.1, np.nan, 0.3]}, index=idx)
    out = m1.asof_values(post, pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-09", "2019-12-31"]))
    assert np.isnan(out["p0"].iloc[0])                   # the date is in the index and marked NaN: stays NaN
    assert out["p0"].iloc[1] == 0.3
    assert out["p0"].iloc[2] == 0.3                      # 3 calendar days after the last row
    assert np.isnan(out["p0"].iloc[3])                   # 6 days: too old
    assert np.isnan(out["p0"].iloc[4])                   # before the first row


def test_attach_regime_uses_asof_values():
    run_m1 = pytest.importorskip("scripts.run_m1")
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    post = pd.DataFrame({"p0": [0.2, np.nan], "p1": [0.3, np.nan], "p2": [0.5, np.nan], "hmm_version": ["v", "v"]}, index=idx)
    cands = pd.DataFrame({"decision_date": pd.to_datetime(["2020-01-02", "2020-01-04"]), "symbol": ["A", "B"]})
    out = run_m1.attach_regime(cands, post)
    assert np.isnan(out["regime_posterior"].iloc[0]).all()           # NaN on the posterior's own row is not replaced by 01-01
    assert np.isnan(out["regime_posterior"].iloc[1]).all()           # carried from the NaN row 01-02, not from 01-01


def test_block_bootstrap_uses_adjacent_weeks_only():
    y = [0.0, 10.0, 100.0, 1000.0]
    wk = ["w1", "w2", "w3", "w4"]
    d1 = m1.weekly_block_bootstrap(y, wk, 400, 5)
    d2 = m1.weekly_block_bootstrap(y, wk, 400, 5, block_len=2)
    # block_len=2, n=4: two blocks, each an adjacent pair (0,10), (10,100) or (100,1000)
    pairs = [(0, 10), (10, 100), (100, 1000)]
    allowed = {round((a1 + a2 + b1 + b2) / 4, 9) for (a1, a2) in pairs for (b1, b2) in pairs}
    assert set(np.round(d2, 9)) <= allowed
    assert not set(np.round(d1, 9)) <= allowed                  # iid weeks do produce means no adjacent-pair design can
    assert np.array_equal(d1, m1.weekly_block_bootstrap(y, wk, 400, 5, block_len=1))      # default unchanged


def test_paired_block_bootstrap_shares_blocks():
    ya, wa = [1, 1, 1, -1], ["w1", "w2", "w3", "w4"]
    d = m1.weekly_block_bootstrap_diff(ya, wa, ya, wa, 100, 1, block_len=2)
    assert np.allclose(d, 0.0)                                  # identical arms on identical resamples: difference exactly 0


def test_no_data_is_not_counted_as_no_fill():
    log = pd.DataFrame({"decision_date": pd.to_datetime(["2013-01-07"] * 4), "published": True,
                        "fill_status": ["FILLED", "NO_FILL_UPPER_LOCK", "NO_DATA", "NO_FILL_NO_TRADE"], "y_net": [0.1, np.nan, np.nan, np.nan]})
    r = m1.realised_fdr(log)
    assert (r["n_no_fill"], r["n_no_data"], r["n_filled"]) == (2, 1, 1)
    assert "NO_DATA" in m1.fdr_report_text(r, 0.3)


def test_build_labels_cache_is_keyed_on_the_request_and_returns_only_requested_rows(tmp_path, monkeypatch):
    run_m1 = pytest.importorskip("scripts.run_m1")
    import scripts.report_nel_coverage as rep
    calls = []
    bars = pd.DataFrame(index=pd.to_datetime(["2013-01-04", "2013-01-11", "2013-01-18"]))
    monkeypatch.setattr(rep, "load_symbol", lambda sym: (bars, pd.Series(1.0, index=bars.index), None, None))
    monkeypatch.setattr(run_m1, "simulate_trade",
                        lambda sym, b, t, h, ks, kt, atr, hits, cov, sessions=None, **kw: calls.append(t) or
                        dict(symbol=sym, decision_date=t, fill_status="FILLED", y_net=0.01))
    monkeypatch.setattr(run_m1, "_code_version", lambda: "v1")
    d = pd.to_datetime(["2013-01-04", "2013-01-11"])
    c1 = pd.DataFrame({"symbol": "A", "decision_date": d})
    r1 = run_m1.build_labels(c1, 5, 1.5, 2.0, tmp_path, None, {})
    assert len(r1) == 2 and len(calls) == 2
    run_m1.build_labels(c1, 5, 1.5, 2.0, tmp_path, None, {})
    assert len(calls) == 2                                                # same request: cache hit
    c2 = pd.DataFrame({"symbol": "A", "decision_date": d[:1]})            # smaller request must not reuse the 2-row chunk
    r2 = run_m1.build_labels(c2, 5, 1.5, 2.0, tmp_path, None, {})
    assert len(r2) == 1 and len(calls) == 3
    monkeypatch.setattr(run_m1, "_code_version", lambda: "v2")             # code change: new key, no stale reuse
    run_m1.build_labels(c1, 5, 1.5, 2.0, tmp_path, None, {})
    assert len(calls) == 5
    assert len(list(tmp_path.glob("A_*.parquet"))) == 3                    # old files kept, nothing deleted
