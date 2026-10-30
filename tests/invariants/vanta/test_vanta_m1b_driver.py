"""
M1b driver invariants (vanta.m1b arms + scripts.run_m1b helpers) on synthetic inputs: the ungated arm is the 10 highest-percentile
candidates (not every candidate), the gates see ALL candidates of the week and calibrate on ALL matured scored candidates, scores are
week-invariant, the regime lookup never fills a missing posterior, k is frozen from M1 and never re-chosen, the report opens with the
"already seen in M1" statement.
"""

import numpy as np
import pandas as pd
import pytest

from scripts import run_m1b
from vanta import m1, m1b

TS = pd.Timestamp
PER = 30


def _cands(n_weeks=70, per=PER, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-06", periods=n_weeks, freq="7D")
    rows = []
    for d in dates:
        mom = rng.normal(0.1, 0.3, per)
        for i in range(per):
            rows.append(dict(decision_date=d, symbol=f"S{i}", mom=mom[i], horizon=5, fill_status="FILLED",
                             label_end_date=d + pd.Timedelta(days=7), regime_posterior=[1 / 3] * 3, hmm_version="h"))
    c = pd.DataFrame(rows)
    c["score"] = m1b.weekly_scores(c, "mom")
    c["y_net"] = 0.04 * (c["score"] - 0.5) + rng.normal(0, 0.05, len(c))
    c["matured_at"] = m1.matured_at(c["label_end_date"], pd.DatetimeIndex(pd.date_range("2011-12-01", periods=len(dates) * 8, freq="D")))
    return c, dates


def test_ungated_arm_is_top_ten_by_percentile_not_every_candidate():
    c, dates = _cands()
    ung = m1b.run_arms_b(c, dates)["ungated"]
    assert len(ung) == len(c)
    pub = ung[ung["published"]]
    assert (pub.groupby("decision_date").size() == m1b.UNGATED_TOP_N).all()
    for d, g in ung.groupby("decision_date"):
        assert g.loc[g["published"], "score"].min() >= g.loc[~g["published"], "score"].max()


def test_gate_arms_see_all_candidates_and_cap_publication_at_ten():
    c, dates = _cands()
    logs = m1b.run_arms_b(c, dates)
    for arm in ("gate_age", "gate_age_regime"):
        log = logs[arm]
        assert (log.groupby("decision_date").size() == PER).all()                  # BH universe = every candidate, not 10
        assert (log.groupby("decision_date")["published"].sum() <= m1b.CAP).all()
        assert log["published"].any()


def test_calibration_count_is_every_matured_scored_candidate_in_52_weeks():
    c, dates = _cands()
    log = m1b.run_arms_b(c, dates)["gate_age"]
    for T in dates[[30, 55, 69]]:
        want = len(m1b.calibration_rows(c[["decision_date", "matured_at", "y_net"]], T))
        got = log.loc[log["decision_date"] == T, "n_calibration"].iloc[0]
        assert got == want and want > 10 * PER                                     # hundreds of rows, not the 10 published


def test_gate_arms_cannot_see_unmatured_outcomes():
    c, dates = _cands()
    T = dates[60]
    c2 = c.copy()
    late = (c2["decision_date"] >= T - pd.Timedelta(days=7)) & (c2["decision_date"] < T)
    c2.loc[late, "y_net"] = -9.0
    assert late.any() and (c2.loc[late, "matured_at"] > T).all()
    a = m1b.run_arms_b(c, dates[:61])["gate_age"]
    b = m1b.run_arms_b(c2, dates[:61])["gate_age"]
    ga, gb = a[a["decision_date"] == T].reset_index(drop=True), b[b["decision_date"] == T].reset_index(drop=True)
    pd.testing.assert_series_equal(ga["p_value"], gb["p_value"])
    assert (ga["published"] == gb["published"]).all()


def test_score_is_unchanged_by_a_market_wide_momentum_shift():
    c, _ = _cands()
    shifted = c.copy()
    shifted["mom"] = shifted["mom"] + shifted["decision_date"].rank(method="dense") * 0.05      # drifting market
    pd.testing.assert_series_equal(m1b.weekly_scores(shifted, "mom"), c["score"], check_names=False)


def test_select_candidates_keeps_every_gate_passing_name_and_never_a_missing_verdict():
    pool = pd.DataFrame({"decision_date": TS("2013-01-04"), "symbol": [f"S{i}" for i in range(25)], "mom": np.arange(25.0)})
    pool["gate_allowed"] = pd.Series([True] * 25, dtype=object)
    pool.loc[3, "gate_allowed"] = False
    pool.loc[4, "gate_allowed"] = np.nan
    pool.loc[5, "mom"] = np.nan
    out = m1b.select_candidates(pool)
    assert sorted(set(range(25)) - set(out.index)) == [3, 4, 5] and len(out) == 22 > m1b.CAP


def test_attach_regime_shares_lists_keeps_nan_and_does_not_fill_across_gaps():
    dts = pd.to_datetime(["2013-01-04", "2013-01-11", "2013-03-01"])
    c = pd.DataFrame({"decision_date": np.repeat(dts, 3), "symbol": list("ABC") * 3})
    post = pd.DataFrame({"p0": [0.2, np.nan], "p1": [0.3, np.nan], "p2": [0.5, np.nan], "hmm_version": ["v", None]},
                        index=pd.DatetimeIndex(["2013-01-04", "2013-01-11"]))
    out = run_m1b.attach_regime(c, post)
    first = out[out["decision_date"] == dts[0]]["regime_posterior"]
    assert first.iloc[0] is first.iloc[1] and list(first.iloc[0]) == [0.2, 0.3, 0.5]
    assert np.isnan(np.array(out[out["decision_date"] == dts[1]]["regime_posterior"].iloc[0], float)).all()    # NaN row stays NaN
    assert np.isnan(np.array(out[out["decision_date"] == dts[2]]["regime_posterior"].iloc[0], float)).all()    # too old: NaN, not carried


def test_frozen_k_is_read_from_m1_and_missing_file_refuses_to_choose(tmp_path, monkeypatch):
    monkeypatch.setattr(run_m1b, "M1_DIR", tmp_path)
    with pytest.raises(SystemExit):
        run_m1b.frozen_k(5)
    (tmp_path / "k_h5_cliff200.json").write_text('{"k_s": 1.5, "k_t": 2.0}')
    assert run_m1b.frozen_k(5) == (1.5, 2.0)


def test_report_opens_with_the_corrected_run_statement_and_never_calls_q_a_guarantee():
    log = pd.DataFrame({"decision_date": pd.to_datetime(["2013-01-04"] * 4 + ["2012-01-06"] * 2), "published": True,
                        "fill_status": "FILLED", "y_net": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03]})
    dates = pd.DatetimeIndex(["2012-01-06", "2013-01-04"])
    s = {"ungated": m1.summarise_arm(log, log, dates, 5, n_boot=50)}
    txt = run_m1b.report_text(5, "floor", (1.5, 1.5), s, [])
    head = txt.split("\n\n")[1]
    assert txt.startswith("# M1b report") and "already seen in M1" in head and "not a fresh out-of-sample test" in head
    assert "engineering target" in txt and "is a guarantee" not in txt


# ------------------------------------------------------------------ review fixes: real gate path, atomic/resumable chunks, caches
def _gate_frame():
    """Rows in the shape gates.io.load_gate_inputs returns, one week, then run through the real evaluate_gates."""
    names = ["OK1", "OK2", "BAN", "UP", "LO", "NAFLAG", "R201", "NOHITS"]
    f = pd.DataFrame({"symbol": names, "decision_date": TS("2013-01-04"),
                      "hit_upper_circuit": [0.0] * 8, "hit_lower_circuit": [0.0] * 8, "is_missing_circuit_hits": [False] * 8,
                      "price_band_pct": [20.0] * 8, "fno_in_ban": [0.0] * 8, "is_missing_ban": [False] * 8,
                      "liquidity_rank": [5.0, 200.0, 7.0, 8.0, 9.0, 10.0, 201.0, 11.0], "is_missing_liquidity_rank": [False] * 8,
                      "mom": [0.3, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]})
    f.loc[2, "fno_in_ban"] = 1.0
    f.loc[3, "hit_upper_circuit"] = 1.0
    f.loc[4, "hit_lower_circuit"] = 1.0
    f.loc[5, ["fno_in_ban", "is_missing_ban"]] = [np.nan, True]                    # ban list unavailable
    f.loc[7, ["hit_upper_circuit", "hit_lower_circuit", "is_missing_circuit_hits"]] = [np.nan, np.nan, True]
    return f


def _verdict(f):
    from gates.override_gates import BUY_SIGNAL_GATES, GateConfig, evaluate_gates
    v = evaluate_gates(f, GateConfig(enabled=BUY_SIGNAL_GATES, liquidity_top_n=200, unavailable_policy="block"))
    return pd.concat([f, v[["gate_status", "gate_allowed"]]], axis=1)


def test_real_gate_path_excludes_banned_circuit_hit_unavailable_and_beyond_cliff_names():
    pool = _verdict(_gate_frame())
    out = m1b.select_candidates(pool)
    assert list(out["symbol"]) == ["OK1", "OK2"]                       # rank 200 stays (inclusive); BAN/UP/LO/NA-flag/rank 201/no-hits go
    assert set(pool.loc[pool["symbol"].isin(["NAFLAG", "NOHITS"]), "gate_status"]) == {"UNAVAILABLE"}


def test_real_scoring_path_is_per_week_and_follows_the_gate_survivors_only():
    f1 = _verdict(_gate_frame())
    f2 = f1.assign(decision_date=TS("2013-01-11"), mom=f1["mom"] + 5.0)           # a later week, whole market drifted up
    cands = m1b.add_scores(m1b.select_candidates(pd.concat([f1, f2], ignore_index=True)))
    wk = {d: g.set_index("symbol")["score"] for d, g in cands.groupby("decision_date")}
    assert wk[TS("2013-01-04")].to_dict() == {"OK1": 1.0, "OK2": 0.5}              # percentile among the 2 survivors, not the 8 names
    assert wk[TS("2013-01-11")].to_dict() == wk[TS("2013-01-04")].to_dict()


def _patch_labels(monkeypatch):
    calls = []
    monkeypatch.setattr(run_m1b, "terminal_kwargs", lambda s, t: {})
    monkeypatch.setattr(run_m1b, "simulate_trade",
                        lambda symbol, bars, t, *a, **k: dict(symbol=symbol, decision_date=t, fill_status="FILLED", y_net=0.01))
    return calls


def test_truncated_label_chunk_is_rebuilt_and_writes_are_atomic(tmp_path, monkeypatch):
    calls = _patch_labels(monkeypatch)
    monkeypatch.setattr(run_m1b.config, "ADJUSTED_PRICES_DIR", tmp_path / "adj")
    monkeypatch.setattr(run_m1b.config, "CIRCUIT_FEATURES_DIR", tmp_path / "circ")

    class Bars:
        index = pd.DatetimeIndex([TS("2013-01-04")])
    import scripts.report_nel_coverage as nel
    monkeypatch.setattr(nel, "load_symbol", lambda s: calls.append(s) or (Bars(), pd.Series(1.0, index=Bars.index), None, None))
    cands = pd.DataFrame({"symbol": ["AAA", "BBB"], "decision_date": TS("2013-01-04")})
    cache = tmp_path / "labels"
    first = run_m1b.build_labels(cands, 5, 1.5, 1.5, cache, None, {})
    assert calls == ["AAA", "BBB"] and not list(cache.glob("*.tmp"))
    again = run_m1b.build_labels(cands, 5, 1.5, 1.5, cache, None, {})
    assert calls == ["AAA", "BBB"]                                       # fully cached: nothing recomputed
    victim = sorted(cache.glob("AAA_*.parquet"))[0]
    victim.write_bytes(victim.read_bytes()[:40])                         # a killed write leaves a stub
    rebuilt = run_m1b.build_labels(cands, 5, 1.5, 1.5, cache, None, {})
    assert calls == ["AAA", "BBB", "AAA"]                                # only the damaged chunk was rebuilt
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, rebuilt)


def test_label_chunk_key_changes_when_the_symbols_price_file_changes(tmp_path, monkeypatch):
    adj = tmp_path / "adj"
    adj.mkdir()
    (adj / "AAA.parquet").write_bytes(b"1234")
    a = run_m1b._stat_fp(adj / "AAA.parquet")
    (adj / "AAA.parquet").write_bytes(b"12345")
    assert run_m1b._stat_fp(adj / "AAA.parquet") != a
    assert run_m1b._stat_fp(adj) != run_m1b._stat_fp(tmp_path / "nothing")


def test_atomic_write_replaces_whole_file_and_unreadable_cache_reads_as_none(tmp_path):
    p = tmp_path / "x.parquet"
    run_m1b._atomic_to_parquet(pd.DataFrame({"a": [1, 2]}), p)
    assert list(run_m1b._read_or_none(p)["a"]) == [1, 2] and not list(tmp_path.glob("*.tmp"))
    p.write_bytes(p.read_bytes()[:20])
    assert run_m1b._read_or_none(p) is None
    assert run_m1b._read_or_none(tmp_path / "missing.parquet") is None


def test_run_m1_globals_are_restored_after_use(tmp_path):
    from scripts import run_m1
    before = (run_m1.M1_DIR, run_m1.FORCE)
    with run_m1b._run_m1_dir(tmp_path, True):
        assert (run_m1.M1_DIR, run_m1.FORCE) == (tmp_path, True)
    assert (run_m1.M1_DIR, run_m1.FORCE) == before
    with pytest.raises(RuntimeError):
        with run_m1b._run_m1_dir(tmp_path, True):
            raise RuntimeError
    assert (run_m1.M1_DIR, run_m1.FORCE) == before


def test_diagnostics_block_is_labelled_not_judged_and_excludes_weeks_without_index_data():
    dts = pd.to_datetime(["2012-01-06", "2013-01-04", "2013-01-11", "2013-01-18"])
    log = pd.DataFrame({"decision_date": np.repeat(dts, 2), "symbol": list("AB") * 4, "published": True, "fill_status": "FILLED",
                        "y_net": [0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1]})
    df = log[["decision_date", "symbol"]].assign(entry_date=log["decision_date"], label_end_date=log["decision_date"] + pd.Timedelta(days=7))
    nifty = pd.Series([100.0, 101.0, 100.0, 103.0], index=pd.to_datetime(["2013-01-04", "2013-01-11", "2013-01-18", "2013-01-25"]))
    txt = "\n".join(run_m1b.diagnostics_lines({"ungated": log}, df, nifty, pd.DatetimeIndex(dts)))
    assert "NOT JUDGED" in txt and "DIAGNOSTIC ONLY" in txt
    assert "in_sample: corr(weekly FDP, market return) nan over 0 weeks" in txt          # 2012 week has no Nifty data: excluded, not 0
