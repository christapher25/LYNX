"""Driver-level guards for scripts.run_m2 / scripts.run_baselines (reviewer findings on row 15d)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_baselines, run_m2

TS = pd.Timestamp


def test_cell_is_current_requires_matching_fingerprint(tmp_path):
    meta, log = tmp_path / "meta.json", tmp_path / "log.parquet"
    log.write_text("x")
    assert not run_m2.cell_is_current(meta, [log], "abc")                      # no meta
    meta.write_text(json.dumps({"fingerprint": "abc"}))
    assert run_m2.cell_is_current(meta, [log], "abc")
    assert not run_m2.cell_is_current(meta, [log], "abd")                      # inputs or code changed: never reused
    assert not run_m2.cell_is_current(meta, [tmp_path / "gone.parquet"], "abc")
    meta.write_text("{broken")
    assert not run_m2.cell_is_current(meta, [log], "abc")


def test_fingerprint_changes_with_inputs(tmp_path):
    f = tmp_path / "scored.parquet"
    f.write_text("a")
    a = run_m2._fingerprint(f, extra=(5, "floor"))
    assert a == run_m2._fingerprint(f, extra=(5, "floor"))
    assert a != run_m2._fingerprint(f, extra=(5, "zero"))
    f.write_text("bb")
    assert a != run_m2._fingerprint(f, extra=(5, "floor"))


def test_attach_scores_keeps_unscored_candidates_as_nan():
    d = [TS("2020-01-03"), TS("2020-01-10"), TS("2020-01-17")]
    df = pd.DataFrame({"decision_date": [d[0]] * 3 + [d[1]] * 2 + [d[2]], "symbol": list("ABC") + list("AB") + ["A"]})
    scored = pd.DataFrame({"decision_date": [d[0], d[0], d[1]], "symbol": ["A", "B", "A"], "b1": [1.0, 2.0, np.nan]})
    out, n_unscored = run_m2.attach_scores(df, scored)
    assert set(out["decision_date"]) == {d[0]}                    # weeks with no finite score are not scored weeks
    assert len(out) == 3 and n_unscored == 1                      # C has no scored row: kept, NaN, counted
    assert np.isnan(out.loc[out["symbol"] == "C", "b1"]).all()


def test_hold_label_dir_follows_source_hash(monkeypatch):
    assert run_m2.hold_source_hash() in str(run_m2.hold_label_dir(5))
    monkeypatch.setattr(run_m2, "hold_source_hash", lambda: "deadbeef00")
    assert "deadbeef00" in str(run_m2.hold_label_dir(5))


def test_tune_refused_for_horizon_60():
    with pytest.raises(SystemExit):
        run_baselines.refuse_tune_for_diagnostic("tune", 60)
    run_baselines.refuse_tune_for_diagnostic("tune", 5)
    run_baselines.refuse_tune_for_diagnostic("b1", 60)


def test_no_judged_week_is_stated_explicitly():
    ok = ({"first_judged_week": "2017-01-06"}, {})
    none = ({"first_judged_week": None}, {})
    lines = run_m2.no_judged_week_lines({(5, "floor", "atr"): ok, (60, "floor", "atr"): none, (20, "zero", "hold"): None})
    assert len(lines) == 1 and "h=60" in lines[0] and "NO JUDGED WEEK" in lines[0]


def test_ungated_tie_order_is_reproducible():
    from vanta import m2
    g = pd.DataFrame({"symbol": list("DCBA"), "score": [1.0, 1.0, 1.0, 0.5]})
    a = list(m2._top_n(g, 2)["symbol"])
    b = list(m2._top_n(g.iloc[::-1], 2)["symbol"])
    assert a == b == ["B", "C"]


def test_arm_b_abstention_is_not_an_age_only_fallback():
    from vanta import m2
    cal = pd.DataFrame(dict(decision_date=[TS("2020-06-19")] * 3, symbol=list("XYZ"), horizon=20, score=0.5, model_version=m2.MODEL_VERSION,
                            fold_id="m2", is_oos=True, regime_posterior=[[0.9, 0.05, 0.05]] * 3, hmm_version="h", y_net=0.01,
                            matured_at=TS("2020-06-24"), candidate_filter_version=m2.FILTER_VERSION))
    out = m2.gate_arm("b", pd.DataFrame({"symbol": ["C"], "score": [0.9]}), cal, TS("2020-06-26"), [0.9, 0.05, 0.05])
    assert bool(out["abstained"].iloc[0]) and not bool(out["age_only_fallback"].iloc[0])
    log = out.assign(decision_date=TS("2020-06-26"))
    c = m2.arm_counts(log)
    assert c["n_abstain_weeks"] == 1 and c["n_age_only_fallback_weeks"] == 0
