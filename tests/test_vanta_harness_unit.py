"""Unit tests for vanta.walkforward helpers (the invariants live in tests/invariants/vanta/test_vanta_walkforward_harness.py)."""

import numpy as np
import pandas as pd
import pytest

from vanta import walkforward as wf

SESS = pd.bdate_range("2008-01-01", "2016-06-30")


def test_incomplete_last_test_window_is_dropped():
    fl = wf.make_folds(SESS, 20, val_years=1)
    assert [f.test_start.year for f in fl] == [2013, 2014, 2015]       # 2016 is not complete in the calendar
    assert all(f.val_start > f.train_start and f.test_start > f.val_start for f in fl)
    assert not fl[0].include_val_in_fit and fl[0].train_stop == fl[0].val_start


def test_bad_embargo_and_window_arguments():
    with pytest.raises(ValueError):
        wf.make_folds(SESS, 20, embargo=-1)
    with pytest.raises(ValueError):
        wf.make_folds(SESS, 20, window="sliding")
    with pytest.raises(ValueError):
        wf.make_folds(SESS[::-1], 20)


def test_ic_summary_and_rank_ic():
    ic = pd.Series([0.04, 0.02, 0.06, 0.0, 0.05])
    s = wf.ic_summary(ic, horizon=20, step=5)
    assert s["mean"] == pytest.approx(0.034) and s["ic_ir"] == pytest.approx(s["mean"] / s["std"])
    assert s["t"] == pytest.approx(s["mean"] / (s["std"] / np.sqrt(5 / 4)))     # overlap-adjusted n_eff
    assert np.isnan(wf.ic_summary(pd.Series([0.1]), 20)["mean"])
    d = pd.Timestamp("2010-01-04")
    frame = pd.DataFrame({"decision_date": d, "symbol": [f"S{i}" for i in range(12)], "score": range(12), "y_net": range(12)})
    assert wf.rank_ic_by_date(frame).iloc[0] == pytest.approx(1.0)
    assert wf.rank_ic_by_date(frame.head(5)).empty                               # fewer than MIN_NAMES: no IC, not 0


def test_shifted_labels_look_n_dates_ahead_per_symbol():
    d = pd.date_range("2010-01-04", periods=4, freq="7D")
    lab = pd.DataFrame({"symbol": ["A"] * 4 + ["B"] * 4, "decision_date": list(d) * 2, "y_net": [1, 2, 3, 4, 10, 20, 30, 40.0]})
    out = wf.shifted_labels(lab, 2)
    a = out[out["symbol"] == "A"].set_index("decision_date")["y_shift"]
    assert a.iloc[0] == 3 and a.iloc[1] == 4 and a.iloc[2:].isna().all()


def test_reader_as_of_lead_and_end_of_calendar():
    rd = wf.PitReader(SESS, before_nightly_run=False, lead_sessions=2)
    assert rd.as_of(SESS[10]) == SESS[12]
    with pytest.raises(ValueError):
        rd.as_of(SESS[-1])
    assert not wf.PitReader(SESS, before_nightly_run=True).leaky


def test_score_fn_length_mismatch_raises():
    lab = pd.DataFrame({"symbol": "A", "decision_date": [SESS[1500]], "horizon": 5, "label_end_date": [SESS[1505]], "y_net": [0.0]})
    f = wf.make_folds(SESS, 5, include_val_in_fit=True, val_years=0)[0]
    lab["decision_date"] = f.test_start
    with pytest.raises(ValueError, match="one score per test row"):
        wf.run_fold(f, lab, SESS, lambda r, fo: None, lambda m, rows, fo, rd: [])


def test_verifier_detector_check_fails_for_a_never_flag_detector():
    from scripts.verify_stage13 import detector_passes, never_flags
    labels = pd.DataFrame([(f"S{i:02d}", d, 5, SESS[int(SESS.get_loc(d)) + 5], float(np.sin(i * 7 + k)))
                           for i in range(15) for k, d in enumerate(SESS[1300:2000:5])],
                          columns=["symbol", "decision_date", "horizon", "label_end_date", "y_net"])
    fl = wf.make_folds(SESS, 5, first_year=2008, val_years=0, include_val_in_fit=True)
    noise = lambda m, rows, f, rd: np.random.default_rng(f.fold_id).normal(size=len(rows))      # noqa: E731
    memo = lambda m, rows, f, rd: rows.merge(labels, on=["symbol", "decision_date"], how="left")["y_net"].to_numpy()   # noqa: E731
    clean = wf.run_walkforward(fl, labels, SESS, lambda r, f: None, noise)
    leaky = wf.run_walkforward(fl, labels, SESS, lambda r, f: None, memo, fit_rows_policy="include_test")
    assert detector_passes(wf.detect_leak, clean, [leaky], labels, 5)
    assert not detector_passes(never_flags, clean, [leaky], labels, 5)
