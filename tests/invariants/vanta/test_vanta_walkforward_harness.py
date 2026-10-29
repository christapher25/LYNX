"""
tests/invariants/vanta/test_vanta_walkforward_harness.py

Stage 13 harness invariants (docs/vanta/VANTA_SPEC.md line 77-79, N11; v6 10.2 and 11.4):

  H1  N11 in the harness   no row handed to a fit has label_end_date >= test_start - embargo, in sessions, on every fold, even
                           when a planted boundary row tries to get through; a fit policy that skips the purge is caught by
                           the independent audit (it reports violations; the safe policy raises instead of fitting).
  H2  Fit isolation        a planted sentinel (label ending on the cutoff session, a test-window row, an absurd y_net) never
                           reaches the model fit; the HMM fit never reads a feature row at or after test_start - embargo.
  H3  Leak detector        a leaky run (fit sees test rows) is flagged, a clean run is not, and a mutant detector that never
                           flags FAILS the same assertion.
  H4  Explicit store read  every store read in the harness passes before_nightly_run explicitly (no default, no omission).
  H5  Embargo >= horizon   a fold schedule with embargo below the label horizon is refused.
"""

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

wf = pytest.importorskip("vanta.walkforward")

H = 20
SESS = pd.bdate_range("2008-01-01", "2019-12-31")


def make_labels(h=H, n_sym=30, step=5, seed=0):
    rng = np.random.default_rng(seed)
    dec = SESS[::step][:-(h // step + 2)]
    rows = []
    for s in range(n_sym):
        for d in dec:
            p = int(SESS.get_loc(d))
            rows.append((f"S{s:02d}", d, h, SESS[p + h], rng.normal(0, 0.05)))
    return pd.DataFrame(rows, columns=["symbol", "decision_date", "horizon", "label_end_date", "y_net"])


LABELS = make_labels()


def folds(**kw):
    kw.setdefault("val_years", 0)
    kw.setdefault("include_val_in_fit", True)
    return wf.make_folds(SESS, H, **kw)


def noise_score(model, rows, fold, reader):
    return np.random.default_rng(fold.fold_id + 11).normal(size=len(rows))


class Memoriser:
    """A fully flexible fit: remembers y by (symbol, decision_date). On rows it has seen, it is exact."""

    def __init__(self, rows):
        self.table = {(r.symbol, r.decision_date): r.y_net for r in rows.itertuples()}


def memorise_fit(rows, fold):
    return Memoriser(rows)


def memorise_score(model, rows, fold, reader):
    return [model.table.get((r.symbol, r.decision_date), 0.0) for r in rows.itertuples()]


def never_flags(result, labels_all, **kw):
    """Mutant detector: reports everything as clean."""
    return wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, 0)


def check_detector(detect):
    """The detector must flag the leaky run and must not flag the clean one."""
    fl = folds()
    clean = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score)
    leaky = wf.run_walkforward(fl, LABELS, SESS, memorise_fit, memorise_score, fit_rows_policy="include_test")
    assert not detect(clean, LABELS, horizon=H).flagged, "clean run flagged"
    assert detect(leaky, LABELS, horizon=H).flagged, "leaky run not flagged"


# ------------------------------------------------------------------ H5 schedule
def test_schedule_is_derived_rolling_and_in_sessions():
    fl = folds()
    assert len(fl) >= 3
    for f in fl:
        assert f.test_start in SESS and f.train_start in SESS
        assert f.test_start.year - f.train_start.year == 4          # 4y rolling train window
        assert f.test_end.year == f.test_start.year
        assert f.embargo == wf.embargo_for_horizon(H) >= H
    assert [f.train_start.year for f in fl] == sorted({f.train_start.year for f in fl})   # rolled by one year
    assert [f.test_start.year for f in fl] == [f.train_start.year + 4 for f in fl]
    exp = folds(window="expanding")
    assert {f.train_start for f in exp} == {SESS[0]}


def test_embargo_below_horizon_refused_unless_explicitly_unsafe():
    with pytest.raises(ValueError, match="leaks"):
        wf.make_folds(SESS, 20, embargo=2)
    assert wf.make_folds(SESS, 20, embargo=2, allow_unsafe_embargo=True)
    assert wf.embargo_for_horizon(5) >= 5 and wf.embargo_for_horizon(60) >= 60 and wf.embargo_for_horizon(20) >= 20
    assert wf.embargo_for_horizon(5, {"swing": 3}) == 5                       # a short config value cannot undercut the horizon
    assert wf.make_folds(SESS, (5, 20, 60), include_val_in_fit=True)[0].embargo >= 60   # longest horizon governs


# ------------------------------------------------------------------ H1 N11
@pytest.mark.parametrize("h", [5, 20, 60])
def test_every_fold_training_row_ends_before_the_cutoff(h):
    labels = make_labels(h=h, n_sym=5, step=1)
    for f in wf.make_folds(SESS, h, val_years=0, include_val_in_fit=True):
        rows = wf.fold_training_rows(labels, f, SESS)
        cutoff = SESS[int(SESS.get_loc(f.test_start)) - f.embargo]
        assert len(rows) > 0
        assert (rows["label_end_date"] < cutoff).all()
        assert (rows["decision_date"] < f.test_start).all()
        # tight boundary (a decision every session): the latest kept label ends exactly one session before the cutoff
        assert rows["label_end_date"].max() == SESS[int(SESS.get_loc(cutoff)) - 1]
        assert wf.audit_training_rows(rows, f, SESS) == 0


def test_planted_boundary_row_does_not_reach_the_fit():
    f = folds()[0]
    cutoff = SESS[int(SESS.get_loc(f.test_start)) - f.embargo]
    plant = pd.DataFrame([("PLANT_EQ", cutoff - pd.Timedelta(days=40), H, cutoff, 9e9),
                          ("PLANT_AFTER", f.test_start - pd.Timedelta(days=3), H, f.test_start + pd.Timedelta(days=20), 9e9),
                          ("PLANT_OK", cutoff - pd.Timedelta(days=60), H, SESS[int(SESS.get_loc(cutoff)) - 1], 7e9)],
                         columns=LABELS.columns)
    seen = {}

    def fit(rows, fold):
        seen["rows"] = rows
        return None

    wf.run_fold(f, pd.concat([LABELS, plant], ignore_index=True), SESS, fit, noise_score)
    syms = set(seen["rows"]["symbol"])
    assert "PLANT_EQ" not in syms and "PLANT_AFTER" not in syms        # ends on the cutoff / after: dropped
    assert "PLANT_OK" in syms                                          # ends one session before the cutoff: kept


def test_unpurged_policy_is_caught_and_purged_policy_refuses_a_short_embargo():
    fl = wf.make_folds(SESS, H, embargo=0, val_years=0, include_val_in_fit=True, allow_unsafe_embargo=True)
    res = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score, fit_rows_policy="unpurged")
    assert res.audit_violations > 0
    assert wf.detect_leak(res, LABELS, horizon=H).flagged
    with pytest.raises(wf.LeakageError):                               # embargo 0 + the purge: the harness refuses to fit
        wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score)


def test_calendar_day_embargo_is_wrong_mutant_would_leak():
    """A purge counting calendar days instead of sessions keeps rows the session purge drops; the audit sees them."""
    f = folds()[0]
    cal_cut = f.test_start - pd.Timedelta(days=f.embargo)
    mutant = LABELS[(LABELS["decision_date"] >= f.train_start) & (LABELS["decision_date"] < f.train_stop)]
    mutant = mutant[mutant["label_end_date"] < cal_cut]
    assert wf.audit_training_rows(mutant, f, SESS) > 0


# ------------------------------------------------------------------ H2 isolation
def test_sentinels_never_reach_the_fit():
    f = folds()[0]
    cutoff = SESS[int(SESS.get_loc(f.test_start)) - f.embargo]
    sent = pd.DataFrame([("SENT", f.test_start + pd.Timedelta(days=10), H, f.test_start + pd.Timedelta(days=40), 1e12),
                         ("SENT", cutoff - pd.Timedelta(days=10), H, cutoff, 1e12)], columns=LABELS.columns)
    peaks = []

    def fit(rows, fold):
        peaks.append(rows["y_net"].abs().max())
        assert (rows["decision_date"] < fold.test_start).all()
        return None

    wf.run_fold(f, pd.concat([LABELS, sent], ignore_index=True), SESS, fit, noise_score)
    assert peaks and peaks[0] < 1e6


def _regime_frame():
    from vanta.regime import FEATURES
    idx = SESS[SESS >= "2014-01-01"]
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(len(idx), len(FEATURES))), index=idx, columns=list(FEATURES))
    X.iloc[:, 0] += np.where((np.arange(len(idx)) // 200) % 2 == 0, 0.0, 2.0)
    return X


def test_hmm_fit_never_sees_rows_at_or_after_test_start_minus_embargo():
    X = _regime_frame()
    f = [x for x in wf.make_folds(SESS, H, first_year=2008, val_years=0, include_val_in_fit=True) if x.test_start.year == 2018][0]
    boundary = SESS[int(SESS.get_loc(f.test_start)) - f.embargo]        # first FORBIDDEN session
    base = wf.fit_regime_for_fold(X, f, SESS)
    planted = X.copy()
    planted.loc[boundary:, :] = 1e6                                     # sentinel from the first forbidden session onward
    again = wf.fit_regime_for_fold(planted, f, SESS)
    assert np.array_equal(base.mu, again.mu) and np.array_equal(base.means, again.means)
    assert base.train_end < boundary
    # sensitivity: a fit that looked one session further WOULD move the parameters
    leaky_cut = X.loc[: boundary]
    moved = planted.loc[: boundary]
    from vanta.regime import fit_hmm
    assert not np.array_equal(fit_hmm(leaky_cut, boundary).mu, fit_hmm(moved, boundary).mu)


def test_run_fold_hmm_receives_only_cut_features(monkeypatch):
    X = _regime_frame()
    f = [x for x in wf.make_folds(SESS, H, first_year=2008, val_years=0, include_val_in_fit=True) if x.test_start.year == 2018][0]
    seen = {}

    def spy(features, train_end, **kw):
        seen["max"], seen["train_end"] = features.index.max(), train_end
        return "params"

    monkeypatch.setattr(wf, "fit_hmm", spy)
    res = wf.run_fold(f, LABELS, SESS, lambda r, fo: None, noise_score, regime_features=X)
    boundary = SESS[int(SESS.get_loc(f.test_start)) - f.embargo]
    assert seen["max"] < boundary and seen["train_end"] < boundary and res.regime == "params"


# ------------------------------------------------------------------ H3 detector
def test_detector_flags_leaky_not_clean():
    check_detector(wf.detect_leak)


def test_mutant_detector_that_never_flags_fails_the_check():
    with pytest.raises(AssertionError, match="leaky run not flagged"):
        check_detector(never_flags)


def test_lookahead_reader_is_flagged_by_the_detector():
    fl = folds()
    res = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score)
    rd = wf.PitReader(SESS, before_nightly_run=False, lead_sessions=21)
    assert rd.leaky
    assert wf.detect_leak(res, LABELS, horizon=H, reader=rd).flagged


def test_ic_is_one_for_perfect_scores_and_near_zero_for_noise():
    fl = folds()
    perfect = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None,
                                 lambda m, rows, f, rd: rows.merge(LABELS, on=["symbol", "decision_date"])["y_net"].to_numpy())
    assert wf.rank_ic_by_date(perfect.scored).mean() > 0.99
    noise = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score)
    assert abs(wf.rank_ic_by_date(noise.scored).mean()) < 0.03


# ------------------------------------------------------------------ H4 explicit before_nightly_run
def test_every_read_as_of_call_in_the_harness_is_explicit():
    tree = ast.parse(inspect.getsource(wf))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "read_as_of"]
    assert calls, "no store read found: the check is vacuous"
    for c in calls:
        kws = {k.arg: k.value for k in c.keywords}
        assert "before_nightly_run" in kws, "read_as_of called without before_nightly_run"
        assert not isinstance(kws["before_nightly_run"], ast.Constant) or isinstance(kws["before_nightly_run"].value, bool)


def test_reader_has_no_default_and_passes_the_value_through(monkeypatch):
    with pytest.raises(TypeError):
        wf.PitReader(SESS)
    got = []
    monkeypatch.setattr(wf, "read_as_of", lambda *a, **k: got.append(k) or pd.DataFrame())
    for flag in (True, False):
        wf.PitReader(SESS, before_nightly_run=flag).read(["A"], SESS[100], ["delivery"])
    assert [k["before_nightly_run"] for k in got] == [True, False]


def test_absent_rows_are_unavailable_not_values():
    df = pd.DataFrame({"symbol": list("ab"), "delivery_row_absent": [True, False], "x": [np.nan, 1.0]})
    out = wf.drop_absent(df, ["delivery"])
    assert list(out["symbol"]) == ["b"] and out["x"].notna().all()


def test_hmm_fit_stops_at_train_window_end_validation_year_untouched():
    X = _regime_frame()
    f = [x for x in wf.make_folds(SESS, H, first_year=2008, val_years=1) if x.test_start.year == 2018][0]
    assert not f.include_val_in_fit and f.val_start < f.test_start
    last_train = SESS[int(SESS.searchsorted(f.val_start)) - 1]
    base = wf.fit_regime_for_fold(X, f, SESS)
    assert base.train_end == last_train
    planted = X.copy()
    planted.loc[f.val_start:, :] = 1e6                                   # sentinel across the validation year onward
    again = wf.fit_regime_for_fold(planted, f, SESS)
    assert np.array_equal(base.mu, again.mu) and np.array_equal(base.means, again.means)
    # mutant: the old window (fold_train_end only) reads the validation year and moves the parameters
    from vanta.regime import fit_hmm, fold_train_end
    old_end = fold_train_end(SESS, f.test_start, f.embargo)
    assert old_end >= f.val_start
    assert not np.array_equal(fit_hmm(X.loc[:old_end], old_end).mu, fit_hmm(planted.loc[:old_end], old_end).mu)


def test_audit_alone_flags_unpurged_rows_under_a_normal_embargo():
    """Fold.embargo == required_embargo, rows NOT purged: the independent audit is the only rule that fires."""
    fl = folds()
    assert all(f.embargo == f.required_embargo for f in fl)
    res = wf.run_walkforward(fl, LABELS, SESS, lambda r, f: None, noise_score, fit_rows_policy="unpurged")
    rep = wf.detect_leak(res, LABELS, horizon=H)
    assert rep.audit_violations > 0 and not rep.config_embargo_short
    assert len(rep.reasons) == 1 and "N11" in rep.reasons[0], rep.reasons
