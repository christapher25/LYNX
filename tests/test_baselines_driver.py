"""
Stage 14 baselines: tests for the pieces tests-15c did not cover (factories without the libraries, A1 publish threshold, all-missing
scoring, the driver's point-in-time feature build and JSON helper). Model libraries are not installed, so toy models stand in.
"""

import types

import numpy as np
import pandas as pd
import pytest

import vanta.baselines as bl

TS = pd.Timestamp
NAN = np.nan


def test_factories_name_the_missing_package_and_refuse_a_seed_override():
    for name, make in (("catboost", bl.b1_model_factory), ("xgboost", bl.a1_model_factory)):
        try:
            __import__(name)
        except ImportError:
            with pytest.raises(ImportError, match=name):
                make()
        with pytest.raises(ValueError):
            make(random_seed=3)


def test_b1_params_hold_no_seed_and_grid_does_not_touch_one():
    assert not {"random_seed", "random_state", "seed"} & set(bl.B1_PARAMS)
    assert all(not {"random_seed", "random_state", "seed"} & set(g) for g in bl.B1_PARAM_GRID)
    assert len(bl.B1_PARAM_GRID) == len({tuple(sorted(g.items())) for g in bl.B1_PARAM_GRID})


class _First:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X, float)[:, 0]


def test_row_with_every_feature_missing_scores_nan_not_a_model_output():
    d = TS("2020-01-03")
    feats = pd.DataFrame({"decision_date": d, "symbol": list("ABC"), "f0": [1.0, NAN, 3.0], "f1": [NAN, NAN, 5.0]})
    rows = pd.DataFrame({"decision_date": d, "symbol": list("ABCD")})          # D is not in the panel at all
    model = types.SimpleNamespace(models=[_First(0)])
    s = bl.make_score_fn(feats, ["f0", "f1"])(model, rows, None, None)
    assert s[0] == 1.0 and np.isnan(s[1]) and s[2] == 3.0 and np.isnan(s[3])


def test_duplicate_feature_rows_are_refused():
    d = TS("2020-01-03")
    feats = pd.DataFrame({"decision_date": [d, d], "symbol": ["A", "A"], "f0": [1.0, 2.0]})
    with pytest.raises(ValueError):
        bl.make_fit_fn(feats, ["f0"], lambda s: _First(s))


def test_fit_with_no_finite_label_raises_instead_of_fitting_on_nothing():
    d = TS("2020-01-03")
    feats = pd.DataFrame({"decision_date": [d], "symbol": ["A"], "f0": [1.0]})
    rows = pd.DataFrame({"decision_date": [d], "symbol": ["A"], "y_net": [NAN]})
    with pytest.raises(ValueError):
        bl.make_fit_fn(feats, ["f0"], lambda s: _First(s))(rows, None)


def _result(scored):
    f = types.SimpleNamespace(fold_id=0, train_start=TS("2016-01-04"), val_start=TS("2020-01-02"), test_start=TS("2020-01-02"),
                              test_end=TS("2020-12-31"), embargo=20)
    return types.SimpleNamespace(fold=f, n_train=10, scored=scored)


def test_fold_report_publish_threshold_cuts_selection_but_not_rank_ic():
    d = TS("2020-01-03")
    sc = pd.DataFrame({"decision_date": d, "symbol": list("ABCDE"), "score": [0.9, 0.6, 0.5, 0.4, 0.2],
                       "y_net": [0.10, 0.02, 0.30, -0.10, -0.20], "fill_status": "FILLED"})
    plain = bl.fold_report([_result(sc)], n=5, min_names=3)[0]
    cut = bl.fold_report([_result(sc)], n=5, min_names=3, publish_above=0.5)[0]
    assert plain["n_published"] == 5 and cut["n_published"] == 2               # 0.5 is not strictly above 0.5
    assert cut["net_per_signal"] == pytest.approx(0.06)
    assert cut["rank_ic"] == plain["rank_ic"]


def test_choose_hyperparameters_empty_inner_split_raises():
    import vanta.walkforward as wf
    sess = pd.bdate_range("2008-01-01", "2019-12-31")
    f = wf.make_folds(sess, 20)[0]
    lab = pd.DataFrame({"symbol": ["A"], "decision_date": [TS("2030-01-01")], "label_end_date": [TS("2030-02-01")], "y_net": [0.1]})
    feats = lab[["decision_date", "symbol"]].assign(f0=1.0)
    with pytest.raises(ValueError):
        bl.choose_hyperparameters([{"a": 1}], lambda p, s: _First(s), feats, lab, f, sess)


# ------------------------------------------------------------------------------------------------ driver helpers
def test_jsonable_handles_timestamps_numpy_and_nan():
    run_baselines = pytest.importorskip("scripts.run_baselines")
    out = run_baselines._jsonable({"d": TS("2020-01-02"), "n": np.int64(3), "x": np.float64("nan"), "t": (1, 2)})
    assert out == {"d": "2020-01-02", "n": 3, "x": None, "t": [1, 2]}


def test_build_features_is_point_in_time_and_keeps_missing_as_nan(monkeypatch):
    """build_features reads each decision date as of that date only: poisoning every row after it changes nothing, and a NaN stays NaN."""
    rb = pytest.importorskip("scripts.run_baselines")
    from pitstore.store import BLOCK_SPECS
    sess = pd.bdate_range("2020-01-01", "2020-03-31")
    spec = BLOCK_SPECS["breadth"]
    dates = sess[:40]
    base = pd.DataFrame({"date": dates, "universe_pct_above_200dma": np.linspace(10, 50, 40), "universe_n_evaluable": 100.0,
                         "is_missing_breadth": False})
    base.loc[5, "universe_pct_above_200dma"] = NAN

    def run(frame):
        monkeypatch.setattr(rb, "indicator_panel", lambda d: d.assign(**{c: np.nan for c in rb.INDICATOR_FEATURES}))
        monkeypatch.setattr(rb, "_cached_loader", lambda: (lambda sp, symbol, upto: frame[frame["date"] <= upto]))
        df = pd.DataFrame({"decision_date": [sess[10], sess[20]] * 2, "symbol": ["A", "A", "B", "B"], "mom": [0.1, 0.2, NAN, 0.4],
                           "liquidity_rank": 1.0, "univ_rank": 2.0})
        return rb.build_features(df, sess, blocks=("breadth",))

    (clean, cols), (poisoned, _) = run(base), run(base.assign(universe_pct_above_200dma=lambda d: np.where(
        d["date"] > sess[20], 1e9, d["universe_pct_above_200dma"])))
    pd.testing.assert_frame_equal(clean, poisoned)
    assert cols == ["mom", "liquidity_rank", "univ_rank", "universe_pct_above_200dma", "universe_n_evaluable", *rb.INDICATOR_FEATURES]
    assert clean["mom"].isna().sum() == 1                                         # the missing momentum stayed missing
    assert spec.scope == "market"


# ------------------------------------------------------------------------------------------------ indicator features
def _ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.bdate_range("2018-01-01", periods=n), "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1_000, 5_000, n).astype(float)})


def _ind(rb, ohlcv, dates):
    df = pd.DataFrame({"decision_date": pd.DatetimeIndex(dates), "symbol": "A"})
    return rb.indicator_panel(df, loader=lambda s: ohlcv if s == "A" else None)


def test_indicator_features_are_causal_truncation_and_poison():
    rb = pytest.importorskip("scripts.run_baselines")
    o = _ohlcv()
    dates = o["date"].iloc[[250, 300, 350]]
    full = _ind(rb, o, dates)
    cut = _ind(rb, o.iloc[:351], dates)
    poisoned = o.copy()
    for c in ("open", "high", "low", "close", "volume"):
        poisoned.loc[351:, c] = 1e9
    pd.testing.assert_frame_equal(full, cut)
    pd.testing.assert_frame_equal(full, _ind(rb, poisoned, dates))
    early = _ind(rb, o.iloc[:251], dates[:1])                      # the value at the last available bar needs no later bar
    pd.testing.assert_frame_equal(full.iloc[:1], early)


def test_indicator_warmup_and_missing_symbol_stay_nan_never_zero():
    rb = pytest.importorskip("scripts.run_baselines")
    o = _ohlcv()
    early = _ind(rb, o, o["date"].iloc[[3]])
    num = early[list(rb.INDICATOR_FEATURES)]
    assert num.isna().any(axis=None) and not (num == 0.0).all(axis=None)
    df = pd.DataFrame({"decision_date": [o["date"].iloc[300]], "symbol": ["ZZ"]})
    gone = rb.indicator_panel(df, loader=lambda s: None)
    assert len(gone) == 1 and gone[list(rb.INDICATOR_FEATURES)].isna().all(axis=None)
    off = pd.DataFrame({"decision_date": [pd.Timestamp("2030-01-01")], "symbol": ["A"]})   # no bar on that date: NaN, no carry-forward
    assert rb.indicator_panel(off, loader=lambda s: o)[list(rb.INDICATOR_FEATURES)].isna().all(axis=None)


def test_indicator_source_has_no_zero_fill():
    import inspect
    rb = pytest.importorskip("scripts.run_baselines")
    from tests.test_baselines import check_source_has_no_zero_fill
    check_source_has_no_zero_fill(inspect.getsource(rb))


def test_scorer_feature_is_per_date_cross_section_only():
    rb = pytest.importorskip("scripts.run_baselines")
    o1, o2 = _ohlcv(seed=1), _ohlcv(seed=2)
    d = o1["date"].iloc[[300, 350]]
    df = pd.DataFrame({"decision_date": list(d) * 2, "symbol": ["A", "A", "B", "B"]})
    load = lambda s: o1 if s == "A" else o2          # noqa: E731
    ind = rb.indicator_panel(df, loader=load)
    full = rb.add_scorer_feature(ind)
    first = rb.add_scorer_feature(ind[ind["decision_date"] == d.iloc[0]])
    a = full[full["decision_date"] == d.iloc[0]][rb.A1_SCORER_COL].to_numpy()
    np.testing.assert_array_equal(a, first[rb.A1_SCORER_COL].to_numpy())      # a later date in the frame changes nothing


def test_params_by_horizon_hold_no_seed_and_cover_both_horizons():
    assert set(bl.B1_PARAMS_BY_HORIZON) == {5, 20}
    for p in bl.B1_PARAMS_BY_HORIZON.values():
        assert not {"random_seed", "random_state", "seed"} & set(p)


def test_first_scored_and_judged_week():
    weeks = pd.date_range("2012-01-06", periods=80, freq="7D")
    sc = pd.DataFrame({"decision_date": weeks, "symbol": "A", "score": 1.0, "y_net": 0.0})
    sc.loc[0, "score"] = np.nan                                              # an unscored first week is not the first scored week
    r = bl.first_scored_and_judged_week(sc, "2013-01-01")
    assert r["first_scored_week"] == weeks[1]
    assert r["first_judged_week"] == weeks[53]                               # 52 weeks after the first scored week, and >= 2013-01-01
    assert pd.isna(bl.first_scored_and_judged_week(sc.iloc[:40], "2013-01-01")["first_judged_week"])      # history genuinely too short
    assert pd.isna(bl.first_scored_and_judged_week(sc.iloc[:0], "2013-01-01")["first_scored_week"])


def test_judged_week_is_horizon_independent_and_satisfiable_for_long_labels():
    """Regression for the unsatisfiable trailing-window reading: it never judged a week when labels mature slowly (h=20). The rule
    mirrors the calibration-window bound (decision_date >= T - 52 weeks) and does not look at maturity at all."""
    weeks = pd.date_range("2011-01-07", periods=120, freq="7D")
    sc = pd.DataFrame({"decision_date": weeks, "symbol": "A", "score": 1.0, "y_net": 0.0})
    r = bl.first_scored_and_judged_week(sc, "2013-01-01")
    assert pd.notna(r["first_judged_week"]) and r["first_judged_week"] >= pd.Timestamp("2013-01-01")
    assert r["first_judged_week"] - r["first_scored_week"] >= pd.Timedelta(weeks=52)


# ------------------------------------------------------------------------------------------------ reviewer-15c fixes
def test_scorer_feature_is_nan_when_every_scorer_input_is_missing_and_on_zero_variance_dates():
    rb = pytest.importorskip("scripts.run_baselines")
    d1, d2 = TS("2020-01-03"), TS("2020-01-10")
    cols = {c: np.nan for c in ("close", *rb.INDICATOR_SCORER_INPUTS, *rb.INDICATOR_FEATURES)}
    base = pd.DataFrame([dict(decision_date=d1, symbol=s, **cols) for s in "ABC"])
    for c in ("rsi_14", "macd_delta", "bb_position", "atr_normalized", "obv_trend", "ema_200", "volume_ratio"):
        base[c] = [20.0, 50.0, 80.0] if c == "rsi_14" else [0.0, 1.0, 2.0] if c != "ema_200" else [90.0, 100.0, 110.0]
    base["close"] = [100.0, 100.0, 100.0]
    allnan = pd.DataFrame([dict(decision_date=d1, symbol="D", **cols)])                 # D: every scorer input missing
    flat = pd.DataFrame([dict(decision_date=d2, symbol=s, **{**cols, **{c: 1.0 for c in cols}}) for s in "AB"])   # zero variance date
    out = rb.add_scorer_feature(pd.concat([base, allnan, flat], ignore_index=True)).set_index(["decision_date", "symbol"])[rb.A1_SCORER_COL]
    assert np.isnan(out[(d1, "D")])                                    # not 50.0 z-scored
    assert out.loc[d1].drop("D").notna().all()                         # the real names still score
    assert out.loc[d2].isna().all()                                    # zero variance: unavailable, not 0.0


def test_b1_run_is_refused_while_params_are_the_placeholder(monkeypatch):
    rb = pytest.importorskip("scripts.run_baselines")
    monkeypatch.setitem(bl.B1_PARAMS_BY_HORIZON, 5, dict(bl.B1_PARAMS))   # the committed dict is tuned since 2026-11-01; simulate the placeholder
    with pytest.raises(SystemExit) as e:
        rb.require_tuned_b1_params(5, mechanics=False)
    assert e.value.code not in (0, None)
    rb.require_tuned_b1_params(5, mechanics=True)                      # a labelled mechanics check is allowed
    monkeypatch.setitem(bl.B1_PARAMS_BY_HORIZON, 5, {**bl.B1_PARAMS, "depth": 4})
    rb.require_tuned_b1_params(5, mechanics=False)                     # a tuned dict passes


def test_flagged_leak_check_gets_a_prefix_and_a_nonzero_exit():
    rb = pytest.importorskip("scripts.run_baselines")
    import vanta.walkforward as wf
    clean = wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, 0)
    flagged = wf.LeakReport(True, ["x"], 0.2, 0.0, 0.0, 0.0, 0.0, 0, False, 0)
    audit = wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 3, False, 0)
    assert rb.leak_prefix(clean) == ""
    assert rb.leak_prefix(flagged) == "LEAKFLAGGED_" and rb.leak_prefix(audit) == "LEAKFLAGGED_"


def test_no_data_is_counted_apart_from_no_fill():
    d = TS("2020-01-03")
    sel = pd.DataFrame({"decision_date": d, "symbol": list("ABCD"), "y_net": [0.1, NAN, NAN, NAN],
                        "fill_status": ["FILLED", "NO_FILL_UPPER_LOCK", "NO_DATA", "NO_DATA"]})
    r = bl.net_per_signal(sel)
    assert (r["n_no_fill"], r["n_no_data"], r["n_filled"], r["n_published"]) == (1, 2, 1, 4)
    rep = bl.fold_report([_result(sel.assign(score=[0.9, 0.8, 0.7, 0.6]))], n=4, min_names=2)[0]
    assert rep["n_no_data"] == 2 and rep["n_no_fill"] == 1

