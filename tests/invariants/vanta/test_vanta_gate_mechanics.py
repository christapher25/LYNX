"""
tests/invariants/vanta/test_vanta_gate_mechanics.py

VANTA Stage 13C invariants N10 and N13 (docs/vanta/VANTA_SPEC.md).

  N10  Normalisation and winsorisation come from the training fold only    changing test-fold labels does not change training gains
                                                                           (and the clip bounds are exactly the train 2nd/98th pct).
  N13  p-value mechanics     p in (0, 1] and non-increasing in score; equal weights reduce to the unweighted conformal p-value;
                             published names are a subset of the BH set, at most 10, with cap_binding logged.
       (supporting)          the weights are exp(-age/tau) * exp(-L1 distance of regime posteriors); BH at q is the step-UP procedure.

Same pattern as the other VANTA tests: real module (skipped until it exists), a correct toy, deliberately broken toys.
Not tested here because the spec is silent: mean/std feature normalisation (only the gain's winsorisation has a defined function).
"""

import math

import numpy as np
import pandas as pd
import pytest

from vanta_toys_gate import FILTER, MODEL, build_fold_gains, cal_rows, toy_api

T = pd.Timestamp("2024-06-03")
PI_T = (1.0, 0.0, 0.0)
KW = dict(model_version=MODEL, candidate_filter_version=FILTER)
HORIZON, ATR_PCT = 20, 0.02
UNIT = ATR_PCT * math.sqrt(HORIZON)               # y = k * UNIT  <=>  gain = k


# ---------------------------------------------------------------- N10
def _fold_labels(test_y):
    k = np.arange(51)                                          # train gains 0..50: 2nd pct = 1, 98th pct = 49 exactly
    y = np.concatenate([k * UNIT, np.asarray(test_y, float) * UNIT])
    n_tr = len(k)
    df = pd.DataFrame({"y_net": y, "atr_pct": ATR_PCT, "horizon": HORIZON})
    mask = np.arange(len(df)) < n_tr
    return df, mask


def check_n10(build):
    """N10: winsorisation bounds are the training fold's 2nd/98th percentile; test-fold labels never move training gains."""
    df, mask = _fold_labels([-400, 7, 800, 25])
    g = build(df, mask, HORIZON)
    train = g[mask].to_numpy()
    np.testing.assert_allclose(train[0], 1.0, atol=1e-9)         # clipped up to the train 2nd percentile
    np.testing.assert_allclose(train[50], 49.0, atol=1e-9)       # clipped down to the train 98th percentile
    np.testing.assert_allclose(train[1:50], np.arange(1, 50), atol=1e-9)
    # test rows are clipped with the TRAIN bounds, not their own
    np.testing.assert_allclose(g[~mask].to_numpy(), [1.0, 7.0, 49.0, 25.0], atol=1e-9)

    # change the test-fold labels arbitrarily: training gains are bit-for-bit unchanged
    df2, mask2 = _fold_labels([-9e5, 3e6, 1e4, -2e4])
    g2 = build(df2, mask2, HORIZON)
    np.testing.assert_array_equal(g2[mask2].to_numpy(), train)
    np.testing.assert_allclose(g2[~mask2].to_numpy(), [1.0, 49.0, 49.0, 1.0], atol=1e-9)


def test_n10_real():
    build = pytest.importorskip("vanta.gains").build_fold_gains
    check_n10(build)


def test_n10_reference_toy_passes():
    check_n10(build_fold_gains)


def test_n10_broken_toy_fails():
    broken = lambda df, m, h: build_fold_gains(df, m, h, bugs=frozenset({"fit_on_all_rows"}))   # noqa: E731
    with pytest.raises(AssertionError):
        check_n10(broken)


# ---------------------------------------------------------------- N13
CAL_S = [1.0, 2.0, 3.0, 4.0, 5.0]
CAL_Y = [-1.0, 1.0, 0.0, 1.0, -2.0]               # losers (y <= 0, y == 0 counts): scores 1, 3, 5
TEST_S = [0.0, 3.0, 6.0]                           # 3.0 ties a losing calibration score


def check_n13_pvalues(api):
    """N13: known-answer p-values; p in (0,1]; non-increasing in score; equal weights == unweighted conformal p."""
    ones = np.ones(5)
    p = api.conformal_pvalues(CAL_S, CAL_Y, ones, TEST_S)
    np.testing.assert_allclose(p, [(1 + 3) / 6, (1 + 2) / 6, (1 + 0) / 6])         # score 3 ties a loser: counted (conservative)
    # weighted known answer
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pw = api.conformal_pvalues(CAL_S, CAL_Y, w, TEST_S)
    np.testing.assert_allclose(pw, [(1 + 1 + 3 + 5) / 16, (1 + 3 + 5) / 16, 1 / 16])

    rng = np.random.default_rng(7)
    s, y = rng.normal(size=300), rng.normal(size=300) * 0.05
    wts = rng.uniform(0.01, 1.0, 300)
    grid = np.sort(rng.normal(size=200))
    for ww in (np.ones(300), wts):
        pp = api.conformal_pvalues(s, y, ww, grid)
        assert ((pp > 0) & (pp <= 1)).all()
        assert (np.diff(pp) <= 1e-15).all()                                          # non-increasing in score
    # even with zero losers the p-value is positive (the +1) and a score below every loser gives exactly 1
    assert (api.conformal_pvalues([1.0, 2.0], [1.0, 2.0], np.ones(2), [0.0, 9.0]) > 0).all()
    np.testing.assert_allclose(api.conformal_pvalues([1.0, 2.0], [-1.0, -1.0], np.ones(2), [0.0]), [1.0])
    # equal weights reduce to the unweighted form (1 + #{y<=0, s_i>=s_j}) / (1 + n)
    pu = api.conformal_pvalues(s, y, np.ones(300), grid)
    ref = [(1 + np.sum((y <= 0) & (s >= g))) / (1 + 300) for g in grid]
    np.testing.assert_allclose(pu, ref)


def check_n13_weights_and_bh(api):
    """Supporting: weights exp(-age/tau)*exp(-L1); BH is step-UP at q."""
    w = api.regime_age_weights([0.0, 26.0, 13.0], [[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]], (1.0, 0.0, 0.0), 26)
    np.testing.assert_allclose(w, [1.0, math.exp(-1) * math.exp(-2), math.exp(-0.5) * math.exp(-1.0)])
    # m=4, thresholds q*i/m = .075 .15 .225 .3 ; input deliberately unsorted
    sel = api.bh_select([0.5, 0.01, 0.2, 0.04], 0.3)
    assert list(sel) == [False, True, True, True]
    # step-up: the 2nd p (0.16) fails its own threshold (.15) but the 3rd (0.2) passes (.225), so ranks 1..3 are all selected
    sel = api.bh_select([0.20, 0.5, 0.01, 0.16], 0.3)
    assert list(sel) == [True, False, True, True]
    assert not api.bh_select([0.9, 0.8], 0.3).any()                                  # zero selected is valid


def _gate_cal(n_cal=100):
    """All-loser calibration with low scores: any candidate scoring above them gets the minimum p = 1/(1+sum w), equal for all."""
    d = T - pd.Timedelta(weeks=1)
    return cal_rows(d, [f"C{i:03d}" for i in range(n_cal)], np.linspace(0.0, 1.0, n_cal), [-0.01] * n_cal,
                    [d + pd.Timedelta(days=6)] * n_cal)


def check_n13_selection(api):
    # NOTE: the spec says only "ties broken by score"; HIGHER score first (user-confirmed 2026-10-27, decisions log Q4).
    """N13: published subset of BH set, at most 10, ties by higher score, cap_binding logged; zero published is valid."""
    cal = _gate_cal()

    def gate(scores):
        c = pd.DataFrame({"symbol": [f"X{i:02d}" for i in range(len(scores))], "score": scores})
        return api.run_gate(c, cal, T, PI_T, tau_weeks=26, q=0.3, cap=10, ess_floor=0, **KW)

    # 20 candidates, all strongly significant (identical p) -> BH set of 20, cap binds, top-10 by score published
    log = gate([float(s) for s in range(10, 30)])
    assert log["in_bh_set"].sum() == 20 and log["published"].sum() == 10
    assert set(log.loc[log["published"], "score"]) == set(float(s) for s in range(20, 30))
    assert (log["cap_binding"]).all()
    assert (log.loc[log["published"], "in_bh_set"]).all()
    assert (log["p_value"] > 0).all() and (log["p_value"] <= 1).all()
    assert (log["n_calibration"] == 100).all()

    # 5 significant candidates -> all published, cap not binding
    log = gate([10.0, 11.0, 12.0, 13.0, 14.0])
    assert log["published"].sum() == 5 and not log["cap_binding"].any()

    # a mix: 3 significant, 4 not (scores below every calibration loser give p = 1) -> only the 3 are published
    log = gate([10.0, 11.0, 12.0, -1.0, -2.0, -3.0, -4.0])
    assert list(log["published"]) == [True] * 3 + [False] * 4
    assert (log.loc[log["published"], "in_bh_set"]).all()
    assert (log.loc[~log["in_bh_set"], "published"] == False).all()                 # noqa: E712  published subset of BH set
    assert (log["in_bh_set"] == (log["p_value"] <= log["bh_threshold"])).all()

    # nobody significant: publishing zero names is valid, not an error
    log = gate([-1.0, -2.0, -3.0])
    assert log["published"].sum() == 0 and not log["in_bh_set"].any()

    # p non-increasing in score through the gate as well
    log = gate([0.5, 0.2, 0.9, -1.0, 2.0]).sort_values("score")
    assert (np.diff(log["p_value"].to_numpy()) <= 1e-15).all()


def _real():
    return pytest.importorskip("vanta.gate")


def test_n13_pvalues_real():
    check_n13_pvalues(_real())


def test_n13_weights_and_bh_real():
    check_n13_weights_and_bh(_real())


def test_n13_selection_real():
    check_n13_selection(_real())


def test_n13_reference_toy_passes():
    api = toy_api()
    check_n13_pvalues(api)
    check_n13_weights_and_bh(api)
    check_n13_selection(api)


@pytest.mark.parametrize("bug", ["no_plus_one", "strict_ties", "ignore_y", "wrong_direction", "ignore_weights"])
def test_n13_pvalue_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n13_pvalues(toy_api(bug))


@pytest.mark.parametrize("bug", ["age_only", "bonferroni", "step_down"])
def test_n13_weights_bh_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n13_weights_and_bh(toy_api(bug))


@pytest.mark.parametrize("bug", ["no_cap", "published_not_subset_of_bh", "cap_not_logged", "tie_lowest_score"])
def test_n13_selection_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n13_selection(toy_api(bug))
