"""
Extra Stage 13C checks (deferred from earlier reviews): N8 at the level of what run_gate computes from, and the Kish ESS fallback.
"""

import numpy as np
import pandas as pd
import pytest

from vanta_toys_gate import FILTER, MODEL, cal_rows, calibration_world, oracle_p

gate = pytest.importorskip("vanta.gate")

T = pd.Timestamp("2024-06-03")
PI_T = (1.0, 0.0, 0.0)
KW = dict(model_version=MODEL, candidate_filter_version=FILTER)
CANDS = pd.DataFrame({"symbol": list("ABCDEF"), "score": [-1.0, 5.5, 19.5, 25.5, 29.5, 50.0]})


def _gate(cal, **kw):
    return gate.run_gate(CANDS, cal, T, PI_T, tau_weeks=26, q=0.3, cap=10, ess_floor=kw.pop("ess_floor", 0), **KW)


def test_n8_writer_level_no_unusable_row_reaches_pvalues():
    """Rows with y_net NaN, unmatured, NaT matured_at, rejected versions or non-OOS are poison: p-values and n_calibration
    (the logged count) are identical with and without them, and equal the independent oracle on the clean rows."""
    cal = calibration_world()
    d = T - pd.Timedelta(weeks=1)
    m = d + pd.Timedelta(days=6)
    poison = pd.concat([
        cal_rows(d, ["P1"], [100.0], [np.nan], [m]),                                     # matured_at set but y unknown
        cal_rows(d, ["P2"], [100.0], [-0.5], [T + pd.Timedelta(days=1)]),                # unmatured
        cal_rows(d, ["P3"], [100.0], [-0.5], [pd.NaT]),
        cal_rows(d, ["P4"], [100.0], [-0.5], [m], model="other"),
        cal_rows(d, ["P5"], [100.0], [-0.5], [m], filt="other"),
        cal_rows(d, ["P6"], [100.0], [-0.5], [m], oos=False),
    ], ignore_index=True)
    clean, dirty = _gate(cal), _gate(pd.concat([cal, poison], ignore_index=True))
    assert (dirty["n_calibration"] == 120).all()
    np.testing.assert_array_equal(dirty["p_value"].to_numpy(), clean["p_value"].to_numpy())
    np.testing.assert_allclose(dirty["p_value"].to_numpy(), oracle_p(cal, T, PI_T, 26, CANDS["score"]))
    # the log covers every scored candidate, not only the published ones
    assert len(dirty) == len(CANDS) and dirty["published"].sum() < len(CANDS)


def test_n8_published_column_is_ignored():
    cal = calibration_world()
    flipped = cal.assign(published=~cal["published"])
    np.testing.assert_array_equal(_gate(cal)["p_value"].to_numpy(), _gate(flipped)["p_value"].to_numpy())


def _regime_world():
    """Half the calibration rows sit in a regime far from pi_T, so regime weighting shrinks ESS well below age-only ESS."""
    cal = calibration_world(weeks=4, n=30)
    cal["regime_posterior"] = [[0.0, 0.0, 1.0] if i % 2 else [1.0, 0.0, 0.0] for i in range(len(cal))]
    return cal


def test_kish_fallback_fires_below_floor_and_is_logged():
    cal = _regime_world()
    off = _gate(cal, ess_floor=0)
    assert not off["age_only_fallback"].any()
    ess = float(off["effective_n"].iloc[0])
    assert 0 < ess < 120

    # floor above the ESS: age-only weights are used, and the log says so; effective_n is still the age x regime ESS
    on = _gate(cal, ess_floor=ess + 1)
    assert on["age_only_fallback"].all()
    np.testing.assert_allclose(on["effective_n"].to_numpy(), ess)
    sel = gate.select_calibration(cal, T, **KW)
    age = (T - sel["decision_date"]).dt.days.to_numpy() / 7.0
    expected = gate.conformal_pvalues(sel["score"], sel["y_net"], np.exp(-age / 26), CANDS["score"])
    np.testing.assert_allclose(on["p_value"].to_numpy(), expected)
    assert not np.allclose(on["p_value"].to_numpy(), off["p_value"].to_numpy())

    # floor at or below the ESS: no fallback
    at = _gate(cal, ess_floor=ess - 1e-6)
    assert not at["age_only_fallback"].any()
    np.testing.assert_allclose(at["p_value"].to_numpy(), off["p_value"].to_numpy())


def test_kish_ess_formula_and_empty_calibration():
    assert gate.kish_ess(np.ones(10)) == pytest.approx(10.0)
    assert gate.kish_ess([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert gate.kish_ess([]) == 0.0
    empty = _gate(calibration_world().iloc[0:0])
    assert (empty["p_value"] == 1.0).all() and not empty["published"].any() and (empty["n_calibration"] == 0).all()


# ---- missing is missing, stale rows, gains: each check runs on the real module AND fails on a source-mutated copy of the OLD behaviour
import inspect
import types

from vanta import gains as _gains


def _mutant(module, edits):
    """A fresh copy of `module` compiled from its source with each `old` replaced by `new` (the old, buggy behaviour)."""
    src = inspect.getsource(module)
    for _fn, old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = {"__name__": module.__name__ + "_mutant"}
    exec(compile(src, "<mutant>", "exec"), ns)
    return types.SimpleNamespace(**ns)


def _raises_value(f):
    try:
        f()
    except ValueError:
        return True
    return False


G_MUTANTS = {
    "nan_cand_best_p": [("conformal_pvalues", "    return np.where(bad_t, 1.0, p)", "    return p"),
                        ("conformal_pvalues", "    t = np.where(bad_t, 0.0, t)\n", "")],
    "nan_cand_not_excluded": [("run_gate", " & np.isfinite(scores)", "")],
    "nan_cal_score_kept": [("_usable_mask", '    m &= np.isfinite(cal["score"].astype(float))\n', "")],
    "nan_cal_score_not_rejected": [("validate_calibration", 'if not np.isfinite(cal["score"].astype(float)).all():',
                                    "if False:")],
    "nan_posterior_no_fallback": [("run_gate", "(not np.isfinite(w).all()) or ", "")],
    "no_window_filter": [("_usable_mask", '    m &= cal["decision_date"] >= T - pd.Timedelta(weeks=window_weeks)\n', "")],
}


def check_nan_candidate(api):
    s = [1.0, 2.0, 3.0, 4.0]
    p = api.conformal_pvalues(s, [-1, 1, -1, 1], np.ones(4), [np.nan, 2.5, np.inf])
    assert p[0] == 1.0 and p[2] == 1.0 and p[1] < 1.0
    cal = calibration_world()
    for q in (0.05, 0.3, 0.9, 1.0):
        c = pd.DataFrame({"symbol": list("ABC"), "score": [np.nan, 50.0, 40.0]})
        out = api.run_gate(c, cal, T, PI_T, model_version=MODEL, candidate_filter_version=FILTER, q=q, ess_floor=0)
        assert not out.loc[0, "published"] and not out.loc[0, "in_bh_set"] and out.loc[0, "p_value"] == 1.0
        assert list(out["score_missing"]) == [True, False, False]
    # k == m: every other candidate is in the set and the NaN one still is not
    c = pd.DataFrame({"symbol": list("ABC"), "score": [50.0, 40.0, np.nan]})
    out = api.run_gate(c, cal, T, PI_T, model_version=MODEL, candidate_filter_version=FILTER, q=1.0, ess_floor=0)
    assert list(out["in_bh_set"]) == [True, True, False]


def check_nan_cal_score(api):
    cal = calibration_world()
    d = T - pd.Timedelta(weeks=1)
    mixed = pd.concat([cal, cal_rows(d, ["NANS"], [np.nan], [-0.5], [d + pd.Timedelta(days=6)])], ignore_index=True)
    assert "NANS" not in set(api.select_calibration(mixed, T, **KW)["symbol"])
    assert _raises_value(lambda: api.validate_calibration(mixed, T, **KW))
    clean = api.run_gate(CANDS, cal, T, PI_T, ess_floor=0, **KW)
    dirty = api.run_gate(CANDS, mixed, T, PI_T, ess_floor=0, **KW)
    np.testing.assert_array_equal(dirty["p_value"].to_numpy(), clean["p_value"].to_numpy())


def check_nan_posterior(api):
    cal = calibration_world()
    cal.at[0, "regime_posterior"] = [np.nan] * 3
    out = api.run_gate(CANDS, cal, T, PI_T, ess_floor=0, **KW)
    assert out["age_only_fallback"].all() and np.isfinite(out["p_value"]).all()
    sel = api.select_calibration(cal, T, **KW)
    age = (T - sel["decision_date"]).dt.days.to_numpy() / 7.0
    np.testing.assert_allclose(out["p_value"].to_numpy(),
                               api.conformal_pvalues(sel["score"], sel["y_net"], np.exp(-age / 26), CANDS["score"]))
    assert (out["effective_n_age_only"] > 0).all()


def check_stale_row(api):
    cal = calibration_world()
    old = T - pd.Timedelta(weeks=60)
    mixed = pd.concat([cal, cal_rows(old, ["STALE"], [100.0], [-0.5], [old + pd.Timedelta(days=6)])], ignore_index=True)
    assert "STALE" not in set(api.select_calibration(mixed, T, **KW)["symbol"])
    out = api.run_gate(CANDS, mixed, T, PI_T, ess_floor=0, **KW)
    assert (out["n_calibration"] == 120).all()
    np.testing.assert_array_equal(out["p_value"].to_numpy(),
                                  api.run_gate(CANDS, cal, T, PI_T, ess_floor=0, **KW)["p_value"].to_numpy())


CHECKS = {
    "nan_cand": (check_nan_candidate, ["nan_cand_best_p", "nan_cand_not_excluded"]),
    "nan_cal": (check_nan_cal_score, ["nan_cal_score_kept", "nan_cal_score_not_rejected"]),
    "nan_post": (check_nan_posterior, ["nan_posterior_no_fallback"]),
    "stale": (check_stale_row, ["no_window_filter"]),
}


@pytest.mark.parametrize("name", list(CHECKS))
def test_missing_and_stale_real(name):
    CHECKS[name][0](gate)


@pytest.mark.parametrize("name,mut", [(n, m) for n, (_, ms) in CHECKS.items() for m in ms])
def test_missing_and_stale_old_behaviour_fails(name, mut):
    with pytest.raises(AssertionError):
        CHECKS[name][0](_mutant(gate, G_MUTANTS[mut]))


def check_gains(build):
    n = 51
    df = pd.DataFrame({"y_net": [k * 0.02 for k in range(n)] + [0.5, 0.5, 0.5, 0.3],
                       "atr_pct": [1.0] * n + [0.0, -1.0, np.nan, 1.0]})
    df.loc[len(df)] = [np.inf, 1.0]                                                       # an infinite label is missing too
    g = build(df, np.arange(len(df)) < n, 1)
    assert g.iloc[n:n + 3].isna().all() and np.isfinite(g.iloc[n + 3]) and np.isnan(g.iloc[n + 4])
    np.testing.assert_allclose(g.iloc[1:n - 1], np.arange(1, n - 1) * 0.02, atol=1e-12)    # fit unaffected by the bad rows


def test_gains_nonfinite_missing_real():
    check_gains(_gains.build_fold_gains)


@pytest.mark.parametrize("old", [
    ('    atr = labels["atr_pct"].where(labels["atr_pct"] > 0)', '    atr = labels["atr_pct"]'),
    ('    g = g.where(np.isfinite(g))', '    g = g'),
])
def test_gains_old_behaviour_fails(old):
    m = _mutant(_gains, [("build_fold_gains", old[0], old[1])])
    with pytest.raises(AssertionError):
        check_gains(m.build_fold_gains)
