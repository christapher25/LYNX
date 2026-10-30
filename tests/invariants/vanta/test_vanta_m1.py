"""
Milestone M1 (Stage 13M) invariants, from the VANTA_SPEC M1 section and the 2026-10-30 USER DECISIONS in
logs/agent_team_decisions.md (M1.1-M1.extra, items 5, 6, 19). Interface: see the proposal sent to the lead (vanta.m1).

Each check_* takes an `api` (vanta.m1, or the toy from vanta_toys_m1) and asserts one invariant on a small hand-built input.
The real module is skipped until it exists; the toy must pass and each deliberately broken toy must fail.
"""

import numpy as np
import pandas as pd
import pytest

from vanta_toys_m1 import FILTER, MODEL, strip_negated_guarantee, toy_api

TS = pd.Timestamp


def _log(rows):
    df = pd.DataFrame(rows, columns=["decision_date", "symbol", "published", "fill_status", "y_net"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    return df


F, NFU = "FILLED", "NO_FILL_UPPER_LOCK"
NAN = np.nan


# ---------------------------------------------------------------- M1.2 realised FDR ----------------------------------
def check_fdr_filled_published_only(api):
    """M1.2: loser = y_net <= 0 (zero is a loser); realised FDR counts FILLED published trades only; NO_FILL reported alongside;
    unpublished candidates and unmatured (NaN) outcomes never enter the numerator or denominator."""
    log = _log([
        ("2013-01-07", "A", True, F, 0.05),
        ("2013-01-07", "B", True, F, 0.0),          # exactly zero: a loser
        ("2013-01-07", "C", True, F, -0.02),
        ("2013-01-07", "D", True, F, 0.01),
        ("2013-01-07", "E", True, NFU, NAN),        # published, no fill: reported, not in the FDR
        ("2013-01-07", "F", True, F, NAN),          # published, not yet matured: neither
        ("2013-01-07", "G", False, F, -0.50),       # not published: ignored
    ])
    r = api.realised_fdr(log)
    assert (r["n_filled"], r["n_losers"], r["n_no_fill"]) == (4, 2, 1)
    assert r["n_unmatured"] == 1                    # published FILLED with NaN y_net is reported, not silently dropped
    assert r["fdr"] == pytest.approx(0.5)


def check_fdr_by_year(api):
    """M1 / GO criterion 3: FDR is reported per calendar year (by decision_date); a pooled number must not stand in for it."""
    log = _log([
        ("2013-03-04", "A", True, F, 0.02), ("2013-03-04", "B", True, F, -0.01),
        ("2014-03-03", "C", True, F, -0.02), ("2014-03-03", "D", True, F, -0.03),
        ("2014-03-03", "E", True, F, 0.04), ("2014-03-03", "G", True, F, -0.01),
    ])
    r = api.fdr_by_year(log)
    assert r[2013] == pytest.approx(0.5)
    assert r[2014] == pytest.approx(0.75)


def check_publication_rate(api):
    """GO criterion 4: publication rate = share of WEEKS with >= 1 published signal, not the share of candidate rows; a week
    with no candidates at all still counts in the denominator when the caller supplies the week list."""
    log = _log([
        ("2013-01-07", "A", True, F, 0.1), ("2013-01-07", "B", True, F, 0.1), ("2013-01-07", "C", True, F, 0.1),
        ("2013-01-14", "A", False, F, 0.1), ("2013-01-14", "B", False, F, 0.1),
        ("2013-01-21", "A", True, F, 0.1),
        ("2013-01-28", "A", False, F, 0.1),
    ])
    assert api.publication_rate(log) == pytest.approx(0.5)           # 2 of 4 weeks (rows would give 4/7)
    weeks = pd.date_range("2013-01-07", periods=5, freq="7D")        # 2013-02-04 has no candidates
    assert api.publication_rate(log, weeks) == pytest.approx(0.4)


# ---------------------------------------------------------------- statistics -----------------------------------------
def check_newey_west(api):
    """Newey-West (Bartlett) SE of a mean: x=[1,-1,1,-1] has gamma0=1, gamma1=-0.75, so L=0 gives 0.5 and L=1 gives
    sqrt((1 + 2*0.5*(-0.75))/4) = 0.25. Autocorrelation must change the SE."""
    x = [1.0, -1.0, 1.0, -1.0]
    assert api.newey_west_se(x, 0) == pytest.approx(0.5)
    assert api.newey_west_se(x, 1) == pytest.approx(0.25)
    y = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]        # positively autocorrelated: SE rises with lags
    assert api.newey_west_se(y, 2) > api.newey_west_se(y, 0)
    # Bartlett weights keep the variance non-negative where flat weights would not
    assert np.isfinite(api.newey_west_se(x + x, 3))


def check_block_bootstrap_resamples_weeks(api):
    """Weekly block bootstrap: whole weeks are resampled. Week 1 has three +1 signals, week 2 one -1 signal, so the pooled mean
    of any 2-week resample can only be 1, 0.5 or -1. Resampling signals iid would also produce 0 and -0.5."""
    y, wk = [1, 1, 1, -1], ["w1", "w1", "w1", "w2"]
    d = np.asarray(api.weekly_block_bootstrap(y, wk, 300, 7))
    assert len(d) == 300
    assert set(np.round(d, 9)) <= {1.0, 0.5, -1.0}
    assert {1.0, 0.5, -1.0} <= set(np.round(d, 9))                  # and all three occur
    lo, hi = api.bootstrap_ci(d)
    assert (lo, hi) == (pytest.approx(np.percentile(d, 2.5)), pytest.approx(np.percentile(d, 97.5)))
    assert -1.0 <= lo <= hi <= 1.0


def check_bootstrap_ci_is_two_sided_95(api):
    """The CI is the 2.5th and 97.5th percentiles of the draws (95%, two-sided)."""
    d = np.arange(1001, dtype=float)
    lo, hi = api.bootstrap_ci(d)
    assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)


def check_paired_bootstrap_shares_weeks(api):
    """GO criterion 5: gated vs ungated is a paired weekly difference: both arms see the SAME resampled weeks. With the data
    below the only possible differences are 0, 0.25 and -1; independent resampling would give others (e.g. 1)."""
    ya, wa = [1, 1, 1, -1], ["w1", "w1", "w1", "w2"]
    yb, wb = [1, 0, 0, 0], ["w1", "w2", "w2", "w2"]
    d = np.asarray(api.weekly_block_bootstrap_diff(ya, wa, yb, wb, 300, 3))
    assert set(np.round(d, 9)) <= {0.0, 0.25, -1.0}


# ---------------------------------------------------------------- k selection ----------------------------------------
def _grid_trades(extra=()):
    rows = [
        # (decision_date, k_s, k_t, y_net)
        ("2010-02-01", 1.0, 2.0, 0.01), ("2011-02-07", 1.0, 2.0, 0.01),
        ("2010-02-01", 1.5, 3.0, 0.00), ("2012-12-31", 1.5, 3.0, 0.03),   # wins only thanks to the last in-window day
        ("2011-02-07", 2.0, 1.5, 0.015), ("2011-03-07", 2.0, 1.5, NAN),   # NaN (no fill) is missing, not 0
    ]
    return pd.DataFrame(list(rows) + list(extra), columns=["decision_date", "k_s", "k_t", "y_net"]).assign(
        decision_date=lambda d: pd.to_datetime(d["decision_date"]))


def check_choose_k_window_and_nan(api):
    """M1.extra: k_s,k_t chosen on 2010-2012 (inclusive both ends) by mean y_net per signal; NaN is ignored, never zero-filled.
    Means: (1.0,2.0)=0.010, (1.5,3.0)=0.015, (2.0,1.5)=0.015 (NaN ignored) -- tie broken so (1.5,3.0) vs (2.0,1.5) is
    avoided by giving (2.0,1.5) 0.0151 below."""
    t = _grid_trades()
    t.loc[(t.k_s == 2.0) & t.y_net.notna(), "y_net"] = 0.016         # NaN ignored -> 0.016 beats 0.015; zero-filled -> 0.008
    assert tuple(api.choose_k(t)) == (2.0, 1.5)
    t2 = _grid_trades()
    t2.loc[(t2.k_s == 2.0) & t2.y_net.notna(), "y_net"] = 0.010
    assert tuple(api.choose_k(t2)) == (1.5, 3.0)                       # 2012-12-31 row is inside the window (inclusive)


def check_choose_k_frozen_after_2012(api):
    """M1.extra: data after 2012 (and before 2010) cannot change the choice; the choice is frozen."""
    base = _grid_trades()
    base.loc[(base.k_s == 2.0) & base.y_net.notna(), "y_net"] = 0.010
    before = tuple(api.choose_k(base))
    extra = _grid_trades([("2013-01-01", 1.0, 2.0, 5.0), ("2015-06-01", 1.0, 2.0, 5.0), ("2009-12-31", 1.0, 2.0, 5.0)])
    extra.loc[(extra.k_s == 2.0) & extra.y_net.notna(), "y_net"] = 0.010
    assert tuple(api.choose_k(extra)) == before
    assert before in {(1.5, 3.0), (2.0, 1.5), (1.0, 2.0)}


def check_choose_k_tie_break(api):
    """choose_k tie-break (pinned by the lead): equal means pick the first combo in grid order (k_s then k_t ascending)."""
    t = pd.DataFrame([("2011-01-03", 2.0, 1.5, 0.02), ("2011-01-03", 1.0, 3.0, 0.02), ("2011-01-03", 1.5, 2.0, 0.02)],
                     columns=["decision_date", "k_s", "k_t", "y_net"]).assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))
    assert tuple(api.choose_k(t)) == (1.0, 3.0)


# ---------------------------------------------------------------- GO split --------------------------------------------
def check_go_report_split(api):
    """M1.extra: GO criteria are evaluated on 2013+ only; 2010-2012 is reported separately as in-sample. The in-sample losers
    must not leak into the out-of-sample FDR (nor vice versa), and the same for publication rate."""
    log = _log([
        ("2012-12-31", "A", True, F, -0.05), ("2012-12-31", "B", True, F, -0.05),     # in-sample: all losers
        ("2013-01-07", "C", True, F, 0.05), ("2013-01-07", "D", True, F, 0.02),
        ("2013-01-14", "E", False, F, 0.0),
    ])
    r = api.go_report(log)
    assert r["in_sample"]["fdr"] == pytest.approx(1.0)
    assert r["oos"]["fdr"] == pytest.approx(0.0)
    assert r["oos"]["n_filled"] == 2 and r["in_sample"]["n_filled"] == 2
    assert r["oos"]["publication_rate"] == pytest.approx(0.5)
    assert set(r["oos"]["fdr_by_year"]) == {2013} and set(r["in_sample"]["fdr_by_year"]) == {2012}


# ---------------------------------------------------------------- v4 reconstruction ----------------------------------
def _cal(T, extra=()):
    d = T - pd.Timedelta(weeks=3)
    m = d + pd.Timedelta(days=6)
    rows = []
    for s, y in [(0, -1.0), (0, 1.0), (1, 0.0), (1, 2.0)]:   # balanced design: OLS line is y = x, shortfalls [1,-1,1,-1]
        rows.append((d, f"S{len(rows)}", 5, float(s), MODEL, "f1", True, [1, 0, 0], "h", y, m, FILTER))
    rows += list(extra)
    return pd.DataFrame(rows, columns=["decision_date", "symbol", "horizon", "score", "model_version", "fold_id", "is_oos",
                                       "regime_posterior", "hmm_version", "y_net", "matured_at", "candidate_filter_version"])


def _v4(api, cands, cal, T):
    return api.v4_gate_reconstruction(pd.DataFrame({"symbol": [f"C{i}" for i in range(len(cands))], "score": cands}),
                                      cal, T, model_version=MODEL, candidate_filter_version=FILTER, q=0.30, cap=10)


def check_v4_bound_and_strict_threshold(api):
    """M1.3: bound = fitted - (1-q) quantile of calibration shortfalls (fitted - y), published iff bound > 0 (strict).
    Fit y=x, shortfalls [1,-1,1,-1], q=0.3: the 0.7 quantile is 1, so bound = score - 1. Score 1.0 -> bound 0 -> NOT published;
    using the q quantile (-1) would publish score 0.5."""
    T = TS("2024-06-03")
    r = _v4(api, [2.5, 1.5, 1.0, 0.5], _cal(T), T)
    np.testing.assert_allclose(r["fitted"], [2.5, 1.5, 1.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(r["bound"], [1.5, 0.5, 0.0, -0.5], atol=1e-9)
    assert list(r["published"]) == [True, True, False, False]


def check_v4_cap_by_score(api):
    """M1.3: at most 10 published, the 10 highest scores."""
    T = TS("2024-06-03")
    scores = [5.0 + i for i in range(15)]
    r = _v4(api, scores, _cal(T), T)
    assert r["published"].sum() == 10
    assert set(r.loc[r["published"], "score"]) == set(sorted(scores)[-10:])


def check_v4_matured_window_only(api):
    """M1.3 / matured rule: the calibration window is the same 52-week MATURED set (matured_at <= T). A row whose outcome is
    not yet known at T must not move the fit, even if its y_net happens to be filled in."""
    T = TS("2024-06-03")
    d = T - pd.Timedelta(weeks=1)
    leak = [(d, f"L{i}", 5, 1.0, MODEL, "f1", True, [1, 0, 0], "h", -50.0, T + pd.Timedelta(days=2), FILTER) for i in range(8)]
    clean = _v4(api, [2.5, 1.5, 1.0, 0.5], _cal(T), T)
    dirty = _v4(api, [2.5, 1.5, 1.0, 0.5], _cal(T, leak), T)
    np.testing.assert_allclose(dirty[["fitted", "bound"]].to_numpy(), clean[["fitted", "bound"]].to_numpy(), atol=1e-9)


# ---------------------------------------------------------------- impact floor ---------------------------------------
def check_impact_floor(api):
    """Item 5 (user's assumption): 0.05%/side for turnover ranks 1-200, 0.20%/side for 201-700; outside the universe or an
    unknown rank is NaN, never silently a default or 0."""
    f = api.impact_floor_per_side
    assert f(1) == pytest.approx(0.0005) and f(200) == pytest.approx(0.0005)
    assert f(201) == pytest.approx(0.0020) and f(700) == pytest.approx(0.0020)
    assert np.isnan(f(701)) and np.isnan(f(float("nan")))
    with pytest.raises(ValueError):
        f(0)


def check_turnover_ranks(api):
    """Rank 1 = highest trailing turnover; ties share the lowest rank; missing turnover stays NaN (not rank-last by zero fill)."""
    s = pd.Series([50.0, 90.0, np.nan, 90.0, 10.0], index=list("abcde"))
    r = api.turnover_ranks(s)
    assert r["b"] == 1 and r["d"] == 1 and r["a"] == 3 and r["e"] == 4
    assert np.isnan(r["c"])


# ---------------------------------------------------------------- constants & report text ----------------------------
def check_constants(api):
    """M1.1, item 6, M1.4, M1.extra: q=0.30, cap 10, notional Rs 1,00,000, horizons 5 and 20, the k grids."""
    assert api.Q == 0.30 and api.CAP == 10 and api.REFERENCE_NOTIONAL == 100_000
    assert tuple(api.HORIZONS) == (5, 20)
    assert tuple(api.K_S_GRID) == (1.0, 1.5, 2.0) and tuple(api.K_T_GRID) == (1.5, 2.0, 3.0)


def check_report_text(api):
    """Item 14 / 19: realised FDR is an empirical number; q is never described as a guarantee; the NO_FILL count is shown."""
    stats = dict(fdr=0.31, n_filled=40, n_no_fill=7, n_losers=12, n_published=47)
    txt = api.fdr_report_text(stats, 0.30)
    assert "empirical" in txt.lower()
    assert "NO_FILL" in txt and "7" in txt
    assert not any(w in strip_negated_guarantee(txt).lower() for w in ("guarantee", "guaranteed", "guarantees"))


# ---------------------------------------------------------------- harness --------------------------------------------
CHECKS = {
    "fdr_filled_published_only": (check_fdr_filled_published_only, ["fdr_all_candidates", "no_fill_as_loser", "strict_loser", "nan_as_loser", "no_fill_hidden", "unmatured_hidden"]),
    "fdr_by_year": (check_fdr_by_year, ["pooled_years"]),
    "publication_rate": (check_publication_rate, ["per_row_rate", "ignore_given_weeks"]),
    "newey_west": (check_newey_west, ["no_lags", "flat_weights"]),
    "block_bootstrap_weeks": (check_block_bootstrap_resamples_weeks, ["iid_signals"]),
    "bootstrap_ci": (check_bootstrap_ci_is_two_sided_95, ["one_sided"]),
    "paired_bootstrap": (check_paired_bootstrap_shares_weeks, ["independent_arms"]),
    "choose_k_window_nan": (check_choose_k_window_and_nan, ["exclusive_end", "nan_as_zero"]),
    "choose_k_tie": (check_choose_k_tie_break, ["last_tie"]),
    "choose_k_frozen": (check_choose_k_frozen_after_2012, ["uses_all_data"]),
    "go_split": (check_go_report_split, ["oos_includes_insample"]),
    "v4_bound": (check_v4_bound_and_strict_threshold, ["wrong_quantile", "nonstrict", "bound_is_fitted"]),
    "v4_cap": (check_v4_cap_by_score, ["no_cap"]),
    "v4_matured": (check_v4_matured_window_only, ["uses_unmatured"]),
    "impact_floor": (check_impact_floor, ["off_by_one", "default_tail", "nan_as_zero"]),
    "turnover_ranks": (check_turnover_ranks, ["nan_as_zero", "ascending"]),
    "constants": (check_constants, []),
    "report_text": (check_report_text, ["guarantee", "no_nofill"]),
}
IDS = list(CHECKS)


@pytest.mark.parametrize("name", IDS)
def test_m1_real(name):
    api = pytest.importorskip("vanta.m1")
    CHECKS[name][0](api)


@pytest.mark.parametrize("name", IDS)
def test_m1_toy_passes(name):
    CHECKS[name][0](toy_api())


MUTANTS = [(n, b) for n in IDS for b in CHECKS[n][1]]


@pytest.mark.parametrize("name,bug", MUTANTS, ids=[f"{n}-{b}" for n, b in MUTANTS])
def test_m1_mutant_fails(name, bug):
    with pytest.raises(Exception) as ei:
        CHECKS[name][0](toy_api(bug))
    assert not isinstance(ei.value, (ImportError, NameError, TypeError)), repr(ei.value)
