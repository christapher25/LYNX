"""
Milestone M1b (Stage 13M, corrected M1) invariants, from the 2026-10-30 user diagnosis in logs/agent_team_decisions.md:
(a) candidates = EVERY point-in-time-universe name passing the ban, circuit (upper + lower) and liquidity-cliff (rank <= 200)
gates that week; BH runs over all of them, at most 10 published; calibration = ALL those scored candidates, matured
(matured_at <= T), last 52 weeks (FCC). (b) score = cross-sectional PERCENTILE RANK of 12-1 momentum within that week.
Interface: docs/vanta/INTERFACES.md section 6 (vanta.m1b, INVENTED names).

Each check_* takes an `api` (vanta.m1b, or the toy from vanta_toys_m1b) and asserts one invariant on a small hand-built input.
The real module is skipped until it exists; the toy must pass and each deliberately broken toy must fail.
"""

import numpy as np
import pandas as pd
import pytest

from vanta_toys_m1b import toy_api

TS = pd.Timestamp
NAN = np.nan


def _arr(x):
    return np.asarray(x, float)


# ---------------------------------------------------------------- score = weekly percentile rank ----------------------
def check_percentile_known_answer(api):
    """M1b(b): percentile = rank / n (range (0,1]), not the raw value: [30,10,40,20] -> [.75,.25,1,.5]."""
    r = _arr(api.percentile_rank([30.0, 10.0, 40.0, 20.0]))
    assert r == pytest.approx([0.75, 0.25, 1.0, 0.5])


def check_percentile_ties_share_average_rank(api):
    """M1b(b) ties: tied momentum shares the AVERAGE rank (order-independent): [1,2,2,3] -> [.25,.625,.625,1]."""
    r = _arr(api.percentile_rank([1.0, 2.0, 2.0, 3.0]))
    assert r == pytest.approx([0.25, 0.625, 0.625, 1.0])
    r2 = _arr(api.percentile_rank([2.0, 3.0, 2.0, 1.0]))     # same data, shuffled: tied names still equal
    assert r2[0] == pytest.approx(r2[2]) and r2[0] == pytest.approx(0.625)


def check_percentile_missing_stays_nan(api):
    """No-zero-fill / M1b: missing momentum stays NaN, is excluded from n, and does not take rank 0:
    [5, nan, 1, 3] -> [1, nan, 1/3, 2/3]."""
    r = _arr(api.percentile_rank([5.0, NAN, 1.0, 3.0]))
    assert np.isnan(r[1])
    assert r[[0, 2, 3]] == pytest.approx([1.0, 1 / 3, 2 / 3])


def check_percentile_degenerate_weeks(api):
    """Pinned by the lead: an all-NaN week returns all NaN (not 0, not an error); a single finite name has n_finite == 1 and
    scores 1.0."""
    assert np.isnan(_arr(api.percentile_rank([NAN, NAN, NAN]))).all()
    r = _arr(api.percentile_rank([NAN, 0.07, NAN]))
    assert np.isnan(r[[0, 2]]).all() and r[1] == pytest.approx(1.0)


def check_weekly_scores_are_per_week_not_raw(api):
    """M1b(b): score is the rank WITHIN the decision week. Week 2 momentum is uniformly higher (market drift) yet has the same
    scores as week 1; neither raw momentum nor a rank pooled over the whole history may be returned."""
    f = pd.DataFrame({
        "decision_date": pd.to_datetime(["2013-01-07"] * 3 + ["2013-01-14"] * 3),
        "symbol": list("ABCABC"),
        "momentum": [0.30, 0.10, 0.20, 0.70, 0.50, 0.60],
    })
    s = _arr(api.weekly_scores(f))
    assert s == pytest.approx([1.0, 1 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3])


def check_weekly_scores_missing_momentum_excluded(api):
    """No-zero-fill: a NaN momentum row stays NaN in weekly_scores and does not change the other names' percentiles."""
    f = pd.DataFrame({
        "decision_date": pd.to_datetime(["2013-01-07"] * 4),
        "symbol": list("ABCD"),
        "momentum": [0.30, NAN, 0.10, 0.20],
    })
    s = _arr(api.weekly_scores(f))
    assert np.isnan(s[1])
    assert s[[0, 2, 3]] == pytest.approx([1.0, 1 / 3, 2 / 3])


# ---------------------------------------------------------------- candidates = all eligible names ---------------------
def _week():
    rows = []
    for i in range(12):                                          # 12 eligible names, more than the publish cap of 10
        rows.append((f"E{i:02d}", False, False, False, float(i + 1), 0.01 * (i + 1)))
    rows[-1] = ("E11", False, False, False, 200.0, 0.12)         # exactly on the cliff: still eligible
    rows += [
        ("BAN", True, False, False, 5.0, 0.99),
        ("UP", False, True, False, 6.0, 0.98),
        ("LO", False, False, True, 7.0, 0.97),
        ("R201", False, False, False, 201.0, 0.96),
        ("RNAN", False, False, False, NAN, 0.95),
        ("MNAN", False, False, False, 8.0, NAN),
    ]
    return pd.DataFrame(rows, columns=["symbol", "banned", "hit_upper", "hit_lower", "liquidity_rank", "momentum"])


def check_candidates_all_passing_names(api):
    """M1b(a): candidates = EVERY name passing ban, upper-circuit, lower-circuit, liquidity rank <= 200 (inclusive; NaN rank
    fails) and finite momentum; not the top 10 by momentum. Input order preserved."""
    out = api.eligible_candidates(_week())
    assert list(out["symbol"]) == [f"E{i:02d}" for i in range(12)]


def check_candidates_na_flag_excluded(api):
    """Block policy (lead 2026-10-30, as M1 and CLAUDE.md): a missing (NA) ban / upper / lower flag means the list was unavailable,
    so the row is EXCLUDED, never treated as 'allowed'."""
    week = _week().iloc[:4].copy()               # E00..E03, all clear
    for c in ("banned", "hit_upper", "hit_lower"):
        week[c] = pd.array([False] * 4, dtype="boolean")
    week.loc[week.index[1], "banned"] = pd.NA
    week.loc[week.index[2], "hit_upper"] = pd.NA
    week.loc[week.index[3], "hit_lower"] = pd.NA
    out = api.eligible_candidates(week)
    assert list(out["symbol"]) == ["E00"]


# ---------------------------------------------------------------- calibration window --------------------------------
def _scored():
    T = TS("2014-01-06")
    rows = [
        # symbol, decision_date, matured_at, y_net, published
        ("EDGE", "2013-01-07", "2013-01-20", 0.01, False),     # exactly T - 52 weeks: inclusive, and unpublished still counts
        ("OLD", "2013-01-06", "2013-01-20", 0.02, True),       # one day outside 52 weeks
        ("MID", "2013-06-03", "2013-06-20", 0.03, True),       # inside 52 weeks, outside 26 weeks
        ("UNMAT", "2013-12-30", "2014-01-07", 0.04, True),     # matures after T: unusable at T
        ("ATT", "2013-12-23", "2014-01-06", 0.05, True),       # matures exactly at T: usable
        ("NOY", "2013-12-02", "2013-12-20", NAN, True),        # no outcome: unusable
        ("REC", "2013-12-16", "2013-12-30", -0.06, False),     # unpublished candidate, recent: usable
    ]
    df = pd.DataFrame(rows, columns=["symbol", "decision_date", "matured_at", "y_net", "published"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["matured_at"] = pd.to_datetime(df["matured_at"])
    df["score"] = 0.5
    return df, T


def check_calibration_matured_52w_all_candidates(api):
    """M1b(a) + SPEC query rule: calibration at T = ALL scored candidates (published or not) with matured_at <= T (inclusive),
    decision_date >= T - 52 weeks (inclusive) and a finite y_net; unmatured rows, rows 52 weeks + 1 day old, and NaN outcomes
    are out."""
    df, T = _scored()
    out = api.calibration_rows(df, T)
    assert sorted(out["symbol"]) == sorted(["EDGE", "MID", "ATT", "REC"])


# ---------------------------------------------------------------- publish at most 10 ---------------------------------
def check_publish_cap_ten_by_score(api):
    """M1b(a): BH over all candidates, then at most 10 published: the top 10 by score among in_bh_set; never outside the BH set."""
    scores = [3.0, 15.0, 1.0, 9.0, 12.0, 6.0, 2.0, 14.0, 5.0, 11.0, 8.0, 4.0, 13.0, 7.0, 10.0]
    f = pd.DataFrame({"symbol": [f"S{i}" for i in range(15)], "score": scores, "in_bh_set": True})
    pub = np.asarray(api.select_published(f), bool)
    assert pub.shape == (15,) and pub.sum() == 10
    assert sorted(np.array(scores)[pub]) == [float(x) for x in range(6, 16)]


def check_publish_only_inside_bh_set(api):
    """M1b(a): a high-scoring name outside the BH set is never published; fewer than 10 BH names publish all of them."""
    scores = [3.0, 15.0, 1.0, 9.0, 12.0, 6.0, 2.0, 14.0, 5.0, 11.0, 8.0, 4.0, 13.0, 7.0, 10.0]
    in_bh = [False, False, True, True, False, True, True, False, False, False, False, False, False, False, False]
    f = pd.DataFrame({"symbol": [f"S{i}" for i in range(15)], "score": scores, "in_bh_set": in_bh})
    pub = np.asarray(api.select_published(f), bool)
    assert list(pub) == in_bh


# ---------------------------------------------------------------- harness --------------------------------------------
CHECKS = {
    "percentile_known": (check_percentile_known_answer, []),
    "percentile_ties": (check_percentile_ties_share_average_rank, ["tie_first", "tie_min", "tie_max"]),
    "percentile_nan": (check_percentile_missing_stays_nan, ["nan_as_zero"]),
    "percentile_degenerate": (check_percentile_degenerate_weeks, ["nan_as_zero", "single_not_one"]),
    "weekly_scores": (check_weekly_scores_are_per_week_not_raw, ["raw_momentum", "pooled_history"]),
    "weekly_scores_nan": (check_weekly_scores_missing_momentum_excluded, ["nan_as_zero", "raw_momentum"]),
    "candidates_all": (check_candidates_all_passing_names, [
        "top10_only", "ignore_ban", "ignore_upper", "ignore_lower", "no_cliff", "exclusive_cliff",
        "nan_rank_passes", "nan_momentum_kept"]),
    "candidates_na_flag": (check_candidates_na_flag_excluded, ["na_allowed"]),
    "calibration": (check_calibration_matured_52w_all_candidates, [
        "unmatured", "window_26", "window_104", "exclusive_window_start", "published_only", "nan_y_kept"]),
    "publish_cap": (check_publish_cap_ten_by_score, ["no_cap", "cap_11"]),
    "publish_bh_set": (check_publish_only_inside_bh_set, ["ignore_bh"]),
}
IDS = list(CHECKS)


@pytest.mark.parametrize("name", IDS)
def test_m1b_real(name):
    api = pytest.importorskip("vanta.m1b")
    CHECKS[name][0](api)


@pytest.mark.parametrize("name", IDS)
def test_m1b_toy_passes(name):
    CHECKS[name][0](toy_api())


MUTANTS = [(n, b) for n in IDS for b in CHECKS[n][1]]


@pytest.mark.parametrize("name,bug", MUTANTS, ids=[f"{n}-{b}" for n, b in MUTANTS])
def test_m1b_mutant_fails(name, bug):
    with pytest.raises(Exception) as ei:
        CHECKS[name][0](toy_api(bug))
    assert not isinstance(ei.value, (ImportError, NameError, TypeError)), repr(ei.value)
