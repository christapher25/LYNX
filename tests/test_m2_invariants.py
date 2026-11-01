"""
M2 (Stage 13M, row 15d) invariants, from docs/vanta/M2_PREREG.md (sections 3-7 and the section 9 amendment) and N11.
Module under test: vanta.m2 (does not exist yet; every name below is INVENTED and proposed for docs/vanta/INTERFACES.md section 8).

  M2-G1  arm (a)      exactly vanta.gate.run_gate: weights exp(-age/26) * exp(-L1), age-only fallback on a missing posterior, counted
  M2-G2  arm (b)      calibration restricted to rows with the SAME argmax filtered state as T, age-only weights; Kish ESS < 200 abstains
                      (200 exactly does not); posterior missing at T -> (a)'s age-only calibration, flagged, NOT an abstention
  M2-G3  arm (c)      arm (a) plus abstain iff P(state 2) >= 0.5 (0.5 abstains; not argmax; not 0.6); missing -> as (a), flagged
  M2-G4  arms         exactly three: ("a", "b", "c"); the publish cap is 10; counts are per WEEK
  M2-S1  score        weekly cross-sectional percentile of the B1 score, average ties, NaN stays NaN, never raw, never pooled
  M2-S2  comparator   ungated = the 10 highest B1 percentile each week
  M2-W1  window       judged from the first scored week >= oos_start with 52 weeks of scored history; earlier weeks reported, not judged
  M2-C1..C5          the five GO criteria on judged weeks only (FDR <= 0.35, no year > 0.5, publication in >= 20% of weeks, net per
                      published signal >= ungated top 10 with a paired block-bootstrap CI), Newey-West lag ceil(h/5)
  M2-T1  table        arm x horizon (5, 20); the floor run is judged, the zero run reported (not judged); no multiplicity correction
  M2-D1  diagnostics  13b diagnostics, A11 hold-to-horizon (next to ATR k = 1.5) and horizon 60 are NOT judged; none enters a GO criterion
  M2-A11 hold label   no stop and no target, exit at the close of the horizon session, same costs, N1 still applies
  M2-N11 no look-ahead  the decision at T is unchanged when everything after T, and every label not matured by T, is poisoned

Each check_* takes an `api` (vanta.m2 or the toy in tests/test_m2_toy.py). The real module skips until it exists; the toy must pass and
each deliberately broken toy (MUTANTS below) must fail. Shared fixtures use Fridays as decision dates.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from test_m2_toy import toy_api
from vanta import gate as G
from vanta import m1

TS = pd.Timestamp
NAN = np.nan
T0 = TS("2020-06-26")                  # a Friday
FILTER, MODEL = "m2-eligibility@v1", "b1@m2"


def _weeks(start, n):
    return pd.DatetimeIndex([TS(start) + pd.Timedelta(weeks=k) for k in range(n)])


# =============================================================================================================== gate fixtures
def _cal(groups):
    """groups: (n, score, y_net, posterior, decision_date). matured_at is a week after the decision date unless it would pass T0."""
    rows = []
    for gi, (n, score, y, post, d) in enumerate(groups):
        for i in range(n):
            rows.append(dict(decision_date=TS(d), symbol=f"G{gi}S{i}", horizon=20, score=score, model_version=MODEL, fold_id="m2",
                             is_oos=True, regime_posterior=list(post), hmm_version="h", y_net=y,
                             matured_at=min(TS(d) + pd.Timedelta(weeks=1), T0), candidate_filter_version=FILTER))
    return pd.DataFrame(rows)


def _cands(*scores):
    return pd.DataFrame({"symbol": [f"C{i}" for i in range(len(scores))], "score": list(scores)})


AGE = np.exp(-2 / 26)                                          # calibration two weeks old
D2 = T0 - pd.Timedelta(weeks=2)
# losers in calm state 0 with a HIGHER score than the candidate; winners in stressed state 2 (below the candidate's score)
REGIME_CAL = _cal([(300, 0.96, -0.05, (1, 0, 0), D2), (400, 0.50, 0.02, (0, 0, 1), D2)])
STRESS = [0.0, 0.0, 1.0]


def _reg_p(pi_T):
    wx = AGE * np.exp(-(abs(1 - pi_T[0]) + abs(pi_T[1]) + abs(pi_T[2])))
    wy = AGE * np.exp(-(abs(pi_T[0]) + abs(pi_T[1]) + abs(1 - pi_T[2])))
    return (1 + 300 * wx) / (1 + 300 * wx + 400 * wy)


def check_arm_a_is_run_gate(api):
    """M2-G1 (a): age x regime weights exp(-age/26)*exp(-L1(pi_i, pi_T)) exactly (known p-value), published when BH accepts; a missing
    posterior forces the age-only fallback, which is COUNTED (age_only_fallback, state_missing) and uses exp(-age/26) only."""
    out = api.gate_arm("a", _cands(0.95), REGIME_CAL, T0, STRESS)
    assert out["p_value"].iloc[0] == pytest.approx(_reg_p(STRESS))
    assert bool(out["published"].iloc[0]) and not out["age_only_fallback"].iloc[0] and not out["abstained"].iloc[0]
    ref = G.run_gate(_cands(0.95), REGIME_CAL, T0, STRESS, model_version=MODEL, candidate_filter_version=FILTER)
    assert out["p_value"].iloc[0] == pytest.approx(ref["p_value"].iloc[0])
    miss = api.gate_arm("a", _cands(0.95), REGIME_CAL, T0, [NAN, NAN, NAN])
    assert bool(miss["age_only_fallback"].iloc[0]) and bool(miss["state_missing"].iloc[0]) and not miss["abstained"].iloc[0]
    assert miss["p_value"].iloc[0] == pytest.approx((1 + 300 * AGE) / (1 + 700 * AGE))       # age-only weights
    assert not bool(miss["published"].iloc[0])                                                 # and it does not publish here


def check_cap_ten(api):
    """M2-G4: at most 10 published per week even when BH accepts all 12 (q = 0.30, cap 10)."""
    cal = _cal([(400, 0.50, 0.02, (0, 0, 1), D2)])
    out = api.gate_arm("a", _cands(*[0.99] * 12), cal, T0, STRESS)
    assert int(out["in_bh_set"].sum()) == 12 and int(out["published"].sum()) == 10


STATE2 = (0.05, 0.05, 0.90)


def _cond_cal(n_state2, mixed=False):
    """Winners in the SAME argmax state as T (age 0, so every age-only weight is exactly 1 and ESS is exactly n), plus high-score losers
    in another state and in rows whose posterior is missing."""
    grp = [(n_state2, 0.50, 0.02, STATE2, T0), (300, 0.96, -0.05, (0.9, 0.05, 0.05), T0), (100, 0.96, -0.05, (NAN, NAN, NAN), T0)]
    if mixed:                                              # same argmax, different L1 distance to pi_T: regime weights would shrink ESS
        grp[0] = (n_state2 // 2, 0.50, 0.02, STATE2, T0)
        grp.insert(1, (n_state2 - n_state2 // 2, 0.50, 0.02, (0.0, 0.0, 1.0), T0))
    return _cal(grp)


def check_arm_b_conditions_on_same_state(api):
    """M2-G2 (b): only rows whose argmax filtered state equals T's, age-only weights. 200 same-state winners vs 300 other-state and 100
    missing-posterior losers: p = 1/201 and the candidate is published (pooling all rows would give p ~ 0.67)."""
    out = api.gate_arm("b", _cands(0.95), _cond_cal(200), T0, STATE2)
    assert out["p_value"].iloc[0] == pytest.approx(1 / 201)
    assert bool(out["published"].iloc[0]) and not out["abstained"].iloc[0] and not out["state_missing"].iloc[0]
    assert int(out["n_calibration"].iloc[0]) == 200 and out["effective_n"].iloc[0] == pytest.approx(200.0)


def check_arm_b_age_only_weights(api):
    """M2-G2 (b): weights are exp(-age/26) and NOT the L1 regime weights: 200 same-argmax rows with different L1 distance to pi_T keep
    ESS 200 under age-only weights (regime weights would give ~198 and abstain)."""
    out = api.gate_arm("b", _cands(0.95), _cond_cal(200, mixed=True), T0, STATE2)
    assert not out["abstained"].iloc[0] and bool(out["published"].iloc[0])
    cal = _cal([(1, 0.97, -0.05, STATE2, T0 - pd.Timedelta(weeks=26)), (1, 0.50, 0.02, STATE2, T0), (1, 0.99, -0.05, (0.9, 0.05, 0.05), T0)])
    out = api.gate_arm("b", _cands(0.95), cal, T0, STATE2, ess_floor=1)          # one loser aged 26 weeks: weight exp(-1)
    assert out["p_value"].iloc[0] == pytest.approx((1 + np.exp(-1)) / (1 + np.exp(-1) + 1))


def check_arm_b_abstains_below_200(api):
    """M2-G2 (b): Kish ESS 199 < 200 -> publishes nothing, counted as an abstention; ESS exactly 200 does not abstain."""
    out = api.gate_arm("b", _cands(0.95), _cond_cal(199), T0, STATE2)
    assert not out["published"].any() and bool(out["abstained"].iloc[0])
    assert out["effective_n"].iloc[0] == pytest.approx(199.0)
    ok = api.gate_arm("b", _cands(0.95), _cond_cal(200), T0, STATE2)
    assert not ok["abstained"].iloc[0] and bool(ok["published"].iloc[0])


def check_arm_b_missing_posterior_falls_back(api):
    """M2-G2 (b): posterior missing at T -> arm (a)'s age-only calibration over ALL rows (p = 401/601 here), flagged
    (age_only_fallback, state_missing), and NOT an abstention and not 'state 0'."""
    out = api.gate_arm("b", _cands(0.95), _cond_cal(200), T0, [NAN, NAN, NAN])
    assert out["p_value"].iloc[0] == pytest.approx(401 / 601)
    assert bool(out["age_only_fallback"].iloc[0]) and bool(out["state_missing"].iloc[0]) and not out["abstained"].iloc[0]


def check_arm_c_abstains_at_half(api):
    """M2-G3 (c): abstain iff P(state 2) >= 0.5: exactly 0.5 abstains, 0.55 abstains, 0.4 does not, and an argmax-2 state with
    P(2) = 0.48 does not (so neither '>' nor 'argmax' nor a 0.6 threshold)."""
    for p2, pi in ((0.5, [0.25, 0.25, 0.5]), (0.55, [0.2, 0.25, 0.55])):
        out = api.gate_arm("c", _cands(0.95), REGIME_CAL, T0, pi)
        assert bool(out["abstained"].iloc[0]) and not out["published"].any(), p2
    for pi in ([0.0, 0.6, 0.4], [0.3, 0.22, 0.48]):
        c, a = api.gate_arm("c", _cands(0.95), REGIME_CAL, T0, pi), api.gate_arm("a", _cands(0.95), REGIME_CAL, T0, pi)
        assert not c["abstained"].iloc[0] and c["published"].tolist() == a["published"].tolist()
        assert c["p_value"].iloc[0] == pytest.approx(a["p_value"].iloc[0])
    assert bool(api.gate_arm("c", _cands(0.95), REGIME_CAL, T0, [0.0, 0.6, 0.4])["published"].iloc[0])      # sanity: it can publish


def check_arm_c_missing_probability_is_arm_a(api):
    """M2-G3 (c): P(state 2) missing at T -> handled as (a) (age-only fallback), counted and flagged, no abstention."""
    c = api.gate_arm("c", _cands(0.95), REGIME_CAL, T0, [NAN, NAN, NAN])
    a = api.gate_arm("a", _cands(0.95), REGIME_CAL, T0, [NAN, NAN, NAN])
    assert not c["abstained"].iloc[0] and bool(c["age_only_fallback"].iloc[0]) and bool(c["state_missing"].iloc[0])
    assert c["p_value"].iloc[0] == pytest.approx(a["p_value"].iloc[0]) and c["published"].tolist() == a["published"].tolist()


def check_exactly_three_arms(api):
    """M2-G4: exactly three arms ('a', 'b', 'c'); a fourth is refused and run_m2_arms returns those three plus the ungated comparator."""
    assert tuple(api.ARMS) == ("a", "b", "c")
    with pytest.raises(ValueError):
        api.gate_arm("d", _cands(0.95), REGIME_CAL, T0, STRESS)
    logs = api.run_m2_arms(_panel(weeks=1), _weeks("2020-01-03", 1), ess_floor=1)
    assert set(logs) == {"a", "b", "c", "ungated"}


def check_constants_frozen(api):
    """M2 sections 3/4/6: q 0.30, cap 10, tau 26 weeks, ESS floor 200, 52-week window, k_s = k_t = 1.5, seeds 0-4, impact floors 0.05% /
    0.20% per side, judged horizons (5, 20), diagnostic horizons (5, 20, 60), three arms."""
    assert (api.Q, api.CAP, api.TAU_WEEKS, api.ESS_FLOOR, api.WINDOW_WEEKS) == (0.30, 10, 26, 200, 52)
    assert (api.K_S, api.K_T) == (1.5, 1.5) and tuple(api.B1_SEEDS) == (0, 1, 2, 3, 4)
    assert (api.IMPACT_FLOOR_TOP200, api.IMPACT_FLOOR_REST) == (0.0005, 0.0020)
    assert tuple(api.JUDGED_HORIZONS) == (5, 20) and tuple(api.DIAGNOSTIC_HORIZONS) == (5, 20, 60) and tuple(api.ARMS) == ("a", "b", "c")


# =============================================================================================================== score (S1, S2)
def check_percentile_known_answers(api):
    """M2-S1: the gate score is the WEEKLY percentile rank of the B1 score (rank/n, average ties), never the raw score, never pooled:
    week 2 drifts up yet scores the same; ties [1,2,2,3] -> [.25,.625,.625,1]; NaN stays NaN and takes no slot."""
    f = pd.DataFrame({"decision_date": pd.to_datetime(["2020-01-03"] * 3 + ["2020-01-10"] * 3), "symbol": list("ABCABC"),
                      "b1": [0.30, 0.10, 0.20, 0.70, 0.50, 0.60]})
    assert np.asarray(api.weekly_percentile(f), float) == pytest.approx([1, 1 / 3, 2 / 3, 1, 1 / 3, 2 / 3])
    t = pd.DataFrame({"decision_date": TS("2020-01-03"), "symbol": list("ABCD"), "b1": [1.0, 2.0, 2.0, 3.0]})
    assert np.asarray(api.weekly_percentile(t), float) == pytest.approx([0.25, 0.625, 0.625, 1.0])
    n = pd.DataFrame({"decision_date": TS("2020-01-03"), "symbol": list("ABCD"), "b1": [5.0, NAN, 1.0, 3.0]})
    r = np.asarray(api.weekly_percentile(n), float)
    assert np.isnan(r[1]) and r[[0, 2, 3]] == pytest.approx([1.0, 1 / 3, 2 / 3])


def _panel(weeks=10, per_week=8, start="2020-01-03", seed=3, horizon=20):
    """A labelled candidate panel: raw B1 score b1, y_net, matured_at three weeks after the decision, a filtered posterior."""
    rng = np.random.default_rng(seed)
    rows = []
    vecs = [(0.8, 0.15, 0.05), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7)]
    for k, d in enumerate(_weeks(start, weeks)):
        for i in range(per_week):
            rows.append(dict(decision_date=d, symbol=f"S{i}", horizon=horizon, b1=float(rng.normal()) * 10 + 100, y_net=float(rng.normal(0.01, 0.05)),
                             fill_status="FILLED", matured_at=d + pd.Timedelta(weeks=3), regime_posterior=list(vecs[k % 3]),
                             hmm_version="h"))
    return pd.DataFrame(rows)


def check_ungated_is_top10_b1_percentile(api):
    """M2-S2: the ungated comparator is the 10 highest B1 percentile names each week (12 candidates -> 10; a NaN score is never one)."""
    p = _panel(weeks=1, per_week=13)
    p["b1"] = [float(i) for i in range(13)]
    p.loc[12, "b1"] = NAN
    logs = api.run_m2_arms(p, _weeks("2020-01-03", 1), ess_floor=1)
    u = logs["ungated"]
    assert set(u.loc[u["published"], "symbol"]) == {f"S{i}" for i in range(2, 12)}
    for arm in ("a", "b", "c"):
        assert logs[arm]["score"].dropna().between(0, 1).all()                      # the gate sees percentiles, never raw B1 scores


def check_counts_are_per_week(api):
    """M2-G4/reporting: the age-only fallback share and the abstention count are per WEEK (week 0 has 5 rows): 1 of 4 weeks = 0.25."""
    d = _weeks("2020-01-03", 4)
    rows = [(d[0], True, False)] * 5 + [(d[1], False, True), (d[2], False, True), (d[3], False, False)]
    log = pd.DataFrame(rows, columns=["decision_date", "age_only_fallback", "abstained"]).assign(state_missing=False)
    c = api.arm_counts(log)
    assert (c["n_weeks"], c["n_age_only_fallback_weeks"], c["n_abstain_weeks"]) == (4, 1, 2)
    assert c["share_age_only_fallback"] == pytest.approx(0.25)


# =============================================================================================================== judged window
def _scored():
    rows = []
    for d in _weeks("2015-01-02", 160):
        for s in "ABC":
            rows.append(dict(decision_date=d, symbol=s, score=0.5 if d >= TS("2016-01-08") else NAN))
    return pd.DataFrame(rows)


def check_judged_window(api):
    """M2-W1: scores exist from 2016-01-08 (NaN-score weeks are not scored); the first judged week is the first scored week >= oos_start
    with 52 weeks of scored history behind it: 2017-01-06 (2016-01-08 + 52 weeks), or 2017-06-02 when oos_start is 2017-06-01."""
    w = api.judged_window(_scored(), oos_start=TS("2013-01-01"))
    assert w["first_scored_week"] == TS("2016-01-08") and w["first_judged_week"] == TS("2017-01-06")
    assert api.judged_window(_scored(), oos_start=TS("2017-06-01"))["first_judged_week"] == TS("2017-06-02")


def check_earlier_weeks_reported_not_judged(api):
    """M2-W1: weeks before the first judged week are kept in the report (reported_only) and never judged; nothing is dropped."""
    log = pd.DataFrame({"decision_date": _weeks("2016-01-08", 80), "y_net": 0.01})
    s = api.split_judged(log, TS("2017-01-06"))
    assert len(s["judged"]) + len(s["reported_only"]) == len(log) == 80
    assert (s["judged"]["decision_date"] >= TS("2017-01-06")).all() and (s["reported_only"]["decision_date"] < TS("2017-01-06")).all()
    assert s["judged"]["decision_date"].min() == TS("2017-01-06")                # the first judged week itself is judged
    assert (s["n_reported_only_weeks"], s["n_judged_weeks"], s["first_judged_week"]) == (52, 28, TS("2017-01-06"))


# =============================================================================================================== GO criteria
FJW = TS("2017-01-06")
JW = _weeks(FJW, 10)
EARLY = _weeks(FJW - pd.Timedelta(weeks=5), 5)


def _rows(dates_ys, published=True, fill="FILLED"):
    return pd.DataFrame([dict(decision_date=d, symbol=f"X{i}", fill_status=fill if np.isfinite(y) else "NO_FILL_UPPER_LOCK",
                              y_net=y, published=published) for i, (d, y) in enumerate(dates_ys)])


def _ungated(weeks, ys=(0.01, -0.01)):
    return _rows([(d, ys[i % 2]) for d in weeks for i in range(2)])


def _gate_log(n_losers, weeks=JW, per_week=10, loser_y=-0.02, win_y=0.02):
    """per_week trades in each week, `n_losers` losers in total, spread from the first week on."""
    ys = [loser_y] * n_losers + [win_y] * (len(weeks) * per_week - n_losers)
    return _rows([(weeks[i // per_week], ys[i]) for i in range(len(ys))])


def _is_true(x):
    return x is not None and not pd.isna(x) and bool(x)


def _is_false(x):
    return x is not None and not pd.isna(x) and not bool(x)


def _go(api, log, ung, h=5, **kw):
    dd = kw.pop("decision_dates", EARLY.append(JW))
    return api.go_criteria(log, ung, dd, h, first_judged_week=FJW, invariants_passed=True, **kw)


def check_c2_fdr_threshold(api):
    """M2-C2: realised FDR over JUDGED weeks <= q + 0.05 = 0.35 (0.35 passes, 0.36 fails); loser = y_net <= 0 (exactly 0 loses);
    NO_FILL published rows are not trades; earlier (reported-only) weeks do not count."""
    early = _rows([(d, -0.05) for d in EARLY for _ in range(5)])
    nofill = _rows([(d, NAN) for d in JW for _ in range(2)])
    for n_lose, ok in ((35, True), (36, False)):
        log = pd.concat([early, nofill, _gate_log(n_lose)], ignore_index=True)
        c = _go(api, log, _ungated(JW))
        assert c["c2"]["fdr"] == pytest.approx(n_lose / 100) and _is_true(c["c2"]["passed"]) == ok and c["c2"]["threshold"] == pytest.approx(0.35)
    zero = _gate_log(35, loser_y=0.0)                                              # y_net == 0 is a loser, not a winner
    assert _go(api, zero, _ungated(JW))["c2"]["fdr"] == pytest.approx(0.35)
    allwin = _gate_log(0)
    assert _go(api, allwin, _ungated(JW))["c2"]["fdr"] == 0.0
    assert not _is_true(_go(api, _rows([(JW[0], NAN)]), _ungated(JW))["c2"]["passed"])       # nothing filled: undefined, never a pass


def _two_years(l17, l18):
    w17, w18 = _weeks("2017-01-06", 10), _weeks("2018-01-05", 10)
    ys = [-0.02] * l17 + [0.02] * (10 - l17), [-0.02] * l18 + [0.02] * (10 - l18)
    return _rows(list(zip(w17, ys[0])) + list(zip(w18, ys[1]))), w17.append(w18)


def check_c3_no_year_above_half(api):
    """M2-C3: no calendar year (judged weeks) with realised FDR above 0.5: a year at exactly 0.5 passes, 0.6 fails; a bad year before
    the first judged week is ignored."""
    ok, dd = _two_years(5, 2)
    assert _is_true(_go(api, ok, _ungated(dd), decision_dates=dd)["c3"]["passed"])
    bad, dd = _two_years(6, 2)
    c = _go(api, bad, _ungated(dd), decision_dates=dd)
    assert _is_false(c["c3"]["passed"]) and c["c3"]["worst_year_fdr"] == pytest.approx(0.6)
    old = _rows([(TS("2016-02-05") + pd.Timedelta(weeks=i), -0.05) for i in range(8)])
    assert _is_true(_go(api, pd.concat([old, ok], ignore_index=True), _ungated(dd), decision_dates=dd)["c3"]["passed"])


def check_c4_publication_rate(api):
    """M2-C4: the gate publishes in >= 20% of judged weeks: 2 of 10 passes, 1 of 10 fails. The denominator is every judged decision week,
    not only the weeks that appear in the log."""
    unpub = lambda ds: _rows([(d, 0.01) for d in ds], published=False)           # noqa: E731
    two = pd.concat([_rows([(JW[0], 0.02), (JW[1], 0.02)]), unpub(JW[2:4])], ignore_index=True)
    c = _go(api, two, _ungated(JW))
    assert c["c4"]["publication_rate"] == pytest.approx(0.2) and _is_true(c["c4"]["passed"])
    one = pd.concat([_rows([(JW[0], 0.02)]), unpub(JW[1:4])], ignore_index=True)
    c = _go(api, one, _ungated(JW))
    assert c["c4"]["publication_rate"] == pytest.approx(0.1) and _is_false(c["c4"]["passed"])


def check_c5_net_vs_ungated_top10(api):
    """M2-C5: net return per published signal >= the ungated top 10 (published rows only), reported with a paired block-bootstrap CI of
    the difference: +0.02 vs 0 passes, equal passes, below fails; an unpublished ungated row with a huge return does not count."""
    ung = pd.concat([_ungated(JW), _rows([(d, 0.5) for d in JW], published=False)], ignore_index=True)
    up = _rows([(d, y) for d in JW for y in (0.03, 0.01)])                         # mean 0.02
    c = _go(api, up, ung)
    assert c["c5"]["diff"] == pytest.approx(0.02) and _is_true(c["c5"]["passed"])
    lo, hi = c["c5"]["ci95"]
    assert lo - 1e-12 <= c["c5"]["diff"] <= hi + 1e-12
    assert _is_true(_go(api, _ungated(JW), _ungated(JW))["c5"]["passed"])            # equal means: >=
    down = _rows([(d, y) for d in JW for y in (-0.01, 0.0)])
    assert _is_false(_go(api, down, _ungated(JW))["c5"]["passed"])


def _weekly_series_log():
    ys = np.sin(np.arange(24) * 0.9) * 0.03 + np.arange(24) * 0.001
    ws = _weeks(FJW, 24)
    return _rows([(d, y + e) for d, y in zip(ws, ys) for e in (0.004, -0.004)]), ws


def check_newey_west_and_block_bootstrap(api):
    """M2-C: Newey-West lag = ceil(h/5) weekly rows (h=5 -> 1, h=20 -> 4, h=7 -> 2, h=60 -> 12) and the weekly block bootstrap uses
    moving blocks of that length; both reported beside the headline, matching vanta.m1's reference functions."""
    assert [api.nw_lag_weeks(h) for h in (5, 20, 7, 60)] == [1, 4, 2, 12]
    log, ws = _weekly_series_log()
    for h in (5, 20):
        lag = int(np.ceil(h / 5))
        c = _go(api, log, _ungated(ws), h=h, decision_dates=ws, n_boot=300, seed=11)
        wk = log.groupby("decision_date")["y_net"].mean().to_numpy()
        assert c["nw_lag_weeks"] == lag and c["nw_se"] == pytest.approx(m1.newey_west_se(wk, lag))
        ref = m1.bootstrap_ci(m1.weekly_block_bootstrap(log["y_net"], m1.weekly_ids(log), 300, 11, lag))
        assert c["boot_ci95_block"] == pytest.approx(ref)


# =============================================================================================================== the table
def _table_logs():
    ung = _ungated(JW)
    mk = lambda n: (_gate_log(n), ung)                                             # noqa: E731
    h2 = lambda n: {5: mk(n), 20: mk(n)}                                           # noqa: E731
    return {"floor": {"a": h2(40), "b": h2(30), "c": h2(35)}, "zero": {"a": h2(30), "b": h2(30), "c": h2(30)}}


def check_go_table(api):
    """M2-T1: one row per arm x horizon (5, 20) x run. The FLOOR run is judged (go_all a bool); the ZERO run is reported with its own
    numbers (judged False, go_all None). Arm a fails c2 under floor (0.40) though it passes under zero (0.30); arms b and c pass all
    five; thresholds are identical for every arm (no multiplicity correction); invariants_passed False fails everything."""
    logs = _table_logs()
    t = api.go_table(logs, JW, FJW, True)
    assert len(t) == 12 and set(zip(t.impact, t.arm, t.horizon)) == {(i, a, h) for i in ("floor", "zero") for a in "abc" for h in (5, 20)}
    fl, ze = t[t.impact == "floor"], t[t.impact == "zero"]
    assert fl["judged"].all() and not ze["judged"].any() and ze["go_all"].isna().all()
    row = lambda df, a, h: df[(df.arm == a) & (df.horizon == h)].iloc[0]           # noqa: E731
    assert row(fl, "a", 5).c2_fdr == pytest.approx(0.40) and row(ze, "a", 5).c2_fdr == pytest.approx(0.30)
    assert _is_false(row(fl, "a", 5).go_all) and _is_false(row(fl, "a", 5).c2)
    assert _is_true(row(fl, "b", 20).go_all) and _is_true(row(fl, "c", 5).go_all)
    assert np.allclose(t["c2_threshold"], 0.35)
    assert [row(fl, "a", h).nw_lag_weeks for h in (5, 20)] == [1, 4]
    assert not any(bool(x) for x in api.go_table(logs, JW, FJW, False).query("judged")["go_all"])


def check_go_table_refuses_diagnostics(api):
    """M2-D1: only arms a/b/c and horizons 5/20 can enter the GO table: horizon 60, an A11 / unknown arm key and an unknown run are refused."""
    ung = _ungated(JW)
    base = {"a": {5: (_gate_log(30), ung)}}
    for bad in ({"floor": {"a": {60: (_gate_log(30), ung)}}}, {"floor": {"a11": base["a"]}}, {"floor": {"d": base["a"]}},
                {"hold": base}):
        with pytest.raises(ValueError):
            api.go_table(bad, JW, FJW, True)


# =============================================================================================================== diagnostics
def check_diagnostics_not_judged(api):
    """M2-D1: ATR vs hold-to-horizon and horizons 5/20/60 come back as reported rows with judged False, known FDR and mean, and NO
    GO-criterion column; an unknown variant is refused."""
    logs = {}
    for v in ("atr", "hold"):
        for h in (5, 20, 60):
            for arm in ("a", "b", "c", "ungated"):
                logs[(v, h, arm)] = _rows([(JW[0], 0.04), (JW[1], 0.02), (JW[2], 0.0), (JW[3], 0.06)])    # one loser (y == 0) of 4
    t = api.diagnostics_table(logs)
    assert len(t) == 24 and set(t["variant"]) == {"atr", "hold"} and set(t["horizon"]) == {5, 20, 60}
    assert not t["judged"].any() and np.allclose(t["fdr"], 0.25) and np.allclose(t["mean_y_net"], 0.03)
    bad = [c for c in t.columns if c in {"c1", "c2", "c3", "c4", "c5", "go_all", "go"} or c.startswith("go_")]
    assert not bad, bad
    with pytest.raises(ValueError):
        api.diagnostics_table({("a12", 5, "a"): _gate_log(1)})


def check_no_diagnostic_value_enters_go(api):
    """M2-D1: go_table / go_criteria take no diagnostic input, and running the diagnostics (even on poisoned data) leaves the GO table
    unchanged."""
    for fn in (api.go_table, api.go_criteria):
        names = set(inspect.signature(fn).parameters)
        assert not [n for n in names if any(k in n.lower() for k in ("diag", "a11", "hold", "h60", "nifty"))], names
    logs = _table_logs()
    before = api.go_table(logs, JW, FJW, True)
    poisoned = {(v, h, "a"): _gate_log(0, loser_y=-9.0, win_y=9.0) for v in ("atr", "hold") for h in (5, 20, 60)}
    api.diagnostics_table(poisoned)
    pd.testing.assert_frame_equal(before, api.go_table(logs, JW, FJW, True))


def check_diagnostics_13b(api):
    """M2-D1: the 13b diagnostics (FDP vs Nifty weekly return, FDR up vs down weeks, FDR vs index) are reported, equal vanta.m1b's own
    numbers, and are marked not judged."""
    from vanta import m1b
    dd = _weeks("2020-01-03", 6)
    log = pd.concat([_rows([(dd[0], 0.02), (dd[0], -0.02)]), _rows([(dd[1], -0.01)]), _rows([(dd[2], 0.03)]), _rows([(dd[3], -0.02)]),
                     _rows([(dd[4], 0.02)])], ignore_index=True)
    log["entry_date"], log["label_end_date"] = log["decision_date"], log["decision_date"] + pd.Timedelta(days=7)
    close = pd.Series(np.linspace(100, 110, 60), index=pd.bdate_range("2020-01-01", periods=60)).reindex(
        dd.append(dd + pd.Timedelta(days=7)).unique().sort_values()).ffill()
    r = api.diagnostics_13b(log, close, dd)
    fdp = m1b.weekly_fdp(log)
    ref = m1b.fdp_vs_market(fdp, m1b.market_weekly_returns(close, dd))
    assert r["judged"] is False and r["fdp_vs_market"]["n_weeks"] == ref["n_weeks"]
    assert r["fdr_vs_index"] == m1b.realised_fdr_vs_index(log, close)


# =============================================================================================================== A11 hold-to-horizon
def _bars(ohlc):
    idx = pd.bdate_range("2024-01-01", periods=len(ohlc))
    cols = ("open", "high", "low", "close")
    d = {f"raw_{c}": [r[i] for r in ohlc] for i, c in enumerate(cols)}
    d.update({c: [r[i] for r in ohlc] for i, c in enumerate(cols)})
    return pd.DataFrame(d, index=idx)


A11_BARS = [(99, 101, 98, 100), (100, 101, 99.5, 100),           # t, entry day (entry 100, ATR 2: k = 1.5 -> stop 97, target 103)
            (100, 102, 90, 95),                                   # low 90 <= stop 97: the ATR run is stopped out at 97
            (95, 104, 94, 100),                                   # high 104 >= target 103
            (100, 101, 99, 100),                                  # close 100: the previous-close decoy
            (101, 150, 60, 104),                                  # horizon session t+5: close 104 (decoys: open 101, high 150, low 60)
            (200, 201, 199, 200)]                                 # close 200: the next-close decoy
NO_HITS = (pd.DataFrame({"date": [], "symbol": [], "side": []}), set())


def check_a11_hold_to_horizon(api):
    """M2-A11: no stop and no target: the dip through the 1.5 ATR stop (low 90) and the touch of the target do not exit; the exit is the
    CLOSE of the horizon session (104, not its open, high, or a neighbouring close): gross 0.04, TIME; the costs are the same cost_fn
    applied to that exit (y_net = 0.04 - 0.0035); the ATR k = 1.5 run on the same bars is stopped at 97 (y -0.03)."""
    bars, hits = _bars(A11_BARS), NO_HITS
    calls = []

    def cost(sym, ed, xd, ep, xp):
        calls.append((ep, xp))
        return (0.001, 0.002, 0.0005)

    row = api.hold_to_horizon_label("T", bars, bars.index[0], 5, 2.0, *hits, cost_fn=cost)
    assert row["fill_status"] == "FILLED" and row["exit_reason"] == "TIME" and row["exit_date"] == bars.index[5]
    assert row["exit_price"] == 104 and row["gross_return"] == pytest.approx(0.04) and row["label_end_date"] == bars.index[5]
    assert row["y_net"] == pytest.approx(0.04 - 0.0035) and calls[-1] == (100, 104)
    from vanta import labels as L
    atr = L.simulate_trade("T", bars, bars.index[0], 5, api.K_S, api.K_T, 2.0, *hits, cost_fn=lambda *a: (0.0, 0.0, 0.0))
    assert atr["exit_reason"] == "STOP" and atr["y_net"] == pytest.approx(-0.03)


def check_a11_upper_lock_still_no_fill(api):
    """M2-A11 / N1: an upper-lock entry (hit list, raw open == raw high) is still NO_FILL_UPPER_LOCK under hold-to-horizon, no payoff."""
    ohlc = list(A11_BARS)
    ohlc[1] = (100, 100, 97, 98)
    bars = _bars(ohlc)
    hits = pd.DataFrame({"date": [bars.index[1]], "symbol": ["T"], "side": ["UPPER"]})
    row = api.hold_to_horizon_label("T", bars, bars.index[0], 5, 2.0, hits, set(bars.index), cost_fn=lambda *a: (0.0, 0.0, 0.0))
    assert row["fill_status"] == "NO_FILL_UPPER_LOCK" and np.isnan(row["y_net"])


# =============================================================================================================== N11
def check_no_lookahead_poison(api):
    """M2-N11: the decision at T uses only what is known at T. Poison (a) every B1 score, label and posterior of weeks after T and (b)
    the labels of earlier rows that mature after T; every arm's p-values, publication and flags for weeks <= T are unchanged."""
    p = _panel(weeks=12, per_week=8)
    dates = _weeks("2020-01-03", 12)
    t0 = dates[7]
    q = p.copy()
    fut = q["decision_date"] > t0
    q.loc[fut, "b1"] = 1e6 + np.arange(int(fut.sum()))
    q.loc[fut, "y_net"] = -0.5
    q["regime_posterior"] = [[1.0, 0.0, 0.0] if f else v for f, v in zip(fut, q["regime_posterior"])]
    unm = (~fut) & (q["matured_at"] > t0)
    q.loc[unm, "y_net"] = -0.5
    base, pois = (api.run_m2_arms(x, dates, ess_floor=1) for x in (p, q))
    for arm in ("a", "b", "c"):
        a, b = (lg[lg["decision_date"] <= t0].reset_index(drop=True) for lg in (base[arm], pois[arm]))
        assert len(a) == len(b) > 0
        np.testing.assert_allclose(a["p_value"].to_numpy(float), b["p_value"].to_numpy(float), equal_nan=True, err_msg=arm)
        for col in ("published", "abstained", "age_only_fallback", "state_missing", "score"):
            assert a[col].tolist() == b[col].tolist() or np.allclose(a[col].astype(float), b[col].astype(float), equal_nan=True), (arm, col)


# =============================================================================================================== harness
CHECKS = {
    "arm_a": (check_arm_a_is_run_gate, ["no_regime", "nan_as_zero"]),
    "cap": (check_cap_ten, ["cap5"]),
    "arm_b_state": (check_arm_b_conditions_on_same_state, ["all_rows", "nan_rows_match"]),
    "arm_b_weights": (check_arm_b_age_only_weights, ["regime_weights"]),
    "arm_b_abstain": (check_arm_b_abstains_below_200, ["no_abstain", "floor_inclusive", "ess100"]),
    "arm_b_missing": (check_arm_b_missing_posterior_falls_back, ["b_missing_abstains", "nan_as_zero"]),
    "arm_c_half": (check_arm_c_abstains_at_half, ["strict", "argmax", "thr06"]),
    "arm_c_missing": (check_arm_c_missing_probability_is_arm_a, ["missing_abstains", "missing_flag_dropped", "nan_as_zero"]),
    "three_arms": (check_exactly_three_arms, ["four_arms"]),
    "constants": (check_constants_frozen, ["q25", "cap5", "ess100", "kt2", "four_arms"]),
    "percentile": (check_percentile_known_answers, ["raw", "pooled", "tie_min", "nan_zero"]),
    "ungated": (check_ungated_is_top10_b1_percentile, ["ung_all", "ung_top5", "ung_bottom", "raw"]),
    "counts": (check_counts_are_per_week, ["row_counts"]),
    "window": (check_judged_window, ["nan_scored", "no_buffer", "w51", "w53", "oos_ignored"]),
    "split": (check_earlier_weeks_reported_not_judged, ["drop_early", "gt", "judged_all"]),
    "c2": (check_c2_fdr_threshold, ["thr_q", "c2_strict", "no_fill_counted", "unjudged_included"]),
    "c3": (check_c3_no_year_above_half, ["c3_ge", "unjudged_included"]),
    "c4": (check_c4_publication_rate, ["c4_strict", "pr_denominator_log"]),
    "c5": (check_c5_net_vs_ungated_top10, ["c5_strict", "ung_unfiltered"]),
    "nw_boot": (check_newey_west_and_block_bootstrap, ["lag1", "lag_h", "no_block"]),
    "table": (check_go_table, ["floor_uses_zero", "judge_zero", "judge_none", "thr_q"]),
    "table_refuses": (check_go_table_refuses_diagnostics, ["accept_h60", "accept_any_arm"]),
    "diag": (check_diagnostics_not_judged, ["judged_true", "go_col", "drop_hold", "drop_h60", "any_variant"]),
    "diag_not_in_go": (check_no_diagnostic_value_enters_go, []),
    "diag_13b": (check_diagnostics_13b, ["judged_true"]),
    "a11": (check_a11_hold_to_horizon, ["uses_atr", "exit_prev_close", "exit_high", "exit_open", "no_costs"]),
    "a11_lock": (check_a11_upper_lock_still_no_fill, []),
    "n11": (check_no_lookahead_poison, ["future_cal", "last_posterior", "pooled"]),
}
IDS = list(CHECKS)


@pytest.mark.parametrize("name", IDS)
def test_m2_real(name):
    api = pytest.importorskip("vanta.m2")
    CHECKS[name][0](api)


@pytest.mark.parametrize("name", IDS)
def test_m2_toy_passes(name):
    CHECKS[name][0](toy_api())


MUTANTS = [(n, b) for n in IDS for b in CHECKS[n][1]]


@pytest.mark.parametrize("name,bug", MUTANTS, ids=[f"{n}-{b}" for n, b in MUTANTS])
def test_m2_mutant_fails(name, bug):
    with pytest.raises((Exception, pytest.fail.Exception)) as ei:
        CHECKS[name][0](toy_api(bug))
    assert not isinstance(ei.value, (ImportError, NameError, TypeError, AttributeError)), repr(ei.value)
