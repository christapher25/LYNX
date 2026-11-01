"""VANTA Milestone M2 (Stage 13M, row 15d): the VANTA gate with the Stage 14 B1 score, three gate arms, the five GO criteria.

Authority: docs/vanta/M2_PREREG.md (sections 3-7 and the section 9 amendment). Nothing here may be changed after an M2 or Stage 14 result
has been seen. Contract: tests/test_m2_invariants.py and tests/test_m2_driver.py (names proposed for docs/vanta/INTERFACES.md section 8).

  * the gate consumes the WEEKLY cross-sectional percentile of the B1 score (average ties, NaN stays NaN, never raw, never pooled);
  * arm (a) is exactly vanta.gate.run_gate; arm (b) calibrates on same-argmax-state rows with age-only weights and abstains below a Kish
    ESS of 200; arm (c) is arm (a) plus an abstention when P(state 2) >= 0.5; a missing posterior at T is handled as in (a), flagged;
  * the ungated comparator is the 10 highest B1 percentiles each week;
  * only the floor run is judged; A11 hold-to-horizon and horizon 60 are diagnostics and never enter a GO criterion;
  * q = 0.30 is an engineering target; realised FDR is an empirical number, never a guarantee. Missing is missing: NaN is never zero.

vanta.m1 and vanta.labels are deliberately NOT edited: scripts.run_m1._code_version() and run_m1b._candidate_key() hash their source,
so any edit would invalidate every cached candidate table and label chunk. Everything M2 adds therefore lives here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from vanta import baselines as bl
from vanta import gate as G
from vanta import labels as L
from vanta import m1, m1b

# ---------------------------------------------------------------------------------------------------------------- frozen constants
Q = 0.30
CAP = 10
TAU_WEEKS = 26
ESS_FLOOR = 200
WINDOW_WEEKS = 52
K_S = 1.5
K_T = 1.5
B1_SEEDS = bl.B1_SEEDS
IMPACT_FLOOR_TOP200 = m1.IMPACT_FLOOR_TOP200
IMPACT_FLOOR_REST = m1.IMPACT_FLOOR_REST
ARMS = ("a", "b", "c")
JUDGED_HORIZONS = (5, 20)
DIAGNOSTIC_HORIZONS = (5, 20, 60)
ALL_HORIZONS = DIAGNOSTIC_HORIZONS
VARIANTS = ("atr", "hold")
IMPACTS = ("floor", "zero")
MODEL_VERSION = "b1@m2"
FILTER_VERSION = "m2-eligibility@v1"
GO_FDR_THRESHOLD = Q + 0.05
GO_YEAR_FDR_LIMIT = 0.5
GO_PUBLICATION_RATE = 0.20
STRESSED_STATE = 2
STRESS_ABSTAIN_P = 0.5
FLAT = [1 / 3] * 3
OOS_START = m1.OOS_START


def nw_lag_weeks(h) -> int:
    return int(math.ceil(h / 5))


# ---------------------------------------------------------------------------------------------------------------- score (S1, S2)
def weekly_percentile(frame: pd.DataFrame, score_col: str = "b1") -> pd.Series:
    """Percentile rank (0, 1] of the B1 score within each decision_date (average ties); NaN stays NaN and takes no slot. Aligned to
    frame.index."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, g in frame.groupby("decision_date"):
        out.loc[g.index] = m1b.percentile_rank(g[score_col].to_numpy(float))
    return out


def _top_n(g: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n highest percentiles of one week. Ties (equal raw B1 score = equal percentile) break by symbol ascending, so the comparator
    does not depend on row order. The gate's own tie rule differs by design: p-value, then higher score, then input order (gate.py)."""
    ok = g[np.isfinite(g["score"].astype(float))].sort_values("symbol", kind="stable")
    return ok.sort_values("score", ascending=False, kind="stable").head(n)


# ---------------------------------------------------------------------------------------------------------------- the three gate arms
def gate_arm(arm, cands, cal, T, pi_T, *, q=Q, cap=CAP, tau_weeks=TAU_WEEKS, ess_floor=ESS_FLOOR, window_weeks=WINDOW_WEEKS,
             model_version=MODEL_VERSION, candidate_filter_version=FILTER_VERSION) -> pd.DataFrame:
    """One gate arm at decision date T. Returns run_gate's frame plus `abstained` and `state_missing` (bool per row). `pi_T` is the
    filtered posterior at T (length 3; any non-finite entry = missing). A missing posterior is never read as a state."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    T = pd.Timestamp(T)
    pi = np.full(3, np.nan) if pi_T is None else np.asarray(pi_T, float)
    missing = not np.isfinite(pi).all()
    gk = dict(model_version=model_version, candidate_filter_version=candidate_filter_version, tau_weeks=tau_weeks, q=q, cap=cap,
              ess_floor=ess_floor, window_weeks=window_weeks)
    base = cands.reset_index(drop=True)

    def decorate(out, abstained, state_missing):
        out = out.copy()
        out["abstained"] = bool(abstained)
        out["state_missing"] = bool(state_missing)
        if abstained:
            out["published"] = False
        return out

    if arm == "a":
        return decorate(G.run_gate(base, cal, T, pi, **gk), False, missing)

    if arm == "c":
        out = G.run_gate(base, cal, T, pi, **gk)
        p2 = pi[STRESSED_STATE]
        if not np.isfinite(p2):                                  # missing: handled as (a), flagged, never an abstention
            return decorate(out, False, missing)
        return decorate(out, p2 >= STRESS_ABSTAIN_P, missing)

    # arm b
    if missing:                                                  # (a)'s age-only calibration, counted and flagged
        return decorate(G.run_gate(base, cal, T, [np.nan] * 3, **gk), False, True)
    sel = G.select_calibration(cal, T, model_version=model_version, candidate_filter_version=candidate_filter_version,
                               window_weeks=window_weeks)
    post = np.array([list(x) for x in sel["regime_posterior"]], float).reshape(len(sel), 3)
    okp = np.isfinite(post).all(axis=1)
    same = np.zeros(len(sel), bool)
    same[okp] = post[okp].argmax(axis=1) == int(np.argmax(pi))   # rows with a missing posterior never match a state
    sub = sel[same]
    age = (T - sub["decision_date"]).dt.days.to_numpy() / 7.0
    ess = G.kish_ess(np.exp(-age / tau_weeks))                   # age-only weights; 0 rows -> ESS 0 -> abstain
    out = G.run_gate(base, sub.assign(regime_posterior=[FLAT] * len(sub)), T, FLAT, **gk)
    out["age_only_fallback"] = False       # age-only weights are arm (b)'s design, not a fallback; a thin ESS is an abstention, counted apart
    out["n_calibration"], out["effective_n"] = len(sub), ess
    return decorate(out, ess < ess_floor, False)


# ---------------------------------------------------------------------------------------------------------------- walk over weeks
def run_m2_arms(cands: pd.DataFrame, decision_dates, *, q=Q, cap=CAP, tau_weeks=TAU_WEEKS, ess_floor=ESS_FLOOR,
                window_weeks=WINDOW_WEEKS) -> dict:
    """Walk the weekly decision dates. `cands` columns: decision_date, symbol, horizon, b1 (raw B1 score), y_net (impact applied),
    fill_status, matured_at, regime_posterior (list of 3, NaN when unavailable), hmm_version. At T the gate sees only calibration rows
    with matured_at <= T (vanta.gate enforces it). Returns {"a","b","c","ungated"} logs, one row per decision date x candidate."""
    c = cands.copy()
    c["score"] = weekly_percentile(c, "b1")
    cal = pd.DataFrame({"decision_date": c["decision_date"], "symbol": c["symbol"], "horizon": c["horizon"], "score": c["score"],
                        "model_version": MODEL_VERSION, "fold_id": "m2", "is_oos": True, "regime_posterior": c["regime_posterior"],
                        "hmm_version": c["hmm_version"], "y_net": c["y_net"], "matured_at": c["matured_at"],
                        "candidate_filter_version": FILTER_VERSION}).reset_index(drop=True)
    logs = {k: [] for k in (*ARMS, "ungated")}
    by_date = {d: g.reset_index(drop=True) for d, g in c.groupby("decision_date")}
    for T in pd.DatetimeIndex(decision_dates):
        g = by_date.get(T)
        if g is None or g.empty:
            continue
        pi_T = g["regime_posterior"].iloc[0]
        pi_T = np.asarray(pi_T if pi_T is not None else [np.nan] * 3, float)
        base = g[["decision_date", "symbol", "horizon", "score", "fill_status", "y_net"]]
        for arm in ARMS:
            r = gate_arm(arm, g[["symbol", "score"]], cal, T, pi_T, q=q, cap=cap, tau_weeks=tau_weeks, ess_floor=ess_floor,
                         window_weeks=window_weeks)
            logs[arm].append(base.assign(published=r["published"].to_numpy(), abstained=r["abstained"].to_numpy(),
                                         age_only_fallback=r["age_only_fallback"].to_numpy(),
                                         state_missing=r["state_missing"].to_numpy(), p_value=r["p_value"].to_numpy(),
                                         n_calibration=r["n_calibration"].to_numpy(), effective_n=r["effective_n"].to_numpy()))
        u = base.assign(published=False)
        u.loc[_top_n(u, cap).index, "published"] = True
        logs["ungated"].append(u)
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in logs.items()}


def arm_counts(log: pd.DataFrame, decision_dates=None) -> dict:
    """Per-WEEK counts (a week's flags are constant across its rows): weeks, abstention weeks, age-only-fallback weeks (and their share),
    state-missing weeks."""
    by_week = log.groupby("decision_date").agg(fb=("age_only_fallback", "first"), ab=("abstained", "first"), sm=("state_missing", "first"))
    return dict(n_weeks=int(len(by_week)), n_abstain_weeks=int(by_week["ab"].sum()), n_age_only_fallback_weeks=int(by_week["fb"].sum()),
                share_age_only_fallback=float(by_week["fb"].mean()) if len(by_week) else float("nan"),
                n_state_missing_weeks=int(by_week["sm"].sum()))


# ---------------------------------------------------------------------------------------------------------------- judged window
def judged_window(scored: pd.DataFrame, oos_start=OOS_START, window_weeks: int = WINDOW_WEEKS) -> dict:
    """First scored week (finite B1 score) and the first JUDGED week: the first scored week >= oos_start with `window_weeks` weeks of
    scored history behind it (the same rule as vanta.baselines.first_scored_and_judged_week, M2_PREREG sections 4 and 9.1)."""
    return bl.first_scored_and_judged_week(scored, oos_start, window_weeks)


def split_judged(log: pd.DataFrame, first_judged_week) -> dict:
    """Weeks >= first_judged_week are judged; earlier weeks are kept as reported_only. Nothing is dropped."""
    f = pd.Timestamp(first_judged_week)
    j, rep = log[log["decision_date"] >= f], log[log["decision_date"] < f]
    return dict(judged=j, reported_only=rep, first_judged_week=f, n_judged_weeks=int(j["decision_date"].nunique()),
                n_reported_only_weeks=int(rep["decision_date"].nunique()))


# ---------------------------------------------------------------------------------------------------------------- the five GO criteria
def _pub_filled(l: pd.DataFrame) -> pd.DataFrame:
    return l[l["published"].astype(bool) & (l["fill_status"] == "FILLED") & l["y_net"].notna()]


def go_criteria(log, ungated, decision_dates, horizon, *, first_judged_week, invariants_passed, q=Q, n_boot=300, seed=11) -> dict:
    """The five GO criteria on JUDGED weeks only (decision_date >= first_judged_week). c1 invariants, c2 realised FDR <= q + 0.05,
    c3 no calendar year above 0.5, c4 publication in >= 20% of judged decision weeks (denominator = `decision_dates`, not the log),
    c5 net per published signal >= the ungated top 10 with a paired weekly block-bootstrap CI. Undefined is None, never a pass."""
    f = pd.Timestamp(first_judged_week)
    lg, ug = log[log["decision_date"] >= f], ungated[ungated["decision_date"] >= f]
    dd = pd.DatetimeIndex(decision_dates)
    dd = dd[dd >= f]
    lag = nw_lag_weeks(horizon)
    s = m1.realised_fdr(lg)
    fdr = s["fdr"]
    thr = q + 0.05
    c2 = None if not np.isfinite(fdr) else bool(fdr <= thr + 1e-12)
    by_year = m1.fdr_by_year(lg)
    c3 = None if not by_year else bool(all(v <= GO_YEAR_FDR_LIMIT for v in by_year.values()))
    pr = m1.publication_rate(lg, dd)
    c4 = None if not np.isfinite(pr) else bool(pr >= GO_PUBLICATION_RATE - 1e-12)
    a, b = _pub_filled(lg), _pub_filled(ug)
    out = dict(c1=dict(passed=bool(invariants_passed)), c2=dict(fdr=fdr, threshold=thr, passed=c2, **{k: s[k] for k in (
        "n_filled", "n_losers", "n_no_fill", "n_no_data", "n_unmatured", "n_published")}),
        c3=dict(fdr_by_year=by_year, worst_year_fdr=max(by_year.values()) if by_year else float("nan"), passed=c3),
        c4=dict(publication_rate=pr, passed=c4), nw_lag_weeks=lag, n_judged_weeks=int(len(dd)))
    out["mean_y_net"] = float(a["y_net"].mean()) if len(a) else float("nan")
    out["ungated_mean_y_net"] = float(b["y_net"].mean()) if len(b) else float("nan")
    out["ungated_fdr"] = m1.realised_fdr(ug)["fdr"]
    if len(a) > 1:
        wk = a.groupby("decision_date")["y_net"].mean()
        out["nw_se"] = m1.newey_west_se(wk.to_numpy(), lag)
        out["boot_ci95_block"] = m1.bootstrap_ci(m1.weekly_block_bootstrap(a["y_net"], m1.weekly_ids(a), n_boot, seed, lag))
    if len(a) > 1 and len(b) > 1:
        diff = float(a["y_net"].mean() - b["y_net"].mean())
        ci = m1.bootstrap_ci(m1.weekly_block_bootstrap_diff(a["y_net"], m1.weekly_ids(a), b["y_net"], m1.weekly_ids(b), n_boot, seed, lag))
        out["c5"] = dict(diff=diff, ci95=ci, passed=bool(diff >= 0))
    else:
        out["c5"] = dict(diff=float("nan"), ci95=(float("nan"),) * 2, passed=None)
    out["all_pass"] = all(out[k]["passed"] is True for k in ("c1", "c2", "c3", "c4", "c5"))
    return out


def go_table(logs: dict, decision_dates, first_judged_week, invariants_passed, n_boot: int = 300, seed: int = 11) -> pd.DataFrame:
    """logs = {impact: {arm: {horizon: (gate_log, ungated_log)}}}. One row per impact x arm x horizon. The FLOOR run is judged (go_all a
    bool); the ZERO run is reported with its own numbers (judged False, go_all None). Only arms a/b/c, horizons 5/20 and impacts
    floor/zero are accepted (A11 / horizon 60 / any other arm are diagnostics and are refused here). No multiplicity correction."""
    rows = []
    for impact, per_arm in logs.items():
        if impact not in IMPACTS:
            raise ValueError("impact must be 'floor' or 'zero'")
        for arm, per_h in per_arm.items():
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm!r}")
            for h, (log, ung) in per_h.items():
                if h not in JUDGED_HORIZONS:
                    raise ValueError(f"horizon {h} is not judged")
                c = go_criteria(log, ung, decision_dates, h, first_judged_week=first_judged_week, invariants_passed=invariants_passed,
                                n_boot=n_boot, seed=seed)
                judged = impact == "floor"
                ci5 = c["c5"]["ci95"]
                nb = c.get("boot_ci95_block", (float("nan"),) * 2)
                rows.append(dict(impact=impact, arm=arm, horizon=h, judged=judged, c1=c["c1"]["passed"], c2=c["c2"]["passed"],
                                 c3=c["c3"]["passed"], c4=c["c4"]["passed"], c5=c["c5"]["passed"], c2_fdr=c["c2"]["fdr"],
                                 c2_threshold=c["c2"]["threshold"], c3_worst_year_fdr=c["c3"]["worst_year_fdr"],
                                 c4_publication_rate=c["c4"]["publication_rate"], c5_diff=c["c5"]["diff"],
                                 c5_ci_lo=ci5[0], c5_ci_hi=ci5[1], mean_y_net=c["mean_y_net"], ungated_mean_y_net=c["ungated_mean_y_net"],
                                 ungated_fdr=c["ungated_fdr"], nw_se=c.get("nw_se", float("nan")), boot_lo=nb[0], boot_hi=nb[1],
                                 nw_lag_weeks=c["nw_lag_weeks"], n_filled=c["c2"]["n_filled"], n_no_fill=c["c2"]["n_no_fill"],
                                 n_no_data=c["c2"]["n_no_data"], n_judged_weeks=c["n_judged_weeks"],
                                 go_all=c["all_pass"] if judged else None))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- diagnostics (NOT judged)
def diagnostics_table(logs: dict) -> pd.DataFrame:
    """logs = {(variant, horizon, arm): log}, variant atr/hold, horizon 5/20/60, arm a/b/c/ungated. Reported rows only: judged False,
    no GO-criterion column."""
    rows = []
    for (variant, h, arm), log in logs.items():
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if h not in DIAGNOSTIC_HORIZONS:
            raise ValueError(f"unknown horizon {h}")
        if arm not in (*ARMS, "ungated"):
            raise ValueError(f"unknown arm {arm!r}")
        s = m1.realised_fdr(log)
        m = _pub_filled(log)
        pub = log[log["published"].astype(bool)]
        rows.append(dict(variant=variant, horizon=h, arm=arm, fdr=s["fdr"], n_filled=s["n_filled"], n_no_fill=s["n_no_fill"],
                         n_no_data=s["n_no_data"], mean_y_net=float(m["y_net"].mean()) if len(m) else float("nan"),
                         n_weeks_published=int(pub["decision_date"].nunique()), judged=False))
    return pd.DataFrame(rows)


def diagnostics_13b(log: pd.DataFrame, index_close: pd.Series, decision_dates) -> dict:
    """The 13b diagnostics, NOT judged: weekly FDP vs the index weekly return, and realised FDR with loser = underperformed the index
    after costs. Equal to vanta.m1b's own functions."""
    fdp = m1b.weekly_fdp(log)
    mkt = m1b.market_weekly_returns(index_close, decision_dates)
    return dict(fdp_vs_market=m1b.fdp_vs_market(fdp, mkt), fdr_vs_index=m1b.realised_fdr_vs_index(log, index_close), judged=False)


# ---------------------------------------------------------------------------------------------------------------- A11 hold-to-horizon
def hold_to_horizon_label(symbol, bars, decision_date, horizon, atr, hits, hit_coverage, cost_fn=None, delisted_after=None, *,
                          sessions=None, terminal_value=None) -> dict:
    """Hold-to-horizon label (v6 ablation A11): no stop and no target; entry exactly as simulate_trade (raw open of t+1, an upper-lock
    open is NO_FILL_UPPER_LOCK, N1); the exit is the CLOSE of the horizon session (first bar on/after t+h) with the same cost_fn.
    Built on simulate_trade with an infinite stop distance and target distance, so every entry / calendar / delisting / cost rule is the
    ATR label's own; stop_price, target_price, k_s and k_t are reported as NaN."""
    row = L.simulate_trade(symbol, bars, decision_date, horizon, math.inf, math.inf, atr, hits, hit_coverage, cost_fn=cost_fn,
                           delisted_after=delisted_after, sessions=sessions, terminal_value=terminal_value)
    for k in ("k_s", "k_t", "stop_price", "target_price"):
        row[k] = float("nan")
    return row


# ---------------------------------------------------------------------------------------------------------------- report text
def report_header() -> str:
    return ("The 2013-onward window was seen in M1 and again in M1b (both NO-GO: logs/m1_report.md, logs/m1b_report.md), and the Stage 14 "
            "baselines were scored on the same window. It has already been inspected. M2 is therefore a re-test on that window with a new "
            "score (B1), not a fresh out-of-sample test. A pass here is weaker evidence than a pass on unseen data. The only truly fresh "
            "test is the forward paper-trading log.")


def report_footer() -> str:
    return ("Realised FDR is an empirical number, not a guarantee, and q = 0.30 is an engineering target. The GO/NO-GO is the owner's "
            "call; this report does not make it.")
