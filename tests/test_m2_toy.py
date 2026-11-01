"""
Reference ("toy") implementation of the vanta.m2 contract with switchable deliberate bugs. NOT a VANTA module and holds no tests.

The tests in tests/test_m2_invariants.py and tests/test_m2_driver.py run every check against the real vanta.m2 (skipped until it
exists), against `toy_api()` (must pass: the check is not over-strict) and against `toy_api(<bug>)` (must fail: the check can fail).

The toy is built on the REAL vanta.gate / vanta.m1 / vanta.m1b / vanta.labels primitives, so it is what an implementer would write.
The contract (all names INVENTED by the test writer, proposed for docs/vanta/INTERFACES.md) is the set of functions below.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from vanta import gate as G
from vanta import labels as L
from vanta import m1, m1b

ARMS = ("a", "b", "c")
JUDGED_HORIZONS = (5, 20)
DIAGNOSTIC_HORIZONS = (5, 20, 60)
VARIANTS = ("atr", "hold")
MODEL_VERSION = "b1@m2"
FILTER_VERSION = "m2-eligibility@v1"
GO_FDR_THRESHOLD = 0.35
FLAT = [1 / 3] * 3


def nw_lag_weeks(h):
    return int(math.ceil(h / 5))


def toy_api(*bugs):
    bugs = frozenset(bugs)

    # ------------------------------------------------------------------ constants
    consts = dict(Q=0.30 if "q25" not in bugs else 0.25, CAP=10 if "cap5" not in bugs else 5,
                  TAU_WEEKS=26, ESS_FLOOR=200 if "ess100" not in bugs else 100, WINDOW_WEEKS=52,
                  K_S=1.5, K_T=1.5 if "kt2" not in bugs else 2.0, B1_SEEDS=(0, 1, 2, 3, 4),
                  IMPACT_FLOOR_TOP200=0.0005, IMPACT_FLOOR_REST=0.0020, ARMS=ARMS if "four_arms" not in bugs else ARMS + ("d",),
                  JUDGED_HORIZONS=JUDGED_HORIZONS, DIAGNOSTIC_HORIZONS=DIAGNOSTIC_HORIZONS)

    # ------------------------------------------------------------------ B1 percentile (the gate's score)
    def weekly_percentile(frame, score_col="b1"):
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        if "raw" in bugs:
            return frame[score_col].astype(float).copy()
        if "pooled" in bugs:
            return pd.Series(m1b.percentile_rank(frame[score_col].to_numpy(float)), index=frame.index)
        for _, g in frame.groupby("decision_date"):
            v = g[score_col].to_numpy(float)
            if "nan_zero" in bugs:
                v = np.where(np.isfinite(v), v, 0.0)
            if "tie_min" in bugs:
                ok = np.isfinite(v)
                r = np.full(len(v), np.nan)
                r[ok] = pd.Series(v[ok]).rank(method="min").to_numpy() / ok.sum()
                out.loc[g.index] = r
            else:
                out.loc[g.index] = m1b.percentile_rank(v)
        return out

    # ------------------------------------------------------------------ the three gate arms
    def gate_arm(arm, cands, cal, T, pi_T, *, q=consts["Q"], cap=consts["CAP"], tau_weeks=26, ess_floor=consts["ESS_FLOOR"],
                 window_weeks=52, model_version=MODEL_VERSION, candidate_filter_version=FILTER_VERSION):
        if arm not in consts["ARMS"]:
            raise ValueError(f"arm must be one of {ARMS}")
        T = pd.Timestamp(T)
        pi = np.full(3, np.nan) if pi_T is None else np.asarray(pi_T, float)
        if "nan_as_zero" in bugs:
            pi = np.where(np.isfinite(pi), pi, 0.0)
        missing = not np.isfinite(pi).all()
        gk = dict(model_version=model_version, candidate_filter_version=candidate_filter_version, tau_weeks=tau_weeks, q=q, cap=cap,
                  ess_floor=ess_floor, window_weeks=window_weeks)
        base = cands.reset_index(drop=True)

        def via_run_gate(cal_use, pi_use):
            return G.run_gate(base, cal_use, T, pi_use, **gk)

        def decorate(out, abstained, state_missing):
            out = out.copy()
            out["abstained"] = abstained
            out["state_missing"] = state_missing
            if abstained:
                out["published"] = False
            return out

        if arm == "a":
            if "no_regime" in bugs:
                out = via_run_gate(cal.assign(regime_posterior=[FLAT] * len(cal)), FLAT)
            else:
                out = via_run_gate(cal, pi)
            return decorate(out, False, missing)

        if arm == "c":
            out = via_run_gate(cal, pi)
            p2 = pi[2]
            if "missing_flag_dropped" in bugs and missing:
                out["age_only_fallback"] = False
            if not np.isfinite(p2):
                return decorate(out, "missing_abstains" in bugs, missing)
            thr = 0.6 if "thr06" in bugs else 0.5
            hit = (p2 > thr) if "strict" in bugs else (p2 >= thr)
            if "argmax" in bugs:
                hit = int(np.argmax(pi)) == 2
            return decorate(out, bool(hit), missing)

        # arm b
        if missing:
            if "b_missing_abstains" in bugs:
                out = via_run_gate(cal, [np.nan] * 3)
                return decorate(out, True, True)
            out = via_run_gate(cal, [np.nan] * 3)                       # (a)'s age-only fallback, counted and flagged
            return decorate(out, False, True)
        sel = G.select_calibration(cal, T, model_version=model_version, candidate_filter_version=candidate_filter_version,
                                   window_weeks=window_weeks)
        post = np.array([list(x) for x in sel["regime_posterior"]], float).reshape(len(sel), 3)
        okp = np.isfinite(post).all(axis=1)
        same = np.zeros(len(sel), bool)
        same[okp] = post[okp].argmax(axis=1) == int(np.argmax(pi))
        if "all_rows" in bugs:
            same = np.ones(len(sel), bool)
        if "nan_rows_match" in bugs:
            same = same | ~okp
        sub = sel[same]
        age = (T - sub["decision_date"]).dt.days.to_numpy() / 7.0
        w = np.exp(-age / tau_weeks)
        if "regime_weights" in bugs:
            w = G.regime_age_weights(age, post[same], pi, tau_weeks)
        ess = G.kish_ess(w)
        abstain = (ess <= ess_floor) if "floor_inclusive" in bugs else (ess < ess_floor)
        if "no_abstain" in bugs:
            abstain = False
        out = via_run_gate(sub.assign(regime_posterior=[FLAT] * len(sub)), FLAT)
        out["n_calibration"], out["effective_n"] = len(sub), ess
        return decorate(out, bool(abstain), False)

    # ------------------------------------------------------------------ walk over weeks
    def _top_n(g, n):
        ok = g[np.isfinite(g["score"].astype(float))]
        return ok.sort_values("score", ascending=False, kind="stable").head(n)

    def run_m2_arms(cands, decision_dates, *, q=consts["Q"], cap=consts["CAP"], tau_weeks=26, ess_floor=consts["ESS_FLOOR"],
                    window_weeks=52):
        c = cands.copy()
        c["score"] = weekly_percentile(c, "b1")
        cal = pd.DataFrame({"decision_date": c["decision_date"], "symbol": c["symbol"], "horizon": c["horizon"], "score": c["score"],
                            "model_version": MODEL_VERSION, "fold_id": "m2", "is_oos": True, "regime_posterior": c["regime_posterior"],
                            "hmm_version": c["hmm_version"], "y_net": c["y_net"], "matured_at": c["matured_at"],
                            "candidate_filter_version": FILTER_VERSION}).reset_index(drop=True)
        if "future_cal" in bugs:
            cal["matured_at"] = cal["decision_date"]
        logs = {k: [] for k in (*ARMS, "ungated")}
        by_date = {d: g.reset_index(drop=True) for d, g in c.groupby("decision_date")}
        for T in pd.DatetimeIndex(decision_dates):
            g = by_date.get(T)
            if g is None or g.empty:
                continue
            pi_T = g["regime_posterior"].iloc[0]
            if "last_posterior" in bugs:
                pi_T = c["regime_posterior"].iloc[-1]
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
            if "ung_all" in bugs:
                u["published"] = True
            else:
                n = 5 if "ung_top5" in bugs else 10
                idx = _top_n(u, n).index
                if "ung_bottom" in bugs:
                    idx = u[np.isfinite(u["score"])].sort_values("score", kind="stable").head(10).index
                u.loc[idx, "published"] = True
            logs["ungated"].append(u)
        return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in logs.items()}

    def arm_counts(log, decision_dates=None):
        by_week = log.groupby("decision_date").agg(fb=("age_only_fallback", "first"), ab=("abstained", "first"),
                                                   sm=("state_missing", "first"))
        if "row_counts" in bugs:
            return dict(n_weeks=int(len(log)), n_abstain_weeks=int(log["abstained"].sum()),
                        n_age_only_fallback_weeks=int(log["age_only_fallback"].sum()),
                        share_age_only_fallback=float(log["age_only_fallback"].mean()),
                        n_state_missing_weeks=int(log["state_missing"].sum()))
        return dict(n_weeks=int(len(by_week)), n_abstain_weeks=int(by_week["ab"].sum()),
                    n_age_only_fallback_weeks=int(by_week["fb"].sum()), share_age_only_fallback=float(by_week["fb"].mean()),
                    n_state_missing_weeks=int(by_week["sm"].sum()))

    # ------------------------------------------------------------------ judged window
    def judged_window(scored, oos_start=m1.OOS_START, window_weeks=52):
        s = scored if "nan_scored" in bugs else scored[np.isfinite(scored["score"].to_numpy(float))]
        weeks = pd.DatetimeIndex(sorted(s["decision_date"].unique()))
        if not len(weeks):
            return {"first_scored_week": pd.NaT, "first_judged_week": pd.NaT}
        ww = {"w51": 51, "w53": 53, "no_buffer": 0}.get(next((b for b in bugs if b in ("w51", "w53", "no_buffer")), ""), window_weeks)
        keep = weeks - pd.Timedelta(weeks=ww) >= weeks[0]
        if "oos_ignored" not in bugs:
            keep &= weeks >= pd.Timestamp(oos_start)
        cand = weeks[keep]
        return {"first_scored_week": weeks[0], "first_judged_week": cand[0] if len(cand) else pd.NaT}

    def split_judged(log, first_judged_week):
        f = pd.Timestamp(first_judged_week)
        if "judged_all" in bugs:
            j = log
        elif "gt" in bugs:
            j = log[log["decision_date"] > f]
        else:
            j = log[log["decision_date"] >= f]
        rep = log[log["decision_date"] < f]
        if "drop_early" in bugs:
            rep = log.iloc[0:0]
        return dict(judged=j, reported_only=rep, first_judged_week=f, n_judged_weeks=int(j["decision_date"].nunique()),
                    n_reported_only_weeks=int(rep["decision_date"].nunique()))

    # ------------------------------------------------------------------ the five GO criteria
    def go_criteria(log, ungated, decision_dates, horizon, *, first_judged_week, invariants_passed, q=consts["Q"], n_boot=300,
                    seed=11):
        f = pd.Timestamp(first_judged_week)
        if "unjudged_included" in bugs:
            f = pd.Timestamp.min
        lg, ug = log[log["decision_date"] >= f], ungated[ungated["decision_date"] >= f]
        dd = pd.DatetimeIndex(decision_dates)
        dd = dd[dd >= f]
        lag = {"lag1": 1, "lag_h": horizon}.get(next((b for b in bugs if b in ("lag1", "lag_h")), ""), nw_lag_weeks(horizon))
        s = m1.realised_fdr(lg)
        fdr = s["fdr"]
        thr = q if "thr_q" in bugs else GO_FDR_THRESHOLD
        if "no_fill_counted" in bugs:
            pub = lg[lg["published"].astype(bool)]
            fdr = float((pub["y_net"].fillna(-1) <= 0).mean()) if len(pub) else float("nan")
        c2 = None if not np.isfinite(fdr) else bool(fdr < thr if "c2_strict" in bugs else fdr <= thr + 1e-12)
        by_year = m1.fdr_by_year(lg)
        lim = 0.5
        c3 = None if not by_year else bool(all((v < lim) if "c3_ge" in bugs else (v <= lim) for v in by_year.values()))
        pr_weeks = dd if "pr_denominator_log" not in bugs else pd.DatetimeIndex(lg["decision_date"].unique())
        pr = m1.publication_rate(lg, pr_weeks)
        c4 = None if not np.isfinite(pr) else bool(pr > 0.20 if "c4_strict" in bugs else pr >= 0.20 - 1e-12)

        def pf(l):
            return l[l["published"].astype(bool) & (l["fill_status"] == "FILLED") & l["y_net"].notna()]
        a = pf(lg)
        b = pf(ug) if "ung_unfiltered" not in bugs else ug[ug["y_net"].notna()]
        bl = 1 if "no_block" in bugs else lag
        out = dict(c1=dict(passed=bool(invariants_passed)), c2=dict(fdr=fdr, threshold=thr, passed=c2),
                   c3=dict(fdr_by_year=by_year, worst_year_fdr=max(by_year.values()) if by_year else float("nan"), passed=c3),
                   c4=dict(publication_rate=pr, passed=c4), nw_lag_weeks=lag, n_judged_weeks=int(len(dd)))
        if len(a) > 1:
            wk = a.groupby("decision_date")["y_net"].mean()
            out["nw_se"] = m1.newey_west_se(wk.to_numpy(), lag)
            out["boot_ci95_block"] = m1.bootstrap_ci(m1.weekly_block_bootstrap(a["y_net"], m1.weekly_ids(a), n_boot, seed, bl))
        if len(a) > 1 and len(b) > 1:
            diff = float(a["y_net"].mean() - b["y_net"].mean())
            ci = m1.bootstrap_ci(m1.weekly_block_bootstrap_diff(a["y_net"], m1.weekly_ids(a), b["y_net"], m1.weekly_ids(b), n_boot,
                                                                seed, bl))
            out["c5"] = dict(diff=diff, ci95=ci, passed=bool(diff > 0 if "c5_strict" in bugs else diff >= 0))
        else:
            out["c5"] = dict(diff=float("nan"), ci95=(float("nan"),) * 2, passed=None)
        out["all_pass"] = all(out[k]["passed"] is True for k in ("c1", "c2", "c3", "c4", "c5"))
        return out

    def go_table(logs, decision_dates, first_judged_week, invariants_passed, n_boot=300, seed=11):
        rows = []
        for impact, per_arm in logs.items():
            if impact not in ("floor", "zero"):
                raise ValueError("impact must be 'floor' or 'zero'")
            for arm, per_h in per_arm.items():
                if arm not in ARMS and "accept_any_arm" not in bugs:
                    raise ValueError(f"unknown arm {arm!r}")
                for h, (log, ung) in per_h.items():
                    if h not in JUDGED_HORIZONS and "accept_h60" not in bugs:
                        raise ValueError(f"horizon {h} is not judged")
                    if "floor_uses_zero" in bugs and impact == "floor":
                        log, ung = logs["zero"][arm][h]
                    c = go_criteria(log, ung, decision_dates, h, first_judged_week=first_judged_week,
                                    invariants_passed=invariants_passed, n_boot=n_boot, seed=seed)
                    judged = impact == "floor" or "judge_zero" in bugs
                    if "judge_none" in bugs:
                        judged = False
                    rows.append(dict(impact=impact, arm=arm, horizon=h, judged=judged, c1=c["c1"]["passed"], c2=c["c2"]["passed"],
                                     c3=c["c3"]["passed"], c4=c["c4"]["passed"], c5=c["c5"]["passed"], c2_fdr=c["c2"]["fdr"],
                                     c2_threshold=c["c2"]["threshold"], c4_publication_rate=c["c4"]["publication_rate"],
                                     c5_diff=c["c5"]["diff"], nw_lag_weeks=c["nw_lag_weeks"],
                                     go_all=c["all_pass"] if judged else None))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ diagnostics (NOT judged)
    def diagnostics_table(logs):
        rows = []
        for (variant, h, arm), log in logs.items():
            if variant not in VARIANTS and "any_variant" not in bugs:
                raise ValueError(f"unknown variant {variant!r}")
            if h not in DIAGNOSTIC_HORIZONS:
                raise ValueError(f"unknown horizon {h}")
            if "drop_hold" in bugs and variant == "hold":
                continue
            if "drop_h60" in bugs and h == 60:
                continue
            s = m1.realised_fdr(log)
            m = log[log["published"].astype(bool) & (log["fill_status"] == "FILLED") & log["y_net"].notna()]
            row = dict(variant=variant, horizon=h, arm=arm, fdr=s["fdr"], n_filled=s["n_filled"], n_no_fill=s["n_no_fill"],
                       mean_y_net=float(m["y_net"].mean()) if len(m) else float("nan"), judged="judged_true" in bugs)
            if "go_col" in bugs:
                row["go_all"] = bool(s["fdr"] <= GO_FDR_THRESHOLD)
            rows.append(row)
        return pd.DataFrame(rows)

    def diagnostics_13b(log, index_close, decision_dates):
        fdp = m1b.weekly_fdp(log)
        mkt = m1b.market_weekly_returns(index_close, decision_dates)
        return dict(fdp_vs_market=m1b.fdp_vs_market(fdp, mkt), fdr_vs_index=m1b.realised_fdr_vs_index(log, index_close),
                    judged="judged_true" in bugs)

    # ------------------------------------------------------------------ A11 hold-to-horizon
    def hold_to_horizon_label(symbol, bars, decision_date, horizon, atr, hits, hit_coverage, cost_fn=None, delisted_after=None, *,
                              sessions=None, terminal_value=None):
        big = 1.5 if "uses_atr" in bugs else 1e9
        kt = 1.5 if "uses_atr" in bugs else 1e9
        row = L.simulate_trade(symbol, bars, decision_date, horizon, big, kt, atr, hits, hit_coverage, cost_fn=cost_fn,
                               delisted_after=delisted_after, sessions=sessions, terminal_value=terminal_value)
        if row["fill_status"] != "FILLED":
            return row
        ac = bars["close"].to_numpy(float)
        j = int(bars.index.get_loc(row["exit_date"]))
        if "exit_prev_close" in bugs:
            j -= 1
        if "exit_high" in bugs:
            px = float(bars["high"].iloc[j])
        elif "exit_open" in bugs:
            px = float(bars["open"].iloc[j])
        else:
            px = float(ac[j])
        if px != row["exit_price"]:
            cf = cost_fn or L.default_cost_fn
            ce, cx, ci = cf(symbol, row["entry_date"], bars.index[j], row["entry_price"], px)
            gross = px / row["entry_price"] - 1.0
            row.update(exit_price=px, exit_date=bars.index[j], label_end_date=bars.index[j], gross_return=gross, cost_entry=ce,
                       cost_exit=cx, cost_impact=ci, y_net=gross - ce - cx - ci)
        if "no_costs" in bugs:
            row["y_net"] = row["gross_return"]
        return row

    # ------------------------------------------------------------------ report text
    def report_header():
        return ("The 2013-onward window was seen in M1 and again in M1b (both NO-GO). M2 is therefore a re-test on a window already "
                "inspected, with a new score, not a fresh out-of-sample test. A pass here is weaker evidence than a pass on unseen "
                "data. The only truly fresh test is the forward paper-trading log." if "header_silent" not in bugs else
                "M2 tests the VANTA gate with the B1 score on 2013-onward data.")

    def report_footer():
        return ("Realised FDR is an empirical number, not a guarantee, and q = 0.30 is an engineering target. The GO/NO-GO is the "
                "owner's call." if "footer_guarantee" not in bugs else
                "Realised FDR is a guarantee at q = 0.30. The verdict is GO.")

    return SimpleNamespace(**consts, nw_lag_weeks=nw_lag_weeks, weekly_percentile=weekly_percentile, gate_arm=gate_arm,
                           run_m2_arms=run_m2_arms, arm_counts=arm_counts, judged_window=judged_window, split_judged=split_judged,
                           go_criteria=go_criteria, go_table=go_table, diagnostics_table=diagnostics_table,
                           diagnostics_13b=diagnostics_13b, hold_to_horizon_label=hold_to_horizon_label,
                           report_header=report_header, report_footer=report_footer)
