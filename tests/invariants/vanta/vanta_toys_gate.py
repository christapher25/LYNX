"""
Reference ("good") and deliberately broken toy implementations of the Stage 13C gate and the 13C gain builder, plus builders for
tiny calibration tables. Test support only; runs with no `vanta` package present.

Interface assumed (see the V1 note to the lead): vanta.gate.{regime_age_weights, conformal_pvalues, bh_select,
select_calibration, validate_calibration, run_gate} and vanta.gains.build_fold_gains.

Every function takes `bugs` (a frozenset of names) to inject exactly one defect; the empty set is the spec's reference behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NO_BUGS = frozenset()
CAL_COLS = ["decision_date", "symbol", "horizon", "score", "model_version", "fold_id", "is_oos", "regime_posterior",
            "hmm_version", "y_net", "matured_at", "candidate_filter_version"]
MODEL, FILTER = "vanta-ranker@f07:2018-12-31:a1b2c3d:9f2e01", "cfilter@f07:2018-12-31:a1b2c3d:77aa11"


def regime_age_weights(age_weeks, pi_cal, pi_T, tau_weeks, bugs=NO_BUGS):
    age = np.asarray(age_weeks, float)
    pi_cal, pi_T = np.asarray(pi_cal, float), np.asarray(pi_T, float)
    w_age = np.exp(-age / tau_weeks)
    if "age_only" in bugs:
        return w_age
    dist = np.abs(pi_cal - pi_T).sum(axis=1)
    return w_age * np.exp(-dist)


def conformal_pvalues(cal_scores, cal_y, cal_w, test_scores, test_w=1.0, bugs=NO_BUGS):
    s, y, w = np.asarray(cal_scores, float), np.asarray(cal_y, float), np.asarray(cal_w, float)
    if "ignore_weights" in bugs:
        w = np.ones_like(w)
    out = []
    for sj in np.asarray(test_scores, float):
        ge = s > sj if "strict_ties" in bugs else s >= sj
        if "wrong_direction" in bugs:
            ge = s <= sj
        lose = np.ones_like(y, bool) if "ignore_y" in bugs else y <= 0
        num = w[ge & lose].sum() + (0.0 if "no_plus_one" in bugs else test_w)
        out.append(num / (test_w + w.sum()))
    return np.array(out)


def bh_select(pvals, q, bugs=NO_BUGS):
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    if "bonferroni" in bugs:
        return p <= q / m
    order = np.argsort(p, kind="stable")
    ok = p[order] <= q * (np.arange(1, m + 1)) / m
    if "step_down" in bugs:                      # stops at the first failure instead of taking the LARGEST passing rank
        k = 0
        while k < m and ok[k]:
            k += 1
    else:
        k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def select_calibration(cal, T, *, model_version, candidate_filter_version, window_weeks=52, bugs=NO_BUGS):
    T = pd.Timestamp(T)
    m = cal["decision_date"] >= T - pd.Timedelta(weeks=window_weeks)
    if "no_maturity_filter" not in bugs:
        m &= (cal["matured_at"] <= T + (pd.Timedelta(days=1) if "maturity_off_by_one" in bugs else pd.Timedelta(0)))
    if "strict_maturity" in bugs:
        m &= cal["matured_at"] < T
    if "no_version_filter" not in bugs:
        m &= cal["model_version"] == model_version
    if "no_filter_version_check" not in bugs:
        m &= cal["candidate_filter_version"] == candidate_filter_version
    if "no_oos_filter" not in bugs:
        m &= cal["is_oos"].astype(bool)
    if "published_only" in bugs and "published" in cal.columns:
        m &= cal["published"].astype(bool)
    return cal[m]


def validate_calibration(cal, T, *, model_version, candidate_filter_version, bugs=NO_BUGS):
    T = pd.Timestamp(T)
    if "no_maturity_filter" not in bugs and ((cal["matured_at"].isna()) | (cal["matured_at"] > T)).any():
        raise ValueError("calibration row not matured by T")
    if "no_version_filter" not in bugs and (cal["model_version"] != model_version).any():
        raise ValueError("model_version mismatch")
    if "no_filter_version_check" not in bugs and (cal["candidate_filter_version"] != candidate_filter_version).any():
        raise ValueError("candidate_filter_version mismatch")
    if "no_oos_filter" not in bugs and not cal["is_oos"].astype(bool).all():
        raise ValueError("non-OOS calibration row")


def run_gate(cands, cal, T, pi_T, *, model_version, candidate_filter_version, tau_weeks=26, q=0.3, cap=10, ess_floor=200,
             bugs=NO_BUGS):
    T = pd.Timestamp(T)
    sel = select_calibration(cal, T, model_version=model_version, candidate_filter_version=candidate_filter_version, bugs=bugs)
    age = (T - sel["decision_date"]).dt.days.to_numpy() / 7.0
    pi_cal = np.array([list(x) for x in sel["regime_posterior"]]).reshape(len(sel), 3)
    w = regime_age_weights(age, pi_cal, pi_T, tau_weeks)
    ess = (w.sum() ** 2) / (w ** 2).sum() if len(w) else 0.0
    if ess < ess_floor:
        w = regime_age_weights(age, pi_cal, pi_T, tau_weeks, bugs={"age_only"})
    p = conformal_pvalues(sel["score"], sel["y_net"], w, cands["score"], bugs=bugs)
    in_bh = bh_select(p, q, bugs=bugs)
    m = len(p)
    k = int(in_bh.sum())
    out = cands.reset_index(drop=True).copy()
    out["p_value"], out["in_bh_set"] = p, in_bh
    out["bh_threshold"] = q * k / m if k else float("nan")
    # Tie-break direction (higher score first) is our reading: the spec says only "ties broken by score".
    order = sorted(range(m), key=lambda i: (p[i], -out["score"].iloc[i] if "tie_lowest_score" not in bugs else out["score"].iloc[i]))
    if "published_not_subset_of_bh" in bugs:
        pool = order
    else:
        pool = [i for i in order if in_bh[i]]
    capped = pool if "no_cap" in bugs else pool[:cap]
    out["published"] = False
    out.loc[capped, "published"] = True
    out["cap_binding"] = (k > cap) and "cap_not_logged" not in bugs
    out["q"], out["n_calibration"], out["effective_n"] = q, len(sel), ess
    return out


def cal_rows(decision_date, symbols, scores, ys, matured_at, *, model=MODEL, filt=FILTER, oos=True, pi=(1.0, 0.0, 0.0),
             published=None):
    n = len(symbols)
    df = pd.DataFrame({
        "decision_date": pd.Timestamp(decision_date), "symbol": symbols, "horizon": 5, "score": scores,
        "model_version": model, "fold_id": "f07", "is_oos": oos, "regime_posterior": [list(pi)] * n, "hmm_version": "hmm@f07",
        "y_net": ys, "matured_at": matured_at, "candidate_filter_version": filt,
    })
    if published is not None:
        df["published"] = published
    return df


def build_fold_gains(labels, train_mask, horizon, bugs=NO_BUGS):
    g = labels["y_net"] / (labels["atr_pct"] * np.sqrt(horizon))
    basis = g if "fit_on_all_rows" in bugs else g[np.asarray(train_mask, bool)]
    lo, hi = np.percentile(basis, [2, 98])
    return g.clip(lo, hi)


def toy_api(*bugs):
    """A namespace with the same function names as vanta.gate, bound to the named bugs (none = reference)."""
    import functools
    from types import SimpleNamespace

    b = frozenset(bugs)
    return SimpleNamespace(
        regime_age_weights=functools.partial(regime_age_weights, bugs=b),
        conformal_pvalues=functools.partial(conformal_pvalues, bugs=b),
        bh_select=functools.partial(bh_select, bugs=b),
        select_calibration=functools.partial(select_calibration, bugs=b),
        validate_calibration=functools.partial(validate_calibration, bugs=b),
        run_gate=functools.partial(run_gate, bugs=b),
    )


def oracle_p(sel, T, pi_T, tau_weeks, test_scores):
    """Independent plain-loop restatement of the spec's weighted p-value (age x regime weights), for known-answer checks."""
    import math

    ws, ys, ss = [], [], []
    for r in sel.itertuples():
        age = (pd.Timestamp(T) - r.decision_date).days / 7.0
        l1 = sum(abs(a - b) for a, b in zip(r.regime_posterior, pi_T))
        ws.append(math.exp(-age / tau_weeks) * math.exp(-l1))
        ys.append(r.y_net)
        ss.append(r.score)
    out = []
    for sj in test_scores:
        lose_w = sum(w for w, y, s in zip(ws, ys, ss) if y <= 0 and s >= sj)
        out.append((1.0 + lose_w) / (1.0 + sum(ws)))
    return out


def calibration_world(T="2024-06-03", weeks=4, n=30):
    """`weeks` prior weekly decision dates x `n` scored candidates each. score = 0..n-1; losers are score < 20 (y=-0.01), winners
    y=+0.02. matured_at = decision_date + 6 days (so every week is matured by T). `published` flags the top-3 scores per week."""
    parts = []
    for w in range(1, weeks + 1):
        d = pd.Timestamp(T) - pd.Timedelta(weeks=w)
        scores = np.arange(n, dtype=float)
        ys = np.where(scores < 20, -0.01, 0.02)
        parts.append(cal_rows(d, [f"S{i:02d}" for i in range(n)], scores, ys, d + pd.Timedelta(days=6),
                              published=scores >= n - 3))
    return pd.concat(parts, ignore_index=True)
