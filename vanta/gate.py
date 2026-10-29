"""VANTA Stage 13C publish gate: regime-weighted conformal selection (RW-CS) with the Kish fallback (FCC).

Weights are exp(-age/tau) * exp(-L1(pi_i, pi_T)), pi being the filtered HMM posterior (soft weights, nothing learned).
q is an engineering target for the false-discovery proportion, never a guarantee.

Honest scope: the synthetic proof (scripts/verify_stage13c.py) covers EQUAL weights on exchangeable data only. The weighted
age x regime gate has NO proven FDR control here: weighted conformal p-values are exact only for true likelihood-ratio weights.
With the test point at weight 1 against calibration weights below 1 the weighted p-values are conservative.

Missing is missing: a non-finite candidate score gets p = 1 (never selectable, but kept in the BH denominator m); a non-finite
calibration score or y_net is excluded by select_calibration and rejected by validate_calibration; a non-finite regime posterior
forces the age-only fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_age_weights(age_weeks, pi_cal, pi_T, tau_weeks) -> np.ndarray:
    age = np.asarray(age_weeks, float)
    pi_cal = np.asarray(pi_cal, float).reshape(len(age), 3 if len(age) == 0 else -1)
    dist = np.abs(pi_cal - np.asarray(pi_T, float)).sum(axis=1)
    return np.exp(-age / tau_weeks) * np.exp(-dist)


def conformal_pvalues(cal_scores, cal_y, cal_w, test_scores, test_w=1.0) -> np.ndarray:
    """p_j = (test_w + sum_i w_i 1[y_i <= 0, s_i >= s_j]) / (test_w + sum_i w_i); sort once, cumulative loser weights."""
    s = np.asarray(cal_scores, float)
    y = np.asarray(cal_y, float)
    w = np.asarray(cal_w, float)
    t = np.asarray(test_scores, float)
    bad_t = ~np.isfinite(t)
    t = np.where(bad_t, 0.0, t)
    denom = test_w + w.sum()
    order = np.argsort(-s, kind="stable")                        # descending score
    s_desc = s[order]
    cum = np.concatenate([[0.0], np.cumsum(np.where(y[order] <= 0, w[order], 0.0))])
    # number of calibration scores >= t_j, from the ascending view of the descending array
    n_ge = len(s) - np.searchsorted(s_desc[::-1], t, side="left")
    p = (test_w + cum[n_ge]) / denom
    return np.where(bad_t, 1.0, p)                                # a missing candidate score is never selectable


def bh_select(pvals, q) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p, kind="stable")
    ok = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def _usable_mask(cal, T, model_version, candidate_filter_version, window_weeks):
    T = pd.Timestamp(T)
    m = cal["matured_at"].notna() & (cal["matured_at"] <= T)
    m &= cal["decision_date"] >= T - pd.Timedelta(weeks=window_weeks)
    m &= cal["model_version"] == model_version
    m &= cal["candidate_filter_version"] == candidate_filter_version
    m &= cal["is_oos"].astype(bool)
    m &= np.isfinite(cal["score"].astype(float))
    m &= cal["y_net"].notna()                                    # an outcome that is not known is not calibration data
    return m


def select_calibration(cal, T, *, model_version, candidate_filter_version, window_weeks=52) -> pd.DataFrame:
    """Rows usable at T: matured_at <= T, inside the window, same model/filter version, OOS, outcome known. Only these columns of
    `cal` are read for the decision (an extra `published` column is ignored by construction: N8)."""
    return cal[_usable_mask(cal, T, model_version, candidate_filter_version, window_weeks)]


def validate_calibration(cal, T, *, model_version, candidate_filter_version) -> None:
    T = pd.Timestamp(T)
    if (cal["matured_at"].isna() | (cal["matured_at"] > T)).any():
        raise ValueError("calibration row not matured by T")
    if (cal["model_version"] != model_version).any():
        raise ValueError("model_version mismatch")
    if (cal["candidate_filter_version"] != candidate_filter_version).any():
        raise ValueError("candidate_filter_version mismatch")
    if not cal["is_oos"].astype(bool).all():
        raise ValueError("non-OOS calibration row")
    if cal["y_net"].isna().any():
        raise ValueError("calibration row with unknown y_net")
    if not np.isfinite(cal["score"].astype(float)).all():
        raise ValueError("calibration row with non-finite score")


def kish_ess(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def run_gate(cands, cal, T, pi_T, *, model_version, candidate_filter_version, tau_weeks=26, q=0.3, cap=10,
             ess_floor=200, window_weeks=52) -> pd.DataFrame:
    """One row per candidate, in `cands` order. Extra columns beyond the interface: `age_only_fallback` (bool; True when the
    Kish ESS of the age x regime weights fell below `ess_floor` and age-only weights were used). `effective_n` is the ESS of the
    age x regime weights BEFORE any fallback, so the log shows why the fallback fired."""
    T = pd.Timestamp(T)
    sel = select_calibration(cal, T, model_version=model_version, candidate_filter_version=candidate_filter_version,
                             window_weeks=window_weeks)
    age = (T - sel["decision_date"]).dt.days.to_numpy() / 7.0
    pi_cal = np.array([list(x) for x in sel["regime_posterior"]], float).reshape(len(sel), 3)
    w = regime_age_weights(age, pi_cal, pi_T, tau_weeks)
    w_age = np.exp(-age / tau_weeks)
    ess = kish_ess(w) if np.isfinite(w).all() else float("nan")
    fallback = bool((not np.isfinite(w).all()) or ess < ess_floor)    # non-finite posterior -> deterministic age-only fallback
    if fallback:
        w = w_age
    out = cands.reset_index(drop=True).copy()
    scores = out["score"].to_numpy(float)
    p = conformal_pvalues(sel["score"], sel["y_net"], w, scores)
    in_bh = bh_select(p, q) & np.isfinite(scores)             # a missing score is never in the set, whatever q
    m, k = len(p), int(in_bh.sum())
    out["p_value"], out["in_bh_set"] = p, in_bh
    out["bh_threshold"] = q * k / m if k else float("nan")
    # ties broken by HIGHER score first (user-confirmed 2026-10-27, decisions log Q4); then input order for determinism
    order = sorted((i for i in range(m) if in_bh[i]), key=lambda i: (p[i], -scores[i], i))
    out["published"] = False
    out.loc[order[:cap], "published"] = True
    out["cap_binding"] = k > cap
    out["q"], out["n_calibration"], out["effective_n"] = q, len(sel), ess
    out["age_only_fallback"] = fallback
    out["effective_n_age_only"] = kish_ess(w_age)
    out["score_missing"] = ~np.isfinite(scores)
    return out
