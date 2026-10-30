"""
Reference ("toy") implementation of the vanta.m1b pure functions, with switchable deliberate bugs.

Not a VANTA module. The tests run every check against `toy_api()` (must pass) and `toy_api(<bug>)` (must fail).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

NO_BUGS = frozenset()
CLIFF = 200


def percentile_rank(values, bugs=NO_BUGS):
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if "nan_as_zero" in bugs:
        v = np.where(np.isfinite(v), v, 0.0)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    if "single_not_one" in bugs and ok.sum() == 1:
        out[ok] = 0.5
        return out
    method = {"tie_first": "first", "tie_min": "min", "tie_max": "max"}.get(
        next((b for b in bugs if b.startswith("tie_")), ""), "average")
    r = pd.Series(v[ok]).rank(method=method).to_numpy()
    out[ok] = r / ok.sum()
    return out


def weekly_scores(frame, momentum_col="momentum", bugs=NO_BUGS):
    if "raw_momentum" in bugs:
        return frame[momentum_col].astype(float).copy()
    if "pooled_history" in bugs:
        return pd.Series(percentile_rank(frame[momentum_col].to_numpy(float), bugs), index=frame.index)
    out = pd.Series(np.nan, index=frame.index)
    for _, g in frame.groupby("decision_date"):
        out.loc[g.index] = percentile_rank(g[momentum_col].to_numpy(float), bugs)
    return out


def eligible_candidates(week, cliff=CLIFF, bugs=NO_BUGS):
    keep = pd.Series(True, index=week.index)

    def clear(col):                                  # True only for an explicit False; NA means "list unavailable"
        f = week[col].astype("boolean")
        return f.fillna(False).eq(False) if "na_allowed" in bugs else f.eq(False).fillna(False).astype(bool)

    if "ignore_ban" not in bugs:
        keep &= clear("banned")
    if "ignore_upper" not in bugs:
        keep &= clear("hit_upper")
    if "ignore_lower" not in bugs:
        keep &= clear("hit_lower")
    rank = week["liquidity_rank"].astype(float)
    if "no_cliff" not in bugs:
        ok = (rank < cliff) if "exclusive_cliff" in bugs else (rank <= cliff)
        if "nan_rank_passes" in bugs:
            ok = ok | rank.isna()
        keep &= ok
    if "nan_momentum_kept" not in bugs:
        keep &= np.isfinite(week["momentum"].astype(float))
    out = week[keep]
    if "top10_only" in bugs:
        out = out.sort_values("momentum", ascending=False, kind="stable").head(10)
    return out


def calibration_rows(scored, T, window_weeks=52, bugs=NO_BUGS):
    T = pd.Timestamp(T)
    if "window_26" in bugs:
        window_weeks = 26
    if "window_104" in bugs:
        window_weeks = 104
    keep = scored["decision_date"] >= T - pd.Timedelta(weeks=window_weeks)
    if "exclusive_window_start" in bugs:
        keep = scored["decision_date"] > T - pd.Timedelta(weeks=window_weeks)
    if "unmatured" not in bugs:
        keep &= scored["matured_at"] <= T
    else:
        keep &= scored["decision_date"] <= T
    if "published_only" in bugs:
        keep &= scored["published"].astype(bool)
    if "nan_y_kept" not in bugs:
        keep &= scored["y_net"].notna()
    return scored[keep]


def select_published(frame, cap=10, bugs=NO_BUGS):
    if "no_cap" in bugs:
        cap = 10**9
    if "cap_11" in bugs:
        cap = 11
    cand = frame if "ignore_bh" in bugs else frame[frame["in_bh_set"].astype(bool)]
    top = cand.sort_values("score", ascending=False, kind="stable").index[:cap]
    return frame.index.isin(top)


def toy_api(*bugs):
    b = frozenset(bugs)
    return SimpleNamespace(
        CAP=10, LIQUIDITY_CLIFF_RANK=CLIFF,
        percentile_rank=lambda v: percentile_rank(v, b),
        weekly_scores=lambda f, momentum_col="momentum": weekly_scores(f, momentum_col, b),
        eligible_candidates=lambda w, cliff=CLIFF: eligible_candidates(w, cliff, b),
        calibration_rows=lambda s, T, window_weeks=52: calibration_rows(s, T, window_weeks, b),
        select_published=lambda f, cap=10: select_published(f, cap, b),
    )
