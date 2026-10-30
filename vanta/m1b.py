"""VANTA Milestone M1b (Stage 13M, corrected): the M1 pipeline with the two departures from v6 removed.

M1 (logs/m1_report.md, NO-GO as run) published from a pool of the weekly top-10 momentum names and scored them with the raw 12-1
momentum LEVEL. v6 runs the BH step across ALL candidates at T, publishes at most CAP, calibrates on ALL scored candidates of the
last 52 weeks, and needs a score that is comparable across weeks. M1b therefore uses

  candidates   every point-in-time universe name that passes the ban, upper/lower circuit and liquidity-cliff gates that week and
               has a finite 12-1 momentum (no top-N truncation anywhere);
  score        cross-sectional percentile rank (0, 1] of 12-1 momentum among that week's candidates (a market-wide drift in momentum
               moves every name together and leaves the percentile unchanged);
  calibration  every scored candidate, labelled, matured by T, of the last 52 weeks (published or not: N8);
  publication  BH over all m candidates at T, at most CAP published.

Everything else is M1's and is reused from vanta.m1: labels, impact modes, statistics, the GO split and the report format. q = 0.30 is
an engineering target; realised FDR is an empirical number, never a guarantee. Missing is missing: NaN is never zero-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vanta import m1

Q = m1.Q
CAP = m1.CAP
UNGATED_TOP_N = m1.CAP
# Calibration rows are stamped with vanta.m1's MODEL_VERSION / FILTER_VERSION inside m1.run_arms; the gate filters on the same pair, so the
# stamp is consistent by construction (M1b outputs live in their own directory and logs).
K_SOURCE_CLIFF = m1.LIQUIDITY_CLIFF_RANK        # k_s / k_t are the values frozen in M1 (data/processed/m1/k_h{h}_cliff200.json)


LIQUIDITY_CLIFF_RANK = m1.LIQUIDITY_CLIFF_RANK


def percentile_rank(values) -> np.ndarray:
    """One week. pct = average rank / number of finite values, range (0, 1]; ties share the average rank. NaN / inf stays NaN, takes no
    slot in the denominator and is never 0."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = pd.Series(v[ok]).rank(method="average").to_numpy() / ok.sum()
    return out


def weekly_scores(frame: pd.DataFrame, momentum_col: str = "momentum") -> pd.Series:
    """percentile_rank applied separately within each decision_date, aligned to frame.index. Never raw momentum, never pooled over weeks."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, g in frame.groupby("decision_date"):
        out.loc[g.index] = percentile_rank(g[momentum_col].to_numpy(float))
    return out


def eligible_candidates(week: pd.DataFrame, cliff: int = LIQUIDITY_CLIFF_RANK) -> pd.DataFrame:
    """Every row that is not banned, hit no upper or lower circuit, has a finite liquidity rank <= cliff (inclusive) and a finite
    momentum. No top-N truncation; row order preserved. Flags are plain booleans (the driver resolves 'list unavailable' first)."""
    rank = week["liquidity_rank"].astype(float)

    def clear(col):                      # NA / unknown flag = list unavailable = NOT clear (block policy), never an exception or a pass
        return week[col].astype("boolean").eq(False).fillna(False).astype(bool)

    keep = (clear("banned") & clear("hit_upper") & clear("hit_lower")
            & np.isfinite(rank) & (rank <= cliff) & np.isfinite(week["momentum"].astype(float)))
    return week[keep]


def calibration_rows(scored: pd.DataFrame, T, window_weeks: int = 52) -> pd.DataFrame:
    """ALL scored candidates (published or not) with matured_at <= T, decision_date >= T - window_weeks (both inclusive) and a finite y_net."""
    T = pd.Timestamp(T)
    keep = (scored["matured_at"].notna() & (scored["matured_at"] <= T) & (scored["decision_date"] >= T - pd.Timedelta(weeks=window_weeks))
            & np.isfinite(scored["y_net"].astype(float)))
    return scored[keep]


def select_published(frame: pd.DataFrame, cap: int = CAP) -> np.ndarray:
    """True iff in_bh_set and among the `cap` highest scores (descending, stable); aligned to frame."""
    in_bh = frame["in_bh_set"].astype(bool).to_numpy()
    out = np.zeros(len(frame), bool)
    idx = np.flatnonzero(in_bh)
    order = idx[np.argsort(-frame["score"].to_numpy(float)[idx], kind="stable")]
    out[order[:cap]] = True
    return out


def select_candidates(pool: pd.DataFrame, gate_allowed_col: str = "gate_allowed", mom_col: str = "mom") -> pd.DataFrame:
    """Driver-side form of eligible_candidates: the gate verdict comes from gates.override_gates (ban, upper, lower circuit, cliff).
    Every pool row whose gates allow a buy and whose momentum is finite; NO truncation to a top-N. A missing verdict is not allowed."""
    allowed = pool[gate_allowed_col].fillna(False).astype(bool)
    finite = np.isfinite(pool[mom_col].astype(float))
    return pool[allowed & finite].copy()


def add_scores(cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    out["score"] = weekly_scores(out, "mom")
    return out


def top_n_by_score(g: pd.DataFrame, n: int = UNGATED_TOP_N) -> pd.DataFrame:
    """The n highest-score rows of one week, ties in input order. A NaN score is never ranked."""
    ok = g[np.isfinite(g["score"].astype(float))]
    return ok.sort_values("score", ascending=False, kind="stable").head(n)


def run_arms_b(cands: pd.DataFrame, decision_dates, *, q: float = Q, cap: int = CAP, tau_weeks: float = 26, ess_floor: int = 200,
               min_cal_v4: int = 20) -> dict[str, pd.DataFrame]:
    """M1's walk with the M1b arm definitions. The gate arms (gate_age, gate_age_regime) and the v4 reconstruction see ALL candidates
    of the week as the BH universe and ALL matured scored candidates of the last 52 weeks as calibration (m1.run_arms builds the
    calibration store from the full candidate table, published or not). The ungated arm is the `cap` highest-percentile candidates
    of each week, not every candidate. Logs keep one row per decision date x candidate, with `published` marking the final set."""
    logs = m1.run_arms(cands, decision_dates, q=q, cap=cap, tau_weeks=tau_weeks, ess_floor=ess_floor, min_cal_v4=min_cal_v4)
    ung = logs.get("ungated")
    if ung is not None and len(ung):
        ung = ung.copy()
        ung["published"] = False
        for _, g in ung.groupby("decision_date"):
            ung.loc[top_n_by_score(g, cap).index, "published"] = True
        logs["ungated"] = ung
    return logs


# ================================================================== 13b diagnostics (reported, NOT JUDGED)
# None of these feed a GO criterion. They describe how the false-discovery proportion behaves, they do not decide anything.
def weekly_fdp(log: pd.DataFrame) -> pd.Series:
    """Per decision date: losers / filled-with-known-outcome among PUBLISHED trades (loser = y_net <= 0, as in m1.realised_fdr). A week
    with no such trade is NaN, never 0. Indexed by every decision date present in `log`."""
    weeks = pd.DatetimeIndex(sorted(log["decision_date"].unique()))
    pub = log[log["published"].astype(bool) & (log["fill_status"] == "FILLED") & log["y_net"].notna()]
    g = pub.groupby("decision_date")["y_net"].agg(n="size", lose=lambda y: int((y <= 0).sum()))
    out = (g["lose"] / g["n"]).reindex(weeks)
    out.index.name = "decision_date"
    return out


def market_weekly_returns(index_close: pd.Series, decision_dates) -> pd.Series:
    """Index close change from each decision date to the NEXT decision date (the holding week). Exact-date closes only: a missing close at
    either end gives NaN. The last decision date has no next one: NaN."""
    d = pd.DatetimeIndex(sorted(pd.DatetimeIndex(decision_dates).unique()))
    c = index_close.reindex(d).to_numpy(float)
    r = np.full(len(d), np.nan)
    r[:-1] = c[1:] / c[:-1] - 1.0
    r[~np.isfinite(r)] = np.nan
    return pd.Series(r, index=d)


def fdp_vs_market(fdp: pd.Series, market_ret: pd.Series) -> dict:
    """Pearson correlation of the weekly FDP with the market's weekly return over weeks where both are finite, and the mean weekly FDP
    in up weeks (market_ret > 0) and down weeks (<= 0). A week missing either number is excluded, never treated as 0."""
    df = pd.concat([fdp.rename("fdp"), market_ret.rename("mkt")], axis=1, join="inner").astype(float)
    df = df[np.isfinite(df["fdp"]) & np.isfinite(df["mkt"])]
    up, dn = df[df["mkt"] > 0], df[df["mkt"] <= 0]
    corr = float(np.corrcoef(df["fdp"], df["mkt"])[0, 1]) if len(df) >= 3 and df["fdp"].std() > 0 and df["mkt"].std() > 0 else float("nan")
    return dict(corr=corr, n_weeks=int(len(df)), fdr_up=float(up["fdp"].mean()) if len(up) else float("nan"),
                fdr_down=float(dn["fdp"].mean()) if len(dn) else float("nan"), n_up=int(len(up)), n_down=int(len(dn)))


def nifty_relative_y(log: pd.DataFrame, index_close: pd.Series) -> pd.Series:
    """y_net minus the index return over the trade's own window (entry_date to label_end_date, exact-date closes). NaN when either index
    close, either date or y_net is missing."""
    c = index_close.astype(float)
    a = c.reindex(pd.to_datetime(log["entry_date"])).to_numpy()
    b = c.reindex(pd.to_datetime(log["label_end_date"])).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        idx_ret = b / a - 1.0
    return pd.Series(log["y_net"].to_numpy(float) - idx_ret, index=log.index)


def realised_fdr_vs_index(log: pd.DataFrame, index_close: pd.Series) -> dict:
    """m1.realised_fdr with loser = underperformed the index after costs (y_net - index return <= 0). Diagnostic only."""
    return m1.realised_fdr(log.assign(y_net=nifty_relative_y(log, index_close)))
