"""VANTA Milestone M1 (Stage 13M): momentum 12-1, weekly, through NEL labels and the RW-CS gate, on real NSE data.

Two layers.

Pure functions (top of file; unit-tested by tests/invariants/vanta/test_vanta_m1.py): realised FDR, per-year FDR, publication rate,
Newey-West SE, weekly block bootstrap (plain and paired), k selection, GO split, the v4 per-stock gate RECONSTRUCTION, impact floor,
turnover ranks, report text.

Driver (bottom): candidate construction, labels, matured calibration, the gate arms and the summary. Driven by scripts/run_m1.py.

Definitions fixed by the user (logs/agent_team_decisions.md, 2026-10-30):
  loser = y_net <= 0 after costs; published = in the final capped set (<= CAP); realised FDR counts FILLED published trades only
  with the NO_FILL count shown next to it; q = 0.30 is an engineering TARGET and realised FDR is an empirical number, never a
  guarantee (the weighted age x regime gate has no demonstrated FDR control; see vanta/gate.py).
Missing is missing: NaN y_net / rank / posterior is never zero-filled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

Q = 0.30
CAP = 10
REFERENCE_NOTIONAL = 100_000
HORIZONS = (5, 20)
K_S_GRID = (1.0, 1.5, 2.0)
K_T_GRID = (1.5, 2.0, 3.0)
K_SELECT_START, K_SELECT_END = "2010-01-01", "2012-12-31"
OOS_START = pd.Timestamp("2013-01-01")
LIQUIDITY_CLIFF_RANK = 200
IMPACT_FLOOR_TOP200, IMPACT_FLOOR_REST = 0.0005, 0.0020      # per side, the user's assumption (decision 5)
MODEL_VERSION = "mom12-1@m1"
FILTER_VERSION = "m1-eligibility@v1"


# ================================================================== pure functions
def realised_fdr(log: pd.DataFrame) -> dict:
    """Realised FDR over FILLED published trades. Loser = y_net <= 0. NO_FILL published signals are counted next to it, and
    published FILLED rows whose outcome is not yet known (NaN y_net) are counted in n_unmatured, never as winners or losers."""
    pub = log[log["published"].astype(bool)]
    y = pub["y_net"].astype(float)
    filled = pub["fill_status"] == "FILLED"
    ok = filled & y.notna()
    n_filled = int(ok.sum())
    n_losers = int((ok & (y <= 0)).sum())
    status = pub["fill_status"].astype(str)
    return dict(fdr=n_losers / n_filled if n_filled else float("nan"), n_published=int(len(pub)), n_filled=n_filled,
                n_no_fill=int(status.str.startswith("NO_FILL").sum()), n_no_data=int((status == "NO_DATA").sum()),
                n_losers=n_losers, n_unmatured=int((filled & y.isna()).sum()))


def fdr_by_year(log: pd.DataFrame) -> dict:
    out = {}
    for yr, g in log.groupby(log["decision_date"].dt.year):
        s = realised_fdr(g)
        if s["n_filled"]:
            out[int(yr)] = s["fdr"]
    return out


def publication_rate(log: pd.DataFrame, decision_dates=None) -> float:
    """Share of WEEKS (decision dates) with at least one published signal; `decision_dates` is the full week list."""
    weeks = pd.DatetimeIndex(decision_dates) if decision_dates is not None else pd.DatetimeIndex(log["decision_date"].unique())
    if len(weeks) == 0:
        return float("nan")
    pub = pd.DatetimeIndex(log[log["published"].astype(bool)]["decision_date"].unique())
    return len(pub.intersection(weeks)) / len(weeks)


def newey_west_se(x, lags: int) -> float:
    """Bartlett-weighted Newey-West standard error of the mean."""
    x = np.asarray(x, float)
    n = len(x)
    d = x - x.mean()
    var = float(d @ d) / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * float(d[l:] @ d[:-l]) / n
    var /= n
    return float(np.sqrt(var)) if var >= 0 else float("nan")


def _week_groups(y, week_ids):
    y = np.asarray(y, float)
    w = np.asarray(week_ids)
    keys = list(dict.fromkeys(w.tolist()))
    return keys, {k: y[w == k] for k in keys}


def _block_pick(rng, n: int, block_len: int) -> np.ndarray:
    """Indices of n weeks. block_len 1: plain iid week draw. Otherwise moving blocks of `block_len` consecutive weeks (start
    uniform over all valid starts), concatenated and cut to n."""
    if block_len <= 1 or n <= block_len:
        return rng.integers(0, n, n)
    nb = -(-n // block_len)
    starts = rng.integers(0, n - block_len + 1, nb)
    return (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]


def weekly_block_bootstrap(y, week_ids, n_boot: int, seed: int, block_len: int = 1) -> np.ndarray:
    """Resample whole weeks with replacement (block_len > 1: moving blocks of consecutive weeks, in order of first appearance, so
    pass data in chronological order). Each draw is the pooled mean per signal."""
    rng = np.random.default_rng(seed)
    keys, g = _week_groups(y, week_ids)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = _block_pick(rng, len(keys), block_len)
        out[b] = np.concatenate([g[keys[i]] for i in pick]).mean()
    return out


def weekly_block_bootstrap_diff(y_a, week_a, y_b, week_b, n_boot: int, seed: int, block_len: int = 1) -> np.ndarray:
    """Paired: both arms see the SAME resampled weeks (union of weeks; chronological when block_len > 1). Draw = mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    ka, ga = _week_groups(y_a, week_a)
    kb, gb = _week_groups(y_b, week_b)
    union = list(dict.fromkeys(ka + kb))
    if block_len > 1:
        union = sorted(union)
    empty = np.array([])
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = _block_pick(rng, len(union), block_len)
        ma = np.concatenate([ga.get(union[i], empty) for i in pick])
        mb = np.concatenate([gb.get(union[i], empty) for i in pick])
        out[b] = (ma.mean() if len(ma) else np.nan) - (mb.mean() if len(mb) else np.nan)
    return out


def bootstrap_ci(draws, level: float = 0.95):
    a = (1 - level) / 2 * 100
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, a)), float(np.percentile(d, 100 - a))


def choose_k(trades: pd.DataFrame, start=K_SELECT_START, end=K_SELECT_END):
    """(k_s, k_t) with the highest mean y_net per signal on [start, end] (inclusive). NaN is ignored, never zero-filled.
    Ties go to the first combination in grid order (K_S_GRID major, K_T_GRID minor)."""
    d = trades["decision_date"]
    t = trades[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    best, best_v = None, -np.inf
    for ks in K_S_GRID:
        for kt in K_T_GRID:
            v = t.loc[(t["k_s"] == ks) & (t["k_t"] == kt), "y_net"].mean()      # NaN skipped by mean
            if np.isfinite(v) and v > best_v:
                best, best_v = (ks, kt), v
    if best is None:
        raise ValueError("no usable trade in the k-selection window")
    return best


def _segment_metrics(log: pd.DataFrame, decision_dates=None) -> dict:
    s = realised_fdr(log)
    return dict(fdr=s["fdr"], n_published=s["n_published"], n_filled=s["n_filled"], n_losers=s["n_losers"],
                n_no_fill=s["n_no_fill"], n_no_data=s["n_no_data"], n_unmatured=s["n_unmatured"],
                publication_rate=publication_rate(log, decision_dates), fdr_by_year=fdr_by_year(log))


def go_report(log: pd.DataFrame, oos_start=OOS_START, decision_dates=None) -> dict:
    """GO criteria are evaluated on decision dates >= oos_start only; earlier rows are reported separately as in-sample."""
    oos_start = pd.Timestamp(oos_start)
    dd = pd.DatetimeIndex(decision_dates) if decision_dates is not None else None
    ins, oos = log["decision_date"] < oos_start, log["decision_date"] >= oos_start
    return dict(in_sample=_segment_metrics(log[ins], None if dd is None else dd[dd < oos_start]),
                oos=_segment_metrics(log[oos], None if dd is None else dd[dd >= oos_start]))


def v4_gate_reconstruction(cands: pd.DataFrame, cal: pd.DataFrame, T, *, model_version, candidate_filter_version,
                           q: float = Q, cap: int = CAP, window_weeks: int = 52, min_cal: int = 3) -> pd.DataFrame:
    """RECONSTRUCTION of the per-stock conformal lower-bound gate v6 replaced (user decision M1.3), not the original code.

    Fit y_net = a + b*score by OLS on the same 52-week MATURED calibration window as the RW-CS gate. Lower bound = fitted minus
    the (1-q) quantile of the calibration shortfalls (fitted - y). Publish iff bound > 0 (strict), then the `cap` highest scores.
    With fewer than `min_cal` usable calibration rows nothing is published (bound NaN)."""
    T = pd.Timestamp(T)
    keep = ((cal["model_version"] == model_version) & (cal["candidate_filter_version"] == candidate_filter_version)
            & cal["is_oos"].astype(bool) & (cal["decision_date"] >= T - pd.Timedelta(weeks=window_weeks))
            & cal["matured_at"].notna() & (cal["matured_at"] <= T))
    c = cal[keep & cal["y_net"].notna() & np.isfinite(cal["score"].astype(float))]
    scores = cands["score"].to_numpy(float)
    out = pd.DataFrame({"symbol": cands["symbol"].to_numpy(), "score": scores})
    if len(c) < min_cal:
        out["fitted"], out["bound"], out["published"] = np.nan, np.nan, False
        return out
    x, y = c["score"].to_numpy(float), c["y_net"].to_numpy(float)
    if np.ptp(x) > 0:
        slope, icpt = np.polyfit(x, y, 1)
    else:
        slope, icpt = 0.0, float(y.mean())
    short = (slope * x + icpt) - y
    cut = np.quantile(short, 1 - q)
    out["fitted"] = slope * scores + icpt
    out["bound"] = out["fitted"] - cut
    ok = (out["bound"] > 0).to_numpy()
    out["published"] = False
    top = out[ok].sort_values("score", ascending=False, kind="stable").index[:cap]
    out.loc[top, "published"] = True
    return out


def impact_floor_per_side(turnover_rank) -> float:
    """0.05% per side for trailing-turnover ranks 1-200, 0.20% for 201-700. Outside 700 or unknown rank: NaN, never a default."""
    if turnover_rank is None or not np.isfinite(turnover_rank):
        return float("nan")
    if turnover_rank < 1:
        raise ValueError("rank must be >= 1")
    if turnover_rank <= 200:
        return IMPACT_FLOOR_TOP200
    if turnover_rank <= 700:
        return IMPACT_FLOOR_REST
    return float("nan")


def turnover_ranks(trailing_turnover) -> pd.Series:
    """Rank 1 = highest turnover; ties share the lowest rank; missing stays NaN."""
    return pd.Series(trailing_turnover, dtype=float).rank(ascending=False, method="min", na_option="keep")


def fdr_report_text(stats: dict, q: float) -> str:
    txt = (f"Empirical realised FDR {stats['fdr']:.3f} on {stats['n_filled']} filled published trades "
           f"(losers {stats['n_losers']}), against a target q of {q:.2f}, which is an engineering target and not a guarantee. "
           f"Published signals that were NO_FILL: {stats['n_no_fill']}.")
    if "n_no_data" in stats:
        txt += f" Published signals with NO_DATA (no label could be built, reported separately): {stats['n_no_data']}."
    return txt


# ================================================================== driver
# Everything below touches real data (Parquet under config paths) and is exercised by tests/invariants/vanta/test_vanta_m1_driver.py
# on synthetic inputs and by scripts/run_m1.py on real ones.

def matured_at(label_end_date: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Outcome becomes known at the close of the label end date and is usable from the next session (one-session publication lag).
    NaT stays NaT."""
    sessions = pd.DatetimeIndex(sessions)
    end = pd.to_datetime(label_end_date)
    pos = sessions.searchsorted(end.to_numpy(dtype="datetime64[ns]"), side="right")
    vals = np.where(end.isna().to_numpy() | (pos >= len(sessions)), np.datetime64("NaT", "ns"),
                    sessions.to_numpy()[np.minimum(pos, len(sessions) - 1)])
    return pd.Series(pd.to_datetime(vals), index=label_end_date.index)


def momentum_12_1(close: pd.Series, dates: pd.DatetimeIndex, lookback: int = 252, skip: int = 21) -> pd.Series:
    """close[t-skip] / close[t-lookback] - 1 by the symbol's own bar rows at each date in `dates`. NaN unless both ends exist and the
    calendar span is plausible for 231 sessions (a long suspension makes the row count misleading)."""
    close = close[close > 0]
    idx = close.index
    pos = idx.get_indexer(dates)
    out = np.full(len(dates), np.nan)
    ok = (pos >= lookback)
    p = pos[ok]
    span = (idx[p - skip] - idx[p - lookback]).days.to_numpy()
    a, b = close.to_numpy()[p - skip], close.to_numpy()[p - lookback]
    val = a / b - 1.0
    val[(span < 280) | (span > 400)] = np.nan
    out[np.where(ok)[0]] = val
    return pd.Series(out, index=dates)


def asof_values(frame: pd.DataFrame, dates, max_days: int = 5) -> pd.DataFrame:
    """Value of `frame` (DatetimeIndex) at each date: the row itself if the date is in the index (a NaN there stays NaN, never filled
    from an earlier row); otherwise the last earlier row if it is at most `max_days` calendar days old; otherwise NaN."""
    dates = pd.DatetimeIndex(dates)
    idx = frame.index
    pos = idx.searchsorted(dates, side="right") - 1
    ok = pos >= 0
    posc = np.where(ok, pos, 0)
    age = (dates - idx[posc]).days
    good = ok & (age <= max_days)
    out = frame.iloc[posc].copy()
    out.index = dates
    out[~good] = np.nan
    return out


def apply_impact(labels: pd.DataFrame, rank_by_row: pd.Series, mode: str) -> pd.DataFrame:
    """Return labels with `y_net` for the impact mode. 'zero' leaves the NEL y_net (cost_impact 0). 'floor' subtracts the user's
    pessimistic per-side floor on both sides (round trip = 2 x per-side), by trailing-turnover rank; an unknown rank makes y_net NaN
    for a FILLED trade (missing, not free)."""
    out = labels.copy()
    if mode == "zero":
        out["impact_rt"] = 0.0
        return out
    if mode != "floor":
        raise ValueError(f"impact mode must be 'zero' or 'floor', got {mode!r}")
    if (out.loc[out["fill_status"] == "FILLED", "cost_impact"].fillna(0.0) != 0.0).any():
        raise ValueError("labels already carry an impact cost; the floor would double count")
    rt = 2.0 * rank_by_row.map(impact_floor_per_side)
    out["impact_rt"] = rt
    out["y_net"] = out["y_net"] - rt
    return out


@dataclass
class ArmResult:
    name: str
    log: pd.DataFrame


def _cal_frame(cands: pd.DataFrame) -> pd.DataFrame:
    """Calibration store rows from the labelled candidate table (all in-sample-free: the momentum rule has no fitted model)."""
    cal = pd.DataFrame({
        "decision_date": cands["decision_date"], "symbol": cands["symbol"], "horizon": cands["horizon"], "score": cands["score"],
        "model_version": MODEL_VERSION, "fold_id": "m1", "is_oos": True,
        "regime_posterior": cands["regime_posterior"], "hmm_version": cands["hmm_version"],
        "y_net": cands["y_net"], "matured_at": cands["matured_at"], "candidate_filter_version": FILTER_VERSION,
    })
    return cal.reset_index(drop=True)


def run_arms(cands: pd.DataFrame, decision_dates, *, q: float = Q, cap: int = CAP, tau_weeks: float = 26,
             ess_floor: int = 200, min_cal_v4: int = 20) -> dict[str, pd.DataFrame]:
    """Walk weekly decision dates; at each T the gate sees only calibration rows with matured_at <= T (vanta.gate enforces it).

    cands columns: decision_date, symbol, horizon, score, y_net (impact-mode applied), fill_status, matured_at,
    regime_posterior (list of 3, NaN when unavailable), hmm_version, pi_T via regime_posterior itself.
    Arms: ungated (all candidates published), v4 (reconstruction), gate_age (age-only weights), gate_age_regime (age x regime).
    Returns {arm: log} with one row per decision date x candidate."""
    from vanta.gate import run_gate

    cal = _cal_frame(cands)
    cal_age = cal.assign(regime_posterior=[[1 / 3, 1 / 3, 1 / 3]] * len(cal))      # equal posteriors: distance 0, weights = age only
    flat = [1 / 3] * 3
    logs: dict[str, list] = {"ungated": [], "v4": [], "gate_age": [], "gate_age_regime": []}
    by_date = {d: g.reset_index(drop=True) for d, g in cands.groupby("decision_date")}
    for T in pd.DatetimeIndex(decision_dates):
        g = by_date.get(T)
        if g is None or g.empty:
            continue
        base = g[["decision_date", "symbol", "horizon", "score", "fill_status", "y_net"]]
        u = base.copy()
        u["published"] = True
        logs["ungated"].append(u)
        v = v4_gate_reconstruction(g[["symbol", "score"]], cal, T, model_version=MODEL_VERSION,
                                   candidate_filter_version=FILTER_VERSION, q=q, cap=cap, min_cal=min_cal_v4)
        logs["v4"].append(base.assign(published=v["published"].to_numpy(), bound=v["bound"].to_numpy()))
        gk = dict(model_version=MODEL_VERSION, candidate_filter_version=FILTER_VERSION, tau_weeks=tau_weeks, q=q, cap=cap,
                  ess_floor=ess_floor)
        ra = run_gate(g[["symbol", "score"]], cal_age, T, flat, **gk)
        logs["gate_age"].append(base.assign(published=ra["published"].to_numpy(), p_value=ra["p_value"].to_numpy(),
                                            n_calibration=ra["n_calibration"].to_numpy(), effective_n=ra["effective_n"].to_numpy(),
                                            age_only_fallback=ra["age_only_fallback"].to_numpy()))
        pi_T = g["regime_posterior"].iloc[0]
        pi_T = np.asarray(pi_T if pi_T is not None else [np.nan] * 3, float)
        rr = run_gate(g[["symbol", "score"]], cal, T, pi_T, **gk)
        pub = rr["published"].to_numpy()
        logs["gate_age_regime"].append(base.assign(published=pub, p_value=rr["p_value"].to_numpy(),
                                                   n_calibration=rr["n_calibration"].to_numpy(),
                                                   effective_n=rr["effective_n"].to_numpy(),
                                                   age_only_fallback=rr["age_only_fallback"].to_numpy()))
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in logs.items()}


def weekly_ids(log: pd.DataFrame) -> np.ndarray:
    return log["decision_date"].dt.strftime("%Y-%m-%d").to_numpy()


def summarise_arm(log: pd.DataFrame, ungated: pd.DataFrame, decision_dates, horizon: int, *, q: float = Q,
                  oos_start=OOS_START, n_boot: int = 2000, seed: int = 11) -> dict:
    """GO-relevant numbers for one arm, on 2013+ (oos) and 2010-2012 (in_sample). Net return per signal = mean y_net of FILLED
    published trades with a known outcome. Newey-West lag = ceil(h/5) weekly rows (trades overlap that many weekly decisions)."""
    rep = go_report(log, oos_start, decision_dates)
    lag = int(math.ceil(horizon / 5))
    out = {"report": rep, "nw_lag_weeks": lag}
    for seg, mask_fn in (("oos", lambda d: d >= pd.Timestamp(oos_start)), ("in_sample", lambda d: d < pd.Timestamp(oos_start))):
        def pubfilled(l):
            m = l["published"].astype(bool) & (l["fill_status"] == "FILLED") & l["y_net"].notna() & mask_fn(l["decision_date"])
            return l[m]
        a, b = pubfilled(log), pubfilled(ungated)
        stats = dict(n_signals=int(len(a)), mean_y_net=float(a["y_net"].mean()) if len(a) else float("nan"))
        if len(a) > 1:
            wk = a.groupby("decision_date")["y_net"].mean()           # equal weight per week for the NW series
            stats["nw_se_weekly_mean"] = newey_west_se(wk.to_numpy(), lag)
            stats["weekly_mean_of_weeks"] = float(wk.mean())
            stats["nw_ci95"] = (stats["weekly_mean_of_weeks"] - 1.96 * stats["nw_se_weekly_mean"],
                                stats["weekly_mean_of_weeks"] + 1.96 * stats["nw_se_weekly_mean"])
            stats["boot_ci95"] = bootstrap_ci(weekly_block_bootstrap(a["y_net"], weekly_ids(a), n_boot, seed))
            stats["boot_ci95_block"] = bootstrap_ci(weekly_block_bootstrap(a["y_net"], weekly_ids(a), n_boot, seed, lag))
            if len(b) > 1:
                d = weekly_block_bootstrap_diff(a["y_net"], weekly_ids(a), b["y_net"], weekly_ids(b), n_boot, seed)
                stats["diff_vs_ungated"] = float(a["y_net"].mean() - b["y_net"].mean())
                stats["diff_ci95"] = bootstrap_ci(d)
                stats["diff_ci95_block"] = bootstrap_ci(
                    weekly_block_bootstrap_diff(a["y_net"], weekly_ids(a), b["y_net"], weekly_ids(b), n_boot, seed, lag))
        out[seg] = stats
    o = rep["oos"]
    out["criteria_oos"] = dict(
        c2_fdr_le_q_plus_0_05=bool(o["fdr"] <= q + 0.05) if np.isfinite(o["fdr"]) else None,
        c3_no_year_above_0_5=bool(all(v <= 0.5 for v in o["fdr_by_year"].values())) if o["fdr_by_year"] else None,
        c4_publishes_ge_20pct_weeks=bool(o["publication_rate"] >= 0.20) if np.isfinite(o["publication_rate"]) else None,
        c5_mean_ge_ungated=(bool(out["oos"]["diff_vs_ungated"] >= 0) if "diff_vs_ungated" in out["oos"] else None),
        c5_diff_ci_excludes_zero_below=(bool(out["oos"]["diff_ci95_block"][1] < 0) if "diff_ci95_block" in out["oos"] else None),
    )
    return out
