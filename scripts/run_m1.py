"""
Milestone M1 driver (Stage 13M): momentum 12-1 weekly -> eligibility gates -> NEL labels -> 52-week matured calibration -> gates.

Stages (each writes under config.PROCESSED_DATA_DIR / "m1" and is skipped on re-run if its output exists; --force rebuilds):
  1 candidates   weekly top-N momentum names among PIT-universe members that pass ban / circuit / liquidity-cliff gates
  2 k selection  9 (k_s, k_t) combos on 2010-2012 by mean y_net per signal of the ungated top 10, frozen (per horizon)
  3 labels       NEL labels at the frozen k for every candidate (per-symbol chunks, resumable)
  4 regime       filtered HMM posterior per fold (annual refit, embargo = horizon sessions), only for the age x regime arm
  5 arms         ungated / v4 reconstruction / gate_age / gate_age_regime, then summary + report for one impact mode

Usage:
    python -m scripts.run_m1 --horizon 5 --impact floor
    python -m scripts.run_m1 --horizon 5 --impact zero --start 2013-01-01 --end 2013-12-31   # small check

q is an engineering target for the false-discovery proportion. Realised FDR is reported as an empirical number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from vanta import m1
from vanta.labels import load_terminal_values, simulate_trade, terminal_kwargs

M1_DIR = config.PROCESSED_DATA_DIR / "m1"
POOL = 30                    # momentum names examined per week before the gates; the top CAP survivors are the candidates
N_CANDIDATES = 10
WINDOW_START = "2010-01-04"  # first price-band hit list on disk
FORCE = False                # set from --force: every stage rebuilds and overwrites its own files (nothing is deleted)


def _code_version() -> str:
    """Hash of the label simulator and the cost model: a change to either gives new cache keys, never stale labels."""
    import costs.model
    import vanta.labels
    h = hashlib.sha1()
    for mod in (vanta.labels, costs.model):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:10]


def _hash(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def _logger():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / "run_m1.log", rotation="20 MB")


def weekly_dates(sessions: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    from scripts.report_nel_coverage import weekly_decision_dates
    d = weekly_decision_dates(sessions)
    return d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def universe_ranks(years) -> pd.DataFrame:
    """(year, symbol) -> Stage 5 RANK (trailing-turnover rank as of that year's snapshot)."""
    frames = []
    for y in years:
        f = config.UNIVERSE_DIR / f"{y}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"universe snapshot {f} missing")
        frames.append(pd.read_parquet(f, columns=["SYMBOL", "RANK"]).assign(year=y))
    return pd.concat(frames, ignore_index=True).rename(columns={"SYMBOL": "symbol", "RANK": "univ_rank"})


# ------------------------------------------------------------------ stage 0
def liquidity_rank_frame(start) -> pd.DataFrame:
    """Point-in-time trailing-turnover rank (gates.liquidity.trailing_turnover_rank) for every adjusted-panel row on or after
    `start`. Used for BOTH the liquidity-cliff gate and the impact-floor tiers (top 200 vs 201-700). Cached."""
    from gates.liquidity import trailing_turnover_rank
    f = M1_DIR / f"liquidity_rank_{pd.Timestamp(start).date()}.parquet"
    if f.exists() and not FORCE:
        return pd.read_parquet(f)
    frames = []
    for p in sorted(config.ADJUSTED_PRICES_DIR.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["SYMBOL", "DATE", "ADJ_CLOSE", "ADJ_VOLUME"]).drop_duplicates(["SYMBOL", "DATE"])
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    logger.info(f"liquidity ranks over {panel['SYMBOL'].nunique()} symbols, {len(panel):,} rows")
    out = trailing_turnover_rank(panel, start=start)
    out.to_parquet(f)
    return out


# ------------------------------------------------------------------ stage 1
def build_candidates(dates: pd.DatetimeIndex, cliff: int) -> pd.DataFrame:
    from gates.io import load_gate_inputs
    from gates.override_gates import BUY_SIGNAL_GATES, GateConfig, evaluate_gates

    liq = liquidity_rank_frame(dates.min())
    ranks = universe_ranks(sorted(set(dates.year)))
    symbols = sorted(ranks["symbol"].unique())
    logger.info(f"momentum for {len(symbols)} universe symbols, {len(dates)} weekly dates")
    rows = []
    for i, s in enumerate(symbols):
        f = config.ADJUSTED_PRICES_DIR / f"{s}.parquet"
        if not f.exists():
            continue
        px = pd.read_parquet(f, columns=["DATE", "ADJ_CLOSE"]).drop_duplicates("DATE").sort_values("DATE")
        close = pd.Series(px["ADJ_CLOSE"].to_numpy(float), index=pd.DatetimeIndex(px["DATE"]))
        d = dates.intersection(close.index)
        if len(d):
            sc = m1.momentum_12_1(close, d)
            rows.append(pd.DataFrame({"decision_date": d, "symbol": s, "score": sc.to_numpy()}))
        if (i + 1) % 200 == 0:
            logger.info(f"  momentum {i + 1}/{len(symbols)}")
    mom = pd.concat(rows, ignore_index=True)
    mom["year"] = mom["decision_date"].dt.year
    mom = mom.merge(ranks, on=["year", "symbol"], how="inner")          # point-in-time membership of the decision year
    mom = mom.merge(liq.rename(columns={"date": "decision_date"})[["symbol", "decision_date", "liquidity_rank"]],
                    on=["symbol", "decision_date"], how="left")
    mom = mom[np.isfinite(mom["score"])]
    if cliff:                                                           # same cut the cliff gate applies; keeps the pool useful
        mom = mom[mom["liquidity_rank"] <= cliff]
    mom["mrank"] = mom.groupby("decision_date")["score"].rank(ascending=False, method="first")
    pool = mom[mom["mrank"] <= POOL].copy()
    logger.info(f"pool rows {len(pool):,}; loading gate inputs for {pool['symbol'].nunique()} symbols")
    g = load_gate_inputs(sorted(pool["symbol"].unique()), start=dates.min(), end=dates.max(), dates=dates, liquidity_ranks=liq)
    g = g.rename(columns={"date": "decision_date"})
    enabled = BUY_SIGNAL_GATES if cliff else tuple(x for x in BUY_SIGNAL_GATES if x != "liquidity_cliff")
    verdict = evaluate_gates(g, GateConfig(enabled=enabled, liquidity_top_n=cliff or 700, unavailable_policy="block"))
    g = pd.concat([g[["symbol", "decision_date"]], verdict[["gate_status", "gate_allowed", "gate_reasons"]]], axis=1)
    pool = pool.merge(g, on=["symbol", "decision_date"], how="left")
    pool["gate_allowed"] = pool["gate_allowed"].fillna(False).astype(bool)      # no gate row at all = not allowed, never a pass
    n_pool = pool.groupby("decision_date").size()
    ok = pool[pool["gate_allowed"]].sort_values(["decision_date", "mrank"])
    cands = ok.groupby("decision_date").head(N_CANDIDATES).reset_index(drop=True)
    stats = pool.groupby(pool["decision_date"].dt.year)["gate_status"].value_counts().unstack(fill_value=0)
    logger.info("gate status of pooled names by year:\n" + stats.to_string())
    logger.info(f"candidates {len(cands):,} over {cands['decision_date'].nunique()} of {len(dates)} weeks "
                f"(pool weeks {len(n_pool)})")
    return cands[["decision_date", "symbol", "score", "mrank", "univ_rank", "liquidity_rank", "gate_status"]]


# ------------------------------------------------------------------ stage 2/3
def build_labels(cands: pd.DataFrame, horizon: int, k_s: float, k_t: float, cache_dir: Path, sessions, terminal) -> pd.DataFrame:
    """NEL label per candidate; per-symbol chunk files make an interrupted run resume where it stopped.

    A chunk's file name carries a hash of everything that determines its rows: symbol, the requested decision dates, horizon, k,
    the symbol's terminal value and the label/cost code version. Any change gives a new file (the old one is left alone), so a chunk
    can never be reused for a different request. The output holds exactly the requested (symbol, decision_date) pairs."""
    from scripts.report_nel_coverage import load_symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    code = _code_version()
    out = []
    for symbol, g in cands.groupby("symbol"):
        dates = sorted(pd.Timestamp(d) for d in g["decision_date"])
        tk = terminal_kwargs(symbol, terminal)
        key = _hash(symbol, [d.date().isoformat() for d in dates], horizon, k_s, k_t, sorted((k, str(v)) for k, v in tk.items()), code)
        chunk = cache_dir / f"{symbol}_{key}.parquet"
        if chunk.exists() and not FORCE:
            out.append(pd.read_parquet(chunk))
            continue
        loaded = load_symbol(symbol)
        rows = []
        for t in dates:
            if loaded is None:
                rows.append(dict(symbol=symbol, decision_date=t, fill_status="NO_DATA"))
                continue
            bars, atr, hits, cov = loaded
            a = atr.get(t, np.nan)
            if t not in bars.index or not np.isfinite(a):
                rows.append(dict(symbol=symbol, decision_date=t, fill_status="NO_DATA"))
                continue
            rows.append(simulate_trade(symbol, bars, t, horizon, k_s, k_t, float(a), hits, cov, sessions=sessions, **tk))
        df = pd.DataFrame(rows)
        df.to_parquet(chunk)
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    keep = ["symbol", "decision_date", "fill_status", "y_net", "gross_return", "cost_impact", "label_end_date", "exit_reason",
            "lock_source", "entry_date"]
    for c in keep:
        if c not in res.columns:
            res[c] = np.nan
    res["decision_date"] = pd.to_datetime(res["decision_date"])
    want = cands[["symbol", "decision_date"]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"])).drop_duplicates()
    res = res.merge(want, on=["symbol", "decision_date"], how="inner")             # never rows outside the request
    if len(res) != len(want):
        raise RuntimeError(f"label build returned {len(res)} rows for {len(want)} requested (symbol, date) pairs")
    return res[keep]


def select_k(cands, horizon, cliff, sessions, terminal) -> tuple:
    sel = cands[(cands["decision_date"] >= m1.K_SELECT_START) & (cands["decision_date"] <= m1.K_SELECT_END)]
    code = _code_version()
    ck = _hash(horizon, cliff, [(r.symbol, pd.Timestamp(r.decision_date).date().isoformat()) for r in sel.itertuples()], code)
    f = M1_DIR / f"k_h{horizon}_cliff{cliff}.json"
    if f.exists() and not FORCE:
        k = json.loads(f.read_text())
        if k.get("key") == ck:
            return k["k_s"], k["k_t"]
        logger.warning(f"{f.name} was built for a different candidate set / code version: recomputing")
    parts = []
    for ks in m1.K_S_GRID:
        for kt in m1.K_T_GRID:
            lab = build_labels(sel, horizon, ks, kt, M1_DIR / "labels_sel" / f"h{horizon}_cliff{cliff}_ks{ks}_kt{kt}", sessions, terminal)
            parts.append(lab.assign(k_s=ks, k_t=kt))
    allk = pd.concat(parts, ignore_index=True)
    table = allk.groupby(["k_s", "k_t"])["y_net"].agg(["mean", "count"])
    logger.info(f"h={horizon} k selection on {m1.K_SELECT_START}..{m1.K_SELECT_END} (mean y_net per signal, impact 0):\n{table}")
    k_s, k_t = m1.choose_k(allk)
    f.write_text(json.dumps({"k_s": k_s, "k_t": k_t, "key": ck, "table": table.reset_index().to_dict("records")}, default=float))
    return k_s, k_t


# ------------------------------------------------------------------ stage 4
def regime_posterior(horizon: int, first_year: int, last_year: int, end) -> pd.DataFrame:
    """Filtered posterior p0..p2 by date, from annual folds: fold Y fits on rows <= fold_train_end(Y-01-01, embargo=h sessions)
    and filters forward through the whole feature history (causal). Dates before the first fold are NaN, so the age x regime
    arm falls back to age-only weights there (visible in age_only_fallback)."""
    from vanta import regime as R
    f = M1_DIR / f"regime_h{horizon}_{pd.Timestamp(end).date()}.parquet"
    if f.exists() and not FORCE:
        return pd.read_parquet(f)
    feats = R.build_regime_features(end=end if end >= pd.Timestamp('2015-01-01') else None)
    out = pd.DataFrame(np.nan, index=feats.index, columns=["p0", "p1", "p2"])
    out["hmm_version"] = None
    for y in range(max(first_year, feats.index[0].year + 1), min(last_year, feats.index[-1].year) + 1):
        try:
            te = R.fold_train_end(feats.index, pd.Timestamp(f"{y}-01-01"), horizon)
            params = R.fit_hmm(feats, te)
        except ValueError as e:
            logger.warning(f"HMM fold {y} not fitted: {e}")
            continue
        post = R.filtered_posterior_frame(params, feats)
        mask = (post.index >= pd.Timestamp(f"{y}-01-01")) & (post.index <= pd.Timestamp(f"{y}-12-31"))
        out.loc[mask, ["p0", "p1", "p2"]] = post.loc[mask].to_numpy()
        out.loc[mask, "hmm_version"] = f"hmm@{y}:{te.date()}:5restarts"
        logger.info(f"HMM fold {y}: train_end {te.date()}, n_train {params.n_train}, loglik {params.train_loglik:.1f}")
    out.index.name = "date"
    out.to_parquet(f)
    return out


def attach_regime(cands: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Posterior as of each decision date: the row for that date if the posterior has one (NaN stays NaN), else the last earlier
    row if at most 5 calendar days old (m1.asof_values). Never fills across a row the posterior itself marks missing."""
    dts = pd.DatetimeIndex(cands["decision_date"])
    if post.empty:
        sub = pd.DataFrame(np.nan, index=dts, columns=["p0", "p1", "p2"])
        hv = pd.Series([np.nan] * len(dts))
    else:
        sub = m1.asof_values(post[["p0", "p1", "p2"]], dts)
        hv = m1.asof_values(post[["hmm_version"]], dts)["hmm_version"]
    cands = cands.copy()
    cands["regime_posterior"] = [list(r) for r in sub.to_numpy(float)]
    cands["hmm_version"] = hv.to_numpy()
    return cands


# ------------------------------------------------------------------ report
def report_text(horizon, impact, k, summaries, extra) -> str:
    lines = [f"# M1 report: h={horizon}, impact={impact}, k_s={k[0]}, k_t={k[1]} (frozen on 2010-2012)", "",
             "q = 0.30 is an engineering target. Realised FDR below is an empirical number, not a guarantee. The weighted age x regime "
             "gate has no demonstrated FDR control.", "",
             f"Newey-West lag = ceil(h/5) = {summaries['ungated']['nw_lag_weeks']} weekly rows. v4 gate is a RECONSTRUCTION. "
             "Impact floor (impact=floor) is the user's assumption, 0.05%/0.20% per side, by the decision-date (point-in-time) trailing-turnover "
             "rank from gates.liquidity.", ""]
    lines += extra
    for arm, s in summaries.items():
        for seg in ("oos", "in_sample"):
            r = s["report"][seg]
            lines += [f"## {arm} / {seg}", m1.fdr_report_text(r, m1.Q),
                      f"- publication rate (weeks with >= 1 published): {r['publication_rate']:.3f}; published {r['n_published']}, "
                      f"filled {r['n_filled']}, NO_FILL {r['n_no_fill']}, NO_DATA {r['n_no_data']} (no label could be built; not NO_FILL), "
                      f"unmatured {r['n_unmatured']}",
                      f"- FDR by year: " + ", ".join(f"{y}: {v:.2f}" for y, v in sorted(r["fdr_by_year"].items())),
                      f"- net return per signal: {json.dumps(s[seg], default=float)}", ""]
            st = s[seg]
            if "nw_se_weekly_mean" in st:
                lines.insert(len(lines) - 1, (
                    f"- headline mean {st['mean_y_net']:.5f} per signal; Newey-West SE of the weekly-mean series {st['nw_se_weekly_mean']:.5f} "
                    f"(lag {s['nw_lag_weeks']}), NW 95% CI [{st['nw_ci95'][0]:.5f}, {st['nw_ci95'][1]:.5f}]; weekly block bootstrap 95% CI "
                    f"1-week [{st['boot_ci95'][0]:.5f}, {st['boot_ci95'][1]:.5f}], {s['nw_lag_weeks']}-week moving block "
                    f"[{st['boot_ci95_block'][0]:.5f}, {st['boot_ci95_block'][1]:.5f}]"))
        lines += [f"- criteria (oos): {s['criteria_oos']}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizon", type=int, required=True, choices=m1.HORIZONS)
    ap.add_argument("--impact", choices=["zero", "floor"], required=True)
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=None, help="last decision date (default: last session on disk)")
    ap.add_argument("--liquidity-cliff", type=int, default=m1.LIQUIDITY_CLIFF_RANK, help="max trailing-turnover liquidity rank (gates.liquidity); 0 = cliff off")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="rebuild every stage")
    ap.add_argument("--tag", default="", help="suffix for output dir (small checks use their own, never the full-run files)")
    a = ap.parse_args()
    _logger()
    global M1_DIR, FORCE
    FORCE = a.force
    M1_DIR = config.PROCESSED_DATA_DIR / ("m1" + (f"_{a.tag}" if a.tag else ""))
    M1_DIR.mkdir(parents=True, exist_ok=True)

    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    end = pd.Timestamp(a.end) if a.end else sessions[-1]
    dates = weekly_dates(sessions, a.start, end)
    terminal = load_terminal_values()

    cf = M1_DIR / f"candidates_cliff{a.liquidity_cliff}_{dates.min().date()}_{dates.max().date()}.parquet"
    if cf.exists() and not a.force:
        cands = pd.read_parquet(cf)
    else:
        cands = build_candidates(dates, a.liquidity_cliff)
        cands.to_parquet(cf)

    k = select_k(cands, a.horizon, a.liquidity_cliff, sessions, terminal) if cands["decision_date"].min() <= pd.Timestamp(m1.K_SELECT_END) \
        else None
    if k is None:
        # window entirely after 2012 (small check): use the saved choice from the full run if present, else refuse to guess
        kf = config.PROCESSED_DATA_DIR / "m1" / f"k_h{a.horizon}_cliff{a.liquidity_cliff}.json"
        if not kf.exists():
            raise SystemExit("window has no 2010-2012 data and no frozen k from a full run exists")
        kk = json.loads(kf.read_text())
        k = (kk["k_s"], kk["k_t"])
    logger.info(f"h={a.horizon}: frozen k_s={k[0]}, k_t={k[1]}")

    labels = build_labels(cands, a.horizon, k[0], k[1], M1_DIR / "labels" / f"h{a.horizon}_cliff{a.liquidity_cliff}_ks{k[0]}_kt{k[1]}", sessions, terminal)
    df = cands.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=a.horizon)
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    df = m1.apply_impact(df, df["liquidity_rank"], a.impact)
    df["matured_at"] = m1.matured_at(df["label_end_date"], sessions)

    post = regime_posterior(a.horizon, 2015, end.year, end)
    df = attach_regime(df, post)
    logs = m1.run_arms(df, dates)
    summaries = {name: m1.summarise_arm(l, logs["ungated"], dates, a.horizon, n_boot=a.n_boot) for name, l in logs.items() if len(l)}
    for name, l in logs.items():
        l.assign(arm=name).to_parquet(M1_DIR / f"log_h{a.horizon}_{a.impact}_{name}.parquet")

    from gates.override_gates import BUY_SIGNAL_GATES, NOT_APPLIED_GATES
    extra = ["- gates applied: " + ", ".join(BUY_SIGNAL_GATES if a.liquidity_cliff else [g for g in BUY_SIGNAL_GATES if g != "liquidity_cliff"]),
             "- gates NOT APPLIED: " + "; ".join(f"{k} ({v})" for k, v in NOT_APPLIED_GATES.items()),
             f"- decision weeks in window: {len(dates)}; weeks with candidates: {cands['decision_date'].nunique()}",
             f"- candidates {len(df):,}; fill_status counts: {df['fill_status'].value_counts().to_dict()}"]
    tiers = pd.cut(df["liquidity_rank"], [0, 200, 700, np.inf], labels=["rank 1-200 (0.05%/side)", "rank 201-700 (0.20%/side)", "rank > 700 (NaN)"])
    extra.append(f"- impact-floor tier of candidates (decision-date rank): {tiers.value_counts(dropna=False).to_dict()}")
    if a.liquidity_cliff and a.liquidity_cliff <= 200:
        extra.append(f"- NOTE: the liquidity cliff ({a.liquidity_cliff}) removes every name outside the top 200, so only the 0.05% tier is ever "
                     "applied; the pessimistic run is NOT a 0.20% run.")
    for name in ("gate_age", "gate_age_regime"):
        if name in logs and len(logs[name]):
            fb = logs[name].groupby("decision_date")["age_only_fallback"].first()
            extra.append(f"- {name}: age-only fallback in {fb.mean():.1%} of weeks; median calibration rows "
                         f"{logs[name].groupby('decision_date')['n_calibration'].first().median():.0f}")
    txt = report_text(a.horizon, a.impact, k, summaries, extra)
    out = M1_DIR / f"report_h{a.horizon}_{a.impact}.md"
    out.write_text(txt, encoding="utf-8")
    logger.info(txt)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
