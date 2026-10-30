"""
Milestone M1b driver (Stage 13M, corrected M1): percentile-rank momentum over ALL gate-passing candidates, NEL labels, 52-week matured
calibration on all scored candidates, BH over all candidates at T with at most 10 published.

Stages (each writes under config.PROCESSED_DATA_DIR / "m1b[_tag]" and is skipped on re-run if its output exists; --force rebuilds):
  1 candidates   every PIT-universe name passing ban / circuit / liquidity-cliff gates, percentile-ranked 12-1 momentum per week
  2 k            frozen from data/processed/m1/k_h{h}_cliff200.json (chosen in M1 on 2010-2012; never re-chosen here)
  3 labels       NEL labels at the frozen k for every candidate, per-symbol chunk files (hash-keyed, resumable, one symbol in memory)
  4 regime       filtered HMM posterior (cache file name carries a fingerprint of its inputs)
  5 arms         ungated top 10 / v4 reconstruction / gate_age / gate_age_regime, then summary + report for one impact mode

Usage:
    python -m scripts.run_m1b --horizon 5 --impact floor
    python -m scripts.run_m1b --horizon 5 --impact zero --start 2013-01-01 --end 2013-06-30 --tag chk   # small check, own directory

q is an engineering target for the false-discovery proportion. Realised FDR is reported as an empirical number.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from scripts import run_m1
from vanta import m1, m1b
from vanta.labels import load_terminal_values, simulate_trade, terminal_kwargs

M1_DIR = config.PROCESSED_DATA_DIR / "m1"             # read-only source of the frozen k and the liquidity-rank / regime caches
M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"
WINDOW_START = run_m1.WINDOW_START
FORCE = False

FIRST_PARAGRAPH = (
    "The 2013+ window was already seen in M1 (logs/m1_report.md, NO-GO as run), and k_s, k_t, q, the costs, the gates and the five GO "
    "criteria are the same ones. M1b is therefore a CORRECTED RUN of the same window with the two departures from v6 removed "
    "(candidates = every gate-passing name rather than the weekly top 10; score = weekly cross-sectional percentile rank of 12-1 "
    "momentum rather than its raw level). It is not a fresh out-of-sample test.")


def _logger(tag: str):
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / f"run_m1b{('_' + tag) if tag else ''}.log", rotation="20 MB")


# ------------------------------------------------------------------ cache plumbing
def _stat_fp(*items) -> str:
    """Fingerprint of files / directories by (relative name, size, mtime_ns). A directory contributes every file under it."""
    h = hashlib.sha1()
    for it in items:
        it = Path(it)
        files = sorted(f for f in it.rglob("*") if f.is_file()) if it.is_dir() else ([it] if it.exists() else [])
        h.update(str(it.name).encode())
        for f in files:
            st = f.stat()
            h.update(f"{f.relative_to(it) if it.is_dir() else f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:10]


def _src_fp(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        h.update(Path(o.__file__).read_bytes() if hasattr(o, "__file__") else inspect.getsource(o).encode())
    return h.hexdigest()[:10]


def _candidate_key(dates: pd.DatetimeIndex, cliff: int) -> str:
    """Everything the candidate table depends on: gate / liquidity / momentum code, the build functions themselves, the universe snapshots,
    adjusted prices, circuit and F&O feature files, the cliff and the date grid."""
    import gates.io
    import gates.liquidity
    import gates.override_gates
    code = _src_fp(gates.io, gates.liquidity, gates.override_gates, m1, m1b, build_candidates, liquidity_rank_frame)
    data = _stat_fp(config.UNIVERSE_DIR, config.ADJUSTED_PRICES_DIR, config.CIRCUIT_FEATURES_DIR, config.FNO_FEATURES_DIR)
    return run_m1._hash(code, data, cliff, dates.min().date().isoformat(), dates.max().date().isoformat(), len(dates))


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write beside the target, then os.replace: a killed run leaves the old file or a complete new one, never a half file."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _read_or_none(path: Path, quiet: bool = False):
    """A parquet file's frame, or None when it is missing or unreadable (truncated); an unreadable file is reported and rebuilt."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:                                  # truncated / corrupt: never crash and never skip it
        logger.warning(f"unreadable cache file {path.name} ({type(e).__name__}); rebuilding it")
        return None
    if not quiet:
        logger.info(f"reusing cache {path.name} (mtime {pd.Timestamp(path.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M})")
    return df


@contextlib.contextmanager
def _run_m1_dir(directory: Path, force: bool):
    """Point run_m1's module globals at our directory for one call into it, then put them back."""
    old = (run_m1.M1_DIR, run_m1.FORCE)
    run_m1.M1_DIR, run_m1.FORCE = directory, force
    try:
        yield
    finally:
        run_m1.M1_DIR, run_m1.FORCE = old


# ------------------------------------------------------------------ stage 0
def liquidity_rank_frame(start) -> pd.DataFrame:
    """Point-in-time trailing-turnover rank for every adjusted-panel row on or after `start`. The cache file name carries a fingerprint of
    the adjusted panel and the ranking code. The M1 cache of an earlier-or-equal start is reused (filtered to `start`) only if it is newer
    than every adjusted-price file and than gates/liquidity.py; otherwise the rank is recomputed from the full panel."""
    import gates.liquidity
    start = pd.Timestamp(start)
    fp = run_m1._hash(_stat_fp(config.ADJUSTED_PRICES_DIR), _src_fp(gates.liquidity))
    own = M1B_DIR / f"liquidity_rank_{start.date()}_{fp}.parquet"
    if not FORCE:
        d = _read_or_none(own)
        if d is not None:
            return d
        newest = max([f.stat().st_mtime for f in config.ADJUSTED_PRICES_DIR.glob("*.parquet")] + [Path(gates.liquidity.__file__).stat().st_mtime])
        for f in sorted(M1_DIR.glob("liquidity_rank_*.parquet")):
            try:
                s0 = pd.Timestamp(f.stem.split("_")[2])
            except (IndexError, ValueError):
                continue
            if s0 <= start and f.stat().st_mtime >= newest:
                d = _read_or_none(f)
                if d is not None:
                    return d[pd.to_datetime(d["date"]) >= start].reset_index(drop=True)
    with _run_m1_dir(M1B_DIR, True):                        # computes from the full panel; its own unkeyed file lands in our directory
        out = run_m1.liquidity_rank_frame(start)
    _atomic_to_parquet(out, own)
    return out


# ------------------------------------------------------------------ stage 1
def build_candidates(dates: pd.DatetimeIndex, cliff: int) -> pd.DataFrame:
    """All PIT-universe names with finite 12-1 momentum that pass the buy-signal gates that week. No pool, no top-N."""
    from gates.io import load_gate_inputs
    from gates.override_gates import BUY_SIGNAL_GATES, GateConfig, evaluate_gates

    liq = liquidity_rank_frame(dates.min())
    ranks = run_m1.universe_ranks(sorted(set(dates.year)))
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
            rows.append(pd.DataFrame({"decision_date": d, "symbol": s, "mom": m1.momentum_12_1(close, d).to_numpy()}))
        if (i + 1) % 300 == 0:
            logger.info(f"  momentum {i + 1}/{len(symbols)}")
    mom = pd.concat(rows, ignore_index=True)
    mom["year"] = mom["decision_date"].dt.year
    mom = mom.merge(ranks, on=["year", "symbol"], how="inner")          # point-in-time membership of the decision year
    mom = mom.merge(liq.rename(columns={"date": "decision_date"})[["symbol", "decision_date", "liquidity_rank"]],
                    on=["symbol", "decision_date"], how="left")
    mom = mom[np.isfinite(mom["mom"])]
    if cliff:                                       # the cliff gate's own cut, applied first so gate inputs load for fewer names
        mom = mom[mom["liquidity_rank"] <= cliff]
    logger.info(f"pool rows {len(mom):,} ({mom['symbol'].nunique()} symbols); loading gate inputs")
    g = load_gate_inputs(sorted(mom["symbol"].unique()), start=dates.min(), end=dates.max(), dates=dates, liquidity_ranks=liq)
    g = g.rename(columns={"date": "decision_date"})
    enabled = BUY_SIGNAL_GATES if cliff else tuple(x for x in BUY_SIGNAL_GATES if x != "liquidity_cliff")
    verdict = evaluate_gates(g, GateConfig(enabled=enabled, liquidity_top_n=cliff or 700, unavailable_policy="block"))
    g = pd.concat([g[["symbol", "decision_date"]], verdict[["gate_status", "gate_allowed", "gate_reasons"]]], axis=1)
    del verdict
    pool = mom.merge(g, on=["symbol", "decision_date"], how="left")
    del g
    pool["gate_allowed"] = pool["gate_allowed"].fillna(False).astype(bool)      # no gate row at all = not allowed, never a pass
    stats = pool.groupby(pool["decision_date"].dt.year)["gate_status"].value_counts().unstack(fill_value=0)
    logger.info("gate status of pool names by year:\n" + stats.to_string())
    cands = m1b.select_candidates(pool)
    cands = m1b.add_scores(cands).reset_index(drop=True)
    per_week = cands.groupby("decision_date").size()
    logger.info(f"candidates {len(cands):,} over {len(per_week)} of {len(dates)} weeks; per week min {per_week.min()}, "
                f"median {per_week.median():.0f}, max {per_week.max()}")
    return cands[["decision_date", "symbol", "mom", "score", "univ_rank", "liquidity_rank", "gate_status"]]


# ------------------------------------------------------------------ stage 3
def build_labels(cands: pd.DataFrame, horizon: int, k_s: float, k_t: float, cache_dir, sessions, terminal) -> pd.DataFrame:
    """NEL label per candidate. One chunk file per symbol, named with a hash of everything that determines its rows (symbol, requested
    dates, horizon, k, terminal value, label/cost code version): an interrupted run resumes at the first missing chunk, a changed request
    or code version gives a new file and the old one is left alone. Only one symbol's bars are in memory at a time; finished chunks
    are read back lazily at the end."""
    from scripts.report_nel_coverage import load_symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    code = run_m1._code_version()
    groups = list(cands.groupby("symbol"))
    t0, n_done, n_new = time.time(), 0, 0
    parts = []
    for symbol, g in groups:
        dates = sorted(pd.Timestamp(d) for d in g["decision_date"])
        tk = terminal_kwargs(symbol, terminal)
        inputs = _stat_fp(config.ADJUSTED_PRICES_DIR / f"{symbol}.parquet", config.CIRCUIT_FEATURES_DIR / f"{symbol}.parquet")
        key = run_m1._hash(symbol, [d.date().isoformat() for d in dates], horizon, k_s, k_t,
                           sorted((k, str(v)) for k, v in tk.items()), code, inputs)
        chunk = cache_dir / f"{symbol}_{key}.parquet"
        n_done += 1
        df = None if FORCE else _read_or_none(chunk, quiet=True)
        if df is not None:
            parts.append(df)
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
        _atomic_to_parquet(df, chunk)
        parts.append(df)
        del loaded, rows
        n_new += 1
        if n_new % 25 == 0:
            gc.collect()
            el = time.time() - t0
            logger.info(f"  labels: {n_done}/{len(groups)} symbols ({n_new} computed, {n_done - n_new} cached), {el / 60:.1f} min; "
                        f"estimated {el / n_done * (len(groups) - n_done) / 60:.1f} min left")
    res = pd.concat(parts, ignore_index=True)
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


def frozen_k(horizon: int) -> tuple:
    kf = M1_DIR / f"k_h{horizon}_cliff{m1b.K_SOURCE_CLIFF}.json"
    if not kf.exists():
        raise SystemExit(f"frozen M1 k file {kf} is missing; M1b never re-chooses k")
    k = json.loads(kf.read_text())
    return k["k_s"], k["k_t"]


# ------------------------------------------------------------------ stage 4
def regime_posterior(horizon: int, first_year: int, last_year: int, end) -> pd.DataFrame:
    """Filtered posterior; the cache file name carries a fingerprint of the regime inputs and vanta/regime.py. Never reads an unkeyed file."""
    from vanta import regime as R
    raw = config.RAW_DATA_DIR
    fp = run_m1._hash(_stat_fp(raw / "indices", config.FLOW_FEATURES_DIR / "fii_flows.parquet",
                               config.BREADTH_FEATURES_DIR / "universe_breadth.parquet", raw / "macro" / "usdinr.parquet"), _src_fp(R))
    keyed = M1B_DIR / f"regime_h{horizon}_{pd.Timestamp(end).date()}_{fp}.parquet"
    if not FORCE:
        d = _read_or_none(keyed)
        if d is not None:
            return d
    with _run_m1_dir(M1B_DIR, True):
        out = run_m1.regime_posterior(horizon, first_year, last_year, end)
    _atomic_to_parquet(out, keyed)
    return out


def attach_regime(cands: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """run_m1.attach_regime semantics (row for the date, NaN stays NaN, else last earlier row at most 5 days old), but looked up once per
    unique decision date and shared across that week's candidates, so hundreds of thousands of candidates do not each hold their own list."""
    dts = pd.DatetimeIndex(sorted(cands["decision_date"].unique()))
    if post.empty:
        sub = pd.DataFrame(np.nan, index=dts, columns=["p0", "p1", "p2"])
        hv = pd.Series([np.nan] * len(dts), index=dts)
    else:
        sub = m1.asof_values(post[["p0", "p1", "p2"]], dts)
        hv = m1.asof_values(post[["hmm_version"]], dts)["hmm_version"]
    lists = {d: list(r) for d, r in zip(dts, sub.to_numpy(float))}
    vers = dict(zip(dts, hv.to_numpy()))
    out = cands.copy()
    out["regime_posterior"] = out["decision_date"].map(lists)
    out["hmm_version"] = out["decision_date"].map(vers)
    return out


# ------------------------------------------------------------------ 13b diagnostics (NOT JUDGED)
def load_nifty50_close() -> pd.Series:
    """Nifty 50 daily close from the Stage 8 index files (one parquet per session, from 2012-02-21). Cached with a fingerprint of the files."""
    idx_dir = config.RAW_DATA_DIR / "indices"
    cache = M1B_DIR / f"nifty50_close_{_stat_fp(idx_dir)}.parquet"
    d = _read_or_none(cache)
    if d is None:
        rows = []
        for f in sorted(idx_dir.glob("*.parquet")):
            v = pd.read_parquet(f, columns=["INDEX_NAME", "DATE", "CLOSE"], filters=[("INDEX_NAME", "==", "Nifty 50")])
            if len(v):
                rows.append((pd.Timestamp(v["DATE"].iloc[0]), float(v["CLOSE"].iloc[0])))
        d = pd.DataFrame(rows, columns=["date", "close"])
        _atomic_to_parquet(d, cache)
    s = pd.Series(d["close"].to_numpy(float), index=pd.DatetimeIndex(d["date"]))
    return s[s > 0].sort_index()


def diagnostics_lines(logs: dict, df: pd.DataFrame, nifty: pd.Series, dates: pd.DatetimeIndex, oos_start=m1.OOS_START) -> list[str]:
    """Report block, every number labelled NOT JUDGED: weekly false-discovery proportion against the Nifty 50 holding-week return, and
    realised FDR with loser = underperformed the Nifty 50 over the trade window after costs. Weeks without Nifty data (before 2012-02-21) are
    excluded, not zero-filled."""
    oos_start = pd.Timestamp(oos_start)
    mkt = m1b.market_weekly_returns(nifty, dates)
    lines = ["## Diagnostics (NOT JUDGED; none of these enters a GO criterion)",
             "Market weekly return = Nifty 50 close change from the decision date to the next decision date (price index, no dividends). "
             f"Nifty 50 data start {nifty.index.min().date()}; earlier weeks are excluded. Weekly FDP = losers / filled published trades with a "
             "known outcome in that week (loser = y_net <= 0), NaN when a week has none.", ""]
    extra_cols = df[["decision_date", "symbol", "entry_date", "label_end_date"]]
    for arm in ("ungated", "gate_age", "gate_age_regime", "v4"):
        log = logs.get(arm)
        if log is None or not len(log):
            continue
        fdp = m1b.weekly_fdp(log)
        for seg, sel in (("oos", fdp.index >= oos_start), ("in_sample", fdp.index < oos_start)):
            r = m1b.fdp_vs_market(fdp[sel], mkt)
            lines.append(f"- {arm} / {seg}: corr(weekly FDP, market return) {r['corr']:.3f} over {r['n_weeks']} weeks; mean weekly FDP in up weeks "
                         f"{r['fdr_up']:.3f} (n={r['n_up']}), in down weeks {r['fdr_down']:.3f} (n={r['n_down']}). NOT JUDGED")
        rl = log.merge(extra_cols, on=["decision_date", "symbol"], how="left")
        for seg, sel in (("oos", rl["decision_date"] >= oos_start), ("in_sample", rl["decision_date"] < oos_start)):
            st = m1b.realised_fdr_vs_index(rl[sel], nifty)
            lines.append(f"- {arm} / {seg}: realised FDR with loser = underperformed the Nifty 50 after costs {st['fdr']:.3f} on {st['n_filled']} filled "
                         f"published trades with an index return (losers {st['n_losers']}; trades without one {st['n_unmatured']}). "
                         "DIAGNOSTIC ONLY, not the pre-registered loser definition")
    lines.append("")
    return lines


# ------------------------------------------------------------------ report
def report_text(horizon, impact, k, summaries, extra) -> str:
    txt = run_m1.report_text(horizon, impact, k, summaries, extra)
    lines = txt.split("\n")
    lines[0] = f"# M1b report: h={horizon}, impact={impact}, k_s={k[0]}, k_t={k[1]} (frozen in M1 on 2010-2012)"
    return "\n".join([lines[0], "", FIRST_PARAGRAPH, ""] + lines[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizon", type=int, required=True, choices=m1.HORIZONS)
    ap.add_argument("--impact", choices=["zero", "floor"], required=True)
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=None, help="last decision date (default: last session on disk)")
    ap.add_argument("--liquidity-cliff", type=int, default=m1.LIQUIDITY_CLIFF_RANK, help="max liquidity rank (0 = cliff off)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="rebuild every stage (overwrites its own files; nothing is deleted)")
    ap.add_argument("--tag", default="", help="suffix for the output dir; small checks use their own, never the full-run files")
    a = ap.parse_args()
    _logger(a.tag)
    global M1B_DIR, FORCE
    FORCE = a.force
    M1B_DIR = config.PROCESSED_DATA_DIR / ("m1b" + (f"_{a.tag}" if a.tag else ""))
    M1B_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    end = pd.Timestamp(a.end) if a.end else sessions[-1]
    dates = run_m1.weekly_dates(sessions, a.start, end)
    terminal = load_terminal_values()

    cf = M1B_DIR / (f"candidates_cliff{a.liquidity_cliff}_{dates.min().date()}_{dates.max().date()}_"
                    f"{_candidate_key(dates, a.liquidity_cliff)}.parquet")
    cands = None if a.force else _read_or_none(cf)
    if cands is None:
        cands = build_candidates(dates, a.liquidity_cliff)
        _atomic_to_parquet(cands, cf)

    k = frozen_k(a.horizon)
    logger.info(f"h={a.horizon}: frozen k_s={k[0]}, k_t={k[1]}; {len(cands):,} candidates")
    labels = build_labels(cands, a.horizon, k[0], k[1], M1B_DIR / "labels" / f"h{a.horizon}_cliff{a.liquidity_cliff}_ks{k[0]}_kt{k[1]}",
                          sessions, terminal)
    df = cands.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=a.horizon)
    del labels
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    df = m1.apply_impact(df, df["liquidity_rank"], a.impact)
    df["matured_at"] = m1.matured_at(df["label_end_date"], sessions)

    post = regime_posterior(a.horizon, 2015, end.year, end)
    df = attach_regime(df, post)
    logs = m1b.run_arms_b(df, dates)
    summaries = {name: m1.summarise_arm(l, logs["ungated"], dates, a.horizon, n_boot=a.n_boot) for name, l in logs.items() if len(l)}
    for name, l in logs.items():
        l.assign(arm=name).to_parquet(M1B_DIR / f"log_h{a.horizon}_{a.impact}_{name}.parquet")

    from gates.override_gates import BUY_SIGNAL_GATES, NOT_APPLIED_GATES
    extra = ["- gates applied: " + ", ".join(BUY_SIGNAL_GATES if a.liquidity_cliff else [g for g in BUY_SIGNAL_GATES if g != "liquidity_cliff"]),
             "- gates NOT APPLIED: " + "; ".join(f"{k_} ({v})" for k_, v in NOT_APPLIED_GATES.items()),
             "- candidates: every PIT-universe name passing the applied gates with finite 12-1 momentum; score = weekly percentile rank "
             "of 12-1 momentum among those candidates; ungated arm = the 10 highest-percentile candidates per week; gated arms run BH over "
             "all candidates of the week and publish at most 10; calibration = all matured scored candidates of the last 52 weeks",
             f"- decision weeks in window: {len(dates)}; weeks with candidates: {cands['decision_date'].nunique()}; "
             f"candidates per week median {cands.groupby('decision_date').size().median():.0f}",
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
    txt += "\n" + "\n".join(diagnostics_lines(logs, df, load_nifty50_close(), dates))
    out = M1B_DIR / f"report_h{a.horizon}_{a.impact}.md"
    out.write_text(txt, encoding="utf-8")
    logger.info(txt)
    logger.info(f"wrote {out}; total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
