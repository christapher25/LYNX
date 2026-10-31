"""
Stage 14 baselines driver (queue row 15c): A0 momentum 12-1, A1 SignalForge port, B1 CatBoost, walk-forward out-of-sample.

Universe and labels are the M1b candidate table (every PIT-universe name passing the ban / circuit / liquidity-cliff gates, weekly) with
NEL labels at the frozen k and the pessimistic impact floor, exactly what M2 will score. Features come from the Stage 12 point-in-time
store through PitReader-equivalent reads (nothing after the decision date), plus 12-1 momentum, the liquidity rank and the universe rank.
Folds come from vanta.walkforward.make_folds (rolling 4y train / 1y validate / 1y test, embargo >= the label horizon, in sessions);
every fit goes through wf.run_fold, so the purge (N11) is the harness's.

  --arm tune   B1 hyperparameter search on FOLD 0's inner split only (M2_PREREG section 3). Prints the winning dict; the lead copies
               it into vanta.baselines.B1_PARAMS and commits before any B1 OOS run. Never reads a fold's test window.
  --arm a0 | a1 | b1   walk-forward OOS scores for the whole fold schedule; writes the scored frame, a per-fold report and a leak check.

Usage:
    python -m scripts.run_baselines --arm tune --horizon 5
    python -m scripts.run_baselines --arm b1 --horizon 5
    python -m scripts.run_baselines --arm b1 --horizon 5 --start 2010-01-04 --end 2012-12-31 --first-year 2010 --train-years 1 \\
        --val-years 0 --mechanics-check --tag chk      # small mechanics check with LightGBM; NOT a baseline result

Realised numbers are empirical. The gate's q is an engineering target, not a guarantee.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from pitstore.store import BLOCK_SPECS, default_loader, select_as_of
from scripts import run_m1, run_m1b
from vanta import baselines as bl
from vanta import m1
from vanta import walkforward as wf
from vanta.labels import load_terminal_values

# Stage 12 blocks used as features. flows_dii is left out: forward capture only, 2 populated rows (CLAUDE.md, Stage 10).
FEATURE_BLOCKS = ("delivery", "fno_oi", "fno_ban", "pledging", "circuits", "flows_fii", "breadth")
CAND_FEATURES = ("mom", "liquidity_rank", "univ_rank")       # columns of the candidate table known at the decision date
A1_SCORER_COL = "det_score"
# Stage 9 indicators used as features: the scale-free ones only (obv, ema_200 and atr_14 are price/volume LEVELS, not comparable across names).
INDICATOR_FEATURES = ("rsi_14", "macd_delta", "bb_position", "atr_normalized", "adx_14", "obv_trend", "volume_ratio",
                      "dist_from_52w_high", "dist_from_52w_low")
INDICATOR_SCORER_INPUTS = ("rsi_14", "macd_delta", "bb_position", "atr_normalized", "obv_trend", "ema_200", "volume_ratio")
BEFORE_NIGHTLY_RUN = False        # the decision follows the decision date's 19:00 nightly run; entry is at the next session (NEEDS LEAD)
STEP = 5                          # weekly decisions


def _logger(tag: str) -> None:
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / f"run_baselines{('_' + tag) if tag else ''}.log", rotation="20 MB")


# ----------------------------------------------------------------------------------------------------- inputs
def load_candidate_labels(horizon: int, start, end, cliff: int, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The M1b candidate table with NEL labels (frozen k) and the impact floor applied, one row per (decision_date, symbol).

    Reads the M1b caches; it does NOT build candidates (a long job, run_m1b's): a missing file stops the run."""
    dates = run_m1.weekly_dates(sessions, start, end)
    cf = run_m1b.M1B_DIR / (f"candidates_cliff{cliff}_{dates.min().date()}_{dates.max().date()}_"
                            f"{run_m1b._candidate_key(dates, cliff)}.parquet")
    cands = run_m1b._read_or_none(cf)
    if cands is None:
        raise SystemExit(f"candidate cache {cf} is missing or stale; run scripts/run_m1b.py for this window first (this driver "
                         "never builds candidates)")
    k_s, k_t = run_m1b.frozen_k(horizon)
    labels = run_m1b.build_labels(cands, horizon, k_s, k_t, run_m1b.M1B_DIR / "labels" / f"h{horizon}_cliff{cliff}_ks{k_s}_kt{k_t}",
                                  sessions, load_terminal_values())
    df = cands.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=horizon)
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    df = m1.apply_impact(df, df["liquidity_rank"], "floor")
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])
    return df


def _cached_loader():
    """default_loader that reads each (block, symbol) file once; per call only the rows in (upto - 30 days, upto] are returned.

    lookback_sessions=1 only ever merges the row(s) of the last session at or before `upto`; the 30-day window keeps every row a lagged
    block could need and makes the per-date cost independent of history length (same values as the unwindowed read)."""
    cache: dict = {}
    far = pd.Timestamp("2100-01-01")

    def load(spec, symbol, upto):
        key = (spec.name, spec.path, symbol)
        if key not in cache:
            frame = default_loader(spec, symbol, far)
            if len(frame):
                frame = frame.assign(date=pd.to_datetime(frame["date"]).dt.normalize().astype("datetime64[ns]")).sort_values(
                    "date", kind="stable").reset_index(drop=True)
            cache[key] = (frame, frame["date"].to_numpy() if len(frame) else None)
        frame, dates = cache[key]
        if dates is None:
            return frame
        lo = np.searchsorted(dates, (upto - pd.Timedelta(days=30)).to_datetime64(), side="right")
        hi = np.searchsorted(dates, upto.to_datetime64(), side="right")
        return frame.iloc[lo:hi]
    return load


def build_features(df: pd.DataFrame, sessions: pd.DatetimeIndex, blocks=FEATURE_BLOCKS) -> tuple[pd.DataFrame, list]:
    """Feature panel (decision_date, symbol, <feature columns>) for every candidate row, point-in-time: each decision date is read as of
    that date only (wf.PitReader-equivalent: select_as_of with the date's own cutoff). Missing stays NaN; absent rows are NaN."""
    loader = _cached_loader()
    value_cols = [c for b in blocks for c in BLOCK_SPECS[b].value_columns]
    if len(set(value_cols)) != len(value_cols):
        raise ValueError("two blocks share a value column name")
    parts = []
    t0 = time.time()
    for i, (d, g) in enumerate(df.groupby("decision_date", sort=True)):
        syms = sorted(g["symbol"].unique())
        frames = {}
        for b in blocks:
            spec = BLOCK_SPECS[b]
            if spec.scope == "symbol":
                got = [p for p in (loader(spec, s, d) for s in syms) if len(p)]
                frames[b] = pd.concat(got, ignore_index=True) if got else pd.DataFrame({"symbol": [], "date": []})
            else:
                frames[b] = loader(spec, None, d)
        f = select_as_of(frames, {b: BLOCK_SPECS[b] for b in blocks}, syms, d, sessions, lookback_sessions=1,
                         before_nightly_run=BEFORE_NIGHTLY_RUN)
        f = f[["symbol", *value_cols]].assign(decision_date=d)
        for c in value_cols:
            f[c] = pd.to_numeric(f[c].astype(object).where(f[c].notna(), np.nan), errors="raise").astype(float)
        parts.append(f)
        if (i + 1) % 100 == 0:
            logger.info(f"  features: {i + 1} dates, {(time.time() - t0) / 60:.1f} min")
    panel = pd.concat(parts, ignore_index=True)
    own = df[["decision_date", "symbol", *CAND_FEATURES]].copy()
    panel = own.merge(panel, on=["decision_date", "symbol"], how="left", validate="one_to_one")
    ind = indicator_panel(df[["decision_date", "symbol"]])
    panel = panel.merge(ind[["decision_date", "symbol", *INDICATOR_FEATURES]], on=["decision_date", "symbol"], how="left",
                        validate="one_to_one")
    cols = [*CAND_FEATURES, *value_cols, *INDICATOR_FEATURES]
    return panel[["decision_date", "symbol", *cols]], cols


def _load_ohlcv(symbol: str):
    """Stage 4 adjusted OHLCV in compute_indicators' schema (non-positive prices already NaN), or None when the symbol has no file."""
    from signals.indicators import load_adjusted_prices
    if not (config.ADJUSTED_PRICES_DIR / f"{symbol}.parquet").exists():
        return None
    return load_adjusted_prices(symbol)


def indicator_panel(df: pd.DataFrame, loader=_load_ohlcv) -> pd.DataFrame:
    """Stage 9 technical indicators as of each decision date, one row per (decision_date, symbol) of `df`.

    Computed with signals.indicators.compute_indicators (the function that fills the Postgres table, which the Stage 9 checkpoint holds
    equal to a fresh recompute) straight from the adjusted bars, so there is no DB dependency. Every indicator is causal (a value at t
    reads bars <= t; tests/test_baselines_driver.py truncates and poisons later bars). The row is taken at the decision date's own bar:
    no bar on that date, no symbol file or an unfilled warm-up window gives NaN, never a carried-forward value."""
    from signals.indicators import compute_indicators
    keep = list(dict.fromkeys(["close", *INDICATOR_SCORER_INPUTS, *INDICATOR_FEATURES]))
    parts = []
    for symbol, g in df.groupby("symbol", sort=True):
        want = pd.DatetimeIndex(sorted(g["decision_date"].unique()))
        ohlcv = loader(symbol)
        if ohlcv is None or not len(ohlcv):
            parts.append(pd.DataFrame({"decision_date": want, "symbol": symbol, **{c: np.nan for c in keep}}))
            continue
        ind = compute_indicators(ohlcv)
        ind["close"] = ohlcv["close"].to_numpy()
        ind = ind.assign(decision_date=pd.to_datetime(ind["date"])).drop_duplicates("decision_date").set_index("decision_date")
        got = ind[keep].astype(float).reindex(want)           # a date with no bar is a NaN row
        parts.append(got.assign(symbol=symbol).rename_axis("decision_date").reset_index())
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["decision_date", "symbol", *keep])
    return out[["decision_date", "symbol", *keep]]


def add_scorer_feature(ind: pd.DataFrame) -> pd.DataFrame:
    """A1's deterministic score: signals/scorer.py technical layer on the decision-date indicator row, z-scored across THAT date's
    candidates only, so a score at d reads date d only. NaN (unavailable, not a neutral 50 and not 0.0) when every scorer input of the
    row is missing, and NaN for every row of a date whose scores have zero variance or fewer than two scored names. The market-context
    layer is identical for every name on a date and adds nothing cross-sectionally."""
    from signals.scorer import compute_technical_score
    out = ind[["decision_date", "symbol"]].copy()
    raw = ind.apply(compute_technical_score, axis=1).astype(float)
    raw[ind[list(INDICATOR_SCORER_INPUTS)].isna().all(axis=1).to_numpy()] = np.nan
    out["_raw"] = raw.to_numpy()

    def z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if g.notna().sum() >= 2 and sd > 0 else pd.Series(np.nan, index=g.index)

    out[A1_SCORER_COL] = out.groupby("decision_date")["_raw"].transform(z)
    return out.drop(columns="_raw")


def require_tuned_b1_params(horizon: int, mechanics: bool) -> None:
    """Refuse a B1 OOS run while the horizon's parameters are still the placeholder template (M2_PREREG section 3: recorded before the
    first OOS run). A labelled mechanics check is allowed."""
    if not mechanics and bl.B1_PARAMS_BY_HORIZON[horizon] == bl.B1_PARAMS:
        raise SystemExit(f"B1_PARAMS_BY_HORIZON[{horizon}] is still the placeholder template: run --arm tune --horizon {horizon}, record the "
                         "result in vanta/baselines.py and commit it before any B1 out-of-sample run (or use --mechanics-check)")


def leak_prefix(leak) -> str:
    """Output-name prefix for a run the leak check flagged, or whose fits saw rows inside the embargo."""
    return "LEAKFLAGGED_" if (leak.flagged or leak.audit_violations) else ""


# ----------------------------------------------------------------------------------------------------- models
def mechanics_factory(arm: str):
    """LightGBM stand-in used ONLY with --mechanics-check (catboost / xgboost are not installed). Not a baseline."""
    import lightgbm as lgb

    class _Reg:
        def __init__(self, seed):
            self.m = lgb.LGBMRegressor(n_estimators=40, num_leaves=8, learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=1)

        def fit(self, X, y):
            self.m.fit(X, y)
            return self

        def predict(self, X):
            return self.m.predict(X)

    class _Clf(_Reg):
        def __init__(self, seed):
            self.m = lgb.LGBMClassifier(n_estimators=40, num_leaves=8, learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=1)

        def predict(self, X):
            return self.m.predict_proba(X)[:, 1]
    return (lambda s: _Clf(s)) if arm == "a1" else (lambda s: _Reg(s))


def real_factory(arm: str, horizon: int):
    return bl.a1_model_factory() if arm == "a1" else bl.b1_model_factory(**bl.B1_PARAMS_BY_HORIZON[horizon])


# ----------------------------------------------------------------------------------------------------- output
def _jsonable(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.date().isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (tuple, list)):
        return [_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    return o


def write_outputs(out_dir: Path, name: str, banner: str, result, rep: list, leak, labels_all: pd.DataFrame, df: pd.DataFrame,
                  horizon: int, extra: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = result.scored.merge(df[["decision_date", "symbol", "fill_status"]], on=["decision_date", "symbol"], how="left")
    scored.to_parquet(out_dir / f"scored_{name}.parquet")
    (out_dir / f"fold_report_{name}.json").write_text(json.dumps(_jsonable(rep), indent=1))
    ic_all = bl.fold_rank_ic(result.scored)
    lines = [banner, "", f"# {name}: walk-forward out-of-sample baseline, horizon {horizon}", ""]
    lines += [f"- folds: {len(rep)}; scored rows {len(scored):,}; pooled-by-date rank IC {ic_all['mean']:.4f} over {ic_all['n_dates']} dates",
              f"- leak check (wf.detect_leak): flagged={leak.flagged}; reasons={leak.reasons}; mean IC {leak.mean_ic:.4f}, "
              f"permutation t {leak.perm_t:.2f}, shifted-label IC {leak.shifted_mean_ic:.4f}, "
              f"audit violations {leak.audit_violations}", *extra, "",
              "| fold | train_start | test | n_train | rank IC | IC dates | net/signal | published | filled | NO_FILL | NO_DATA | unmatured |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep:
        lines.append(f"| {r['fold_id']} | {r['train_start']:%Y-%m-%d} | {r['test_start']:%Y-%m-%d}..{r['test_end']:%Y-%m-%d} | {r['n_train']} | "
                     f"{r['rank_ic']:.4f} | {r['n_ic_dates']} | {r['net_per_signal']:.4f} | {r['n_published']} | {r['n_filled']} | "
                     f"{r['n_no_fill']} | {r['n_no_data']} | {r['n_unmatured']} |")
    lines += ["", "Rank IC and net return are empirical numbers on a window already inspected in M1/M1b; nothing here is a guarantee."]
    (out_dir / f"report_{name}.md").write_text("\n".join(lines))
    logger.info(f"wrote {out_dir / ('report_' + name + '.md')}")


# ----------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--arm", choices=["tune", "a0", "a1", "b1"], required=True)
    ap.add_argument("--horizon", type=int, required=True, choices=m1.HORIZONS)
    ap.add_argument("--start", default=run_m1.WINDOW_START)
    ap.add_argument("--end", default=None, help="last decision date (default: last session on disk)")
    ap.add_argument("--liquidity-cliff", type=int, default=m1.LIQUIDITY_CLIFF_RANK)
    ap.add_argument("--first-year", type=int, default=None, help="first fold's train year (default: year of the first candidate)")
    ap.add_argument("--train-years", type=int, default=4)
    ap.add_argument("--val-years", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=None, help="small checks only")
    ap.add_argument("--mechanics-check", action="store_true", help="use LightGBM instead of CatBoost/XGBoost: mechanics only, NOT a result")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    _logger(a.tag)
    run_m1b.M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"                 # read-only source of candidates and label chunks
    out_dir = config.PROCESSED_DATA_DIR / ("baselines" + (f"_{a.tag}" if a.tag else ""))
    mech = a.mechanics_check
    banner = ("MECHANICS CHECK ONLY (LightGBM stand-in, short window): NOT a baseline result and not usable by M2." if mech
              else "Stage 14 baseline run.")
    if a.arm == "b1":
        require_tuned_b1_params(a.horizon, mech)
    if a.arm == "tune" and mech:
        raise SystemExit("--mechanics-check cannot be combined with --arm tune: the tuned parameters are CatBoost's")

    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    end = pd.Timestamp(a.end) if a.end else sessions[-1]
    df = load_candidate_labels(a.horizon, a.start, sessions[-1], a.liquidity_cliff, sessions)   # the cache is keyed on the full window
    df = df[df["decision_date"] <= end].reset_index(drop=True)
    first_year = a.first_year or int(df["decision_date"].min().year)
    folds = wf.make_folds(sessions, a.horizon, train_years=a.train_years, val_years=a.val_years, first_year=first_year,
                          include_val_in_fit=False)      # train years only; the validation year serves fold 0's tuning (lead 2026-10-31)
    if a.max_folds:
        folds = folds[:a.max_folds]
    if not folds:
        raise SystemExit("no complete fold in this window (need train + validate + a full test year inside the calendar)")
    logger.info(f"{a.arm} h={a.horizon}: {len(df):,} candidates {df['decision_date'].min().date()}..{df['decision_date'].max().date()}; "
                f"{len(folds)} folds, first test {folds[0].test_start.date()}, embargo {folds[0].embargo} sessions")

    if a.arm == "tune":
        f0 = folds[0]
        d0 = df[df["decision_date"] < f0.test_start]                     # fold 0's test window and everything later is never read
        panel, cols = build_features(d0, sessions)
        best = bl.choose_hyperparameters(
            bl.B1_PARAM_GRID, lambda p, seed: bl.b1_model_factory(**p)(seed), panel, d0, f0, sessions, seed=0)
        merged = {**bl.B1_PARAMS_BY_HORIZON[a.horizon], **best}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"b1_tuned_h{a.horizon}.json").write_text(json.dumps(merged, indent=1))
        logger.info(f"B1 tuned on fold {f0.fold_id} inner split (fit [{f0.train_start.date()}, {f0.val_start.date()}), evaluate "
                    f"[{f0.val_start.date()}, {f0.test_start.date()})): {best}")
        logger.info(f"copy into vanta.baselines.B1_PARAMS_BY_HORIZON[{a.horizon}] and commit before any OOS run")
        return

    if a.arm == "a0":
        panel = df[["decision_date", "symbol", "mom"]]
        cols = ["mom"]
        fit_fn = lambda rows, fold: None                                  # noqa: E731  A0 has nothing to fit
        mom = panel.set_index(["decision_date", "symbol"])["mom"]

        def score_fn(model, rows, fold, reader):
            return mom.reindex(pd.MultiIndex.from_frame(rows[["decision_date", "symbol"]])).to_numpy(float)
        publish_above = None
        seeds = ()
    else:
        panel, cols = build_features(df, sessions)
        if a.arm == "a1":
            sc = add_scorer_feature(indicator_panel(df[["decision_date", "symbol"]]))
            panel = panel.merge(sc, on=["decision_date", "symbol"], how="left", validate="one_to_one")
            cols = cols + [A1_SCORER_COL]
        factory = mechanics_factory(a.arm) if mech else real_factory(a.arm, a.horizon)
        target = "binary" if a.arm == "a1" else "net"
        fit_fn = bl.make_fit_fn(panel, cols, factory, seeds=bl.B1_SEEDS, target=target)
        score_fn = bl.make_score_fn(panel, cols)
        publish_above = bl.A1_PUBLISH_ABOVE if a.arm == "a1" else None
        seeds = bl.B1_SEEDS

    labels = df[["symbol", "decision_date", "label_end_date", "y_net"]]
    result = wf.run_walkforward(folds, labels, sessions, fit_fn, score_fn)
    rep = bl.fold_report(
        [dataclasses.replace(r, scored=r.scored.merge(df[["decision_date", "symbol", "fill_status"]],
                                                      on=["decision_date", "symbol"], how="left")) for r in result.folds],
        seeds=seeds, publish_above=publish_above)
    leak = wf.detect_leak(result, labels, horizon=a.horizon, step=STEP)
    extra = [f"- features ({len(cols)}): {', '.join(cols)}" if a.arm != "a0" else "- A0 score: 12-1 momentum (vanta.m1.momentum_12_1)",
             f"- seeds: {list(seeds)}; B1_PARAMS: {bl.B1_PARAMS_BY_HORIZON[a.horizon]}" if a.arm == "b1" else f"- seeds: {list(seeds)}",
             f"- candidates {len(df):,}; fill_status {df['fill_status'].value_counts().to_dict()}"]
    if publish_above is not None:
        extra.append(f"- A1 published when P(win) > {publish_above}, top 10 per week; net/signal is over those")
    win = bl.first_scored_and_judged_week(result.scored, m1.OOS_START)
    extra.append(f"- first scored week {win['first_scored_week']:%Y-%m-%d}; first judged week (>= {m1.OOS_START:%Y-%m-%d}, >= 52 weeks of "
                 f"scored history before it) " + (f"{win['first_judged_week']:%Y-%m-%d}" if pd.notna(win['first_judged_week']) else "NONE in this window"))
    prefix = leak_prefix(leak) + ("MECHANICS_" if mech else "")
    write_outputs(out_dir, f"{prefix}{a.arm}_h{a.horizon}", banner, result, rep, leak, labels, df, a.horizon, extra)
    if leak_prefix(leak):
        logger.error(f"LEAK CHECK FLAGGED ({leak.reasons}; audit violations {leak.audit_violations}): outputs written with the "
                     "LEAKFLAGGED_ prefix; do not use them")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
