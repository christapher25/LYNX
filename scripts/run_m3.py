"""
Milestone M3 driver (Stage 13M): the VANTA gate with B1 retrained on HOLD-TO-HORIZON net labels. Pre-registration: docs/vanta/M3_PREREG.md.

Nothing the pre-registration fixes can be changed from the command line (q, cap, tau, ESS floor, seeds, window, liquidity cliff, arms,
labels): only the horizon, the impact mode, the stage, and bookkeeping options. Inputs are READ, never built here: the M1b candidate table
(scripts.run_m1b). Everything M3 builds goes under data/processed/m3[_tag]/ and NEVER into m2/, m1b/ or baselines/.

Stages (each resumable; a finished output is skipped on re-run, --force rebuilds):
  --tune   --horizon H   B1 hyperparameter search on FOLD 0's inner split only, on hold-to-horizon floor-impact labels, M2's grid and rule.
                         Writes data/processed/m3/b1hold_tuned_h{H}.json. The lead copies it into vanta.m3.B1_HOLD_PARAMS_BY_HORIZON
                         and commits it BEFORE --score. Labels and features are built for decision dates before fold 0's test year only.
  --score  --horizon H   walk-forward out-of-sample B1 scores (5 seeds, tuned params) on hold labels, for the whole fold schedule. Refused
                         while vanta.m3.B1_HOLD_PARAMS_BY_HORIZON[H] is still the placeholder. Writes scored_b1hold_h{H}.parquet, the fold
                         report, a leak check report.
  --horizon H --impact {floor,zero}
                         one M3 cell: the three M2 gate arms (a, b, c) and the ungated top-10 on the saved scores. Writes
                         log_h{H}_{impact}_{arm}.parquet and meta_h{H}_{impact}.json.
  --report               assemble logs/m3_report.md from the saved cells and stop. The GO/NO-GO is the owner's.

Usage:
    python -m scripts.run_m3 --tune --horizon 20
    python -m scripts.run_m3 --score --horizon 20
    python -m scripts.run_m3 --horizon 20 --impact floor
    python -m scripts.run_m3 --report
    python -m scripts.run_m3 --score --horizon 5 --mechanics-check --tag chk --end 2012-12-31 --first-year 2010 \\
        --train-years 1 --val-years 0                       # small LightGBM mechanics check, NOT a result

Realised FDR is an empirical number. q = 0.30 is an engineering target, not a guarantee.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from pitstore.store import BLOCK_SPECS
from scripts import run_baselines, run_m1, run_m1b, run_m2
from vanta import baselines as bl
from vanta import m1, m2, m3
from vanta import walkforward as wf
from vanta.labels import load_terminal_values

M3_DIR = config.PROCESSED_DATA_DIR / "m3"
REPORT_PATH = config.LOG_DIR / "m3_report.md"
WINDOW_START = run_m1.WINDOW_START
CLIFF = m1.LIQUIDITY_CLIFF_RANK                      # fixed by the pre-registration (top 200 by trailing turnover)
LOG_ARMS = (*m3.ARMS, "ungated")
NO_LEVEL = math.inf                                  # hold-to-horizon has no stop and no target: k_s = k_t = infinity in the chunk key


def _logger(tag: str) -> None:
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / f"run_m3{('_' + tag) if tag else ''}.log", rotation="20 MB")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizon", type=int, choices=list(m3.ALL_HORIZONS), help="5 (NOT JUDGED), 20 or 60 (judged on the floor run)")
    ap.add_argument("--impact", choices=["zero", "floor"], help="floor = the judged run; zero = reported")
    ap.add_argument("--tune", action="store_true", help="B1 hyperparameter search on fold 0's inner split only (hold labels)")
    ap.add_argument("--score", action="store_true", help="walk-forward OOS B1 scores on hold labels (refused while params are placeholders)")
    ap.add_argument("--report", action="store_true", help="assemble logs/m3_report.md from the saved cells and stop")
    ap.add_argument("--end", default=None, help="last decision date (small checks only; default: last session on disk)")
    ap.add_argument("--first-year", type=int, default=None, help="first fold's train year (default: year of the first candidate)")
    ap.add_argument("--train-years", type=int, default=4)
    ap.add_argument("--val-years", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=None, help="small checks only")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="rebuild this stage's outputs (overwrites its own files; nothing is deleted)")
    ap.add_argument("--mechanics-check", action="store_true", help="LightGBM stand-in, MECHANICS_ prefix: NOT a result")
    ap.add_argument("--tag", default="", help="suffix for the output directory; small checks never touch the full-run files")
    return ap


# ---------------------------------------------------------------------------------------------------------------- paths
def out_dir_for(tag: str) -> Path:
    return config.PROCESSED_DATA_DIR / ("m3" + (f"_{tag}" if tag else ""))


def stem(h: int, impact: str) -> str:
    return f"h{h}_{impact}"


def scored_name(h: int, mechanics: bool) -> str:
    return f"scored_{'MECHANICS_' if mechanics else ''}b1hold_h{h}.parquet"


def hold_label_dir(h: int) -> Path:
    """Shared by every tag: a chunk's file name carries a hash of everything that determines its rows, so tags cannot collide."""
    return M3_DIR / "labels_hold" / f"h{h}_cliff{CLIFF}_hold_{m3.hold_label_source_fp()}"


# ---------------------------------------------------------------------------------------------------------------- labels and features
def load_candidates(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The M1b candidate table (cache; never built here)."""
    run_m1b.M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"            # read-only source of candidates / regime cache
    run_m1b.FORCE = False
    return run_m2.load_candidates(sessions)


def build_hold_labels(cands: pd.DataFrame, h: int, sessions, terminal) -> pd.DataFrame:
    """Hold-to-horizon labels (vanta.m2.hold_to_horizon_label via the same chunked, hash-keyed, resumable build_labels the M1b labels use),
    one chunk file per symbol under data/processed/m3/labels_hold/. Impact is NOT applied here. Rows are exactly the requested
    (symbol, decision_date) pairs; passing the same candidate set again reuses every finished chunk."""
    with run_m2._hold_labels():
        labels = run_m1b.build_labels(cands, h, NO_LEVEL, NO_LEVEL, hold_label_dir(h), sessions, terminal)
    df = cands.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=h)
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    for c in ("decision_date", "label_end_date", "entry_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def _apply_impact(df: pd.DataFrame, impact: str) -> pd.DataFrame:
    return m1.apply_impact(df, df["liquidity_rank"], impact)


def _features_key(df: pd.DataFrame, cand_name: str) -> str:
    """Everything the feature panel depends on: the build code, the block definitions, the data files behind every block and the indicators,
    the candidate cache, and the exact (decision_date, symbol) request."""
    code = hashlib.sha1()
    for o in (run_baselines.build_features, run_baselines.indicator_panel, run_baselines._cached_loader):
        code.update(inspect.getsource(o).encode())
    code.update(repr((run_baselines.FEATURE_BLOCKS, run_baselines.CAND_FEATURES, run_baselines.INDICATOR_FEATURES,
                      run_baselines.BEFORE_NIGHTLY_RUN)).encode())
    paths = [BLOCK_SPECS[b].path for b in run_baselines.FEATURE_BLOCKS if BLOCK_SPECS[b].path is not None]
    data = run_m1b._stat_fp(*paths, config.ADJUSTED_PRICES_DIR)
    req = hashlib.sha1(pd.util.hash_pandas_object(df[["decision_date", "symbol"]].sort_values(["decision_date", "symbol"]),
                                                  index=False).to_numpy().tobytes()).hexdigest()
    return run_m1._hash(code.hexdigest(), data, cand_name, req)


def feature_panel(df: pd.DataFrame, sessions, cand_name: str, out_dir: Path, force: bool = False) -> tuple[pd.DataFrame, list]:
    """The Stage 14 feature panel (run_baselines.build_features, point-in-time, missing stays NaN), cached by a key over its inputs and the
    exact rows requested, so h = 5, 20, 60 share one build. A request for fewer dates (tuning: before fold 0's test year) is its own key
    and never reads a later date."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"features_{_features_key(df, cand_name)}.parquet"
    if not force:
        got = run_m1b._read_or_none(f)
        if got is not None:
            return got, [c for c in got.columns if c not in ("decision_date", "symbol")]
    panel, cols = run_baselines.build_features(df, sessions)
    run_m1b._atomic_to_parquet(panel, f)
    return panel, cols


def _cand_cache_name(sessions) -> str:
    dates = run_m1.weekly_dates(sessions, WINDOW_START, sessions[-1])
    return f"candidates_cliff{CLIFF}_{dates.min().date()}_{dates.max().date()}_{run_m1b._candidate_key(dates, CLIFF)}.parquet"


def _make_folds(sessions, h, df, a) -> list:
    first_year = a.first_year or int(df["decision_date"].min().year)
    folds = wf.make_folds(sessions, h, train_years=a.train_years, val_years=a.val_years, first_year=first_year, include_val_in_fit=False)
    if a.max_folds:
        folds = folds[:a.max_folds]
    if not folds:
        raise SystemExit("no complete fold in this window (need train + validate + a full test year inside the calendar)")
    return folds


def refuse_schedule_overrides(a, what: str) -> None:
    """M3_PREREG section 4: folds are 4 y train / 1 y validation / 1 y test over the whole window. A judged (non-mechanics) stage that
    overrides any of that is refused BEFORE any data is read; only a labelled mechanics check may use another schedule."""
    if getattr(a, "mechanics_check", False):
        return
    bad = [k for k, v in (("train_years", a.train_years != 4), ("val_years", a.val_years != 1), ("first_year", a.first_year is not None),
                          ("max_folds", bool(a.max_folds)), ("end", bool(a.end))) if v]
    if bad:
        raise SystemExit(f"{what}: {', '.join(bad)} may be overridden only in a --mechanics-check (M3_PREREG section 4 fixes the fold schedule "
                         "and the window)")


# ---------------------------------------------------------------------------------------------------------------- tune
def run_tune(h: int, a) -> None:
    refuse_schedule_overrides(a, "tune")
    out_dir = out_dir_for(a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    tuned_f = out_dir / f"b1hold_tuned_h{h}.json"
    if tuned_f.exists() and not a.force:
        logger.info(f"{tuned_f.name} exists; skipping (--force to redo). Contents: {tuned_f.read_text()}")
        return
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    cands = load_candidates(sessions)
    folds = _make_folds(sessions, h, cands, a)
    f0 = folds[0]
    d0 = cands[cands["decision_date"] < f0.test_start].reset_index(drop=True)      # fold 0's test window and everything later is never read
    logger.info(f"tune h={h}: fold 0 fit [{f0.train_start.date()}, {f0.val_start.date()}), evaluate [{f0.val_start.date()}, "
                f"{f0.test_start.date()}), embargo {f0.embargo}; {len(d0):,} candidates before the test year")
    lab = _apply_impact(build_hold_labels(d0, h, sessions, load_terminal_values()), "floor")
    panel, _ = feature_panel(d0, sessions, _cand_cache_name(sessions), out_dir, a.force)
    best = bl.choose_hyperparameters(m3.B1_HOLD_GRID, lambda p, seed: bl.b1_model_factory(**p)(seed), panel, lab, f0, sessions, seed=0)
    merged = {**bl.B1_PARAMS, **best}
    tuned_f.write_text(json.dumps(merged, indent=1))
    logger.info(f"B1 (hold labels) tuned on fold {f0.fold_id} inner split: {best}")
    logger.info(f"copy {tuned_f.name} into vanta.m3.B1_HOLD_PARAMS_BY_HORIZON[{h}] and commit before any --score run")


# ---------------------------------------------------------------------------------------------------------------- score (walk-forward)
def run_score(h: int, a) -> None:
    mech = a.mechanics_check
    try:
        m3.require_tuned_hold_params(h, mech)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    refuse_schedule_overrides(a, "score")
    out_dir = out_dir_for(a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{'MECHANICS_' if mech else ''}b1hold_h{h}"
    if (out_dir / f"scored_{name}.parquet").exists() and not a.force:
        logger.info(f"scored_{name}.parquet exists; skipping (--force to redo)")
        return
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    end = pd.Timestamp(a.end) if a.end else sessions[-1]
    cands = load_candidates(sessions)                          # keyed on the full window; labels are built on the FULL set (chunk reuse)
    if a.end:                                                  # a small check: label only what it needs (its own chunks)
        cands = cands[cands["decision_date"] <= end].reset_index(drop=True)
    df = _apply_impact(build_hold_labels(cands, h, sessions, load_terminal_values()), "floor")
    df = df[df["decision_date"] <= end].reset_index(drop=True)
    folds = _make_folds(sessions, h, df, a)
    logger.info(f"score h={h}: {len(df):,} candidates {df['decision_date'].min().date()}..{df['decision_date'].max().date()}; {len(folds)} folds, "
                f"first test {folds[0].test_start.date()}, embargo {folds[0].embargo} sessions; hold labels, impact floor")
    panel, cols = feature_panel(df, sessions, _cand_cache_name(sessions), out_dir, a.force)
    factory = run_baselines.mechanics_factory("b1") if mech else bl.b1_model_factory(**m3.hold_params(h))
    fit_fn = bl.make_fit_fn(panel, cols, factory, seeds=bl.B1_SEEDS, target="net")
    score_fn = bl.make_score_fn(panel, cols)
    labels = df[["symbol", "decision_date", "label_end_date", "y_net"]]
    result = wf.run_walkforward(folds, labels, sessions, fit_fn, score_fn)
    rep = bl.fold_report([dataclasses.replace(r, scored=r.scored.merge(df[["decision_date", "symbol", "fill_status"]],
                                                                         on=["decision_date", "symbol"], how="left")) for r in result.folds],
                         seeds=bl.B1_SEEDS)
    leak = wf.detect_leak(result, labels, horizon=h, step=run_baselines.STEP)
    win = bl.first_scored_and_judged_week(result.scored, m1.OOS_START)
    extra = [f"- features ({len(cols)}): {', '.join(cols)}", f"- seeds: {list(bl.B1_SEEDS)}; B1 hold params: {m3.hold_params(h)}",
             f"- labels: HOLD-TO-HORIZON net (entry next open, exit horizon-end close), impact floor; candidates {len(df):,}; "
             f"fill_status {df['fill_status'].value_counts().to_dict()}",
             f"- first scored week {win['first_scored_week']:%Y-%m-%d}; first judged week "
             + (f"{win['first_judged_week']:%Y-%m-%d}" if pd.notna(win["first_judged_week"]) else "NONE in this window")]
    prefix = run_baselines.leak_prefix(leak) + ("MECHANICS_" if mech else "")
    banner = ("MECHANICS CHECK ONLY (LightGBM stand-in): NOT a result and not usable by the M3 cells." if mech else "M3 B1 (hold labels) run.")
    run_baselines.write_outputs(out_dir, f"{prefix}b1hold_h{h}", banner, result, rep, leak, labels, df, h, extra)
    if run_baselines.leak_prefix(leak):
        logger.error(f"LEAK CHECK FLAGGED ({leak.reasons}; audit violations {leak.audit_violations}): outputs written with the LEAKFLAGGED_ "
                     "prefix; do not use them")
        raise SystemExit(2)


# ---------------------------------------------------------------------------------------------------------------- one cell
def load_b1_scores(h: int, out_dir: Path, mechanics: bool, end) -> pd.DataFrame:
    f = out_dir / scored_name(h, mechanics)
    if not f.exists():
        raise SystemExit(f"{f} is missing: run `python -m scripts.run_m3 --score --horizon {h}` first (a LEAKFLAGGED_ output is never used)")
    s = pd.read_parquet(f, columns=["decision_date", "symbol", "score"]).rename(columns={"score": "b1"})
    s["decision_date"] = pd.to_datetime(s["decision_date"]).astype("datetime64[ns]")
    if s.duplicated(["decision_date", "symbol"]).any():
        raise SystemExit(f"{f} holds duplicate (decision_date, symbol) rows; refusing to pick one")
    return s[s["decision_date"] <= end]


def _fingerprint(*files, extra=()) -> str:
    """Inputs of a cell: the scored file and the candidate cache (name, size, mtime), the code of vanta.m2 / vanta.m3 / this driver, and the
    run's own arguments."""
    h = hashlib.sha1()
    for f in files:
        f = Path(f)
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    for mod in (m2, m3):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(repr(extra).encode())
    return h.hexdigest()[:10]


cell_is_current = run_m2.cell_is_current


def run_cell(h: int, impact: str, *, end, tag: str, mechanics: bool, force: bool) -> None:
    out_dir = out_dir_for(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = stem(h, impact)
    meta_f = out_dir / f"meta_{st}.json"
    files = [out_dir / f"log_{st}_{arm}.parquet" for arm in LOG_ARMS]
    t0 = time.time()
    scored_f = out_dir / scored_name(h, mechanics)
    if not scored_f.exists():
        raise SystemExit(f"{scored_f} is missing: run `python -m scripts.run_m3 --score --horizon {h}` first")
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    explicit_end = end is not None
    end = pd.Timestamp(end) if end else sessions[-1]
    expected_fp = _fingerprint(scored_f, *sorted((config.PROCESSED_DATA_DIR / "m1b").glob(f"candidates_cliff{CLIFF}_*.parquet")),
                               extra=(h, impact, str(end.date()), mechanics, m3.hold_label_source_fp()))
    if not force and cell_is_current(meta_f, files, expected_fp):
        logger.info(f"{st}: outputs exist and match the current inputs/code ({meta_f.name}); skipping")
        return
    if meta_f.exists() and not force:
        logger.warning(f"{st}: stored outputs do not match the current inputs/code; rebuilding")
    cands = load_candidates(sessions)
    if explicit_end:                                                                # a small check: label only what it needs
        cands = cands[cands["decision_date"] <= end].reset_index(drop=True)
    lab = build_hold_labels(cands, h, sessions, load_terminal_values())             # the FULL set otherwise: reuses the --score chunks
    df = _apply_impact(lab[lab["decision_date"] <= end].reset_index(drop=True), impact)
    scored = load_b1_scores(h, out_dir, mechanics, end)
    logger.info(f"{st}: {len(df):,} candidates, {len(scored):,} B1-scored rows, {scored['decision_date'].nunique()} weeks")
    df["matured_at"] = m1.matured_at(df["label_end_date"], sessions)
    post = run_m1b.regime_posterior(h, 2015, end.year, end)
    df = run_m1b.attach_regime(df, post)
    df, n_unscored = run_m2.attach_scores(df, scored)
    n_nan = int((~np.isfinite(df["b1"].to_numpy(float))).sum())
    logger.info(f"{st}: candidates in scored weeks with no scored row {n_unscored:,}; with a NaN B1 score {n_nan:,} "
                f"({n_nan / max(1, len(df)):.2%} of {len(df):,}); both keep p = 1 and stay in the BH denominator")
    weeks = pd.DatetimeIndex(sorted(df[np.isfinite(df["b1"].to_numpy(float))]["decision_date"].unique()))
    win = m2.judged_window(df.assign(score=df["b1"]), m1.OOS_START)
    logger.info(f"{st}: {len(df):,} rows, {len(weeks)} scored weeks; first scored {win['first_scored_week']}, first judged {win['first_judged_week']}")
    logs = m2.run_m2_arms(df, weeks)
    extra = df[["decision_date", "symbol", "entry_date", "label_end_date"]]
    meta = dict(horizon=h, impact=impact, variant="hold", mechanics=mechanics, end=end, n_rows=len(df), n_scored_weeks=len(weeks),
                scored_weeks=[d.date().isoformat() for d in weeks], first_scored_week=win["first_scored_week"],
                first_judged_week=win["first_judged_week"], fingerprint=expected_fp, judged=m3.is_judged(impact, h),
                n_unscored_candidates=n_unscored, n_nan_score_candidates=n_nan, fill_status=df["fill_status"].value_counts().to_dict(),
                b1_hold_params=m3.hold_params(h), counts={}, judged_counts={})
    for arm in LOG_ARMS:
        lg = logs[arm].merge(extra, on=["decision_date", "symbol"], how="left")
        lg.assign(arm=arm).to_parquet(out_dir / f"log_{st}_{arm}.parquet")
        if arm != "ungated" and len(lg):
            meta["counts"][arm] = m2.arm_counts(lg)
            if pd.notna(win["first_judged_week"]):
                meta["judged_counts"][arm] = m2.arm_counts(m2.split_judged(lg, win["first_judged_week"])["judged"])
    meta_f.write_text(json.dumps(run_m2._json(meta), indent=1))                     # written last: its presence marks a finished cell
    logger.info(f"{st}: done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------------- report
def _load_cell(out_dir: Path, h: int, impact: str):
    st = stem(h, impact)
    mf = out_dir / f"meta_{st}.json"
    if not mf.exists():
        return None
    meta = json.loads(mf.read_text())
    logs = {}
    for arm in LOG_ARMS:
        lg = pd.read_parquet(out_dir / f"log_{st}_{arm}.parquet")
        for c in ("decision_date", "entry_date", "label_end_date"):
            lg[c] = pd.to_datetime(lg[c])
        logs[arm] = lg
    return meta, logs


def no_judged_week_lines(cells: dict) -> list[str]:
    """One explicit line per finished cell whose first judged week does not exist: never a silently absent row."""
    out = []
    for (h, imp), c in sorted(cells.items()):
        if c is not None and not c[0].get("first_judged_week"):
            kind = "judged" if m3.is_judged(imp, h) else "reported"
            out.append(f"- h={h} {imp}: NO JUDGED WEEK in this window (no scored week with {m3.WINDOW_WEEKS} weeks of scored history); "
                       f"no {kind} numbers for arms a, b, c and ungated.")
    return out


def _fmt(x, p=3):
    return run_m2._fmt(x, p)


def _tf(x):
    return run_m2._tf(x)


def _go_rows(tbl: pd.DataFrame, judged: bool) -> list[str]:
    L = ["| impact | arm | h | c1 inv | c2 FDR<=0.35 | FDR | c3 no yr>0.5 | worst yr | c4 pub>=20% | pub rate | c5 net>=ungated | diff (95% CI) | "
         + ("GO |" if judged else "status |"), "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.sort_values(["impact", "horizon", "arm"], ascending=[False, True, True]).iterrows():
        go = _tf(r["go_all"]) if judged else "NOT JUDGED"
        L.append(f"| {r['impact']} | {r['arm']} | {r['horizon']} | {_tf(r['c1'])} | {_tf(r['c2'])} | {_fmt(r['c2_fdr'])} | {_tf(r['c3'])} | "
                 f"{_fmt(r['c3_worst_year_fdr'])} | {_tf(r['c4'])} | {_fmt(r['c4_publication_rate'])} | {_tf(r['c5'])} | "
                 f"{_fmt(r['c5_diff'], 4)} ({_fmt(r['c5_ci_lo'], 4)}, {_fmt(r['c5_ci_hi'], 4)}) | {go} |")
    return L


def _comparator_rows(tbl: pd.DataFrame) -> list[str]:
    L = ["| impact | arm | h | net/signal | ungated top-10 net/signal | ungated FDR | NW SE (weekly means, lag ceil(h/5)) | block-bootstrap 95% CI | "
         "filled | NO_FILL | NO_DATA | judged weeks |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.sort_values(["impact", "horizon", "arm"], ascending=[False, True, True]).iterrows():
        L.append(f"| {r['impact']} | {r['arm']} | {r['horizon']} | {_fmt(r['mean_y_net'], 4)} | {_fmt(r['ungated_mean_y_net'], 4)} | "
                 f"{_fmt(r['ungated_fdr'])} | {_fmt(r['nw_se'], 4)} (lag {r['nw_lag_weeks']}) | ({_fmt(r['boot_lo'], 4)}, {_fmt(r['boot_hi'], 4)}) | "
                 f"{r['n_filled']} | {r['n_no_fill']} | {r['n_no_data']} | {r['n_judged_weeks']} |")
    return L


def report(tag: str, n_boot: int, invariants: tuple[bool, str] | None = None) -> str:
    out_dir = out_dir_for(tag)
    inv_ok, inv_line = invariants if invariants is not None else run_m2.run_invariants()
    cells = {(h, i): _load_cell(out_dir, h, i) for h in m3.ALL_HORIZONS for i in m3.IMPACTS}
    L = ["# M3 report (Stage 13M): the VANTA gate with B1 trained on hold-to-horizon net labels", "", m3.report_header(), "",
         "Pre-registration: docs/vanta/M3_PREREG.md (committed before any M3 training, tuning or run). Labels: hold-to-horizon net (entry at the "
         "next session's open, exit at the horizon-end close, no stop, no target, the Stage 13A cost model, the same impact floor 0.05% per side "
         "rank 1-200 and 0.20% rank 201-700; the liquidity cliff keeps only rank <= 200, so only the 0.05% tier applies). JUDGED: floor run at "
         "h = 20 and h = 60. REPORTED, NOT JUDGED: h = 5 and every zero-impact run. "
         f"q = {m3.Q}, tau = {m3.TAU_WEEKS} weeks, ESS floor {m3.ESS_FLOOR}, cap {m3.CAP}, calibration window {m3.WINDOW_WEEKS} weeks, "
         f"B1 seeds {list(m3.B1_SEEDS)}, Newey-West lag and bootstrap block = ceil(h/5) weeks (4 at h = 20, 12 at h = 60).", ""]
    missing = [f"h{h} {i}" for (h, i), c in cells.items() if c is None and m3.is_judged(i, h)]
    if missing:
        L += [f"**INCOMPLETE: judged cells not yet run: {', '.join(missing)}.**", ""]

    logs_t, wins, dds = {}, {}, {}
    for (h, imp), c in cells.items():
        if c is None:
            continue
        meta, lg = c
        wins[(h, imp)] = pd.Timestamp(meta["first_judged_week"]) if meta["first_judged_week"] else pd.NaT
        dds[(h, imp)] = pd.DatetimeIndex(pd.to_datetime(meta["scored_weeks"]))
    tables = []
    for (h, imp), c in cells.items():
        if c is None or pd.isna(wins[(h, imp)]):
            continue
        sub = {imp: {arm: {h: (c[1][arm], c[1]["ungated"])} for arm in m3.ARMS}}
        tables.append(m3.go_table(sub, {h: dds[(h, imp)]}, {h: wins[(h, imp)]}, inv_ok, n_boot=n_boot))
    cols = ["impact", "arm", "horizon", "judged", "go_all", "c2_fdr", "c2_threshold"]
    tbl = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=cols)
    jt = tbl[tbl["judged"].astype(bool)] if len(tbl) else tbl
    nt = tbl[~tbl["judged"].astype(bool)] if len(tbl) else tbl

    L += ["## Evaluation window", ""]
    for h in m3.ALL_HORIZONS:
        c = cells[(h, "floor")] or cells[(h, "zero")]
        if c is None:
            L.append(f"- h={h}: not run")
            continue
        key = (h, "floor") if cells[(h, "floor")] else (h, "zero")
        w = wins[key]
        L.append(f"- h={h}{'' if h in m3.JUDGED_HORIZONS else ' (NOT JUDGED)'}: first scored week {c[0]['first_scored_week']}; FIRST JUDGED WEEK "
                 + (f"{w:%Y-%m-%d}" if pd.notna(w) else "NONE")
                 + f" (first scored week >= {m1.OOS_START:%Y-%m-%d} with {m3.WINDOW_WEEKS} weeks of scored history behind it); {len(dds[key])} scored "
                 + (f"weeks, {int((dds[key] >= w).sum())} judged." if pd.notna(w) else "weeks.") + " Weeks before the first judged week are reported only.")
    nj = no_judged_week_lines(cells)
    if nj:
        L += ["", *nj]
    L += ["", f"Criterion 1 (invariants N1-N13): `python -m pytest tests/invariants -q` -> {'PASS' if inv_ok else 'FAIL'} ({inv_line}).", ""]

    L += ["## GO table: JUDGED cells (floor run; h = 20 and h = 60; arm x horizon against the five criteria)", ""]
    L += _go_rows(jt, True) if len(jt) else ["(no judged cell has been run)"]
    L += ["", "No multiplicity correction across arms and horizons; thresholds are identical for every arm; each row stands on its own numbers. "
          "Net return per published signal is the pooled mean y_net of FILLED published trades with a known outcome.", "",
          "### Net return per published signal, ungated comparator, Newey-West and block bootstrap (judged cells)", ""]
    L += _comparator_rows(jt) if len(jt) else ["(none)"]

    L += ["", "## NOT JUDGED: h = 5 (floor and zero) and the zero-impact runs at h = 20 and h = 60", "",
          "None of these rows enters a GO decision. They are shown with the same criteria columns only so the reader can see the numbers.", ""]
    L += _go_rows(nt, False) if len(nt) else ["(none run)"]
    L += ["", "### Net return per published signal, ungated comparator, Newey-West and block bootstrap (NOT JUDGED cells)", ""]
    L += _comparator_rows(nt) if len(nt) else ["(none)"]

    # ---- per-year FDR and net per signal
    L += ["", "## Per-year realised FDR and net return per published signal, every arm and the ungated comparator (floor run; judged weeks; NOT JUDGED)", ""]
    year_rows = []
    for h in m3.ALL_HORIZONS:
        c = cells[(h, "floor")]
        if c is None or pd.isna(wins[(h, "floor")]):
            continue
        L += [f"### h = {h}{'' if h in m3.JUDGED_HORIZONS else ' (NOT JUDGED)'}", "", "| year | " + " | ".join(
            f"{a} FDR | {a} net | {a} n" for a in LOG_ARMS) + " |", "|---|" + "---|---|---|" * len(LOG_ARMS)]
        yt = {a: m3.year_table(c[1][a][c[1][a]["decision_date"] >= wins[(h, "floor")]]).set_index("year") for a in LOG_ARMS}
        for y in sorted(set().union(*[set(t.index) for t in yt.values()])):
            cells_ = []
            for a in LOG_ARMS:
                if y in yt[a].index:
                    r = yt[a].loc[y]
                    cells_ += [_fmt(r["fdr"]), _fmt(r["net_per_signal"], 4), str(int(r["n_filled"]))]
                    year_rows.append(dict(horizon=h, impact="floor", arm=a, **{k: r[k] for k in yt[a].columns}, year=y))
                else:
                    cells_ += ["n/a", "n/a", "0"]
            L.append(f"| {y} | " + " | ".join(cells_) + " |")
        L.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(year_rows).to_csv(out_dir / "m3_year_table.csv", index=False)

    # ---- counts, publication by year
    L += ["## Fallbacks, abstentions, publication, NO_FILL / NO_DATA and unscored counts (judged weeks)", "",
          "| impact | arm | h | judged weeks | age-only fallback weeks (share) | state-missing weeks | abstention weeks |", "|---|---|---|---|---|---|---|"]
    for h in m3.ALL_HORIZONS:
        for imp in m3.IMPACTS:
            c = cells[(h, imp)]
            if c is None:
                continue
            for arm in m3.ARMS:
                jc = c[0]["judged_counts"].get(arm)
                if jc:
                    L.append(f"| {imp} | {arm} | {h} | {jc['n_weeks']} | {jc['n_age_only_fallback_weeks']} ({_fmt(jc['share_age_only_fallback'])}) | "
                             f"{jc['n_state_missing_weeks']} | {jc['n_abstain_weeks']} |")
    L += ["", "Publication rate by year (share of judged weeks with at least one published signal; floor run):", ""]
    for h in m3.ALL_HORIZONS:
        c = cells[(h, "floor")]
        if c is None or pd.isna(wins[(h, "floor")]):
            continue
        wk = dds[(h, "floor")][dds[(h, "floor")] >= wins[(h, "floor")]]
        for arm in LOG_ARMS:
            lg = c[1][arm][c[1][arm]["decision_date"] >= wins[(h, "floor")]]
            L.append(f"- h={h} {arm}: " + str({y: f'{v:.2f}' for y, v in m3.publication_rate_by_year(lg, wk).items()}))
    L += ["", "Candidate fill status, unscored and NaN-score candidates (all weeks of each cell):", ""]
    for h in m3.ALL_HORIZONS:
        for imp in m3.IMPACTS:
            c = cells[(h, imp)]
            if c is not None:
                L.append(f"- h={h} {imp}: fill_status {c[0]['fill_status']}; candidates in scored weeks with no scored row "
                         f"{c[0]['n_unscored_candidates']}; candidates with a NaN B1 score {c[0]['n_nan_score_candidates']}")

    # ---- 13b: Nifty 50
    L += ["", "## Against the Nifty 50 (floor run, judged weeks; NOT JUDGED)", "",
          "Loser = underperformed the Nifty 50 over the same holding window, after costs; the weekly false-discovery proportion is also "
          "split into up-weeks and down-weeks of the Nifty 50 weekly return.", ""]
    try:
        run_m1b.M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"
        nifty = run_m1b.load_nifty50_close()
    except Exception as e:                                               # noqa: BLE001 - a missing index file must not hide the rest
        nifty = None
        L.append(f"- Nifty 50 closes unavailable ({type(e).__name__}: {e}); not produced")
    if nifty is not None:
        for h in m3.ALL_HORIZONS:
            c = cells[(h, "floor")]
            if c is None or pd.isna(wins[(h, "floor")]):
                continue
            for arm in LOG_ARMS:
                lg = c[1][arm][c[1][arm]["decision_date"] >= wins[(h, "floor")]]
                r = m2.diagnostics_13b(lg, nifty, dds[(h, "floor")][dds[(h, "floor")] >= wins[(h, "floor")]])
                f, v = r["fdp_vs_market"], r["fdr_vs_index"]
                L.append(f"- h={h} {arm}: FDR vs Nifty 50 {_fmt(v['fdr'])} on {v['n_filled']} filled trades; corr(weekly FDP, market return) "
                         f"{_fmt(f['corr'])} over {f['n_weeks']} weeks; FDP in up weeks {_fmt(f['fdr_up'])} (n={f['n_up']}), down weeks "
                         f"{_fmt(f['fdr_down'])} (n={f['n_down']})")

    # ---- B1 per-fold IC
    L += ["", "## B1 (hold labels) rank IC and net per published signal per fold (reported)", "",
          "Parameters tuned once per horizon on fold 0's inner split only (hold labels, M2 grid and selection rule); h = 60 has its own tuning.", ""]
    for h in m3.ALL_HORIZONS:
        fr = out_dir / f"fold_report_b1hold_h{h}.json"
        L.append(f"- h={h}{'' if h in m3.JUDGED_HORIZONS else ' (NOT JUDGED)'}: B1 params {m3.hold_params(h)}"
                 + ("" if fr.exists() else f"; {fr.name} not found"))
        if fr.exists():
            for r in json.loads(fr.read_text()):
                L.append(f"  - fold {r['fold_id']} test {r['test_start']}..{r['test_end']}: rank IC {_fmt(r['rank_ic'], 4)} over {r['n_ic_dates']} "
                         f"dates; net/signal {_fmt(r['net_per_signal'], 4)} (published {r['n_published']}, filled {r['n_filled']}, "
                         f"NO_FILL {r['n_no_fill']}, NO_DATA {r['n_no_data']})")

    from gates.override_gates import BUY_SIGNAL_GATES, NOT_APPLIED_GATES
    L += ["", "## Gates applied and not applied", "", "- gates applied (as in M1b): " + ", ".join(BUY_SIGNAL_GATES),
          "- gates NOT APPLIED: " + "; ".join(f"{k} ({v})" for k, v in NOT_APPLIED_GATES.items()),
          "- candidates: every point-in-time universe name passing the ban, upper/lower circuit and liquidity-cliff (top 200 by trailing turnover) "
          "gates that week; a missing ban or circuit flag excludes the name. Gate score = weekly cross-sectional percentile of the B1 score "
          "(average ranks). Ungated comparator = the 10 highest B1 percentile each week.", "",
          "Arms: (a) age x regime = vanta.gate.run_gate; (b) regime-conditional calibration (same argmax filtered state, age-only weights, "
          "abstain if Kish ESS < 200; missing posterior at T -> (a)'s age-only calibration, flagged); (c) arm (a) plus abstain when the "
          "filtered (forward-algorithm) P(state 2) at T >= 0.5 (missing -> as (a), flagged). k_s and k_t are not used (no levels).", "",
          m3.report_footer()]
    text = "\n".join(L) + "\n"
    tbl.to_csv(out_dir / "m3_go_table.csv", index=False)
    return text


def main() -> None:
    a = build_parser().parse_args()
    _logger(a.tag)
    if a.report:
        text = report(a.tag, a.n_boot)
        path = REPORT_PATH if not a.tag else config.LOG_DIR / f"m3_report_{a.tag}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info(f"wrote {path}")
        return
    if a.horizon is None:
        raise SystemExit("--horizon is required (or --report)")
    if a.tune and a.score:
        raise SystemExit("--tune and --score are separate stages")
    if a.tune:
        if a.mechanics_check:
            raise SystemExit("--mechanics-check cannot be combined with --tune: the tuned parameters are CatBoost's")
        run_tune(a.horizon, a)
        return
    if a.score:
        run_score(a.horizon, a)
        return
    if a.impact is None:
        raise SystemExit("--impact is required for a cell (or use --tune / --score / --report)")
    if a.end and not a.mechanics_check:
        raise SystemExit("--end may be used only with --mechanics-check (the judged window is the whole window)")
    run_cell(a.horizon, a.impact, end=a.end, tag=a.tag, mechanics=a.mechanics_check, force=a.force)


if __name__ == "__main__":
    main()
