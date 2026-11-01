"""
Milestone M2 driver (Stage 13M, row 15d): the VANTA gate with the Stage 14 B1 score. Pre-registration: docs/vanta/M2_PREREG.md.

Nothing the pre-registration fixes can be changed from the command line (q, k, tau, cap, ESS floor, seeds, window, liquidity cliff,
arms): only the horizon, the impact mode and the label variant, plus bookkeeping options. Inputs are READ, never built here:
  * data/processed/baselines/scored_b1_h{h}.parquet   walk-forward out-of-sample B1 scores (scripts.run_baselines --arm b1)
  * data/processed/m1b/                               the M1b candidate table and NEL label chunks (scripts.run_m1b)

One run = one (horizon, impact, variant) and writes data/processed/m2/{log_h{h}_{impact}_{variant}_{arm}.parquet, meta_...json} for the
arms a, b, c and the ungated comparator; a finished run is skipped on re-run (--force rebuilds). `--report` assembles logs/m2_report.md
from the saved outputs (judged: floor / atr / h5, h20; diagnostics NOT JUDGED: A11 hold-to-horizon and h60).

Usage:
    python -m scripts.run_m2 --horizon 5 --impact floor                       # judged cell (variant atr)
    python -m scripts.run_m2 --horizon 20 --impact zero
    python -m scripts.run_m2 --horizon 5 --impact floor --variant hold        # A11 diagnostic
    python -m scripts.run_m2 --horizon 60 --impact floor --variant atr        # h60 diagnostic
    python -m scripts.run_m2 --report
    python -m scripts.run_m2 --horizon 5 --impact floor --mechanics-check --tag chk --end 2017-12-31   # reads baselines_chk MECHANICS scores

Realised FDR is an empirical number. q = 0.30 is an engineering target, not a guarantee.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from scripts import run_baselines, run_m1, run_m1b
from vanta import m1, m1b, m2
from vanta.labels import load_terminal_values

M2_DIR = config.PROCESSED_DATA_DIR / "m2"
LABEL_DIR = config.PROCESSED_DATA_DIR / "m2" / "labels_hold"          # shared by every tag: chunk keys carry their own inputs
BASELINES_DIR = config.PROCESSED_DATA_DIR / "baselines"
REPORT_PATH = config.LOG_DIR / "m2_report.md"
WINDOW_START = run_m1.WINDOW_START
CLIFF = m1.LIQUIDITY_CLIFF_RANK                                       # fixed by the pre-registration (top 200 by trailing turnover)
LOG_ARMS = (*m2.ARMS, "ungated")


def _logger(tag: str) -> None:
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / f"run_m2{('_' + tag) if tag else ''}.log", rotation="20 MB")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizon", type=int, choices=list(m2.ALL_HORIZONS), help="5, 20 (judged) or 60 (diagnostic, NOT JUDGED)")
    ap.add_argument("--impact", choices=["zero", "floor"], help="floor = the judged run; zero = reported")
    ap.add_argument("--variant", choices=list(m2.VARIANTS), default="atr", help="atr = ATR k=1.5 levels; hold = A11 hold-to-horizon (diagnostic)")
    ap.add_argument("--report", action="store_true", help="assemble logs/m2_report.md from the saved outputs and stop")
    ap.add_argument("--end", default=None, help="last decision date (small checks only; default: last session on disk)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="rebuild this cell's outputs (overwrites its own files; nothing is deleted)")
    ap.add_argument("--mechanics-check", action="store_true", help="read the MECHANICS_ B1 scores of a --tag baselines run: NOT a result")
    ap.add_argument("--tag", default="", help="suffix for the output (and baselines) directory; small checks never touch the full-run files")
    return ap


# ---------------------------------------------------------------------------------------------------------------- inputs
def _dirs(tag: str) -> tuple[Path, Path]:
    return (config.PROCESSED_DATA_DIR / ("m2" + (f"_{tag}" if tag else "")),
            config.PROCESSED_DATA_DIR / ("baselines" + (f"_{tag}" if tag else "")))


def _stem(h: int, impact: str, variant: str) -> str:
    return f"h{h}_{impact}_{variant}"


def load_candidates(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The M1b candidate table (cache; never built here), keyed on the full window exactly as scripts.run_baselines reads it."""
    dates = run_m1.weekly_dates(sessions, WINDOW_START, sessions[-1])
    cf = run_m1b.M1B_DIR / f"candidates_cliff{CLIFF}_{dates.min().date()}_{dates.max().date()}_{run_m1b._candidate_key(dates, CLIFF)}.parquet"
    cands = run_m1b._read_or_none(cf)
    if cands is None:
        raise SystemExit(f"candidate cache {cf} is missing or stale; run scripts/run_m1b.py first (this driver never builds candidates)")
    cands["decision_date"] = pd.to_datetime(cands["decision_date"]).astype("datetime64[ns]")
    return cands


def load_b1_scores(h: int, base_dir: Path, mechanics: bool, end) -> pd.DataFrame:
    name = f"scored_{'MECHANICS_' if mechanics else ''}b1_h{h}.parquet"
    f = base_dir / name
    if not f.exists():
        raise SystemExit(f"{f} is missing: run `python -m scripts.run_baselines --arm b1 --horizon {h}` first (a LEAKFLAGGED_ output is never used)")
    s = pd.read_parquet(f, columns=["decision_date", "symbol", "score"]).rename(columns={"score": "b1"})
    s["decision_date"] = pd.to_datetime(s["decision_date"]).astype("datetime64[ns]")
    if s.duplicated(["decision_date", "symbol"]).any():
        raise SystemExit(f"{f} holds duplicate (decision_date, symbol) rows; refusing to pick one")
    return s[s["decision_date"] <= end]


def hold_source_hash() -> str:
    """Hash of the hold-to-horizon label source (m2.py): an edit gives a new chunk directory, so stale chunks are never reused."""
    return hashlib.sha1(Path(m2.__file__).read_bytes()).hexdigest()[:10]


def hold_label_dir(h: int) -> Path:
    return LABEL_DIR / f"h{h}_cliff{CLIFF}_hold_{hold_source_hash()}"


@contextlib.contextmanager
def _hold_labels():
    """Make run_m1b.build_labels (same chunking, caching, calendar and terminal-value handling) simulate A11 hold-to-horizon labels."""
    orig = run_m1b.simulate_trade

    def hold(symbol, bars, t, horizon, k_s, k_t, atr, hits, cov, **kw):
        return m2.hold_to_horizon_label(symbol, bars, t, horizon, atr, hits, cov, **kw)
    run_m1b.simulate_trade = hold
    try:
        yield
    finally:
        run_m1b.simulate_trade = orig


def build_label_table(cands: pd.DataFrame, scored: pd.DataFrame, h: int, variant: str, sessions, terminal) -> pd.DataFrame:
    """NEL labels (variant atr, at the frozen k, from the cache the baselines driver reads) or A11 hold-to-horizon labels (variant hold,
    only for candidates that have a B1 score), zero impact; the impact mode is applied by the caller."""
    k_s, k_t = run_baselines.frozen_k(h)
    if variant == "atr":
        labels = run_m1b.build_labels(cands, h, k_s, k_t, run_m1b.M1B_DIR / "labels" / f"h{h}_cliff{CLIFF}_ks{k_s}_kt{k_t}", sessions, terminal)
        want = cands
    else:
        want = cands.merge(scored[["decision_date", "symbol"]], on=["decision_date", "symbol"], how="inner")
        with _hold_labels():
            labels = run_m1b.build_labels(want, h, k_s, k_t, hold_label_dir(h), sessions, terminal)
    df = want.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=h)
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def attach_scores(df: pd.DataFrame, scored: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Left-merge the B1 scores onto ALL gate-passing candidates, then keep the weeks that have at least one finite B1 score (the
    walk-forward test weeks). Within such a week a candidate with no scored row keeps b1 = NaN (gate: p = 1, still counted in m and absent
    from the percentile denominator). Returns (frame, number of such unscored candidates)."""
    out = df.merge(scored, on=["decision_date", "symbol"], how="left", validate="one_to_one")
    weeks = out.loc[np.isfinite(out["b1"].to_numpy(float)), "decision_date"].unique()
    out = out[out["decision_date"].isin(weeks)].reset_index(drop=True)
    key = scored.set_index(["decision_date", "symbol"]).index
    n_unscored = int((~pd.MultiIndex.from_frame(out[["decision_date", "symbol"]]).isin(key)).sum())
    return out, n_unscored


def _json(o):
    if isinstance(o, pd.Timestamp):
        return o.date().isoformat() if pd.notna(o) else None
    if o is pd.NaT:
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(x) for x in o]
    return o


def _fingerprint(*files, extra=()) -> str:
    """Inputs of a cell: the B1 scored file and the candidate cache (name, size, mtime), vanta/m2.py and this driver's source, and the
    run's own arguments."""
    h = hashlib.sha1()
    for f in files:
        f = Path(f)
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(Path(m2.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(repr(extra).encode())
    return h.hexdigest()[:10]


def cell_is_current(meta_f: Path, files, expected_fp: str) -> bool:
    """True iff the cell is finished (meta and every log exist) AND its stored fingerprint equals the recomputed one."""
    if not meta_f.exists() or not all(Path(f).exists() for f in files):
        return False
    try:
        return json.loads(meta_f.read_text()).get("fingerprint") == expected_fp
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------------------------------------------- one cell
def run_cell(h: int, impact: str, variant: str, *, end, tag: str, mechanics: bool, force: bool) -> None:
    out_dir, base_dir = _dirs(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(h, impact, variant)
    meta_f = out_dir / f"meta_{stem}.json"
    files = [out_dir / f"log_{stem}_{arm}.parquet" for arm in LOG_ARMS]
    t0 = time.time()
    scored_f = base_dir / f"scored_{'MECHANICS_' if mechanics else ''}b1_h{h}.parquet"
    if not scored_f.exists():
        raise SystemExit(f"{scored_f} is missing: run `python -m scripts.run_baselines --arm b1 --horizon {h}` first")
    run_m1b.M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"                  # read-only source of candidates / label chunks / regime cache
    run_m1b.M1B_DIR.mkdir(parents=True, exist_ok=True)
    run_m1b.FORCE = False
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    end = pd.Timestamp(end) if end else sessions[-1]
    expected_fp = _fingerprint(scored_f, *sorted(run_m1b.M1B_DIR.glob(f"candidates_cliff{CLIFF}_*.parquet")),
                               extra=(h, impact, variant, str(end.date()), mechanics))
    if not force and cell_is_current(meta_f, files, expected_fp):
        logger.info(f"{stem}: outputs exist and match the current inputs/code ({meta_f.name}); skipping")
        return
    if meta_f.exists() and not force:
        logger.warning(f"{stem}: stored outputs do not match the current inputs/code; rebuilding")
    cands = load_candidates(sessions)
    cands = cands[cands["decision_date"] <= end].reset_index(drop=True)
    scored = load_b1_scores(h, base_dir, mechanics, end)
    logger.info(f"{stem}: {len(cands):,} candidates, {len(scored):,} B1-scored rows, {scored['decision_date'].nunique()} weeks")
    df = build_label_table(cands, scored, h, variant, sessions, load_terminal_values())
    df = m1.apply_impact(df, df["liquidity_rank"], impact)
    df["matured_at"] = m1.matured_at(df["label_end_date"], sessions)
    post = run_m1b.regime_posterior(h, 2015, end.year, end)
    df = run_m1b.attach_regime(df, post)
    df, n_unscored = attach_scores(df, scored)
    n_nan = int((~np.isfinite(df["b1"].to_numpy(float))).sum())
    logger.info(f"{stem}: candidates in scored weeks with no scored row {n_unscored:,}; with a NaN B1 score {n_nan:,} "
                f"({n_nan / max(1, len(df)):.2%} of {len(df):,}); both keep p = 1 and stay in the BH denominator")
    wk = df[np.isfinite(df["b1"].to_numpy(float))]["decision_date"]
    weeks = pd.DatetimeIndex(sorted(wk.unique()))
    win = m2.judged_window(df.assign(score=df["b1"]), m1.OOS_START)
    logger.info(f"{stem}: {len(df):,} rows, {len(weeks)} scored weeks; first scored {win['first_scored_week']}, first judged "
                f"{win['first_judged_week']}")
    logs = m2.run_m2_arms(df, weeks)
    extra = df[["decision_date", "symbol", "entry_date", "label_end_date"]]
    meta = dict(horizon=h, impact=impact, variant=variant, mechanics=mechanics, end=end, n_rows=len(df), n_scored_weeks=len(weeks),
                scored_weeks=[d.date().isoformat() for d in weeks], first_scored_week=win["first_scored_week"],
                first_judged_week=win["first_judged_week"], fingerprint=expected_fp,
                n_unscored_candidates=n_unscored, n_nan_score_candidates=n_nan, fill_status=df["fill_status"].value_counts().to_dict(), counts={}, judged_counts={})
    for arm in LOG_ARMS:
        lg = logs[arm].merge(extra, on=["decision_date", "symbol"], how="left")
        lg.assign(arm=arm).to_parquet(out_dir / f"log_{stem}_{arm}.parquet")
        if arm != "ungated" and len(lg):
            meta["counts"][arm] = m2.arm_counts(lg)
            if pd.notna(win["first_judged_week"]):
                meta["judged_counts"][arm] = m2.arm_counts(m2.split_judged(lg, win["first_judged_week"])["judged"])
    meta_f.write_text(json.dumps(_json(meta), indent=1))                  # written last: its presence marks a finished cell
    logger.info(f"{stem}: done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------------- report
def _load_cell(out_dir: Path, h: int, impact: str, variant: str):
    stem = _stem(h, impact, variant)
    mf = out_dir / f"meta_{stem}.json"
    if not mf.exists():
        return None
    meta = json.loads(mf.read_text())
    logs = {}
    for arm in LOG_ARMS:
        lg = pd.read_parquet(out_dir / f"log_{stem}_{arm}.parquet")
        for c in ("decision_date", "entry_date", "label_end_date"):
            lg[c] = pd.to_datetime(lg[c])
        logs[arm] = lg
    return meta, logs


def no_judged_week_lines(cells: dict) -> list[str]:
    """One explicit line per finished cell whose first judged week does not exist (NaT/None): never a silently absent row."""
    out = []
    for (h, imp, v), c in sorted(cells.items()):
        if c is not None and not c[0].get("first_judged_week"):
            kind = "judged" if (h in m2.JUDGED_HORIZONS and imp == "floor" and v == "atr") else "reported"
            out.append(f"- h={h} {imp} {v}: NO JUDGED WEEK in this window (no scored week with {m2.WINDOW_WEEKS} weeks of scored history); "
                       f"no {kind} numbers for arms a, b, c and ungated.")
    return out


def run_invariants() -> tuple[bool, str]:
    """Criterion 1: the invariant suite on the code that produced the run (live)."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/invariants", "-q"], cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    return r.returncode == 0, tail[0]


def _fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{p}f}" if isinstance(x, (float, np.floating)) else str(x))


def _tf(x):
    return "n/a" if x is None or (not isinstance(x, (bool, np.bool_)) and pd.isna(x)) else ("PASS" if bool(x) else "FAIL")


def report(tag: str, n_boot: int, invariants: tuple[bool, str] | None = None) -> str:
    out_dir, base_dir = _dirs(tag)
    inv_ok, inv_line = invariants if invariants is not None else run_invariants()
    cells = {(h, i, v): _load_cell(out_dir, h, i, v) for h in m2.ALL_HORIZONS for i in m2.IMPACTS for v in m2.VARIANTS}
    L = [f"# M2 report (Stage 13M, row 15d): the VANTA gate with the B1 score", "", m2.report_header(), "",
         f"Pre-registration: docs/vanta/M2_PREREG.md (section 9 amendment of 2026-11-01). Floor run = judged (pessimistic impact floor "
         f"0.05% per side rank 1-200, 0.20% rank 201-700; the liquidity cliff keeps only rank <= 200, so only the 0.05% tier applies); "
         f"zero-impact run = reported, not judged. q = {m2.Q}, k_s = k_t = {m2.K_S}, tau = {m2.TAU_WEEKS} weeks, ESS floor {m2.ESS_FLOOR}, "
         f"cap {m2.CAP}, calibration window {m2.WINDOW_WEEKS} weeks, B1 seeds {list(m2.B1_SEEDS)}.", ""]
    missing = [f"h{h} {i} {v}" for (h, i, v), c in cells.items() if c is None and v == "atr" and h in m2.JUDGED_HORIZONS]
    if missing:
        L += [f"**INCOMPLETE: judged cells not yet run: {', '.join(missing)}.**", ""]

    # ---- judged table
    logs_t, wins, dds = {}, {}, {}
    for h in m2.JUDGED_HORIZONS:
        for imp in m2.IMPACTS:
            c = cells[(h, imp, "atr")]
            if c is None:
                continue
            meta, lg = c
            wins[h] = pd.Timestamp(meta["first_judged_week"]) if meta["first_judged_week"] else pd.NaT
            dds[h] = pd.DatetimeIndex(pd.to_datetime(meta["scored_weeks"]))
            for arm in m2.ARMS:
                logs_t.setdefault(imp, {}).setdefault(arm, {})[h] = (lg[arm], lg["ungated"])
    tables = []
    for h in m2.JUDGED_HORIZONS:
        if h in wins and pd.notna(wins[h]):
            sub = {imp: {arm: {h: per[h]} for arm, per in pa.items() if h in per} for imp, pa in logs_t.items()}
            tables.append(m2.go_table(sub, dds[h], wins[h], inv_ok, n_boot=n_boot))
    tbl = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["impact", "arm", "horizon", "judged", "go_all", "c2_fdr", "c2_threshold"])
    L += ["## Evaluation window", ""]
    for h in m2.JUDGED_HORIZONS:
        if h in wins:
            meta = cells[(h, "floor", "atr")][0] if cells[(h, "floor", "atr")] else cells[(h, "zero", "atr")][0]
            L.append(f"- h={h}: first scored week {meta['first_scored_week']}; FIRST JUDGED WEEK {wins[h]:%Y-%m-%d} (first scored week >= "
                     f"{m1.OOS_START:%Y-%m-%d} with {m2.WINDOW_WEEKS} weeks of scored history behind it); {len(dds[h])} scored weeks, "
                     f"{int((dds[h] >= wins[h]).sum())} judged. Weeks before the first judged week are reported only.")
    nj = no_judged_week_lines(cells)
    if nj:
        L += ["", *nj]
    L += ["", f"Criterion 1 (invariants N1-N13): `python -m pytest tests/invariants -q` -> {'PASS' if inv_ok else 'FAIL'} ({inv_line}).", ""]

    L += ["## GO table: arm x horizon against the five criteria (floor run = judged; zero run = reported, not judged)", "",
          "| impact | arm | h | c1 inv | c2 FDR<=0.35 | FDR | c3 no yr>0.5 | worst yr | c4 pub>=20% | pub rate | c5 net>=ungated | diff (95% CI) | GO |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.sort_values(["impact", "arm", "horizon"], ascending=[False, True, True]).iterrows():
        go = _tf(r["go_all"]) if r["judged"] else "not judged"
        L.append(f"| {r['impact']} | {r['arm']} | {r['horizon']} | {_tf(r['c1'])} | {_tf(r['c2'])} | {_fmt(r['c2_fdr'])} | {_tf(r['c3'])} | "
                 f"{_fmt(r['c3_worst_year_fdr'])} | {_tf(r['c4'])} | {_fmt(r['c4_publication_rate'])} | {_tf(r['c5'])} | "
                 f"{_fmt(r['c5_diff'], 4)} ({_fmt(r['c5_ci_lo'], 4)}, {_fmt(r['c5_ci_hi'], 4)}) | {go} |")
    L += ["", "Thresholds are identical for every arm (no multiplicity correction across 3 arms x 2 horizons); each row stands on its own numbers. "
          "Net return per published signal is the pooled mean y_net of FILLED published trades with a known outcome.", "",
          "### Net return per published signal, comparators, Newey-West and block bootstrap (floor / zero)", "",
          "| impact | arm | h | net/signal | ungated top-10 net/signal | ungated FDR | NW SE (weekly means, lag ceil(h/5)) | block-bootstrap 95% CI | filled | NO_FILL | NO_DATA | judged weeks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.sort_values(["impact", "arm", "horizon"], ascending=[False, True, True]).iterrows():
        L.append(f"| {r['impact']} | {r['arm']} | {r['horizon']} | {_fmt(r['mean_y_net'], 4)} | {_fmt(r['ungated_mean_y_net'], 4)} | {_fmt(r['ungated_fdr'])} | "
                 f"{_fmt(r['nw_se'], 4)} (lag {r['nw_lag_weeks']}) | ({_fmt(r['boot_lo'], 4)}, {_fmt(r['boot_hi'], 4)}) | {r['n_filled']} | "
                 f"{r['n_no_fill']} | {r['n_no_data']} | {r['n_judged_weeks']} |")

    # ---- counts, publication by year
    L += ["", "## Fallbacks, abstentions and publication (floor run, judged weeks)", "",
          "| arm | h | judged weeks | age-only fallback weeks (share) | state-missing weeks | abstention weeks |", "|---|---|---|---|---|---|"]
    for h in m2.JUDGED_HORIZONS:
        c = cells[(h, "floor", "atr")]
        if c is None:
            continue
        for arm in m2.ARMS:
            jc = c[0]["judged_counts"].get(arm)
            if jc:
                L.append(f"| {arm} | {h} | {jc['n_weeks']} | {jc['n_age_only_fallback_weeks']} ({_fmt(jc['share_age_only_fallback'])}) | "
                         f"{jc['n_state_missing_weeks']} | {jc['n_abstain_weeks']} |")
    L += ["", "Publication rate by year (share of judged weeks with at least one published signal; floor run):", ""]
    for h in m2.JUDGED_HORIZONS:
        c = cells[(h, "floor", "atr")]
        if c is None:
            continue
        for arm in (*m2.ARMS, "ungated"):
            lg = c[1][arm]
            lg = lg[lg["decision_date"] >= wins[h]]
            wk = pd.DatetimeIndex(dds[h][dds[h] >= wins[h]])
            pub = pd.DatetimeIndex(lg[lg["published"].astype(bool)]["decision_date"].unique())
            by = {int(y): f"{len(pub[pub.year == y]) / max(1, int((wk.year == y).sum())):.2f}" for y in sorted(set(wk.year))}
            L.append(f"- h={h} {arm}: {by}")

    # ---- diagnostics
    L += ["", "## Diagnostics (NOT JUDGED; none of these enters a GO criterion)", "",
          "### A11 hold-to-horizon (no stop, no target, exit at the horizon-end close, same costs) next to ATR k = 1.5, and horizon 60 "
          "(floor run, judged-window weeks of each horizon; B1 score unchanged, trained on the ATR net labels). NOT JUDGED", "",
          "| variant | h | arm | FDR | net/signal | filled | NO_FILL | NO_DATA | weeks with a publication |", "|---|---|---|---|---|---|---|---|---|"]
    dlogs = {}
    for (h, imp, v), c in cells.items():
        if imp != "floor" or c is None:
            continue
        fjw = pd.Timestamp(c[0]["first_judged_week"]) if c[0]["first_judged_week"] else pd.NaT
        if pd.isna(fjw):
            continue
        for arm in LOG_ARMS:
            dlogs[(v, h, arm)] = c[1][arm][c[1][arm]["decision_date"] >= fjw]
    dt = m2.diagnostics_table(dlogs) if dlogs else pd.DataFrame()
    for _, r in dt.sort_values(["horizon", "variant", "arm"]).iterrows():
        L.append(f"| {r['variant']} | {r['horizon']} | {r['arm']} | {_fmt(r['fdr'])} | {_fmt(r['mean_y_net'], 4)} | {r['n_filled']} | "
                 f"{r['n_no_fill']} | {r['n_no_data']} | {r['n_weeks_published']} |")
    L += ["", "### 13b: weekly false-discovery proportion against the Nifty 50 weekly return, and FDR with loser = underperformed the Nifty 50 "
          "(floor / atr, judged weeks). NOT JUDGED", ""]
    try:
        run_m1b.M1B_DIR = config.PROCESSED_DATA_DIR / "m1b"
        nifty = run_m1b.load_nifty50_close()
    except Exception as e:                                               # noqa: BLE001 - a missing index file must not hide the rest
        nifty = None
        L.append(f"- Nifty 50 closes unavailable ({type(e).__name__}: {e}); 13b diagnostics not produced")
    if nifty is not None:
        for h in m2.JUDGED_HORIZONS:
            c = cells[(h, "floor", "atr")]
            if c is None:
                continue
            for arm in LOG_ARMS:
                lg = c[1][arm][c[1][arm]["decision_date"] >= wins[h]]
                r = m2.diagnostics_13b(lg, nifty, dds[h][dds[h] >= wins[h]])
                f, v = r["fdp_vs_market"], r["fdr_vs_index"]
                L.append(f"- h={h} {arm}: corr(weekly FDP, market return) {_fmt(f['corr'])} over {f['n_weeks']} weeks; FDP in up weeks "
                         f"{_fmt(f['fdr_up'])} (n={f['n_up']}), down weeks {_fmt(f['fdr_down'])} (n={f['n_down']}); FDR vs index {_fmt(v['fdr'])} "
                         f"on {v['n_filled']} filled trades. NOT JUDGED")

    # ---- B1 per-fold IC
    L += ["", "### B1 rank IC and net per published signal per fold (Stage 14, from the baselines driver; reported)", "",
          "Horizon 60 is a DIAGNOSTIC horizon: its B1 model reuses the h=20 tuned hyperparameter dict (lead decision, not separately tuned), "
          "with k_s = k_t = 1.5.", ""]
    for h in m2.ALL_HORIZONS:
        fr = base_dir / f"fold_report_b1_h{h}.json"
        if not fr.exists():
            L.append(f"- h={h}: {fr.name} not found")
            continue
        L.append(f"- h={h}{' (DIAGNOSTIC)' if h not in m2.JUDGED_HORIZONS else ''}:")
        for r in json.loads(fr.read_text()):
            L.append(f"  - fold {r['fold_id']} test {r['test_start']}..{r['test_end']}: rank IC {_fmt(r['rank_ic'], 4)} over {r['n_ic_dates']} dates; "
                     f"net/signal {_fmt(r['net_per_signal'], 4)} (published {r['n_published']}, filled {r['n_filled']}, NO_FILL {r['n_no_fill']})")

    from gates.override_gates import BUY_SIGNAL_GATES, NOT_APPLIED_GATES
    L += ["", "## Gates applied and not applied", "", "- gates applied (as in M1b): " + ", ".join(BUY_SIGNAL_GATES),
          "- gates NOT APPLIED: " + "; ".join(f"{k} ({v})" for k, v in NOT_APPLIED_GATES.items()),
          "- candidates: every point-in-time universe name passing the ban, upper/lower circuit and liquidity-cliff (top 200 by trailing turnover) "
          "gates that week; a missing ban or circuit flag excludes the name. Gate score = weekly cross-sectional percentile of the B1 score "
          "(average ranks). Ungated comparator = the 10 highest B1 percentile each week.", "",
          "Arms: (a) age x regime = vanta.gate.run_gate; (b) regime-conditional calibration (same argmax filtered state, age-only weights, "
          "abstain if Kish ESS < 200; missing posterior at T -> (a)'s age-only calibration, flagged); (c) arm (a) plus abstain when the "
          "filtered P(state 2) at T >= 0.5 (missing -> as (a), flagged).", "", m2.report_footer()]
    text = "\n".join(L) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "m2_go_table.csv", index=False)
    dt.to_csv(out_dir / "m2_diagnostics.csv", index=False)
    return text


def main() -> None:
    a = build_parser().parse_args()
    _logger(a.tag)
    if a.report:
        text = report(a.tag, a.n_boot)
        path = REPORT_PATH if not a.tag else config.LOG_DIR / f"m2_report_{a.tag}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info(f"wrote {path}")
        return
    if a.horizon is None or a.impact is None:
        raise SystemExit("--horizon and --impact are required (or --report)")
    run_cell(a.horizon, a.impact, a.variant, end=a.end, tag=a.tag, mechanics=a.mechanics_check, force=a.force)


if __name__ == "__main__":
    main()
