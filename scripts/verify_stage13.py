"""
scripts/verify_stage13.py

Stage 13 checkpoint: the walk-forward harness on a SAMPLE of real data (default 120 symbols, folds testing 2015-2017, horizon 20).
Demonstration scorer: 12-1 momentum from Stage 4 adjusted prices, label = vanta.labels NEL y_net (FILLED trades only). Sample scale:
nothing is materialised beyond the sampled labels; no job.

Configurations (all run through vanta.walkforward):
  CLEAN               embargo = embargo_for_horizon(20), purged fit rows, features as of the decision date.
  LEAK fit+test       the fit is a memoriser handed every row through test_end (fit_rows_policy="include_test").
  LEAK lookahead      features read as of decision date + 41 sessions (the 12-1 window then contains the label window).
  LEAK embargo 0      embargo 0 with the purge skipped (fit_rows_policy="unpurged"); the safe policy refuses (LeakageError).
                      Flagged by two rules (embargo below horizon, and the audit).
  LEAK unpurged       normal embargo 20 but rows NOT purged: the independent audit is the only rule that fires.
Checks
  1 CLEAN not flagged; its audit has 0 training rows inside the required embargo, on every fold.
  2 each LEAK configuration is flagged by detect_leak.
  3 a never-flag mutant FAILS detector_passes (which the real detector passes); 3b the audit rule alone fires for LEAK unpurged.
  5 store path on real files: no value available after the decision time, absent rows dropped not filled.
  4 embargo 0 under the purged policy raises LeakageError.
Prints per-config mean rank IC, t, permutation-placebo t, future-shifted-label IC, audit violations and the flag reasons.
Exit 0 if every check passes.

Usage:  python -m scripts.verify_stage13 [--symbols 120] [--seed 0] [--horizon 20]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CIRCUIT_FEATURES_DIR  # noqa: E402
from data.trading_calendar import trading_sessions  # noqa: E402
from scripts.report_nel_coverage import load_symbol, weekly_decision_dates  # noqa: E402
from vanta.labels import simulate_trade  # noqa: E402
from vanta import walkforward as wf  # noqa: E402

MOM_LONG, MOM_SKIP = 252, 21     # 12-1 momentum in sessions of the symbol's own bars
LEAD = 41                        # look-ahead leak: 21 (skip) + 20 (horizon)


def momentum(bars, as_of):
    """12-1 momentum as of `as_of`: close[p-21]/close[p-252]-1, p = the last bar on or before as_of. NaN if too short."""
    p = int(bars.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if p < MOM_LONG:
        return np.nan
    c = bars["close"].to_numpy(dtype=float)
    a, b = c[p - MOM_SKIP], c[p - MOM_LONG]
    return a / b - 1.0 if a > 0 and b > 0 else np.nan


def make_runners(bars_by_symbol, sessions):
    def feats(rows, reader):
        return np.array([momentum(bars_by_symbol[r.symbol], reader.as_of(r.decision_date)) for r in rows.itertuples()])

    def fit_clean(rows, fold, reader):          # train-fold-only winsorisation limits (2nd / 98th percentile) of the score
        v = feats(rows, reader)
        v = v[~np.isnan(v)]
        return (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (-np.inf, np.inf)

    def score_clean(model, rows, fold, reader):
        return np.clip(feats(rows, reader), *model)

    class Memo:
        def __init__(self, rows):
            self.t = {(r.symbol, r.decision_date): r.y_net for r in rows.itertuples()}

    def fit_memo(rows, fold, reader):
        return Memo(rows)

    def score_memo(model, rows, fold, reader):
        base = feats(rows, reader)
        return [model.t.get((r.symbol, r.decision_date), b) for r, b in zip(rows.itertuples(), base)]

    return fit_clean, score_clean, fit_memo, score_memo


def never_flags(result, labels_all, **kw):
    """Mutant detector: reports everything as clean."""
    return wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, 0)


def detector_passes(detect, clean_result, leak_results, labels, horizon, **kw) -> bool:
    """True iff `detect` leaves the clean run unflagged AND flags every leaky run. Fails for a detector that never flags."""
    return (not detect(clean_result, labels, horizon=horizon, **kw).flagged
            and all(detect(r, labels, horizon=horizon, **kw).flagged for r in leak_results))


def store_path_check(symbols, sessions) -> tuple[bool, str]:
    """Real-data read through PitReader (explicit before_nightly_run): no value is available after the decision time, and rows
    with no knowable data come back NaN with row_absent=True and are DROPPED by drop_absent, never filled."""
    blocks = ["delivery", "circuits", "fno_ban"]
    dates = ["2007-06-01", "2012-03-15", "2018-09-14", "2024-05-10", "2026-07-15", "2026-08-14", "2026-09-04", "2026-09-10",
             "2026-09-17", "2026-09-22"]
    syms = symbols[:10]
    prefix = {"delivery": ("delivery_pct", "delivery_qty"), "circuits": ("price_band", "hit_"), "fno_ban": ("fno_in", "fno_ban_")}
    n_absent = n_rows = n_avail = 0
    for before in (False, True):
        rd = wf.PitReader(sessions, before_nightly_run=before)
        for d in dates:
            d = pd.Timestamp(d)
            df = rd.read(syms, d, blocks, lookback_sessions=3)   # 3 sessions: lagged blocks (delivery) have a knowable earlier row
            gone = pd.Series(False, index=df.index)
            for b in blocks:
                av = pd.to_datetime(df[f"{b}_available_on"])
                bad = av.notna() & ((av >= d) if before else (av > d))
                if bad.any():
                    return False, f"{b} at {d.date()} before_nightly_run={before}: {int(bad.sum())} rows available after the decision"
                absent = df[f"{b}_row_absent"].astype(bool)
                vcols = [c for c in df.columns if c.startswith(prefix[b]) and not c.endswith(("_available_on", "_row_absent"))]
                if absent.any() and df.loc[absent, vcols].notna().any().any():
                    return False, f"{b} at {d.date()}: an absent row carries a value"
                gone |= absent
            kept = wf.drop_absent(df, blocks)
            if len(kept) != len(df) - int(gone.sum()) or kept[[f"{b}_row_absent" for b in blocks]].astype(bool).any().any():
                return False, f"drop_absent at {d.date()} kept an absent row"
            n_absent += int(gone.sum())
            n_avail += sum(int(df[f"{b}_available_on"].notna().sum()) for b in blocks)
            n_rows += len(df)
    return n_absent > 0 and n_avail > 0, (f"{n_rows} rows read, {n_avail} block-rows with an availability date checked, {n_absent} rows with an absent block dropped"
                                         + ("" if n_absent and n_avail else " (VACUOUS)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stdout, format="{message}")

    h = args.horizon
    all_sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    sessions = all_sessions[(all_sessions >= "2005-01-01") & (all_sessions <= "2018-12-31")]
    symbols = sorted(p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet"))
    symbols = random.Random(args.seed).sample(symbols, min(args.symbols, len(symbols)))
    dec_sessions = sessions[(sessions >= "2010-01-04")]           # hit lists start 2010-01-04: locks decidable
    rows, bars = [], {}
    wk = weekly_decision_dates(dec_sessions)                       # decisions 2010-2018 only; simulation uses the full calendar
    for sym in symbols:
        loaded = load_symbol(sym)
        if loaded is None:
            continue
        b, atr, hits, cov = loaded
        bars[sym] = b
        for t in wk.intersection(b.index):
            a = atr.get(t, np.nan)
            if not np.isnan(a):
                rows.append(simulate_trade(sym, b, t, h, 1.5, 2.0, float(a), hits, cov, sessions=all_sessions))
    labels = pd.DataFrame(rows)
    labels = labels[labels["fill_status"] == "FILLED"].reset_index(drop=True)
    logger.info(f"{len(bars)} symbols, {len(labels):,} FILLED weekly decisions 2010-2018, horizon {h}")

    folds = wf.make_folds(sessions, h, first_year=2010)
    logger.info("folds: " + "; ".join(f"f{f.fold_id} train {f.train_start.date()} val {f.val_start.date()} test {f.test_start.date()}.."
                                       f"{f.test_end.date()} embargo {f.embargo}" for f in folds))
    fit_clean, score_clean, fit_memo, score_memo = make_runners(bars, sessions)

    def reader_for(lead=0):
        return wf.PitReader(sessions, before_nightly_run=False, lead_sessions=lead)

    def run(fit, score, lead=0, **kw):
        rd = reader_for(lead)
        res = wf.run_walkforward(kw.pop("folds", folds), labels, sessions, lambda r, f: fit(r, f, rd),
                                 lambda m, rows, f, _rd: score(m, rows, f, rd), reader=rd, **kw)
        return res, rd

    ok = True
    rows_out = []
    runs = {}
    runs["CLEAN"] = run(fit_clean, score_clean)
    runs["LEAK fit+test"] = run(fit_memo, score_memo, fit_rows_policy="include_test")
    runs["LEAK lookahead"] = run(fit_clean, score_clean, lead=LEAD)
    zero = wf.make_folds(sessions, h, embargo=0, first_year=2010, allow_unsafe_embargo=True, val_years=0, include_val_in_fit=True)
    runs["LEAK embargo 0"] = run(fit_clean, score_clean, folds=zero, fit_rows_policy="unpurged")
    adj = wf.make_folds(sessions, h, first_year=2010, val_years=0, include_val_in_fit=True)     # normal embargo, rows unpurged
    runs["LEAK unpurged"] = run(fit_clean, score_clean, folds=adj, fit_rows_policy="unpurged")

    for name, (res, rd) in runs.items():
        rep = wf.detect_leak(res, labels, horizon=h, reader=rd)
        rows_out.append((name, rep))
        logger.info(f"{name:15s} IC {rep.mean_ic:+.4f} t {rep.t_stat:+.2f} | perm t {rep.perm_t:+.2f} | shifted IC {rep.shifted_mean_ic:+.4f} "
                    f"| audit viol {rep.audit_violations:6d} | flagged {rep.flagged} {rep.reasons}")
        logger.info("                per fold n_train " + ", ".join(str(f.n_train) for f in res.folds)
                    + " | per fold IC " + ", ".join(f"{wf.rank_ic_by_date(f.scored).mean():+.3f}" for f in res.folds))

    clean = dict(rows_out)["CLEAN"]
    c1 = (not clean.flagged) and all(f.audit_violations == 0 for f in runs["CLEAN"][0].folds)
    logger.info(f"[1] CLEAN not flagged, 0 audit violations on EACH fold ({[f.audit_violations for f in runs['CLEAN'][0].folds]}): "
                f"{'PASS' if c1 else 'FAIL'}")
    c2 = all(rep.flagged for n, rep in rows_out if n.startswith("LEAK"))
    for n, rep in rows_out:
        if n.startswith("LEAK"):
            logger.info(f"    {n} flagged: {rep.flagged}")
    logger.info(f"[2] every LEAK configuration flagged: {'PASS' if c2 else 'FAIL'}")
    leaks = [r for n, (r, _) in runs.items() if n.startswith("LEAK")]
    real_ok = detector_passes(wf.detect_leak, runs["CLEAN"][0], leaks, labels, h)
    mutant_fails = not detector_passes(never_flags, runs["CLEAN"][0], leaks, labels, h)
    logger.info(f"[3] real detector passes clean+leaky ({real_ok}); never-flag mutant FAILS the same check ({mutant_fails}): "
                f"{'PASS' if real_ok and mutant_fails else 'FAIL'}")
    mutant_misses = real_ok and mutant_fails
    only = dict(rows_out)["LEAK unpurged"]
    c_audit = len(only.reasons) == 1 and "N11" in only.reasons[0] and not only.config_embargo_short
    logger.info(f"[3b] LEAK unpurged (normal embargo) is flagged by the audit rule ALONE: {'PASS' if c_audit else 'FAIL'} {only.reasons}")
    c5, msg5 = store_path_check(symbols, all_sessions)
    logger.info(f"[5] store path (PitReader, explicit before_nightly_run, drop_absent): {'PASS' if c5 else 'FAIL'}: {msg5}")
    try:
        wf.run_walkforward(zero, labels, sessions, lambda r, f: None, lambda m, rows, f, rd: np.zeros(len(rows)))
        c4 = False
    except wf.LeakageError as e:
        c4 = True
        logger.info(f"    refused: {e}")
    logger.info(f"[4] embargo 0 under the purged policy raises LeakageError: {'PASS' if c4 else 'FAIL'}")
    ok = c1 and c2 and mutant_misses and c4 and c_audit and c5
    logger.info("VERIFY STAGE 13: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
