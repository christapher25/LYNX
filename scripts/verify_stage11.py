"""
scripts/verify_stage11.py

Stage 11 checkpoint: the India hard gates (gates/override_gates.py) fire correctly on known historical examples of REAL data.
Expectations are computed from the RAW ban lists (data/raw/fno_ban) and RAW price-band hit lists (data/raw/price_band_hits),
independently of the Stage 10 feature blocks the gates read.

  1. RECENT WINDOW      every universe symbol x the last N sessions: gate verdicts equal the verdicts derived straight from the raw files
                        (0 mismatches), and both BLOCK and PASS are non-empty (the check is not vacuous).
  2. BAN HISTORY        spread-out dates across 2008-2026: every symbol on that day's raw ban list gets fno_ban=BLOCK; a published NIL day
                        gives PASS (not UNAVAILABLE) to an F&O symbol.
  3. NO USABLE LIST     dates with no usable ban list (before 2008-01-01, 2008-11-28, 2014-02-19, 2014-08-25): UNAVAILABLE, not allowed.
                        The absence is itself re-derived from the raw files (no file, or a file carrying another day's date).
  4. CIRCUIT HISTORY    real upper and lower hits from 2010 on give BLOCK; a stock absent from a published list gives PASS; sessions before
                        the first hit list (2010-01-04) are UNAVAILABLE.
  5. MUTATION           the same real rows run through a gate that zero-fills NaN / ignores flags would PASS the unavailable days; proves
                        checks 3 and 4 can fail.

Run:  python -m scripts.verify_stage11 [--window 15]       Exit 0 = all pass.   Runtime: ~1 min, small memory (feature Parquets are
read column- and date-filtered).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CIRCUIT_FEATURES_DIR, FNO_BAN_DIR, PRICE_BAND_HITS_DIR
from gates.io import load_gate_inputs
from gates.override_gates import BLOCK, PASS, UNAVAILABLE, GateConfig, evaluate_gates

ALL = GateConfig(enabled=("upper_circuit", "lower_circuit", "fno_ban"))
NO_LIST_DAYS = ["2007-06-01", "2008-11-28", "2014-02-19", "2014-08-25"]     # CLAUDE.md F&O ban notes, plus a pre-archive day
FIRST_HIT_LIST = pd.Timestamp("2010-01-04")
FNO_PROBES = ["RELIANCE", "SBIN"]                                            # F&O members since before 2008, with rows on every probe day

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        _failures.append(name)


# --- raw readers (no dependency on features/blocks) --------------------------------------------------------------------

def raw_ban_day(day: pd.Timestamp) -> tuple[bool, set[str]]:
    """(usable, banned symbols): usable iff a file exists whose rows carry DATE == day. A NIL day is usable with an empty set."""
    path = FNO_BAN_DIR / f"{day.date()}.parquet"
    if not path.exists():
        return False, set()
    frame = pd.read_parquet(path)
    if frame.empty or not (pd.to_datetime(frame["DATE"]).dt.normalize() == day).all():
        return False, set()
    return True, set(frame["SYMBOL"].dropna())


def raw_hit_day(day: pd.Timestamp) -> tuple[bool, set[str], set[str]]:
    """(published, EQ upper hitters, EQ lower hitters) from the bh file whose rows carry DATE == day."""
    path = PRICE_BAND_HITS_DIR / f"{day.date()}.parquet"
    if not path.exists():
        return False, set(), set()
    frame = pd.read_parquet(path)
    if frame.empty or not (pd.to_datetime(frame["DATE"]).dt.normalize() == day).all():
        return False, set(), set()
    eq = frame[(frame["SERIES"] == "EQ") & frame["SYMBOL"].notna()]
    return True, set(eq.loc[eq["HIT"] == "H", "SYMBOL"]), set(eq.loc[eq["HIT"] == "L", "SYMBOL"])


def _gate_rows(symbols, day: pd.Timestamp) -> pd.DataFrame:
    frame = load_gate_inputs(symbols, day, day)
    return frame.merge(evaluate_gates(frame, ALL), left_index=True, right_index=True)


def _gate_rows_present(symbols, day: pd.Timestamp) -> pd.DataFrame:
    """Gate rows only for symbols whose circuits file really holds `day` (a symbol with no row is UNAVAILABLE by design, not a probe)."""
    rows = _gate_rows(symbols, day)
    held = [s for s in rows["symbol"] if (CIRCUIT_FEATURES_DIR / f"{s}.parquet").exists()
            and len(pd.read_parquet(CIRCUIT_FEATURES_DIR / f"{s}.parquet", columns=["date"], filters=[("date", "==", day)])) > 0]
    return rows[rows["symbol"].isin(held)].reset_index(drop=True)


def _zero_fill_mutant(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_gates(frame.assign(hit_upper_circuit=frame["hit_upper_circuit"].fillna(0.0), hit_lower_circuit=frame["hit_lower_circuit"].fillna(0.0),
                                       fno_in_ban=frame["fno_in_ban"].fillna(0.0), is_missing_circuit_hits=False, is_missing_ban=False), ALL)


# --- checks ------------------------------------------------------------------------------------------------------------

def check_recent_window(n: int) -> None:
    symbols = sorted(p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet"))
    # the window comes from the data: the latest n sessions of a probe symbol's circuits file (no hard-coded start date to rot)
    probe = pd.read_parquet(CIRCUIT_FEATURES_DIR / f"{FNO_PROBES[0]}.parquet", columns=["date"])["date"]
    days = sorted(pd.to_datetime(probe).unique())[-n:]
    lo = pd.Timestamp(days[0])
    present = set()                                  # (symbol, date) pairs the circuits block really holds; an absent pair is UNAVAILABLE by design
    for sym_ in symbols:
        got = pd.read_parquet(CIRCUIT_FEATURES_DIR / f"{sym_}.parquet", columns=["date"], filters=[("date", ">=", lo)])["date"]
        present.update((sym_, pd.Timestamp(x)) for x in got)
    frame = load_gate_inputs(symbols, dates=days)
    absent = [(s_, d_) for s_, d_ in zip(frame["symbol"], frame["date"]) if (s_, d_) not in present]
    all_absent_unavailable = bool(frame.loc[[(s_, d_) in set(absent) for s_, d_ in zip(frame["symbol"], frame["date"])], ["is_missing_circuit_hits", "is_missing_ban"]].all().all())
    check("absent (symbol, session) pairs come back flagged missing", all_absent_unavailable, f"{len(absent):,} absent pairs in the window")
    frame = frame[[(s_, d_) in present for s_, d_ in zip(frame["symbol"], frame["date"])]].reset_index(drop=True)
    verdict = evaluate_gates(frame, ALL)
    mismatches = blocks = passes = 0
    for day in days:
        day = pd.Timestamp(day)
        ban_ok, banned = raw_ban_day(day)
        hit_ok, upper, lower = raw_hit_day(day)
        rows = frame["date"] == day
        sym = frame.loc[rows, "symbol"]
        expect_up = pd.Series(["UNAVAILABLE" if not hit_ok else BLOCK if s in upper else PASS for s in sym], index=sym.index)
        expect_lo = pd.Series(["UNAVAILABLE" if not hit_ok else BLOCK if s in lower else PASS for s in sym], index=sym.index)
        expect_ban = pd.Series([UNAVAILABLE if not ban_ok else BLOCK if s in banned else PASS for s in sym], index=sym.index)
        for column, expected in (("gate_upper_circuit", expect_up), ("gate_lower_circuit", expect_lo), ("gate_fno_ban", expect_ban)):
            mismatches += int((verdict.loc[rows, column] != expected).sum())
    blocks, passes = int((verdict["gate_status"] == BLOCK).sum()), int((verdict["gate_status"] == PASS).sum())
    check("recent window equals raw-file derivation", mismatches == 0 and blocks > 0 and passes > 0,
          f"{len(days)} sessions x {frame['symbol'].nunique():,} symbols = {len(frame):,} rows; mismatches={mismatches}; BLOCK={blocks:,} PASS={passes:,}")


def check_ban_history() -> None:
    dated = sorted(p.stem for p in FNO_BAN_DIR.glob("*.parquet"))
    picks = [d for d in dated[::max(1, len(dated) // 10)] if d >= "2008-01-01"]
    total = missed = 0
    for d in picks:
        day = pd.Timestamp(d)
        usable, banned = raw_ban_day(day)
        if not usable or not banned:
            continue
        rows = _gate_rows_present(sorted(banned), day)
        got = set(rows.loc[rows["gate_fno_ban"] == BLOCK, "symbol"])
        total += len(banned)
        missed += len(banned & set(rows["symbol"]) - got)
        if not (banned & set(rows["symbol"])):
            missed += 1                                                     # a banned list with no joinable symbol proves nothing
    universe = sorted(p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet"))
    n_pass = false_block = bad_unavail = 0
    for d in picks:
        day = pd.Timestamp(d)
        usable, banned = raw_ban_day(day)
        if not usable:
            continue
        clear = [s for s in universe if s not in banned][::max(1, len(universe) // 60)]
        rows = _gate_rows(clear, day)
        assert len(rows) == len(clear), "loader must return a row for every requested symbol"
        n_pass += int((rows["gate_fno_ban"] == PASS).sum())
        false_block += int((rows["gate_fno_ban"] == BLOCK).sum())
        bad_unavail += int(((rows["gate_fno_ban"] == UNAVAILABLE) & ~rows["is_missing_ban"]).sum())
    check("symbols not on a usable raw ban list are never BLOCKed", n_pass > 0 and false_block == 0 and bad_unavail == 0,
          f"PASS={n_pass} false_BLOCK={false_block} UNAVAILABLE-without-missing-flag={bad_unavail}")
    check("every raw-banned symbol is BLOCKed", total > 0 and missed == 0, f"{len(picks)} dates sampled, {total} banned (day, symbol) pairs, missed={missed}")
    nil = next((d for d in dated if d >= "2020-01-01" and not (pd.read_parquet(FNO_BAN_DIR / f"{d}.parquet")["SYMBOL"].notna().any())), None)
    if nil is None:
        check("NIL ban day gives PASS, not UNAVAILABLE", False, "no NIL day found")
        return
    rows = _gate_rows(FNO_PROBES, pd.Timestamp(nil))
    check("NIL ban day gives PASS, not UNAVAILABLE", len(rows) == len(FNO_PROBES) and (rows["gate_fno_ban"] == PASS).all(),
          f"{nil}: {dict(zip(rows['symbol'], rows['gate_fno_ban']))}")


def check_no_usable_list() -> None:
    bad = []
    for d in NO_LIST_DAYS:
        day = pd.Timestamp(d)
        usable, _ = raw_ban_day(day)
        rows = _gate_rows(FNO_PROBES, day)
        derived_absent = not usable
        ok = derived_absent and len(rows) == len(FNO_PROBES) and (rows["gate_fno_ban"] == UNAVAILABLE).all() and not rows["gate_allowed"].any()
        if not ok:
            bad.append((d, derived_absent, list(rows["gate_fno_ban"])))
    check("days with no usable ban list are UNAVAILABLE and not allowed", not bad, f"{NO_LIST_DAYS}; failures={bad}")


def check_circuit_history() -> None:
    files = sorted(PRICE_BAND_HITS_DIR.glob("*.parquet"))
    picks = [p for p in files[::max(1, len(files) // 8)] if p.stem >= "2010-01-04"]
    n_up = n_lo = n_clear = bad = 0
    for path in picks:
        day = pd.Timestamp(path.stem)
        ok, upper, lower = raw_hit_day(day)
        if not ok:
            continue
        sample = sorted(upper)[:8] + sorted(lower)[:8]
        if not sample:
            continue
        # a few symbols in the feature universe that the list does not name
        universe = {p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet")}
        clear = sorted(universe - upper - lower)[:5]
        rows = _gate_rows_present(sorted(set(sample) & universe) + clear, day)
        for _, r in rows.iterrows():
            if r["symbol"] in upper:
                n_up += 1
                bad += r["gate_upper_circuit"] != BLOCK
            if r["symbol"] in lower:
                n_lo += 1
                bad += r["gate_lower_circuit"] != BLOCK
            if r["symbol"] in clear:
                n_clear += 1
                bad += (r["gate_upper_circuit"], r["gate_lower_circuit"]) != (PASS, PASS)
    check("real upper/lower hits BLOCK, unnamed stocks PASS", n_up > 0 and n_lo > 0 and n_clear > 0 and bad == 0,
          f"{len(picks)} hit-list dates; upper={n_up} lower={n_lo} clear={n_clear} violations={bad}")
    before = _gate_rows(FNO_PROBES, pd.Timestamp("2009-12-31"))
    check("session before the first hit list is UNAVAILABLE", len(before) == len(FNO_PROBES) and
          (before["gate_upper_circuit"] == UNAVAILABLE).all() and not before["gate_allowed"].any() and not (PRICE_BAND_HITS_DIR / "2009-12-31.parquet").exists(),
          f"first list on disk is {files[0].stem}")


def check_mutation() -> None:
    days = [pd.Timestamp(d) for d in NO_LIST_DAYS + ["2009-12-31"]]
    frame = pd.concat([load_gate_inputs(FNO_PROBES, d, d) for d in days], ignore_index=True)
    real_pass = int((evaluate_gates(frame, ALL)["gate_status"] == PASS).sum())
    mutant_pass = int((_zero_fill_mutant(frame)["gate_status"] == PASS).sum())
    check("mutation: zero-fill gate would PASS the unavailable days", real_pass == 0 and mutant_pass > 0, f"real PASS={real_pass}, zero-fill mutant PASS={mutant_pass} of {len(frame)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="LYNX Stage 11 checkpoint: India hard gates")
    parser.add_argument("--window", type=int, default=15, help="recent sessions checked exhaustively against the raw files")
    args = parser.parse_args()
    check_recent_window(args.window)
    check_ban_history()
    check_no_usable_list()
    check_circuit_history()
    check_mutation()
    if _failures:
        logger.error(f"Stage 11 verification FAILED: {_failures}")
        return 1
    logger.success("Stage 11 verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
