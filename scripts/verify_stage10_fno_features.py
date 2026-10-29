"""
Stage 10 (F&O feature block) checkpoint verification, against the real files on disk.

features/blocks/fno.py -> FNO_FEATURES_DIR. This is NOT scripts/verify_stage10_fno.py, which verifies the raw F&O ingestion.

Every check is a `*_violations(...) -> int` function. Each is run on the real output (must be 0) and, where a bug can be
injected, on an in-memory copy of real output with one bug deliberately injected (must be > 0), the same discipline as
scripts/verify_stage10_circuits.py. The independent recompute never imports compute_fno_features, summarise_fno_day or the
fno loaders: own raw readers, own plain-loop arithmetic on the raw bhavcopy / ban files and on Stage 4, so a bug shared by the
block and its checker cannot hide. The truncation check is the deliberate exception: it re-runs the REAL builder.

  1. Files: one per universe symbol (full run) with dates identical 1:1 to the symbol's Stage 4 rows; a scoped run
     (--features-dir) checks the files it finds against the same Stage 4 rows over their own date range.
  2. Structure: flags are non-null bool; is_missing_fno <=> every OI/rollover feature NaN; is_missing_ban <=> fno_in_ban NaN;
     fno_day_unavailable implies is_missing_fno and is_missing_opt; values in range; no inf.
  3. Calendar: a spine date with no F&O file is fno_day_unavailable for every symbol, and only for those dates; 2003-10-01
     (documented, unrecoverable) has no file and is flagged wherever it is in range.
  4. Ban availability, every row of every file: is_missing_ban <=> no ban file whose embedded DATE is that session; the
     documented no-list days (2008-11-28, 2014-02-19, 2014-08-25) and everything before 2008-01-01 are NaN, never 0.0;
     fno_in_ban agrees with the raw list.
  5. Independent recompute on sampled (symbol, date) cases, every feature and flag, including the trailing windows: forced
     dates (legacy->UDiFF cutover, a weekend session, documented gap days, an early year) plus random F&O days plus days
     that have bans. Includes symbols the F&O file does not name (must be flagged missing, not zero).
  6. Adjustment factor: on real ex-dates where the Stage 4 factor changes, NSE rescaled raw open interest overnight; the stored
     fut_oi_chg_1d must equal the factor-adjusted change and NOT the raw one.
  7. Truncation: cutting a symbol's spine AND the F&O/ban inputs at several dates never changes an earlier row (real builder).
  8. Injected bugs, each must be caught: zero-fill, stale carry-forward, ban unavailable as none, next-session ban,
     unadjusted OI at an ex-date, a cleared missing flag.

Usage:
    python -m scripts.verify_stage10_fno_features                       # the full block in FNO_FEATURES_DIR
    python -m scripts.verify_stage10_fno_features --features-dir DIR    # a scoped run written by fno.py --output-dir DIR
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import ADJUSTED_PRICES_DIR, FNO_BAN_DIR, FNO_BHAVCOPY_DIR, FNO_FEATURES_DIR
from features.blocks import fno
from features.blocks.fno import MIN_OBS_20D, OUTPUT_COLUMNS
from signals.indicators import list_universe_symbols

SEED = 20260930
SAMPLE_RANDOM_DAYS = 14
SAMPLE_SYMBOLS_PER_DAY = 25
SAMPLE_NON_FNO_SYMBOLS = 4
BAN_DAYS_SAMPLED = 6
EX_DATE_EVENTS = 40
FORCED_DATES = [
    "2005-06-15", "2008-11-28", "2013-10-09", "2014-02-19", "2014-08-25", "2021-03-30", "2024-07-05", "2024-07-08",
    "2025-02-01", "2003-10-01",
]
UNAVAILABLE_BAN_DAYS = ["2008-11-28", "2014-02-19", "2014-08-25"]
OI_FEATURES = ["fut_oi", "fut_oi_chg_1d", "fut_oi_chg_5d", "fut_oi_ratio_20d", "fut_roll_next_share", "fut_days_to_expiry"]
VALUE_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in ("symbol", "date")]
failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


def expect_caught(bug: str, violations: int) -> None:
    caught = violations > 0
    (logger.success if caught else logger.error)(f"{'CAUGHT ' if caught else 'MISSED '} injected bug: {bug} ({violations} violations)")
    if not caught:
        failures.append(f"injected bug not caught: {bug}")


def same(a, b) -> bool:
    a_nan = a is None or (isinstance(a, float) and np.isnan(a))
    b_nan = b is None or (isinstance(b, float) and np.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))


# --------------------------------------------------------------------------- independent raw readers ---------------------------

class Raw:
    """Own readers: Stage 4 spine and factor per symbol, one F&O day, and the ban archive keyed by EMBEDDED date."""

    def __init__(self) -> None:
        self.fno_files = {pd.Timestamp(p.stem) for p in FNO_BHAVCOPY_DIR.glob("*.parquet")}
        self._day: dict[pd.Timestamp, dict | None] = {}
        self._stage4: dict[str, tuple[list, dict]] = {}
        self.ban_days: dict[pd.Timestamp, set] = {}
        for path in sorted(FNO_BAN_DIR.glob("*.parquet")):
            frame = pd.read_parquet(path, columns=["DATE", "SYMBOL"])
            if frame.empty:
                continue
            when = pd.Timestamp(frame["DATE"].iloc[0]).normalize()
            names = self.ban_days.setdefault(when, set())
            for value in frame["SYMBOL"].tolist():
                if value is not None and not pd.isna(value) and str(value).strip():
                    names.add(str(value).strip())

    def stage4(self, symbol: str) -> tuple[list, dict]:
        """(spine dates ascending, {date: factor}); factor = ADJ_VOLUME / VOLUME, NaN where VOLUME is not positive."""
        if symbol not in self._stage4:
            frame = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE", "VOLUME", "ADJ_VOLUME"])
            frame = frame.sort_values("DATE")
            factor = {}
            for date, volume, adjusted in zip(frame["DATE"], frame["VOLUME"], frame["ADJ_VOLUME"]):
                factor[pd.Timestamp(date)] = adjusted / volume if volume > 0 else float("nan")
            self._stage4[symbol] = ([pd.Timestamp(d) for d in frame["DATE"]], factor)
        return self._stage4[symbol]

    def day(self, when: pd.Timestamp) -> dict | None:
        """None if no file. Else {symbol: dict(fut_raw, near, next, dte, put, call)} from plain loops over live contracts."""
        if when in self._day:
            return self._day[when]
        path = FNO_BHAVCOPY_DIR / f"{when:%Y-%m-%d}.parquet"
        if not path.exists():
            self._day[when] = None
            return None
        frame = pd.read_parquet(path, columns=["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "OPTION_TYPE", "OPEN_INT"])
        if len(frame) == 0:
            self._day[when] = None
            return None
        perf: dict[str, dict] = {}
        for symbol, instrument, expiry, option_type, oi in frame.itertuples(index=False):
            if instrument not in ("FUTSTK", "OPTSTK") or pd.Timestamp(expiry) < when:
                continue
            entry = perf.setdefault(symbol, {"fut": {}, "put": None, "call": None})
            value = None if pd.isna(oi) else float(oi)
            if instrument == "FUTSTK":
                key = pd.Timestamp(expiry)
                if key not in entry["fut"]:
                    entry["fut"][key] = None
                if value is not None:
                    entry["fut"][key] = (entry["fut"][key] or 0.0) + value
            else:
                side = "call" if option_type in ("CE", "CA") else "put" if option_type in ("PE", "PA") else None
                if side and value is not None:
                    entry[side] = (entry[side] or 0.0) + value
        out = {}
        for symbol, entry in perf.items():
            expiries = sorted(entry["fut"])
            levels = [entry["fut"][e] for e in expiries if entry["fut"][e] is not None]
            out[symbol] = {
                "fut_raw": sum(levels) if levels else None,
                "near": entry["fut"][expiries[0]] if expiries else None,
                "next": entry["fut"][expiries[1]] if len(expiries) > 1 else None,
                "dte": float((expiries[0] - when).days) if expiries else None,
                "put": entry["put"], "call": entry["call"],
            }
        self._day[when] = out
        return out


def adjusted_oi(raw: Raw, symbol: str, when: pd.Timestamp) -> float:
    """Factor-adjusted futures OI for (symbol, day) from the raw files, NaN when unusable."""
    day = raw.day(when)
    factor = raw.stage4(symbol)[1].get(when, float("nan"))
    if day is None or symbol not in day or day[symbol]["fut_raw"] is None or not factor > 0:
        return float("nan")
    return day[symbol]["fut_raw"] * factor


def expected_row(raw: Raw, symbol: str, when: pd.Timestamp) -> dict:
    """Every output value for (symbol, date), recomputed from raw files with plain loops."""
    spine, factors = raw.stage4(symbol)
    i = spine.index(when)
    day = raw.day(when)
    entry = day.get(symbol) if day else None
    factor = factors[when]
    usable = day is not None and entry is not None and entry["fut_raw"] is not None and factor > 0

    oi = [adjusted_oi(raw, symbol, spine[j]) if j >= 0 else float("nan") for j in range(i - 19, i + 1)]   # oldest .. today
    today = oi[-1]

    def change(k):
        if i < k:
            return float("nan")
        prior = oi[-1 - k] if k <= 19 else adjusted_oi(raw, symbol, spine[i - k])
        return today / prior - 1 if today == today and prior == prior and prior > 0 else float("nan")

    seen = [v for v in oi if v == v]
    mean = sum(seen) / len(seen) if len(seen) >= MIN_OBS_20D else float("nan")
    ratio = today / mean if today == today and mean == mean and mean > 0 else float("nan")

    roll = float("nan")
    if usable and entry["near"] is not None and entry["next"] is not None and entry["near"] + entry["next"] > 0:
        roll = entry["next"] / (entry["near"] + entry["next"])
    opt_ok = day is not None and entry is not None and entry["put"] is not None and entry["call"] is not None
    pcr = entry["put"] / entry["call"] if opt_ok and entry["call"] > 0 else float("nan")

    def banned(j):
        stamp = spine[j]
        return float(symbol in raw.ban_days[stamp]) if stamp in raw.ban_days else float("nan")

    ban_today = banned(i)
    ban_prev = banned(i - 1) if i >= 1 else float("nan")
    entry_flag = float("nan")
    if ban_today == ban_today and ban_prev == ban_prev:
        entry_flag = float(ban_today == 1.0 and ban_prev == 0.0)
    window = [banned(j) for j in range(max(0, i - 19), i + 1)]
    window = [v for v in window if v == v]
    return {
        "fut_oi": today, "fut_oi_chg_1d": change(1), "fut_oi_chg_5d": change(5), "fut_oi_ratio_20d": ratio,
        "fut_roll_next_share": roll, "fut_days_to_expiry": entry["dte"] if usable else float("nan"),
        "opt_pcr_oi": pcr, "fno_in_ban": ban_today, "fno_ban_new_entry": entry_flag,
        "fno_ban_days_20d": float(sum(window)) if len(window) >= MIN_OBS_20D else float("nan"),
        "is_missing_fno": not usable, "fno_day_unavailable": day is None, "is_missing_opt": not opt_ok,
        "is_missing_ban": when not in raw.ban_days,
    }


# --------------------------------------------------------------------------- violation counters --------------------------------

def structure_violations(frame: pd.DataFrame) -> int:
    bad = 0
    for column in fno.FLAG_COLUMNS:
        bad += int(frame[column].isna().sum()) + int(frame[column].dtype != bool)
    flagged = frame["is_missing_fno"].to_numpy()
    bad += int(frame.loc[flagged, OI_FEATURES].notna().to_numpy().sum())
    bad += int(frame.loc[~flagged, "fut_oi"].isna().sum())
    bad += int(frame.loc[frame["is_missing_opt"], "opt_pcr_oi"].notna().sum())
    ban_missing = frame["is_missing_ban"].to_numpy()
    bad += int(frame.loc[ban_missing, ["fno_in_ban", "fno_ban_new_entry"]].notna().to_numpy().sum())
    bad += int(frame.loc[~ban_missing, "fno_in_ban"].isna().sum())
    bad += int((~frame["fno_in_ban"].dropna().isin([0.0, 1.0])).sum())
    bad += int(frame["fno_ban_new_entry"].dropna().isin([0.0, 1.0]).eq(False).sum())
    outage = frame["fno_day_unavailable"].to_numpy()
    bad += int((outage & ~(frame["is_missing_fno"].to_numpy() & frame["is_missing_opt"].to_numpy())).sum())
    share = frame["fut_roll_next_share"].dropna()
    bad += int(((share < 0) | (share > 1)).sum()) + int((frame["opt_pcr_oi"].dropna() < 0).sum()) + int((frame["fut_oi"].dropna() < 0).sum())
    bad += int(np.isinf(frame[[c for c in VALUE_COLUMNS if frame[c].dtype != bool]].to_numpy(dtype=float)).sum())
    return bad


def compare_violations(frames: dict[str, pd.DataFrame], expected: dict[tuple[str, pd.Timestamp], dict]) -> int:
    bad = 0
    for (symbol, when), want in expected.items():
        frame = frames[symbol]
        row = frame.loc[frame["date"] == when]
        if len(row) != 1:
            bad += 1
            continue
        row = row.iloc[0]
        bad += sum(0 if same(row[c], want[c]) else 1 for c in VALUE_COLUMNS)
    return bad


def stale_gap_dates(frame: pd.DataFrame, per_kind: int = 5) -> list:
    """Missing-OI dates a carry-forward bug can actually corrupt: those AFTER the first observed OI row, so a last
    known value exists to be carried. Leading pre-listing gaps are useless (ffill leaves them NaN); on real history
    they are the earliest rows and made the injection a no-op."""
    observed = ~frame["is_missing_fno"]
    after_first = observed.cummax() & ~observed
    unavailable = frame["fno_day_unavailable"].astype(bool)
    return list(frame.loc[after_first & ~unavailable, "date"])[:per_kind] + list(frame.loc[after_first & unavailable, "date"])[:3]


def zero_fill_violations(frame: pd.DataFrame) -> int:
    observed = (~frame["is_missing_ban"]).astype(float).rolling(20).sum()   # lists seen in a full 20-row window (a scoped file lacks the history before its first rows)
    return int(frame.loc[frame["is_missing_fno"], OI_FEATURES].notna().to_numpy().sum()
               + frame.loc[frame["is_missing_ban"], "fno_in_ban"].notna().sum()
               + frame.loc[observed < MIN_OBS_20D, "fno_ban_days_20d"].notna().sum())


def ban_calendar_violations(frames: dict[str, pd.DataFrame], raw: Raw) -> tuple[int, int, int]:
    """(availability/value disagreements with the raw lists, rows with a definite 0.0 on a no-list day, real 1.0 rows)."""
    bad = zero_on_nolist = banned_rows = 0
    for symbol, frame in frames.items():
        dates = pd.DatetimeIndex(frame["date"])
        available = np.array([d in raw.ban_days for d in dates], dtype=bool)
        bad += int((frame["is_missing_ban"].to_numpy() == available).sum())
        bad += int(frame.loc[~available, "fno_in_ban"].notna().sum())
        zero_on_nolist += int((frame.loc[~available, "fno_in_ban"] == 0.0).sum())
        expect = np.array([float(symbol in raw.ban_days[d]) if d in raw.ban_days else np.nan for d in dates])
        bad += int((~np.isclose(expect, frame["fno_in_ban"].to_numpy(dtype=float), equal_nan=True)).sum())
        banned_rows += int((frame["fno_in_ban"] == 1.0).sum())
    return bad, zero_on_nolist, banned_rows


def fno_calendar_violations(frames: dict[str, pd.DataFrame], raw: Raw) -> int:
    bad = 0
    for frame in frames.values():
        has_file = np.array([d in raw.fno_files for d in pd.DatetimeIndex(frame["date"])], dtype=bool)
        bad += int((frame["fno_day_unavailable"].to_numpy() == has_file).sum())
    return bad


# --------------------------------------------------------------------------- checks ----------------------------------------------

def load_frames(directory: Path, universe: list[str], scoped: bool) -> dict[str, pd.DataFrame]:
    present = {p.stem for p in directory.glob("*.parquet")}
    if scoped:
        symbols = sorted(present)
        check(set(symbols) <= set(universe) and symbols, "scoped run: every file is a universe symbol", f"({len(symbols)} files)")
    else:
        extra, missing = sorted(present - set(universe)), sorted(set(universe) - present)
        check(not extra and not missing, "exactly one file per universe symbol", f"(files: {len(present):,}, universe: {len(universe):,}, "
              f"extra: {extra[:5]}, missing: {missing[:5]})")
        symbols = sorted(present & set(universe))
    frames = {s: pd.read_parquet(directory / f"{s}.parquet") for s in symbols}
    empty = [s for s, f in frames.items() if f.empty]
    if empty and scoped:
        logger.info(f"scoped run: {len(empty)} symbol(s) have no rows in this range and are skipped: {empty}")
    elif empty:
        check(False, "every universe symbol has rows", f"(empty: {empty[:5]})")
    return {s: f for s, f in frames.items() if not f.empty}


def check_alignment(frames: dict[str, pd.DataFrame], raw: Raw) -> None:
    bad = 0
    for symbol, frame in frames.items():
        lo, hi = frame["date"].min(), frame["date"].max()
        spine = [d for d in raw.stage4(symbol)[0] if lo <= d <= hi]
        if list(frame.columns) != OUTPUT_COLUMNS or list(pd.DatetimeIndex(frame["date"])) != spine or (frame["symbol"] != symbol).any():
            bad += 1
    check(bad == 0, "columns exact and dates identical 1:1 to the symbol's Stage 4 rows", f"({len(frames):,} files; bad files: {bad})")


def pick_cases(frames: dict[str, pd.DataFrame], raw: Raw) -> list[tuple[str, pd.Timestamp]]:
    rng = random.Random(SEED)
    all_dates = sorted({d for f in frames.values() for d in f["date"]})
    fno_dates = [d for d in all_dates if d in raw.fno_files]
    dates = [pd.Timestamp(d) for d in FORCED_DATES if pd.Timestamp(d) in set(all_dates)]
    dates += rng.sample(fno_dates, min(SAMPLE_RANDOM_DAYS, len(fno_dates)))
    ban_dates = [d for d in all_dates if raw.ban_days.get(d)]
    dates += rng.sample(ban_dates, min(BAN_DAYS_SAMPLED, len(ban_dates)))
    cases = set()
    for when in dict.fromkeys(dates):
        on_day = [s for s, f in frames.items() if when in set(f["date"])]
        day = raw.day(when) or {}
        named = [s for s in on_day if s in day]
        unnamed = [s for s in on_day if s not in day]
        chosen = rng.sample(named, min(SAMPLE_SYMBOLS_PER_DAY, len(named))) + rng.sample(unnamed, min(SAMPLE_NON_FNO_SYMBOLS, len(unnamed)))
        chosen += [s for s in on_day if s in raw.ban_days.get(when, ())][:10]                         # the banned names
        cases.update((s, when) for s in chosen)
    logger.info(f"sampled {len(cases):,} (symbol, date) cases over {len(set(w for _, w in cases))} dates "
                f"(forced dates present: {sorted({d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') in FORCED_DATES})})")
    return sorted(cases)


def find_ex_date_events(frames: dict[str, pd.DataFrame], raw: Raw) -> list[tuple[str, int, pd.Timestamp]]:
    events = []
    for symbol, frame in frames.items():
        if frame["is_missing_fno"].all():
            continue
        spine, factors = raw.stage4(symbol)
        dates = list(pd.DatetimeIndex(frame["date"]))
        for k in range(1, len(dates)):
            f_prev, f_now = factors.get(dates[k - 1], float("nan")), factors.get(dates[k], float("nan"))
            if f_prev == f_prev and f_now == f_now and f_now > 0 and abs(f_prev / f_now - 1) > 0.2 \
                    and not frame["is_missing_fno"].iloc[k] and not frame["is_missing_fno"].iloc[k - 1]:
                events.append((symbol, k, dates[k]))
    return events


def ex_date_violations(frames: dict[str, pd.DataFrame], raw: Raw, events) -> tuple[int, int]:
    """(stored chg_1d disagreements with the factor-adjusted change, events where NSE's raw change differs from it by >20pp)."""
    bad = informative = 0
    for symbol, k, when in events:
        spine = [d for d in raw.stage4(symbol)[0]]
        i = spine.index(when)
        day_now, day_prev = raw.day(spine[i]), raw.day(spine[i - 1])
        raw_chg = day_now[symbol]["fut_raw"] / day_prev[symbol]["fut_raw"] - 1
        adj_chg = adjusted_oi(raw, symbol, spine[i]) / adjusted_oi(raw, symbol, spine[i - 1]) - 1
        informative += int(abs(raw_chg - adj_chg) > 0.2)
        stored = frames[symbol]["fut_oi_chg_1d"].iloc[k]
        bad += int(not same(stored, adj_chg))
    return bad, informative


def check_truncation(frames: dict[str, pd.DataFrame], raw: Raw) -> None:
    lo = min(f["date"].min() for f in frames.values())
    hi = max(f["date"].max() for f in frames.values())
    summaries, fno_days = fno.load_fno_summaries(lo - timedelta(days=fno.WARMUP_DAYS), hi)
    bans, ban_available = fno.load_ban_lists(lo - timedelta(days=fno.WARMUP_DAYS), hi)
    banned = {s: set(g["date"]) for s, g in bans.groupby("symbol")}
    by_symbol = {s: g for s, g in summaries.groupby("symbol")}
    candidates = [s for s, f in frames.items() if (~f["is_missing_fno"]).sum() > 40 and (f["fno_in_ban"] == 1.0).any()] \
        or [s for s, f in frames.items() if (~f["is_missing_fno"]).sum() > 40]
    bad = rows = 0
    for symbol in candidates[:4]:
        panel = fno.load_price_panel(symbol)
        panel = panel[(panel["date"] >= lo - timedelta(days=fno.WARMUP_DAYS)) & (panel["date"] <= hi)].reset_index(drop=True)
        full = fno.build_symbol_features(symbol, panel, by_symbol.get(symbol), fno_days, banned.get(symbol, set()), ban_available)
        for fraction in (0.35, 0.6, 0.85):
            cut = int(len(panel) * fraction)
            cut_date = panel["date"].iloc[cut - 1]
            head = fno.build_symbol_features(
                symbol, panel.iloc[:cut], by_symbol[symbol][by_symbol[symbol]["date"] <= cut_date] if symbol in by_symbol else None,
                {d for d in fno_days if d <= cut_date}, {d for d in banned.get(symbol, set()) if d <= cut_date},
                {d for d in ban_available if d <= cut_date})
            for column in VALUE_COLUMNS:
                a, b = full[column].iloc[:cut].astype(float).to_numpy(), head[column].astype(float).to_numpy()
                bad += int((~np.isclose(a, b, equal_nan=True)).sum())
            rows += cut
    check(bad == 0, "truncation: cutting the spine and the F&O/ban inputs never changes an earlier row (real builder)",
          f"({min(4, len(candidates))} symbols x 3 cuts, {rows:,} rows; violations: {bad})")


def main(argv=()) -> int:
    parser = argparse.ArgumentParser(description="Stage 10 F&O feature block checkpoint")
    parser.add_argument("--features-dir", type=Path, help="verify a scoped run written by fno.py --output-dir")
    args = parser.parse_args(list(argv))
    failures.clear()
    scoped = args.features_dir is not None
    directory = args.features_dir if scoped else FNO_FEATURES_DIR

    universe = list_universe_symbols()
    logger.info(f"F&O feature checkpoint on {directory} ({'scoped' if scoped else 'full'}); loading raw ban archive and files ...")
    raw = Raw()
    frames = load_frames(directory, universe, scoped)
    if not frames:
        logger.error("no feature files found")
        return 1
    lo, hi = min(f["date"].min() for f in frames.values()), max(f["date"].max() for f in frames.values())
    rows = sum(len(f) for f in frames.values())
    with_fno = sum(int((~f["is_missing_fno"]).sum()) for f in frames.values())
    with_ban = sum(int((~f["is_missing_ban"]).sum()) for f in frames.values())
    logger.info(f"{len(frames):,} symbols, {rows:,} rows, {lo.date()} -> {hi.date()}: {with_fno:,} rows with futures OI, "
                f"{with_ban:,} rows with a ban list")
    check(with_fno > 0, "real row counts: there are rows with futures open interest in scope")
    if scoped and with_ban == 0:
        logger.info("no ban lists in this range (the archive starts 2008-01-01): every row must be, and is checked to be, unavailable")
    else:
        check(with_ban > 0, "real row counts: there are rows with a usable ban list in scope")

    check_alignment(frames, raw)
    structure = sum(structure_violations(f) for f in frames.values())
    check(structure == 0, "structure: flags, NaN patterns, ranges, no inf", f"(violations: {structure})")

    calendar_bad = fno_calendar_violations(frames, raw)
    check(calendar_bad == 0, "fno_day_unavailable <=> no F&O file for that date, every row", f"(violations: {calendar_bad})")
    if any(pd.Timestamp("2003-10-01") in set(f["date"]) for f in frames.values()):
        check(pd.Timestamp("2003-10-01") not in raw.fno_files, "2003-10-01 has no F&O file (excluded, no zero-copy rule applied)")
    else:
        logger.info("2003-10-01 is outside this run's date range: not exercised")

    ban_bad, zero_on_nolist, banned_rows = ban_calendar_violations(frames, raw)
    check(ban_bad == 0, "ban lists: availability and value agree with the raw archive, every row", f"(violations: {ban_bad})")
    check(zero_on_nolist == 0, "no session without a list carries a definite ban status", f"(violations: {zero_on_nolist})")
    for day in UNAVAILABLE_BAN_DAYS:
        stamp = pd.Timestamp(day)
        hit = [f.loc[f["date"] == stamp, "is_missing_ban"].iloc[0] for f in frames.values() if (f["date"] == stamp).any()]
        if hit:
            check(all(hit) and stamp not in raw.ban_days, f"documented no-list day {day} is unavailable for every symbol", f"({len(hit)} symbols)")
    if not scoped:
        check(banned_rows > 0, "real 1.0 ban rows exist in the real output (otherwise the ban checks prove nothing)")
    else:
        logger.info(f"banned rows in scope: {banned_rows:,}")

    cases = pick_cases(frames, raw)
    expected = {c: expected_row(raw, *c) for c in cases}
    recompute = compare_violations(frames, expected)
    check(recompute == 0, "independent recompute from raw files matches every feature and flag", f"({len(cases):,} cases x {len(VALUE_COLUMNS)} columns; violations: {recompute})")
    n_non_fno = sum(1 for e in expected.values() if e["is_missing_fno"] and not e["fno_day_unavailable"])
    n_zero_fact = sum(1 for e in expected.values() if e["fno_in_ban"] == 0.0)
    logger.info(f"sampled cases covering a symbol absent from an available F&O day: {n_non_fno}; real 0.0 ban cases: {n_zero_fact}")

    events = find_ex_date_events(frames, raw)[:EX_DATE_EVENTS]
    ex_bad, informative = ex_date_violations(frames, raw, events)
    logger.info(f"{len(events)} ex-date events found (factor jump > 20%), {informative} where NSE's raw OI change differs from the adjusted one by >20pp")
    check(ex_bad == 0, "adjustment factor: stored fut_oi_chg_1d at every ex-date equals the factor-adjusted change", f"(violations: {ex_bad})")
    if events:
        check(informative > 0, "at least one ex-date where raw and adjusted OI changes differ materially (the factor path is exercised)")
    else:
        logger.warning("no ex-date event in this range: the factor path was not exercised; choose a range with a split/bonus (e.g. HDFCBANK 2025-08-26)")

    check_truncation(frames, raw)

    logger.info("--- injected bugs (in-memory copies of real output; each must be caught) ---")
    victims = [s for s, f in frames.items() if (~f["is_missing_fno"]).sum() > 5 and stale_gap_dates(f)]
    if victims:
        symbol = max(victims, key=lambda s: int((~frames[s]["is_missing_fno"]).sum()))
        base = frames[symbol]
        gaps = stale_gap_dates(base)
        own = {(symbol, d): expected_row(raw, symbol, d) for d in gaps}

        zero = base.copy()
        zero.loc[zero["is_missing_fno"], OI_FEATURES] = 0.0
        expect_caught("missing OI rows filled with 0", zero_fill_violations(zero))

        stale = base.copy()
        stale[OI_FEATURES] = stale[OI_FEATURES].ffill()
        expect_caught("stale carry-forward (ffill) of the last known OI", compare_violations({symbol: stale}, own))

        cleared = base.copy()
        cleared.loc[cleared.index[cleared["is_missing_fno"]][-1], "is_missing_fno"] = False
        expect_caught("is_missing_fno cleared on a NaN row", structure_violations(cleared))
    else:
        logger.warning("no symbol in scope has both missing and observed F&O rows: OI zero-fill/stale/flag bugs not injectable here")

    thin = next((s for s, f in frames.items() if f["fno_ban_days_20d"].iloc[19:].isna().any()), None)
    if thin is not None:
        ban_zero = frames[thin].copy()
        ban_zero["fno_ban_days_20d"] = ban_zero["fno_ban_days_20d"].fillna(0.0)
        expect_caught("fno_ban_days_20d filled with 0 where fewer than 15 lists were seen", zero_fill_violations(ban_zero))
    else:
        logger.warning("no fno_ban_days_20d NaN in scope: its zero-fill bug is not injectable here")

    # Ban bugs. A scoped range may hold no genuinely list-less session, so run the real output against a raw world in which
    # one list never existed: every definite status the block reported for that session is then unsupported.
    changing = [s for s, f in frames.items() if f["fno_in_ban"].dropna().nunique() > 1]
    ban_symbol = changing[0] if changing else next(iter(frames))
    ban_frame = frames[ban_symbol]
    listed = ban_frame.loc[ban_frame["fno_in_ban"].notna(), "date"]
    if len(listed):
        lost = listed.iloc[len(listed) // 2]
        world = SimpleNamespace(ban_days={d: v for d, v in raw.ban_days.items() if d != lost})
        expect_caught(f"ban status reported for {lost.date()} though no list exists for it (unavailable read as none)",
                      ban_calendar_violations({ban_symbol: ban_frame}, world)[0])
    else:
        logger.warning("no ban list in scope: the unavailable-read-as-none bug is not injectable here")
    if changing:
        next_ban = ban_frame.copy()
        next_ban["fno_in_ban"] = next_ban["fno_in_ban"].shift(-1)
        expect_caught("next session's ban list used for today", ban_calendar_violations({ban_symbol: next_ban}, raw)[0])
    else:
        logger.warning("no symbol in scope changes ban status: the next-session leak bug is not injectable here")

    if events:
        victim = next((e[0] for e in events), None)
        unadjusted = {victim: frames[victim].copy()}
        raw_oi = [raw.day(d)[victim]["fut_raw"] if raw.day(d) and victim in raw.day(d) else np.nan
                  for d in pd.DatetimeIndex(unadjusted[victim]["date"])]
        unadjusted[victim]["fut_oi_chg_1d"] = pd.Series(raw_oi).pct_change(fill_method=None).to_numpy()
        expect_caught("OI not factor-adjusted (raw NSE OI used across the ex-date)",
                      ex_date_violations(unadjusted, raw, [e for e in events if e[0] == victim])[0])

    if failures:
        logger.error(f"Stage 10 F&O feature checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 F&O feature checkpoint PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
