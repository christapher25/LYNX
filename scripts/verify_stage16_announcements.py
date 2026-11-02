"""
Stage 16 checkpoint: announcement ingestion and entity resolution, verified against the files on disk.

Passing check (v6 section 15.1): "Announcements ticker-tagged, date range verified" -- plus row counts per year.

  1. Files: every Parquet matches announcements_{start}_{end}.parquet, chunk ranges do not overlap, and each file's row
     count equals its last `ok` manifest record.
  2. Date range: every record's broadcast date lies inside its file's range; the overall first/last broadcast date is
     reported.
  3. Tagging: every record is ticker-tagged OR explicitly unresolved with a reason; the invariant checker finds zero
     violations in every file (no ticker on an unresolved record, no resolved record without one, publication time never
     earlier than either source stamp, date-only stamps at end of day, tickers all in the symbol master).
  4. Reported, not judged: rows per year, resolution method mix, unresolved counts by reason; the distinct unresolved
     identities are listed in data/raw/announcements/_unresolved_summary.csv.
  5. Injected bugs: the checker is run on a copy of real output with a guessed ticker and with event-date keying, and must
     fire on both.
  6. --live-crosscheck: one short range is fetched twice (as a single range, and day by day) and the record counts must
     agree, which shows the API's range parameters are honoured and nothing is truncated.

Usage:  python -m scripts.verify_stage16_announcements [--live-crosscheck] [--dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ANNOUNCEMENTS_DIR, BHAVCOPY_DIR
from data import ingest_announcements as ia
from data import isin_bridge
from data.entity_resolution import build_symbol_master

READ_COLUMNS = [c for c in ia.OUTPUT_COLUMNS if c not in ("TEXT", "ATTACHMENT_URL")]


def last_ok_rows(manifest: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["status"] == "ok":
                rows[rec["trade_date"]] = rec["rows"]
            elif rec["trade_date"] in rows and rec["status"] != "ok":
                pass                                       # a later failed retry does not undo a written file
    return rows


def live_crosscheck(start: date, end: date) -> tuple[int, int]:
    fetch = ia.Fetcher()
    whole, _ = fetch(start, end)
    total, day = 0, start
    while day <= end:
        part, _ = fetch(day, day)
        total += len(part)
        day += timedelta(days=1)
        time.sleep(1.0)
    return len(whole), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, help="announcements directory (default config.ANNOUNCEMENTS_DIR)")
    ap.add_argument("--live-crosscheck", action="store_true")
    args = ap.parse_args()
    directory = Path(args.dir) if args.dir else ANNOUNCEMENTS_DIR
    failures: list[str] = []

    files = sorted(directory.glob("announcements_*.parquet"))
    if not files:
        logger.error(f"no announcement files under {directory}")
        return 1
    ranges = []
    for f in files:
        m = ia.FILE_RE.match(f.name)
        if not m:
            failures.append(f"unexpected file name {f.name}")
            continue
        ranges.append((date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2)), f))
    ranges.sort()
    for (a1, b1, f1), (a2, b2, f2) in zip(ranges, ranges[1:]):
        if a2 <= b1:
            failures.append(f"overlapping chunks {f1.name} / {f2.name}")

    manifest_rows = last_ok_rows(ia.MANIFEST_PATH if directory == ANNOUNCEMENTS_DIR else directory / "_manifest.jsonl")
    master = build_symbol_master(BHAVCOPY_DIR)

    per_year: dict[int, int] = {}
    methods: dict[str, int] = {}
    reasons: dict[str, int] = {}
    violations: dict[str, int] = {}
    unresolved_parts = []
    first_day = last_day = None
    total = 0
    sample = None
    bridged_total = 0
    bridge_pit_bad = 0
    bridge_unres: dict[str, int] = {}

    for a, b, f in ranges:
        df = pd.read_parquet(f, columns=READ_COLUMNS)
        total += len(df)
        key = f"{a}_{b}"
        if key in manifest_rows and manifest_rows[key] != len(df):
            failures.append(f"{f.name}: {len(df)} rows on disk, manifest says {manifest_rows[key]}")
        broadcast = pd.to_datetime(df["AN_DT_RAW"].map(ia._parse_stamp)).dt.normalize()
        if (broadcast < pd.Timestamp(a)).any() or (broadcast > pd.Timestamp(b)).any():
            failures.append(f"{f.name}: records broadcast outside {a}..{b}")
        lo, hi = broadcast.min().date(), broadcast.max().date()
        first_day = lo if first_day is None else min(first_day, lo)
        last_day = hi if last_day is None else max(last_day, hi)

        for name, n in ia.announcement_invariant_violations(df, master).items():
            violations[name] = violations.get(name, 0) + n
        for year, n in df["PUBLISHED_DATE"].dt.year.value_counts().items():
            per_year[int(year)] = per_year.get(int(year), 0) + int(n)
        for method, n in df["RESOLUTION_METHOD"].value_counts().items():
            methods[str(method)] = methods.get(str(method), 0) + int(n)
        unresolved = df[df["RESOLUTION_METHOD"] == "unresolved"]
        for reason, n in unresolved["UNRESOLVED_REASON"].value_counts().items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(n)
        unresolved_parts.append(unresolved.groupby(["UNRESOLVED_REASON", "SYMBOL_RAW", "ISIN", "COMPANY_NAME"], dropna=False)
                                .size().rename("RECORDS").reset_index())
        bridged = df[df["RESOLUTION_METHOD"] == "isin_bridge"]
        bridged_total += len(bridged)
        for _, r in bridged.iterrows():                       # independent PIT check: the ISIN's window for TICKER holds the date
            day = pd.Timestamp(r["PUBLISHED_DATE"]).date()
            if not isin_bridge.pit_window_ok(r["TICKER"], r["ISIN"], day, master):
                bridge_pit_bad += 1
        for reason, n in unresolved["UNRESOLVED_REASON"].value_counts().items():
            if str(reason).startswith("bridge_"):
                bridge_unres[str(reason)] = bridge_unres.get(str(reason), 0) + int(n)
        if sample is None and df["TICKER"].notna().any() and (df["RESOLUTION_METHOD"] == "unresolved").any():
            sample = df

    if bridge_pit_bad:
        failures.append(f"{bridge_pit_bad} isin_bridge records whose ISIN window for the ticker does not contain the publication date")
    if violations:
        failures.append(f"invariant violations on real data: {violations}")

    unresolved_all = pd.concat(unresolved_parts, ignore_index=True) if unresolved_parts else pd.DataFrame()
    if len(unresolved_all):
        unresolved_all = (unresolved_all.groupby(["UNRESOLVED_REASON", "SYMBOL_RAW", "ISIN", "COMPANY_NAME"], dropna=False)["RECORDS"]
                          .sum().reset_index().sort_values("RECORDS", ascending=False))
    unresolved_all.to_csv(directory / "_unresolved_summary.csv", index=False)

    untagged = total - methods.get("symbol", 0) - methods.get("isin", 0) - methods.get("name", 0) - methods.get("isin_bridge", 0) - methods.get("unresolved", 0)
    if untagged:
        failures.append(f"{untagged} records are neither ticker-tagged nor flagged unresolved")

    # Injected bugs: the checker must fire on both, using real output as the base.
    base = sample if sample is not None else pd.read_parquet(ranges[0][2], columns=READ_COLUMNS)
    guessed = base.copy()
    unres_mask = guessed["RESOLUTION_METHOD"] == "unresolved"
    if unres_mask.any():
        guessed.loc[unres_mask, "TICKER"] = next(iter(master._symbols))
        if "ticker_on_unresolved" not in ia.announcement_invariant_violations(guessed, master):
            failures.append("injected guessed-ticker bug was NOT detected")
    else:
        logger.warning("no unresolved record in the sample; guessed-ticker injection skipped")
    event_keyed = base.copy()
    event_keyed["PUBLISHED_DATE"] = event_keyed["PUBLISHED_DATE"] - pd.Timedelta(days=1)
    if "published_date_mismatch" not in ia.announcement_invariant_violations(event_keyed, master):
        failures.append("injected wrong-date keying bug was NOT detected")

    if args.live_crosscheck:
        end = date.today() - timedelta(days=10)
        whole, by_day = live_crosscheck(end - timedelta(days=2), end)
        logger.info(f"live cross-check {end - timedelta(days=2)}..{end}: range={whole:,} day-by-day={by_day:,}")
        if whole != by_day:
            failures.append(f"range query returned {whole} but day-by-day sums to {by_day}")

    resolved = total - methods.get("unresolved", 0)
    logger.info(f"files={len(ranges)} records={total:,} broadcast range {first_day} .. {last_day}")
    logger.info(f"ticker-tagged={resolved:,} ({resolved / total:.2%}) methods={methods}")
    logger.info(f"unresolved={methods.get('unresolved', 0):,} by reason={reasons}")
    logger.info(f"resolved by ISIN bridge={bridged_total:,}; unresolved after bridge by reason={bridge_unres}; "
                f"unresolved with no ISIN to bridge={sum(n for r, n in reasons.items() if not r.startswith('bridge_') and r in ('symbol_not_in_master', 'symbol_outside_traded_window')):,}")
    logger.info("rows per publication year: " + ", ".join(f"{y}={n:,}" for y, n in sorted(per_year.items())))
    logger.info(f"distinct unresolved identities listed in {directory / '_unresolved_summary.csv'}: {len(unresolved_all):,}")

    if failures:
        for failure in failures:
            logger.error(f"FAIL: {failure}")
        return 1
    logger.info("Stage 16 announcements: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
