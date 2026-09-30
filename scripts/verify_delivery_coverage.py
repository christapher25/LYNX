"""
Verify delivery data coverage against the authoritative manifest.

Why this exists: grepping "FAILED" from log files double-counts dates that
were re-attempted across multiple ingestion sessions, since every unfinished
run walks the full date range again on restart. The manifest at
data/raw/_manifests/bhavdata.jsonl is the correct source of truth instead —
one JSON line per (date, attempt), which this script deduplicates by date,
keeping only the most recent outcome for each.

Usage:
    python scripts/verify_delivery_coverage.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DELIVERY_DIR, PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifests" / "bhavdata.jsonl"
FIRST_AVAILABLE = date(2019, 9, 30)  # measured in Stage 3, see DATA_NOTES.md


def trading_day_candidates(start: date, end: date):
    """Weekdays only — matches ingest_common.py's own candidate logic."""
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}")
        print("Nothing to verify against — falling back to file count only.")
        files = sorted(DELIVERY_DIR.glob("*.parquet"))
        print(f"Files on disk: {len(files)}")
        return

    # Deduplicate by date, keeping the LAST (most recent) record for each —
    # a date attempted across three sessions should count once, with
    # whatever its final outcome was.
    latest: dict[str, dict] = {}
    total_lines = 0
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            record = json.loads(line)
            latest[record["trade_date"]] = record

    print(f"Manifest: {total_lines:,} total log lines -> {len(latest):,} unique dates\n")

    # Tally by final status
    by_status: dict[str, list[str]] = {}
    for trade_date, record in latest.items():
        by_status.setdefault(record["status"], []).append(trade_date)

    for status in ["ok", "holiday", "failed", "empty"]:
        dates = sorted(by_status.get(status, []))
        print(f"  {status:8s}: {len(dates):5,}")

    # Cross-check: does the manifest's 'ok' count match files actually on disk?
    files_on_disk = {p.stem for p in DELIVERY_DIR.glob("*.parquet")}
    manifest_ok = set(by_status.get("ok", []))

    print(f"\nFiles on disk:        {len(files_on_disk):,}")
    print(f"Manifest says 'ok':    {len(manifest_ok):,}")

    missing_from_disk = manifest_ok - files_on_disk
    extra_on_disk = files_on_disk - manifest_ok
    if missing_from_disk:
        print(f"  WARNING: {len(missing_from_disk)} dates marked 'ok' in manifest "
              f"but no file on disk: {sorted(missing_from_disk)[:5]}...")
    if extra_on_disk:
        print(f"  NOTE: {len(extra_on_disk)} files on disk with no manifest record "
              f"(likely from before manifest logging existed): "
              f"{sorted(extra_on_disk)[:5]}...")
    if not missing_from_disk and not extra_on_disk:
        print("  Manifest and disk agree exactly.")

    # Independent candidate count, computed fresh — not trusting any prior number
    today = date.today()
    candidates = list(trading_day_candidates(FIRST_AVAILABLE, today))
    print(f"\nExpected weekday candidates, {FIRST_AVAILABLE} to {today}: {len(candidates):,}")

    accounted_for = sum(len(v) for v in by_status.values())
    print(f"Accounted for in manifest:                              {accounted_for:,}")

    if accounted_for == len(candidates):
        print("MATCH — every candidate date is accounted for.")
    else:
        diff = len(candidates) - accounted_for
        print(f"MISMATCH — {diff:+d} candidate dates have no manifest record at all "
              f"(never attempted, or attempted before manifest logging existed).")

    # The actual list, for a manual spot-check against NSE's holiday calendar
    failed_dates = sorted(by_status.get("failed", []))
    if failed_dates:
        print(f"\nAll {len(failed_dates)} failed dates (for holiday-calendar spot-check):")
        for d in failed_dates:
            print(f"  {d}")


if __name__ == "__main__":
    main()