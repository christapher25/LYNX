"""
Stage 10 — NSE daily security list: the price band (circuit limit) each security trades under on each date.

Source (probed 2026-09-24, see data/raw/DATA_NOTES.md §13):

    https://nsearchives.nseindia.com/content/equities/sec_list_{ddmmyyyy}.csv

    first available 2019-12-30 (2019-12-27 -> 404; every earlier date probed -> 404). Header:
    Symbol,Series,Security Name,Band,Remarks. Band is one of 2, 5, 10, 20, 40 (percent) or "No Band" (securities in the
    derivatives segment and a few others; they have no static band). Bands change often: the 5% bucket alone ranged from
    295 to 704 securities across 28 sampled dates, so a band is only meaningful for the date of its file.

The file carries NO date inside it, so a stale re-serve of an earlier day cannot be caught from the content. The Last-Modified
header is the only evidence of when the file was produced (sampled 11:30-13:37 GMT = 17:00-19:07 IST on the file's own
date); it is stored with every row as LAST_MODIFIED_UTC, and it is also the publication clock features/blocks/circuits.py
keys on. A missing header is stored as null, never invented.

Output: one Parquet per date, columns OUTPUT_COLS. BAND_PCT is NaN for "No Band" (IS_NO_BAND True); an unknown or blank band is
rejected, not read as "no band": a wrong band would be applied to every price move of that day.

Usage
-----
    python data/ingest_sec_list.py                    # full backfill (resumable)
    python data/ingest_sec_list.py --days 3
    python data/ingest_sec_list.py --weekends         # Saturday/Sunday special sessions
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, MANIFEST_DIR, SEC_LIST_DIR
from data.ingest_common import run_ingestion

FIRST_AVAILABLE = date(2019, 12, 30)       # probed: 2019-12-27 -> 404, 2019-12-30 -> 200
BASE_URL = "https://nsearchives.nseindia.com/content/equities/sec_list_{ddmmyyyy}.csv"
EXPECTED_HEADER = ["Symbol", "Series", "Security Name", "Band", "Remarks"]
EXTENDED_HEADER = EXPECTED_HEADER + ["", "Check"]      # 2020-02-05 only: each row also carries a copy of the band and a check of 0
OUTPUT_COLS = ["DATE", "SYMBOL", "SERIES", "BAND_TEXT", "BAND_PCT", "IS_NO_BAND", "LAST_MODIFIED_UTC"]
NO_BAND = "No Band"
BAND_PERCENTS = {"2": 2.0, "5": 5.0, "10": 10.0, "20": 20.0, "40": 40.0}
MANIFEST_PATH = MANIFEST_DIR / "sec_list.jsonl"


def build_url(trade_date: date) -> str:
    return BASE_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))


def _last_modified_utc(header: str | None) -> str | None:
    if header is None:
        return None
    try:
        stamp = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        raise ValueError(f"unparseable Last-Modified header {header!r}") from None
    if stamp.tzinfo is None:
        raise ValueError(f"Last-Modified header {header!r} has no timezone; refusing to assume one")
    return stamp.astimezone(timezone.utc).isoformat()


def parse_sec_list(text: str, trade_date: date, last_modified: str | None) -> pd.DataFrame:
    """Parse one day's security list. Raises ValueError on an unknown header, band, row shape or duplicate key."""
    stamp = _last_modified_utc(last_modified)
    rows = [row for row in csv.reader(io.StringIO(text.lstrip("﻿"))) if row]
    header = [cell.strip() for cell in rows[0]] if rows else []
    extended = header == EXTENDED_HEADER
    if not rows or (header != EXPECTED_HEADER and not extended):
        raise ValueError(f"unexpected header: {rows[0] if rows else '(empty file)'}")
    body = rows[1:]
    if not body:
        raise ValueError("header only: the file holds no securities")
    width = len(header)
    records = []
    for row in body:
        if len(row) != width:
            raise ValueError(f"row with {len(row)} fields, expected {width}{' (extended layout)' if extended else ''}: {row}")
        symbol, series, _name, band, _remarks = (cell.strip() for cell in row[:5])
        if band.lower() == NO_BAND.lower():
            band = NO_BAND                                            # 2021-12-30 wrote 'No band' for nine securities
        if extended and (row[5].strip().lower() != band.lower() or row[6].strip() != ("#VALUE!" if band == NO_BAND else "0")):
            # measured over all 1,903 rows of 2020-02-05: the copy always equals the band; Check is 0, or #VALUE! for No Band
            raise ValueError(f"extended layout: the copied band or the check column does not match in row {row}")
        if band == NO_BAND:
            pct, no_band = float("nan"), True
        elif band in BAND_PERCENTS:
            pct, no_band = BAND_PERCENTS[band], False
        else:
            raise ValueError(f"unknown band {band!r} for {symbol} {series}")
        if not symbol or not series:
            raise ValueError(f"blank symbol or series in row {row}")
        records.append((pd.Timestamp(trade_date), symbol, series, band, pct, no_band, stamp))
    out = pd.DataFrame(records, columns=OUTPUT_COLS)
    repeated = out.duplicated(["SYMBOL", "SERIES"], keep=False)
    if repeated.any():
        conflicting = out[repeated].groupby(["SYMBOL", "SERIES"])["BAND_TEXT"].nunique().gt(1)
        if conflicting.any():
            raise ValueError(f"duplicate (symbol, series) rows with conflicting bands: {[s for s, _ in conflicting[conflicting].index[:5]]}")
        out = out.drop_duplicates(["SYMBOL", "SERIES"], keep="first").reset_index(drop=True)   # same band repeated (remarks differ)
    return out


def fetch_day(session: requests.Session, trade_date: date) -> tuple[pd.DataFrame | None, str, int]:
    """One day. A None dataframe means no file for that date (weekend / holiday)."""
    resp = session.get(build_url(trade_date), timeout=60)
    if resp.status_code == 404:
        return None, "sec_list", 0
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig", errors="replace")
    return parse_sec_list(text, trade_date, resp.headers.get("Last-Modified")), "sec_list", len(resp.content)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSE's daily security list (price bands).")
    ap.add_argument("--start", type=str, help="start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="end date YYYY-MM-DD")
    ap.add_argument("--days", type=int, help="fetch only the last N days")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests (default 0.4)")
    ap.add_argument("--weekends", action="store_true", help="request Saturdays and Sundays only (special sessions)")
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    if args.days:
        start = end - timedelta(days=args.days)
    elif args.start:
        start = date.fromisoformat(args.start)
    else:
        start = FIRST_AVAILABLE
    start = max(start, FIRST_AVAILABLE)

    run_ingestion(
        name="sec_list", start=start, end=end, output_dir=SEC_LIST_DIR, fetch_fn=fetch_day, manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR, request_delay=args.delay, days="weekends" if args.weekends else "weekdays",
    )


if __name__ == "__main__":
    main()
