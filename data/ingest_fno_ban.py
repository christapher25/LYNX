"""
Stage 10 — NSE F&O ban list ingestion: the securities in the F&O ban period on each trade date.

A stock enters the ban period when its open interest reaches 95% of the market-wide position
limit; new positions are then prohibited until it falls back. Stage 11's override gates read this
to drop banned names before a signal is published.

Source (probed 2026-09-24, see data/raw/DATA_NOTES.md §8):

    https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{ddmmyyyy}.csv

    first available 2008-01-01 (2007-12-31 -> 404). The /content/fo/fo_secban_* path is a 404 for
    every date; only /archives/fo/sec_ban/ serves the files.

The files are tiny text, not CSV proper, in three observed header shapes:

    "Securities in Ban For Trade Date 11-JUN-2020: "  then  "1,BHEL" ...
    "Securities in Ban For Trade Date 03-JUN-2024: NIL"
    "Sr. No.,Securities in Ban For Trade Date 01-Jan-2008"  then  "1,ADLABSFILM" ...

Output: one Parquet per trade date with columns (DATE, SYMBOL), one row per banned symbol. A day on
which the list was published and empty ("NIL") is stored as ONE row with SYMBOL null: "nobody was
banned" is information, and is not the same fact as "no file exists for this date". Filter
SYMBOL.notna() to get the banned names.

The parser rejects, rather than skips, anything it does not recognise: a symbol line silently
dropped would let a banned stock through Stage 11's gate.

Usage
-----
    python data/ingest_fno_ban.py                     # full backfill (resumable)
    python data/ingest_fno_ban.py --days 30
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FNO_BAN_DIR, LOG_DIR, MANIFEST_DIR
from data.ingest_common import run_ingestion

FIRST_AVAILABLE = date(2008, 1, 1)       # probed: 2007-12-31 -> 404, 2008-01-01 -> 200

BASE_URL = "https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{ddmmyyyy}.csv"

OUTPUT_COLS = ["DATE", "SYMBOL"]

# The date is written 11-JUN-2020, 01-Jan-2008 or, in 2008-2009, with a full month name (02-June-2008,
# 24-September-2008). What follows it varies too: ':' or ' :' or ',' or ',Nil', then NIL/Nil or nothing.
HEADER_RE = re.compile(
    r"^(?:Sr\.?\s*No\.?\s*,\s*)?Securities in Ban For Trade Date\s+(\d{1,2})-([A-Za-z]{3,9})-(\d{4})\s*(.*)$",
    re.IGNORECASE,
)
SEPARATOR_CHARS = " \t:,"                 # stripped from both ends of whatever follows the date
# '1,BHEL' or, from 2012, fixed-width CSV padding: '1,HDIL,' / '1,INDIACEM,,,'. Padding is accepted only when EVERY
# field after the symbol is empty (checked against raw bytes: 2012-08-29, 09-21, 09-27, 2015-12-23); any data after the
# symbol, or a missing symbol, is rejected.
ROW_RE = re.compile(r"^\d+\s*,\s*([A-Za-z0-9&_.\-]+)(?:\s*,)*\s*$")

MANIFEST_PATH = MANIFEST_DIR / "fno_ban.jsonl"


def build_url(trade_date: date) -> str:
    return BASE_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))


def parse_ban_list(text: str, trade_date: date) -> pd.DataFrame:
    """
    Parse one day's ban-list text into (DATE, SYMBOL) rows. Raises ValueError on any line it does
    not understand, on a header dated for another day, and on internal contradictions.
    """
    lines = [line.strip() for line in text.lstrip("﻿").splitlines()]
    lines = [line for line in lines if line]
    header = HEADER_RE.match(lines[0]) if lines else None
    if header is None:
        raise ValueError(f"no recognisable ban-list header; first line: {lines[0][:80]!r}" if lines
                         else "no recognisable ban-list header (empty file)")

    day, month_name, year = header.group(1), header.group(2), header.group(3)
    try:                                                                # 'June' -> 'Jun', 'September' -> 'Sep'
        header_date = pd.to_datetime(f"{day}-{month_name[:3]}-{year}", format="%d-%b-%Y").date()
    except ValueError:
        raise ValueError(f"unrecognised month {month_name!r} in the ban-list header") from None
    if header_date != trade_date:
        raise ValueError(f"file dated {header_date}, requested {trade_date}")

    trailing, body = header.group(4).strip(SEPARATOR_CHARS), lines[1:]
    is_nil = trailing.upper() == "NIL"
    if trailing and not is_nil:
        raise ValueError(f"unparseable text after the header: {trailing!r}")
    if is_nil and body:
        raise ValueError("header says NIL but symbol lines follow")

    symbols: list[str] = []
    for line in body:
        match = ROW_RE.match(line)
        if match is None:
            raise ValueError(f"unparseable ban-list line: {line!r}")
        symbols.append(match.group(1))

    duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
    if duplicates:
        raise ValueError(f"duplicate symbol(s) in one day's list: {duplicates}")

    stamp = pd.Timestamp(trade_date)
    if is_nil or not symbols:
        # Published-and-empty. One row with a null symbol keeps the day visible; see module docstring.
        return pd.DataFrame({"DATE": [stamp], "SYMBOL": pd.Series([None], dtype="string")})[OUTPUT_COLS]
    return pd.DataFrame({"DATE": stamp, "SYMBOL": pd.Series(symbols, dtype="string")})[OUTPUT_COLS]


def validate(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    if df.empty:
        raise ValueError("parsed to zero rows")
    dates = df["DATE"].dt.date.unique()
    if len(dates) != 1 or dates[0] != trade_date:
        raise ValueError(f"file dated {list(dates)}, requested {trade_date}")
    named = df["SYMBOL"].dropna()
    if named.duplicated().any():
        raise ValueError("duplicate symbols present")
    if len(df) > 1 and df["SYMBOL"].isna().any():
        raise ValueError("null symbol mixed with named symbols")
    return df


def fetch_day(session: requests.Session, trade_date: date) -> tuple[pd.DataFrame | None, str, int]:
    """Fetch one day. A None dataframe means no file for that date (holiday / non-session day)."""
    resp = session.get(build_url(trade_date), timeout=30)
    if resp.status_code == 404:
        return None, "sec_ban", 0
    resp.raise_for_status()

    text = resp.content.decode("utf-8-sig", errors="replace")
    return validate(parse_ban_list(text, trade_date), trade_date), "sec_ban", len(resp.content)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the NSE F&O ban list (securities in ban period).")
    ap.add_argument("--start", type=str, help="start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="end date YYYY-MM-DD")
    ap.add_argument("--days", type=int, help="fetch only the last N days")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests (default 0.4)")
    ap.add_argument("--weekends", action="store_true",
                    help="request Saturdays and Sundays only (Budget-day, Muhurat and special sessions); "
                         "default is weekdays only")
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
        name="fno_ban",
        start=start,
        end=end,
        output_dir=FNO_BAN_DIR,
        fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR,
        request_delay=args.delay,
        days="weekends" if args.weekends else "weekdays",
    )


if __name__ == "__main__":
    main()
