"""
Stage 10 — securities that hit their price band (upper / lower circuit) on each trade date.

Source (probed 2026-09-24, see data/raw/DATA_NOTES.md §13): the bh file inside NSE's daily PR zip,

    https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip  ->  bh{ddmmyyyy | ddmmyy}.csv

NSE's own readme: "bh...csv contains a list of securities which have hit their price bands during the day".
Columns SYMBOL, SERIES (SR in 2010), SECURITY, HIGH/LOW (H = upper, L = lower), and INDEX FLAG (blank on nearly every row; an
index-membership label such as 'CNX Nifty Junior' on a few in 2010; absent from 2026; ignored). First available 2010-01-04: every
earlier date probed -> 404. Header variants seen: 'SR' vs 'SERIES', 'INDEXFLAG' vs 'INDEX FLAG', trailing empty padding cells.

Facts that shape the parser:
  * The member name is bhDDMMYYYY.csv (2010-11, 2026) or bhDDMMYY.csv (2012-25), sometimes inside a folder
    ('nupr100113/bh100113.csv', 2013). The date in the name is the only date
    embedded in the zip, so it is checked against the requested day: a holiday re-serve of the previous session's zip
    is rejected, not stored under the wrong date.
  * A stock that hit BOTH bands in a day appears twice (H and L; 1-61 per file). The key is (symbol, series, hit).
  * A published file with no hits is stored as ONE row with a null SYMBOL: "nobody hit a band" is information, and is not
    the same fact as "no file exists" (same convention as the F&O ban list).
  * Last-Modified is kept per row: real same-day stamps from 2020 (about 17:35 IST), but every zip before 2020 was
    re-stamped 2023-04-07, so the true publication time of the older hit lists is unrecoverable.

Output: one Parquet per trade date, columns OUTPUT_COLS.

Usage
-----
    python data/ingest_price_band_hits.py                # full backfill (resumable), ~4,000 zips of 0.3-0.7 MB
    python data/ingest_price_band_hits.py --days 3
    python data/ingest_price_band_hits.py --weekends
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, MANIFEST_DIR, PRICE_BAND_HITS_DIR
from data.ingest_common import run_ingestion
from data.ingest_sec_list import _last_modified_utc

FIRST_AVAILABLE = date(2010, 1, 4)         # probed: 2010-01-01 and every earlier date -> 404, 2010-01-04 -> 200
BASE_URL = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip"
OUTPUT_COLS = ["DATE", "SYMBOL", "SERIES", "HIT", "LAST_MODIFIED_UTC"]
MEMBER_RE = re.compile(r"^bh(\d{6}|\d{8})\.csv$", re.IGNORECASE)
SERIES_HEADERS = ("SERIES", "SR")
HEADER_ALIASES = {"INDEXFLAG": "INDEX FLAG"}                    # 2011-04-05 spells it without a space
ALLOWED_HEADERS = ({"SYMBOL", "SECURITY", "HIGH/LOW"}, {"SYMBOL", "SECURITY", "HIGH/LOW", "INDEX FLAG"})
MANIFEST_PATH = MANIFEST_DIR / "price_band_hits.jsonl"


def build_url(trade_date: date) -> str:
    return BASE_URL.format(ddmmyy=trade_date.strftime("%d%m%y"))


def _member_date(name: str) -> date:
    digits = MEMBER_RE.match(name.rsplit("/", 1)[-1]).group(1)
    fmt = "%d%m%Y" if len(digits) == 8 else "%d%m%y"
    return datetime.strptime(digits, fmt).date()


def parse_pr_zip(content: bytes, trade_date: date, last_modified: str | None) -> pd.DataFrame:
    """The band-hit rows of one day's PR zip. Raises ValueError on a non-zip, a missing/ambiguous/misdated bh member, an
    unknown header or token, or a duplicate (symbol, series, hit)."""
    stamp = _last_modified_utc(last_modified)
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise ValueError(f"content is not a zip archive ({len(content)} bytes)") from None
    # some 2013 zips hold every file inside a folder ('nupr100113/bh100113.csv'), so match on the base name; folder entries are not files
    members = [name for name in archive.namelist() if not name.endswith("/") and MEMBER_RE.match(name.rsplit("/", 1)[-1])]
    if not members:
        raise ValueError(f"no bh member in the zip (members: {archive.namelist()[:6]})")
    if len(members) > 1:
        raise ValueError(f"more than one bh member: {members}")
    member = members[0]
    if _member_date(member) != trade_date:
        raise ValueError(f"member {member} is dated {_member_date(member)}, requested {trade_date}: a stale re-serve")

    rows = [row for row in csv.reader(io.StringIO(archive.read(member).decode("latin-1"))) if row]
    header = [HEADER_ALIASES.get(cell.strip().upper(), cell.strip().upper()) for cell in rows[0]] if rows else []
    while header and header[-1] == "":
        header.pop()                                             # the 2010 fixed-width export pads the header with empty cells
    series_col = next((name for name in SERIES_HEADERS if name in header), None)
    if series_col is None or set(header) - {series_col} not in ALLOWED_HEADERS or len(set(header)) != len(header):
        raise ValueError(f"unrecognised header {header}")
    at = {name: header.index(name) for name in header}

    records = []
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))          # a trailing blank INDEX FLAG may be dropped by the exporter
        # INDEX FLAG is blank in almost every row, and 'CNX Nifty Junior' on a few in 2010: an index-membership label, ignored
        hit = row[at["HIGH/LOW"]].strip().upper()
        if hit not in ("H", "L"):
            raise ValueError(f"HIGH/LOW {hit!r} is neither H nor L for {row[at['SYMBOL']]}")
        symbol, series = row[at["SYMBOL"]].strip(), row[at[series_col]].strip()
        if not symbol or not series:
            raise ValueError(f"blank symbol or series in row {row}")
        records.append((pd.Timestamp(trade_date), symbol, series, hit, stamp))

    if not records:
        return pd.DataFrame({"DATE": [pd.Timestamp(trade_date)], "SYMBOL": pd.Series([None], dtype="string"),
                             "SERIES": pd.Series([None], dtype="string"), "HIT": pd.Series([None], dtype="string"),
                             "LAST_MODIFIED_UTC": [stamp]})[OUTPUT_COLS]
    out = pd.DataFrame(records, columns=OUTPUT_COLS)
    if out.duplicated(["SYMBOL", "SERIES", "HIT"]).any():
        raise ValueError(f"duplicate (symbol, series, hit) rows: {out[out.duplicated(['SYMBOL', 'SERIES', 'HIT'], keep=False)]['SYMBOL'].unique()[:5].tolist()}")
    return out


def fetch_day(session: requests.Session, trade_date: date) -> tuple[pd.DataFrame | None, str, int]:
    """One day. A None dataframe means no zip for that date (weekend / holiday)."""
    resp = session.get(build_url(trade_date), timeout=120)
    if resp.status_code == 404:
        return None, "pr_zip_bh", 0
    resp.raise_for_status()
    return parse_pr_zip(resp.content, trade_date, resp.headers.get("Last-Modified")), "pr_zip_bh", len(resp.content)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSE's daily price-band hit list (from the PR zip).")
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
        name="price_band_hits", start=start, end=end, output_dir=PRICE_BAND_HITS_DIR, fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH, log_dir=LOG_DIR, request_delay=args.delay, days="weekends" if args.weekends else "weekdays",
    )


if __name__ == "__main__":
    main()
