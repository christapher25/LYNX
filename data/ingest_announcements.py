"""
Stage 16 — NSE corporate-announcement ingestion with entity resolution.

Source (probed 2026-10-31):

    https://www.nseindia.com/api/corporate-announcements?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY

    JSON list, one object per announcement. Needs the session cookies the NSE home page sets (the home page itself answers
    403 to a script, but still sets them). A month-wide range is served in one response with no cap (Sep 2026: 17,132
    records; a week 2,341). History starts in 2003 and is sparse until 2005 (2003: 142 records, Jan 2004: 9, Jan 2005:
    1,046); dates before 2003 return an empty list. Announcements are issued on weekends too, so every calendar day is
    asked about, not only sessions.

Keying. Every record is keyed by PUBLICATION time (`PUBLISHED_AT`, IST, tz-aware), never by the event the text describes.
`an_dt` is the broadcast stamp; `exchdisstime` (present from about 2020) is the exchange's dissemination stamp. The LATER of
the two is the publication time: a record is never treated as public earlier than either stamp says. Old records carry a
date-only stamp (00:00:00); they are flagged TIME_PRECISION="date" and placed at 23:59:59 IST of that date, so the
point-in-time primitive (data/publication.py) hands them to the next session's run, never the same day's.

Output: one Parquet per chunk, `announcements_{start}_{end}.parquet`, chunks being calendar months (clamped to the
requested range). Each record also carries TICKER / RESOLUTION_METHOD / UNRESOLVED_REASON from data/entity_resolution.py:
resolved records name the method; the rest keep NaN ticker and a reason. Nothing is dropped for being unresolved.
A chunk whose file name exists is done and is skipped on resume; a month still growing (end date moves) rewrites its
file and removes the superseded shorter one. Failed or empty chunks are written nowhere and retried on the next run.
The manifest (data/raw/_manifests/announcements.jsonl) logs every attempt.

BSE: not ingested. Its API is keyed by BSE scrip code, needs a scrip-to-ISIN master we do not hold, and its records would
duplicate NSE's for dual-listed names. Left as a decision for the lead (see the Stage 16 report).

Usage
-----
    python -m data.ingest_announcements --start 2026-09-14 --end 2026-09-20    # small verification range
    python -m data.ingest_announcements                                         # full history, resumable
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ANNOUNCEMENTS_DIR, BHAVCOPY_DIR, LOG_DIR, MANIFEST_DIR
from data.entity_resolution import (
    RESOLUTION_COLUMNS,
    SymbolMaster,
    build_name_index,
    build_symbol_master,
    resolve,
)
from data.ingest_common import DayRecord, Manifest, write_parquet_atomic
from data.publication import IST

FIRST_AVAILABLE = date(2003, 1, 1)
HOME_URL = "https://www.nseindia.com/"
API_URL = ("https://www.nseindia.com/api/corporate-announcements?index=equities"
           "&from_date={start}&to_date={end}")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}
MANIFEST_PATH = MANIFEST_DIR / "announcements.jsonl"
FORMAT_ERA = "corporate-announcements-v1"
FILE_RE = re.compile(r"^announcements_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
STAMP_FORMATS = ("%d-%b-%Y %H:%M:%S",)
END_OF_DAY = (23, 59, 59)

RECORD_COLUMNS = [
    "PUBLISHED_AT", "PUBLISHED_DATE", "TIME_PRECISION", "AN_DT_RAW", "EXCH_DISS_RAW",
    "SYMBOL_RAW", "ISIN", "COMPANY_NAME", "CATEGORY", "TEXT", "ATTACHMENT_URL", "SEQ_ID", "HAS_XBRL", "INDUSTRY",
]
OUTPUT_COLUMNS = RECORD_COLUMNS + RESOLUTION_COLUMNS


# ─────────────────────────────────────────────────────────────────
# Chunking and timestamps
# ─────────────────────────────────────────────────────────────────

def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-month chunks covering [start, end], the first and last clamped to the requested range."""
    chunks, cursor = [], start
    while cursor <= end:
        month_end = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1])
        chunks.append((cursor, min(month_end, end)))
        cursor = month_end + timedelta(days=1)
    return chunks


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() in {"", "-"}


def _parse_stamp(text) -> datetime:
    if _blank(text):
        raise ValueError(f"missing timestamp {text!r}")
    for fmt in STAMP_FORMATS:
        try:
            return datetime.strptime(str(text).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {text!r}")


def parse_published_at(an_dt, exchdisstime) -> tuple[pd.Timestamp, str]:
    """
    (publication time in IST, precision). Later of the two stamps; a date-only stamp (00:00:00) becomes 23:59:59 of its
    date and precision "date". A missing or unparseable an_dt raises: a record that cannot be keyed is not guessed at.
    """
    stamps = [_parse_stamp(an_dt)]
    if not _blank(exchdisstime):
        stamps.append(_parse_stamp(exchdisstime))
    latest = max(stamps)
    precision = "second"
    if latest.hour == latest.minute == latest.second == 0:
        latest = latest.replace(hour=END_OF_DAY[0], minute=END_OF_DAY[1], second=END_OF_DAY[2])
        precision = "date"
    return pd.Timestamp(latest, tz=IST), precision


# ─────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────

def parse_bool(value):
    """True/False for a bool or the strings "true"/"false"; NA for missing. Anything else is an error, never bool(str)."""
    if isinstance(value, bool):
        return value
    if _blank(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise ValueError(f"unrecognised boolean {value!r}")


def _nz(value):
    return pd.NA if _blank(value) else str(value).strip()


def parse_announcements(payload, start: date, end: date) -> pd.DataFrame:
    """
    Normalise one API response into RECORD_COLUMNS. Raises on a non-list payload, on a record that cannot be keyed, and on
    a record whose broadcast date lies outside [start, end] (the API ignoring its date parameters would otherwise load a
    wrong window silently). Exact duplicate records are dropped; same-text records with different ids are kept.
    """
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list of announcements, got {type(payload).__name__}")
    rows = []
    for item in payload:
        published_at, precision = parse_published_at(item.get("an_dt"), item.get("exchdisstime"))
        broadcast_day = _parse_stamp(item.get("an_dt")).date()
        if not start <= broadcast_day <= end:
            raise ValueError(f"record broadcast {broadcast_day} is outside the requested range {start}..{end}")
        rows.append({
            "PUBLISHED_AT": published_at,
            "PUBLISHED_DATE": pd.Timestamp(published_at.date()),
            "TIME_PRECISION": precision,
            "AN_DT_RAW": _nz(item.get("an_dt")),
            "EXCH_DISS_RAW": _nz(item.get("exchdisstime")),
            "SYMBOL_RAW": _nz(item.get("symbol")),
            "ISIN": _nz(item.get("sm_isin")),
            "COMPANY_NAME": _nz(item.get("sm_name")),
            "CATEGORY": _nz(item.get("desc")),
            "TEXT": _nz(item.get("attchmntText")),
            "ATTACHMENT_URL": _nz(item.get("attchmntFile")),
            "SEQ_ID": _nz(item.get("seq_id")),
            "HAS_XBRL": parse_bool(item.get("hasXbrl")),
            "INDUSTRY": _nz(item.get("smIndustry")),
        })
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in RECORD_COLUMNS})
    frame = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    frame["HAS_XBRL"] = frame["HAS_XBRL"].astype("boolean")
    return frame.drop_duplicates().reset_index(drop=True)


def build_chunk_frame(payload, start: date, end: date, master: SymbolMaster) -> pd.DataFrame:
    """Parse and entity-resolve one chunk. Two passes: symbol/ISIN, then the name index those records yield."""
    records = parse_announcements(payload, start, end)
    if records.empty:
        return records.assign(**{c: pd.Series(dtype="string") for c in RESOLUTION_COLUMNS})[OUTPUT_COLUMNS]
    first = resolve(records, master, {})
    return resolve(records, master, build_name_index(first))[OUTPUT_COLUMNS]


# ─────────────────────────────────────────────────────────────────
# Invariants (shared by the tests and the verifier)
# ─────────────────────────────────────────────────────────────────

def announcement_invariant_violations(df: pd.DataFrame, master: SymbolMaster | None = None) -> dict[str, int]:
    """name -> count for every invariant that fails; {} means clean."""
    found: dict[str, int] = {}

    def flag(name: str, mask) -> None:
        count = int(mask.sum())
        if count:
            found[name] = count

    if df.empty:
        return found
    unresolved = df["RESOLUTION_METHOD"] == "unresolved"
    flag("ticker_on_unresolved", unresolved & df["TICKER"].notna())
    flag("resolved_without_ticker", ~unresolved & df["TICKER"].isna())
    flag("unresolved_without_reason", unresolved & df["UNRESOLVED_REASON"].isna())
    flag("resolved_with_reason", ~unresolved & df["UNRESOLVED_REASON"].notna())
    flag("unknown_method", ~df["RESOLUTION_METHOD"].isin(["symbol", "isin", "name", "unresolved"]))
    if master is not None:
        flag("ticker_not_in_master", df["TICKER"].notna() & ~df["TICKER"].fillna("").map(master.has_symbol))

    if getattr(df["PUBLISHED_AT"].dt, "tz", None) is None:
        found["published_at_not_tz_aware"] = len(df)
        return found
    local = df["PUBLISHED_AT"].dt.tz_convert(IST).dt.tz_localize(None)
    flag("published_date_mismatch", df["PUBLISHED_DATE"] != local.dt.normalize())
    flag("date_precision_not_end_of_day", (df["TIME_PRECISION"] == "date") & (local.dt.strftime("%H:%M:%S") != "23:59:59"))

    an_dt = df["AN_DT_RAW"].map(_parse_stamp)
    an_ist = pd.to_datetime(an_dt).dt.tz_localize(IST)
    flag("published_before_source_stamp", df["PUBLISHED_AT"] < an_ist)
    diss = df["EXCH_DISS_RAW"].dropna()
    if len(diss):
        diss_ist = pd.to_datetime(diss.map(_parse_stamp)).dt.tz_localize(IST)
        flag("published_before_source_stamp", df.loc[diss.index, "PUBLISHED_AT"] < diss_ist)
    return found


# ─────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────

def api_session() -> requests.Session:
    """A session carrying NSE's cookies. The home page answers 403 to a script but sets the cookies regardless."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(HOME_URL, timeout=30)
    return session


def decode_payload(resp) -> list:
    """The response body as a JSON list, or an error: an HTML block page or a JSON error object must never read as 'no announcements'."""
    if resp.status_code in (401, 403):
        raise PermissionError(f"HTTP {resp.status_code}: session cookies rejected")
    if resp.status_code != 200:
        raise requests.HTTPError(f"HTTP {resp.status_code}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise ValueError(f"response is not JSON: {resp.content[:80]!r}") from exc
    if not isinstance(body, list):
        raise ValueError(f"response is not a list: {str(body)[:80]!r}")
    return body


class Fetcher:
    """Fetches one chunk per call; renews the cookies when NSE rejects them."""

    def __init__(self) -> None:
        self.session = api_session()

    @retry(retry=retry_if_exception_type((requests.RequestException, PermissionError, ValueError)),
           stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def __call__(self, start: date, end: date) -> tuple[list, int]:
        url = API_URL.format(start=start.strftime("%d-%m-%Y"), end=end.strftime("%d-%m-%Y"))
        try:
            resp = self.session.get(url, timeout=120)
            return decode_payload(resp), len(resp.content)
        except (PermissionError, ValueError):
            self.session = api_session()
            raise


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────

def chunk_path(out_dir: Path, start: date, end: date) -> Path:
    return out_dir / f"announcements_{start.isoformat()}_{end.isoformat()}.parquet"


def existing_chunks(out_dir: Path) -> list[tuple[date, date, Path]]:
    found = []
    for path in out_dir.glob("announcements_*.parquet"):
        match = FILE_RE.match(path.name)
        if match:
            found.append((date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2)), path))
    return found


def classify_overlap(out_dir: Path, start: date, end: date) -> tuple[str, list[Path]]:
    """
    ("covered", [file]) when an existing file already holds [start, end]; ("partial", files) when one overlaps without
    containing or being contained (writing would duplicate records); else ("clear", files inside the range, which the
    new file will supersede).
    """
    inside, partial = [], []
    for a, b, path in existing_chunks(out_dir):
        if a <= start and end <= b:
            return "covered", [path]
        if start <= a and b <= end:
            inside.append(path)
        elif a <= end and start <= b:
            partial.append(path)
    return ("partial", partial) if partial else ("clear", inside)


def _row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def clean_leftovers(out_dir: Path, covering: Path) -> None:
    """
    Remove files lying inside `covering`'s range: what a crash between the atomic write and the unlink leaves behind.
    Only when the covering file holds at least as many rows as they do together; otherwise warn and keep everything.
    """
    match = FILE_RE.match(covering.name)
    start, end = date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2))
    inside = [p for a, b, p in existing_chunks(out_dir) if p != covering and start <= a and b <= end]
    if not inside:
        return
    held, leftover = _row_count(covering), sum(_row_count(p) for p in inside)
    if held < leftover:
        logger.warning(f"{covering.name} has {held} rows, fewer than the {leftover} in {[p.name for p in inside]}; keeping all")
        return
    for old in inside:
        old.unlink()
        logger.info(f"removed leftover {old.name} (inside {covering.name})")


def run(start: date, end: date, *, out_dir: Path, manifest_path: Path, master: SymbolMaster,
        fetch_fn: Callable[[date, date], tuple[list, int]] | None = None, delay: float = 1.0) -> dict:
    """Resumable chunk loop. Returns counts of ok / empty / failed / skipped chunks and rows written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(manifest_path)
    fetch = fetch_fn
    counts = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0, "rows": 0}
    chunks = month_chunks(start, end)
    logger.info(f"announcements: {len(chunks)} chunks from {start} to {end}")

    for i, (a, b) in enumerate(chunks, 1):
        path, key = chunk_path(out_dir, a, b), f"{a}_{b}"
        overlap, others = classify_overlap(out_dir, a, b)
        if overlap == "covered":
            counts["skipped"] += 1
            clean_leftovers(out_dir, others[0])
            continue
        if overlap == "partial":
            counts["failed"] += 1
            names = [p.name for p in others]
            manifest.add(DayRecord(key, "failed", error=f"partially overlaps existing {names}"[:300], format_era=FORMAT_ERA))
            logger.error(f"{key}: partially overlaps {names}; refusing to write duplicates (align the range to those files)")
            manifest.flush()
            continue
        try:
            if fetch is None:
                fetch = Fetcher()
            payload, nbytes = fetch(a, b)
            frame = build_chunk_frame(payload, a, b, master)
        except Exception as exc:                                   # noqa: BLE001
            counts["failed"] += 1
            manifest.add(DayRecord(key, "failed", error=str(exc)[:300], format_era=FORMAT_ERA))
            logger.warning(f"{key}: FAILED: {exc}")
        else:
            if frame.empty:
                counts["empty"] += 1
                manifest.add(DayRecord(key, "empty", bytes_downloaded=nbytes, format_era=FORMAT_ERA))
                logger.warning(f"{key}: zero announcements; nothing written, will be retried")
            elif len(frame) < sum(_row_count(p) for p in others):
                counts["failed"] += 1
                msg = f"new chunk has {len(frame)} rows, fewer than the {sum(_row_count(p) for p in others)} rows in {[p.name for p in others]} it would replace"
                manifest.add(DayRecord(key, "failed", error=msg[:300], format_era=FORMAT_ERA))
                logger.warning(f"{key}: {msg}; nothing written, existing files kept")
            else:
                write_parquet_atomic(frame, path)
                for old in others:
                    old.unlink()
                    logger.info(f"removed superseded {old.name}")
                counts["ok"] += 1
                counts["rows"] += len(frame)
                manifest.add(DayRecord(key, "ok", rows=len(frame), bytes_downloaded=nbytes, format_era=FORMAT_ERA))
                unresolved = int((frame["RESOLUTION_METHOD"] == "unresolved").sum())
                logger.info(f"[{i}/{len(chunks)}] {key}: {len(frame):,} records, {unresolved:,} unresolved")
        manifest.flush()
        if delay:
            time.sleep(delay)

    logger.info(f"announcements complete: {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSE corporate announcements with entity resolution.")
    ap.add_argument("--start", type=str, help="start date YYYY-MM-DD (default 2003-01-01)")
    ap.add_argument("--end", type=str, help="end date YYYY-MM-DD (default yesterday)")
    ap.add_argument("--days", type=int, help="only the last N days")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1.0)")
    ap.add_argument("--out-dir", type=str, help="override the output directory (verification runs)")
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1) if args.days else (
        date.fromisoformat(args.start) if args.start else FIRST_AVAILABLE)
    start = max(start, FIRST_AVAILABLE)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_DIR / "announcements.log", rotation="10 MB")
    master = build_symbol_master(BHAVCOPY_DIR)
    run(start, end, out_dir=Path(args.out_dir) if args.out_dir else ANNOUNCEMENTS_DIR,
        manifest_path=MANIFEST_PATH, master=master, delay=args.delay)


if __name__ == "__main__":
    main()
