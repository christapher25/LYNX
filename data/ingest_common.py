"""
Shared ingestion infrastructure for LYNX data pipelines.

Provides the concerns every ingester needs and none should reimplement:
session management with connection pooling, retry with exponential
backoff, atomic writes, structured logging, and a manifest recording
what was fetched and whether it verified.

Design notes
------------
Sequential, rate-limited fetching is deliberate. NSE's archive publishes
no rate-limit policy, and an IP block mid-backfill costs far more than
concurrency saves. The bottleneck here is politeness, not throughput.

Atomic writes matter because a backfill is thousands of sequential
writes over hours. An interrupted write leaves a truncated Parquet file
that reads as valid-but-wrong — exactly the silent-failure class this
project is built to avoid. Writing to a temp file and renaming makes
partial writes impossible to observe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Console + optional rotating file logger. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"{name}_{stamp}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────────

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def build_session(
    total_retries: int = 4,
    backoff_factor: float = 1.5,
    pool_size: int = 4,
) -> requests.Session:
    """
    Session with connection pooling and transport-level retry.

    Retries on connection errors and 5xx only. A 404 is a legitimate
    answer here ("no trading that day") and must reach the caller
    rather than being retried.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ─────────────────────────────────────────────────────────────────
# Atomic writes
# ─────────────────────────────────────────────────────────────────

def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """
    Write via temp file + rename. On POSIX and NTFS alike, rename is
    atomic, so an interrupted run can never leave a half-written file
    that later reads as valid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ─────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────

@dataclass
class DayRecord:
    """One trading day's ingestion outcome."""
    trade_date: str
    status: str            # ok | holiday | failed | empty
    rows: int = 0
    bytes_downloaded: int = 0
    format_era: str = ""
    error: str = ""


class Manifest:
    """
    Append-only record of what was fetched, when, and whether it
    verified. This is the audit trail — without it, "why does 2011
    look odd" has no answer six months from now.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[DayRecord] = []

    def add(self, record: DayRecord) -> None:
        self._records.append(record)

    def flush(self) -> None:
        """Append accumulated records as JSON lines."""
        if not self._records:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(asdict(rec)) + "\n")
        self._records.clear()

    def summary(self) -> dict:
        """Aggregate the full on-disk manifest, not just this session."""
        if not self.path.exists():
            return {}
        counts: dict[str, int] = {}
        total_rows = 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                total_rows += rec.get("rows", 0)
        return {"by_status": counts, "total_rows": total_rows}


# ─────────────────────────────────────────────────────────────────
# Date iteration
# ─────────────────────────────────────────────────────────────────

DAY_SELECTORS = {
    "weekdays": lambda d: d.weekday() < 5,      # Mon-Fri; the default, and all Stage 3 requested
    "weekends": lambda d: d.weekday() >= 5,     # Sat/Sun: Budget-day Saturdays, Muhurat and special sessions
    "all": lambda d: True,
}


def trading_day_candidates(start: date, end: date, days: str = "weekdays") -> Iterator[date]:
    """
    Days in [start, end] worth asking the archive about: "weekdays" (default), "weekends", or "all".
    Exchange holidays and ordinary weekends still surface as 404s — that is
    the archive's own answer and is more reliable than maintaining a
    holiday calendar that drifts.

    Weekdays alone silently miss every weekend trading session (Budget-day
    Saturdays, Sunday/Saturday Muhurat, special Saturday sessions); a missing
    session breaks continuity of anything measured against the previous session.
    """
    if days not in DAY_SELECTORS:
        raise ValueError(f"days must be one of {sorted(DAY_SELECTORS)}, got {days!r}")
    wanted = DAY_SELECTORS[days]
    current = start
    one_day = timedelta(days=1)
    while current <= end:
        if wanted(current):
            yield current
        current += one_day


# ─────────────────────────────────────────────────────────────────
# Ingestion driver
# ─────────────────────────────────────────────────────────────────

def run_ingestion(
    *,
    name: str,
    start: date,
    end: date,
    output_dir: Path,
    fetch_fn: Callable[[requests.Session, date], tuple[pd.DataFrame | None, str, int]],
    manifest_path: Path,
    log_dir: Path,
    request_delay: float = 0.4,
    progress_every: int = 50,
    days: str = "weekdays",
) -> dict:
    """
    Drive a resumable, rate-limited backfill.

    fetch_fn must return (dataframe_or_None, format_era, bytes_downloaded).
    Returning None means "no data for this date" — a holiday, not an error.
    """
    logger = get_logger(name, log_dir)
    manifest = Manifest(manifest_path)
    session = build_session()

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(trading_day_candidates(start, end, days))
    total = len(candidates)
    logger.info(f"{name}: {total:,} {days} candidates from {start} to {end}")

    counts = {"ok": 0, "holiday": 0, "failed": 0, "skipped": 0}
    total_rows = 0
    started = time.monotonic()

    for i, day in enumerate(candidates, 1):
        out_path = output_dir / f"{day.isoformat()}.parquet"

        if out_path.exists():
            counts["skipped"] += 1
            continue

        try:
            df, era, nbytes = fetch_fn(session, day)
        except Exception as exc:                       # noqa: BLE001
            counts["failed"] += 1
            manifest.add(DayRecord(day.isoformat(), "failed", error=str(exc)[:300]))
            logger.warning(f"{day}: FAILED — {exc}")
            time.sleep(request_delay)
            continue

        if df is None:
            counts["holiday"] += 1
            manifest.add(DayRecord(day.isoformat(), "holiday"))
        elif df.empty:
            counts["failed"] += 1
            manifest.add(DayRecord(day.isoformat(), "empty", format_era=era))
            logger.warning(f"{day}: parsed to zero rows (era={era})")
        else:
            write_parquet_atomic(df, out_path)
            counts["ok"] += 1
            total_rows += len(df)
            manifest.add(DayRecord(
                day.isoformat(), "ok", rows=len(df),
                bytes_downloaded=nbytes, format_era=era,
            ))

        if i % progress_every == 0:
            elapsed = time.monotonic() - started
            done = counts["ok"] + counts["holiday"] + counts["failed"]
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            logger.info(
                f"[{i:,}/{total:,}] ok={counts['ok']:,} "
                f"holiday={counts['holiday']:,} failed={counts['failed']:,} "
                f"skipped={counts['skipped']:,} | "
                f"eta {timedelta(seconds=int(remaining))}"
            )
            manifest.flush()

        time.sleep(request_delay)

    manifest.flush()
    elapsed = time.monotonic() - started

    logger.info(
        f"{name} complete in {timedelta(seconds=int(elapsed))} | "
        f"ok={counts['ok']:,} holiday={counts['holiday']:,} "
        f"failed={counts['failed']:,} skipped={counts['skipped']:,} | "
        f"rows={total_rows:,}"
    )

    if counts["failed"]:
        logger.warning(
            f"{counts['failed']:,} dates failed. Rerun to retry — "
            f"completed dates are skipped automatically."
        )

    return {**counts, "total_rows": total_rows, "elapsed_seconds": elapsed}