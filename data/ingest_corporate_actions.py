"""
Stage 4 — Corporate action ingestion.

Fetches, parses, and stores every corporate action (bonus, split, dividend,
rights, buyback) for every symbol that has ever appeared in the bhavcopy
history. This is the raw action log — NOT the price adjustment itself.
That's a separate step (adjust_prices.py) that reads this output plus the
bhavcopy series and produces adjusted prices, kept separate so each piece
can be tested independently, same discipline as every other stage tonight.

Access pattern differs from bhavcopy/delivery: this hits NSE's dynamic API
(nseindia.com/api/...), which is per-symbol rather than per-date, and
appears capped around 20 records per request (confirmed via probing) — so
each symbol is queried in yearly chunks, not one wide range.

Universe bootstrap: NSE's corporate-actions endpoint has no bulk/all-symbols
mode, so we need a symbol list to iterate over. Stage 5 (proper point-in-time
universe construction) hasn't happened yet, so this bootstraps from the
distinct SYMBOL values already sitting in the completed bhavcopy backfill —
a strict superset including delisted names, which is the historically
CORRECT superset for this purpose (a delisted stock's old corporate actions
still matter for any fold that trained on it while it was listed).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MAX_WORKERS = 4   # each with its own session; total request rate stays similar

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BHAVCOPY_DIR, PROJECT_ROOT
from corporate_action_parser import parse_subject, adjustment_factor
from data.ingest_common import get_logger, write_parquet_atomic, Manifest, DayRecord

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "corporate_actions"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifests" / "corporate_actions.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# Confirmed via probe: requests capped around 20 records. One-year windows
# keep well under that even for the most dividend-active names.
CHUNK_YEARS = 1
REQUEST_DELAY = 0.6   # gentler than the archive fetches — this is NSE's
                       # more fragile dynamic API, not the static archive


def get_universe_from_bhavcopy() -> dict[str, tuple[int, int]]:
    """
    Every distinct EQ symbol that has ever appeared in bhavcopy, mapped to
    its own (first_year, last_year) trading range — not a fixed 2000-2026
    window for all 5,326 symbols regardless of when each actually listed.

    Without this, every symbol queries all 27 years even if it IPO'd in
    2021 or delisted in 2010, which is why the naive version was on pace
    for ~28 hours. Scoping each symbol to its real range cuts total
    requests by roughly the average unused-year fraction across the
    universe — most symbols have far shorter real histories than 27 years.
    """
    files = sorted(BHAVCOPY_DIR.glob("*.parquet"))
    sampled = files[::20]   # monthly-granularity sample, same as before

    first_seen: dict[str, int] = {}
    last_seen: dict[str, int] = {}

    for f in sampled:
        year = int(f.stem[:4])
        df = pd.read_parquet(f, columns=["SYMBOL", "SERIES"])
        eq_symbols = df.loc[df["SERIES"] == "EQ", "SYMBOL"].unique()
        for sym in eq_symbols:
            if sym not in first_seen or year < first_seen[sym]:
                first_seen[sym] = year
            if sym not in last_seen or year > last_seen[sym]:
                last_seen[sym] = year

    # Pad one year either side to cover the monthly-sampling gap.
    return {
        sym: (max(2000, first_seen[sym] - 1), min(date.today().year, last_seen[sym] + 1))
        for sym in first_seen
    }


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass   # confirmed in probing: the API responds even when this 403s
    return session


def fetch_symbol_year(
    session: requests.Session, symbol: str, year: int
) -> list[dict]:
    """One symbol, one calendar year. Returns raw NSE records (unparsed)."""
    url = "https://www.nseindia.com/api/corporates-corporateActions"
    params = {
        "index": "equities",
        "symbol": symbol,
        "from_date": f"01-01-{year}",
        "to_date": f"31-12-{year}",
    }
    resp = session.get(
        url, params=params,
        headers={"Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions"},
        timeout=20,
    )
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}")

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected response shape: {type(data)}")
    return data


def parse_nse_date(s: str) -> pd.Timestamp | None:
    """NSE dates arrive as 'DD-Mon-YYYY' or '-' for not-applicable."""
    if not s or s == "-":
        return None
    try:
        return pd.to_datetime(s, format="%d-%b-%Y")
    except (ValueError, TypeError):
        return None


def ingest_symbol(
    session: requests.Session, symbol: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """Fetch and parse one symbol's full corporate-action history."""
    all_records = []

    for year in range(start_year, end_year + 1):
        try:
            raw_records = fetch_symbol_year(session, symbol, year)
        except (requests.RequestException, ValueError) as e:
            # One bad year for one symbol shouldn't abort the whole symbol —
            # log it and continue to the next year.
            print(f"  {symbol} {year}: FETCH FAILED — {e}")
            time.sleep(REQUEST_DELAY)
            continue

        for raw in raw_records:
            subject = raw.get("subject", "")
            face_val = raw.get("faceVal")
            parsed = parse_subject(subject, face_val)
            factor = adjustment_factor(parsed)

            all_records.append({
                "symbol": symbol,
                "ex_date": parse_nse_date(raw.get("exDate", "")),
                "record_date": parse_nse_date(raw.get("recDate", "")),
                "bc_start_date": parse_nse_date(raw.get("bcStartDate", "")),
                "bc_end_date": parse_nse_date(raw.get("bcEndDate", "")),
                "isin": raw.get("isin"),
                "series": raw.get("series"),
                "face_value": face_val,
                "raw_subject": subject,
                "action_type": parsed.action_type.value,
                "ratio_new": parsed.ratio_new,
                "ratio_old": parsed.ratio_old,
                "face_value_from": parsed.face_value_from,
                "face_value_to": parsed.face_value_to,
                "dividend_amount": parsed.dividend_amount,
                "dividend_percent": parsed.dividend_percent,
                "parse_confidence": parsed.parse_confidence,
                "adjustment_factor": factor,
            })

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["symbol", "ex_date", "raw_subject"]
        ).sort_values("ex_date")
    return df


def main(start_year: int = 2000, end_year: int | None = None) -> None:
    if end_year is None:
        end_year = date.today().year

    logger = get_logger("corporate_actions", LOG_DIR)
    manifest = Manifest(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_symbol(symbol: str, sym_start: int, sym_end: int) -> dict:
    """One symbol, own session — runs inside a worker thread."""
    out_path = OUTPUT_DIR / f"{symbol}.parquet"
    if out_path.exists():
        return {"symbol": symbol, "status": "skipped"}

    session = get_session()
    try:
        df = ingest_symbol(session, symbol, sym_start, sym_end)
    except Exception as e:                               # noqa: BLE001
        return {"symbol": symbol, "status": "failed", "error": str(e)[:300]}

    if df.empty:
        return {"symbol": symbol, "status": "empty"}

    write_parquet_atomic(df, out_path)
    unparseable = int((df["parse_confidence"] == "failed").sum())
    return {"symbol": symbol, "status": "ok", "rows": len(df), "unparseable": unparseable}


def main(start_year: int = 2000, end_year: int | None = None) -> None:
    if end_year is None:
        end_year = date.today().year

    logger = get_logger("corporate_actions", LOG_DIR)
    manifest = Manifest(MANIFEST_PATH)
    manifest_lock = threading.Lock()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Bootstrapping universe from bhavcopy...")
    universe = get_universe_from_bhavcopy()
    total_requests = sum(hi - lo + 1 for lo, hi in universe.values())
    logger.info(
        f"Universe: {len(universe):,} symbols | {total_requests:,} year-requests | "
        f"{MAX_WORKERS} workers | est. {total_requests * REQUEST_DELAY / 60 / MAX_WORKERS:.0f} min"
    )

    flagged_unparseable = 0
    total_actions = 0
    completed = 0
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_symbol, sym, lo, hi): sym
            for sym, (lo, hi) in universe.items()
        }

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            with manifest_lock:
                if result["status"] == "ok":
                    total_actions += result["rows"]
                    flagged_unparseable += result["unparseable"]
                    manifest.add(DayRecord(result["symbol"], "ok", rows=result["rows"]))
                    if result["unparseable"] > 0:
                        logger.warning(
                            f"{result['symbol']}: {result['unparseable']} "
                            f"action(s) unparseable — flagged for review"
                        )
                elif result["status"] == "failed":
                    manifest.add(DayRecord(result["symbol"], "failed", error=result["error"]))
                    logger.warning(f"{result['symbol']}: FAILED — {result['error']}")
                elif result["status"] == "empty":
                    manifest.add(DayRecord(result["symbol"], "empty"))

                if completed % 25 == 0:
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_min = (len(universe) - completed) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{len(universe)}] total_actions={total_actions:,} "
                        f"unparseable={flagged_unparseable} | "
                        f"elapsed {elapsed/60:.1f}m | eta {eta_min:.0f}m"
                    )
                    manifest.flush()

    manifest.flush()
    logger.info(
        f"Complete. {total_actions:,} actions across {len(universe)} symbols. "
        f"{flagged_unparseable} flagged for manual review."
    )


if __name__ == "__main__":
    main()