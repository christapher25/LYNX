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

Modes:
  default (no flags): per-symbol full history; a symbol that already has a file is skipped, never refetched.
  --refresh-from YYYY-MM-DD: for every symbol that ALREADY has a file, request only the year chunks covering
    [refresh_from, today] and append actions not already present (key: symbol, ex_date, raw_subject; the existing row wins a
    tie). Existing rows are never altered or removed; a file is rewritten atomically only when there are new rows dated on or
    after refresh_from (a new row dated earlier is reported and NOT added, so history before refresh_from cannot change). Any
    failed year request or failed write fails that symbol and leaves its file untouched. A fetched factor-bearing row with the
    same ex_date/type/factor as an existing row (reworded subject) is NOT appended (it would double-adjust) and is warned;
    an existing row dated >= refresh_from that is missing from the fetch is warned. Symbols without a file are ignored.
    Expected requests = (symbols with a file) x (today.year - refresh_from.year + 1), at REQUEST_DELAY per request across
    MAX_WORKERS workers. --symbols SYM [SYM ...] restricts the refresh to the named symbols (each must have a file).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
try:
    from corporate_action_parser import parse_subject, adjustment_factor   # run as a script: data/ is on sys.path
except ModuleNotFoundError:
    from data.corporate_action_parser import parse_subject, adjustment_factor   # run as a module / imported by tests
from loguru import logger as log
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


def _today() -> date:
    return date.today()


def ingest_symbol(
    session: requests.Session, symbol: str, start_year: int, end_year: int, raise_on_fetch_error: bool = False
) -> pd.DataFrame:
    """Fetch and parse one symbol's corporate-action history for the given years.

    Default: a failed year is logged and skipped (original behaviour). raise_on_fetch_error=True (refresh mode): the first
    failed year raises, because a silently missing year would look like "no new actions"."""
    all_records = []

    for year in range(start_year, end_year + 1):
        try:
            raw_records = fetch_symbol_year(session, symbol, year)
        except (requests.RequestException, ValueError) as e:
            if raise_on_fetch_error:
                raise RuntimeError(f"{symbol} {year}: fetch failed: {e}") from e
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


# --------------------------------------------------------------------------------------------- refresh mode
_KEY = ["symbol", "ex_date", "raw_subject"]


def _existing_symbols() -> list[str]:
    return sorted(path.stem for path in OUTPUT_DIR.glob("*.parquet"))


def _select_symbols(symbols: list[str] | None) -> list[str]:
    """All symbols with a file, or the named ones (deduplicated, in the given order). A named symbol with no file is an error:
    refresh mode never creates a file."""
    existing = _existing_symbols()
    if symbols is None:
        return existing
    chosen = list(dict.fromkeys(symbols))
    missing = [s for s in chosen if s not in set(existing)]
    if missing:
        raise ValueError(f"no existing corporate-actions file for: {missing} (refresh mode only updates existing files)")
    return chosen


def expected_refresh_requests(refresh_from: date, symbols: list[str] | None = None) -> int:
    """(symbols that already have a file, or the named ones) x (year chunks covering refresh_from..today)."""
    return len(_select_symbols(symbols)) * (_today().year - refresh_from.year + 1)


def _semantic_key(ex_date, action_type, factor):
    """Wording-independent identity of an action: (ex_date, type, adjustment_factor)."""
    d = None if pd.isna(ex_date) else str(pd.Timestamp(ex_date))[:10]
    f = None if pd.isna(factor) else round(float(factor), 12)
    return d, action_type, f


def refresh_symbol(symbol: str, refresh_from: date, end_year: int) -> dict:
    """One symbol with an existing file: fetch refresh_from.year..end_year, append unseen rows dated >= refresh_from.
    Never alters or removes an existing row; writes only when there are new rows; any fetch or write failure leaves the file
    untouched and marks the symbol failed.

    Guards: (1) a fetched row with an adjustment factor is NOT appended if an existing row has the same ex_date, type and factor
    (NSE reworded the subject; appending would double-adjust prices); (2) an existing row dated >= refresh_from that does not
    reappear in the fetch only logs a WARNING (an empty 200 or rate limit), it does not fail the symbol."""
    path = OUTPUT_DIR / f"{symbol}.parquet"
    if not path.exists():
        return {"symbol": symbol, "status": "skipped"}
    try:
        existing = pd.read_parquet(path)
        fetched = ingest_symbol(get_session(), symbol, refresh_from.year, end_year, raise_on_fetch_error=True)
    except Exception as e:                                       # noqa: BLE001
        return {"symbol": symbol, "status": "failed", "error": str(e)[:300]}

    cutoff = pd.Timestamp(refresh_from)
    window = existing[pd.to_datetime(existing["ex_date"]) >= cutoff]
    fetched_keys = set() if fetched.empty else set(map(tuple, fetched[_KEY].itertuples(index=False, name=None)))
    fetched_sem = set() if fetched.empty else {_semantic_key(r.ex_date, r.action_type, r.adjustment_factor)
                                               for r in fetched.itertuples()}
    missing = [r for r in window.itertuples()
               if tuple(getattr(r, c) for c in _KEY) not in fetched_keys
               and _semantic_key(r.ex_date, r.action_type, r.adjustment_factor) not in fetched_sem]
    if missing:
        log.warning("{}: existing action missing from fetch: {} row(s) dated >= {} not returned (empty answer or rate limit?): {}",
                    symbol, len(missing), refresh_from, [(str(r.ex_date)[:10], r.action_type) for r in missing])
    if fetched.empty:
        return {"symbol": symbol, "status": "ok", "added": []}

    unseen = fetched[~pd.MultiIndex.from_frame(fetched[_KEY]).isin(pd.MultiIndex.from_frame(existing[_KEY]))]
    ex = pd.to_datetime(unseen["ex_date"])
    too_early = unseen[ex.isna() | (ex < cutoff)]
    in_range = unseen[ex >= cutoff]
    skipped = []   # (ex_date or "NaT", action_type, factor or None): new actions before refresh_from, deliberately NOT added
    for r in too_early.itertuples():
        factor = None if pd.isna(r.adjustment_factor) else float(r.adjustment_factor)
        skipped.append((str(r.ex_date)[:10], r.action_type, factor))
    if skipped:
        shown = [f"{d} {t}" + (f" factor={f:g}" if f is not None else "") for d, t, f in skipped]
        log.warning("{}: {} new action(s) dated before {} (or with no ex_date) were NOT added: {}", symbol, len(skipped),
                    refresh_from, shown)

    existing_by_sem = {}
    for r in existing.itertuples():
        existing_by_sem.setdefault(_semantic_key(r.ex_date, r.action_type, r.adjustment_factor), r.raw_subject)
    keep = []
    for r in in_range.itertuples():
        sem = _semantic_key(r.ex_date, r.action_type, r.adjustment_factor)
        if sem[2] is not None and sem in existing_by_sem:
            log.warning("{}: possible reworded duplicate NOT added (same ex_date/type/factor {}): existing {!r} vs fetched {!r}",
                        symbol, sem, existing_by_sem[sem], r.raw_subject)
        else:
            keep.append(r.Index)
    added = in_range.loc[keep]
    if added.empty:
        return {"symbol": symbol, "status": "ok", "added": [], "skipped": skipped}

    try:
        merged = pd.concat([existing, added[list(existing.columns)]], ignore_index=True)
        merged = merged.sort_values("ex_date", kind="stable").reset_index(drop=True)
        write_parquet_atomic(merged, path)
    except Exception as e:                                       # noqa: BLE001
        return {"symbol": symbol, "status": "failed", "error": f"write failed: {e}"[:300]}
    return {"symbol": symbol, "status": "ok", "skipped": skipped,
            "added": [(str(row.ex_date)[:10], row.action_type) for row in added.itertuples()]}


def run_refresh(refresh_from: date, workers: int = MAX_WORKERS, symbols: list[str] | None = None) -> dict:
    """Refresh every symbol that already has a file (or only the named `symbols`, each of which must have a file). Returns {"requests", "gained": {symbol: [(ex_date, type)]}, "failed": [...],
    "skipped_by_type": {type: n}, "skipped_with_factor": [(symbol, ex_date, type, factor)]} (the last two: new actions dated
    before refresh_from, which are reported and never added)."""
    today = _today()
    if refresh_from > today:
        raise ValueError(f"--refresh-from {refresh_from} is in the future (today {today})")
    symbols = _select_symbols(symbols)
    requests_planned = len(symbols) * (today.year - refresh_from.year + 1)
    log.info("refresh: {:,} symbols with a file, {} year chunk(s) each = {:,} requests, {} workers, est. {:.0f} min (delay only)",
             len(symbols), today.year - refresh_from.year + 1, requests_planned, workers,
             requests_planned * REQUEST_DELAY / 60 / max(workers, 1))
    manifest = Manifest(MANIFEST_PATH)
    lock = threading.Lock()
    gained: dict[str, list] = {}
    failed: list[str] = []
    skipped_by_type: Counter = Counter()
    skipped_with_factor: list[tuple] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(refresh_symbol, sym, refresh_from, today.year): sym for sym in symbols}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                if result["status"] == "failed":
                    failed.append(result["symbol"])
                    manifest.add(DayRecord(result["symbol"], "failed", format_era="refresh", error=result["error"]))
                    log.warning("{}: refresh FAILED, file untouched: {}", result["symbol"], result["error"])
                elif result["status"] == "ok":
                    manifest.add(DayRecord(result["symbol"], "ok", rows=len(result["added"]), format_era="refresh"))
                    for ex_date, action_type, factor in result.get("skipped", []):
                        skipped_by_type[action_type] += 1
                        if factor is not None:
                            skipped_with_factor.append((result["symbol"], ex_date, action_type, factor))
                    if result["added"]:
                        gained[result["symbol"]] = result["added"]
                        for ex_date, action_type in result["added"]:
                            log.info("{}: NEW action ex_date {} type {}", result["symbol"], ex_date, action_type)
                if done % 100 == 0:
                    manifest.flush()
    manifest.flush()
    log.info("refresh complete: {} symbol(s) gained rows ({} new action(s)), {} failed (files untouched)",
             len(gained), sum(len(v) for v in gained.values()), len(failed))
    skipped_with_factor.sort()
    log.info("pre-refresh_from actions NOT added: {} total, by action_type {}; with an adjustment factor: {}",
             sum(skipped_by_type.values()), dict(sorted(skipped_by_type.items())), skipped_with_factor)
    return {"requests": requests_planned, "gained": dict(sorted(gained.items())), "failed": sorted(failed),
            "skipped_by_type": dict(sorted(skipped_by_type.items())), "skipped_with_factor": skipped_with_factor}


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid YYYY-MM-DD date: {text!r}")


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingest NSE corporate actions (default: new symbols only).")
    parser.add_argument("--refresh-from", type=_iso_date, metavar="YYYY-MM-DD",
                        help="refresh mode: append new actions dated on/after this date to every symbol that already has a file")
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="with --refresh-from: only these symbols (each must already have a file, else exit 2)")
    args = parser.parse_args(argv)
    if args.symbols and args.refresh_from is None:
        parser.error("--symbols is only valid together with --refresh-from")
    if args.refresh_from is None:
        main()
        return 0
    try:
        _select_symbols(args.symbols)
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_refresh(args.refresh_from, symbols=args.symbols)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(cli())