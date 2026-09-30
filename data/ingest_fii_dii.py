"""
Stage 8 — FII/DII forward capture.

No free source has multi-year FII/DII history — confirmed across four
independent attempts: NSE's own API (today-only), NSDL's official portal
(today-only), a third-party aggregator (robots.txt disallowed), and an
MIT-licensed community tracker (well-built, but its own roadmap admits
historical backfill isn't done — ~115 commits, a few months at most).

This is the honest fallback: capture today's real value every time this
runs, building genuine history forward from whenever it starts. It will
NOT have data for the older walk-forward folds — that gap is real and
should be documented, not hidden. Regime detection for those folds runs
on the three confirmed inputs (VIX, Nifty 500 breadth, USDINR) only.

Run this daily (e.g. via Task Scheduler) for it to actually accumulate
anything — a single run just adds one day.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FII_DII_DIR, PROJECT_ROOT
from data.ingest_common import get_logger, write_parquet_atomic

IST = timezone(timedelta(hours=5, minutes=30))
OUTPUT_COLS = ["DATE", "CATEGORY", "BUY_VALUE_CR", "SELL_VALUE_CR", "NET_VALUE_CR", "SOURCE", "CAPTURED_AT_IST"]
ENDPOINT = "fiidiiTradeReact"           # combined NSE+BSE+MSEI cash figures, as on NSE's FII/DII page
OUTPUT_DIR = FII_DII_DIR
LOG_DIR = PROJECT_ROOT / "logs"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def _num(text) -> float:
    try:
        return float(str(text).strip().replace(",", ""))
    except ValueError:
        return float("nan")


def parse_payload(data, captured_at: datetime, source: str) -> pd.DataFrame | None:
    """
    Turn NSE's FII/DII payload into rows, stamping each with WHEN it was captured. That timestamp is the only evidence
    of when a value was public (NSE serves the latest day only), so it is required and must be timezone-aware.
    Returns None unless the payload holds both FII and DII rows for one date: a partial answer is a failed capture.
    """
    if captured_at.tzinfo is None:
        raise ValueError("captured_at has no timezone; refusing to assume one")
    if not isinstance(data, list) or len(data) < 2:
        return None
    rows = []
    for rec in data:
        try:
            rows.append({
                "DATE": pd.to_datetime(rec["date"], format="%d-%b-%Y"),
                "CATEGORY": rec["category"],
                "BUY_VALUE_CR": _num(rec["buyValue"]),
                "SELL_VALUE_CR": _num(rec["sellValue"]),
                "NET_VALUE_CR": _num(rec["netValue"]),
                "SOURCE": source,
                "CAPTURED_AT_IST": captured_at.astimezone(IST).isoformat(),
            })
        except (KeyError, ValueError, TypeError):
            return None
    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    if df["DATE"].nunique() != 1 or not {"FII/FPI", "DII"} <= set(df["CATEGORY"]):
        return None
    return df


def fetch_today() -> pd.DataFrame | None:
    """NSE's own endpoint: FII and DII rows for the latest published session, stamped with the capture time."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass

    resp = session.get(
        f"https://www.nseindia.com/api/{ENDPOINT}",
        headers={"Referer": "https://www.nseindia.com/reports/fii-dii"},
        timeout=20,
    )
    if resp.status_code != 200:
        return None
    return parse_payload(resp.json(), datetime.now(IST), ENDPOINT)


def main() -> None:
    logger = get_logger("fii_dii", LOG_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_today()
    if df is None:
        logger.warning("Fetch failed or returned unexpected shape — nothing written")
        return

    trade_date = df["DATE"].iloc[0].date()
    out_path = OUTPUT_DIR / f"{trade_date.isoformat()}.parquet"

    if out_path.exists():
        logger.info(f"{trade_date}: already captured, skipping")
        return

    write_parquet_atomic(df, out_path)
    logger.info(f"{trade_date}: captured — {dict(zip(df['CATEGORY'], df['NET_VALUE_CR']))}")

    existing = sorted(OUTPUT_DIR.glob("*.parquet"))
    logger.info(f"Total days captured so far: {len(existing)} "
                f"({existing[0].stem} to {existing[-1].stem})")


if __name__ == "__main__":
    main()