"""
Stage 3 — NSE bhavcopy ingestion. Deep price history, 1994 to present.

Handles three distinct file formats across NSE's history:

    Era 1  1994-11-03 → ~2020      12 columns, no TOTALTRADES, no ISIN
    Era 2  ~2020      → 2024-07-05  14 columns, both present
    Era 3  2024-07-08 → present     34 columns, UDiFF, includes derivatives

The 1→2 boundary is detected per-file by column presence rather than a
hardcoded date, because NSE never published when those columns appeared
and probing every date to find out is not worth the requests. The 2→3
boundary is a documented cutover (Circular 62424) and switches on date.

Delivery percentage is NOT in these files. It comes from
sec_bhavdata_full (2019-09-30 onward) via ingest_bhavdata.py.

Usage
-----
    python data/ingest_bhavcopy.py                     # full backfill
    python data/ingest_bhavcopy.py --start 2020-01-01  # from a date
    python data/ingest_bhavcopy.py --days 30           # last 30 days
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BHAVCOPY_DIR, PROJECT_ROOT
from data.ingest_common import run_ingestion

# ─────────────────────────────────────────────────────────────────
# Constants — measured in Stage 3, see data/raw/DATA_NOTES.md
# ─────────────────────────────────────────────────────────────────

FIRST_AVAILABLE = date(1994, 11, 3)      # NSE equity segment launch
UDIFF_CUTOVER = date(2024, 7, 8)         # NSE Circular 62424

LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{yyyy}/{MON}/cm{DD}{MON}{yyyy}bhav.csv.zip"
)
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

OUTPUT_COLS = [
    "SYMBOL", "SERIES", "DATE", "OPEN", "HIGH", "LOW", "CLOSE",
    "LAST", "PREV_CLOSE", "VOLUME", "TURNOVER", "NUM_TRADES", "ISIN",
]

NUMERIC_COLS = [
    "OPEN", "HIGH", "LOW", "CLOSE", "LAST",
    "PREV_CLOSE", "VOLUME", "TURNOVER", "NUM_TRADES",
]

MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifests" / "bhavcopy.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric columns to numeric dtype.

    Missing values stay NaN — never zero. A stock with zero volume and
    a stock with no volume recorded are different facts; conflating them
    corrupts every feature built downstream.
    """
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eras 1 and 2. Column count varies (12 or 14); missing optional
    columns are filled with NA rather than assumed present.
    """
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "SYMBOL": df["SYMBOL"].astype(str).str.strip(),
        "SERIES": df["SERIES"].astype(str).str.strip(),
        "DATE": pd.to_datetime(
            df["TIMESTAMP"].astype(str).str.strip(),
            format="%d-%b-%Y",
            errors="coerce",
        ),
        "OPEN": df["OPEN"],
        "HIGH": df["HIGH"],
        "LOW": df["LOW"],
        "CLOSE": df["CLOSE"],
        "LAST": df["LAST"],
        "PREV_CLOSE": df["PREVCLOSE"],
        "VOLUME": df["TOTTRDQTY"],
        "TURNOVER": df["TOTTRDVAL"],
        # Not published in early years
        "NUM_TRADES": df["TOTALTRADES"] if "TOTALTRADES" in df.columns else pd.NA,
        "ISIN": df["ISIN"].astype(str).str.strip() if "ISIN" in df.columns else pd.NA,
    })
    return _coerce_numeric(out[OUTPUT_COLS])


def parse_udiff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Era 3. UDiFF carries cash equities, derivatives, and other
    instrument types in one file — filter to cash stocks only.
    """
    df.columns = [c.strip() for c in df.columns]
    df = df[df["FinInstrmTp"].astype(str).str.strip() == "STK"].copy()

    out = pd.DataFrame({
        "SYMBOL": df["TckrSymb"].astype(str).str.strip(),
        "SERIES": df["SctySrs"].astype(str).str.strip(),
        "DATE": pd.to_datetime(df["TradDt"], errors="coerce"),
        "OPEN": df["OpnPric"],
        "HIGH": df["HghPric"],
        "LOW": df["LwPric"],
        "CLOSE": df["ClsPric"],
        "LAST": df["LastPric"],
        "PREV_CLOSE": df["PrvsClsgPric"],
        "VOLUME": df["TtlTradgVol"],
        "TURNOVER": df["TtlTrfVal"],
        "NUM_TRADES": df["TtlNbOfTxsExctd"],
        "ISIN": df["ISIN"].astype(str).str.strip(),
    })
    return _coerce_numeric(out[OUTPUT_COLS])


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Structural checks before a file is accepted. Catches format drift
    at ingestion rather than letting it surface as a modelling anomaly
    months later.
    """
    if df.empty:
        raise ValueError("parsed to zero rows")

    if df["DATE"].isna().any():
        raise ValueError(f"{df['DATE'].isna().sum()} rows failed date parsing")

    parsed_dates = df["DATE"].dt.date.unique()
    if len(parsed_dates) != 1:
        raise ValueError(f"expected one date, found {len(parsed_dates)}")
    if parsed_dates[0] != trade_date:
        raise ValueError(f"file dated {parsed_dates[0]}, requested {trade_date}")

    if df["CLOSE"].isna().all():
        raise ValueError("every CLOSE is null — likely a parser mismatch")

    if (df["CLOSE"] < 0).any() or (df["VOLUME"] < 0).any():
        raise ValueError("negative price or volume present")

    return df


# ─────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────

def build_url(trade_date: date) -> tuple[str, str]:
    """Return (url, era_label) for the date's format generation."""
    if trade_date >= UDIFF_CUTOVER:
        return UDIFF_URL.format(yyyymmdd=trade_date.strftime("%Y%m%d")), "udiff"
    return (
        LEGACY_URL.format(
            yyyy=trade_date.strftime("%Y"),
            MON=trade_date.strftime("%b").upper(),
            DD=trade_date.strftime("%d"),
        ),
        "legacy",
    )


def fetch_day(
    session: requests.Session, trade_date: date
) -> tuple[pd.DataFrame | None, str, int]:
    """
    Fetch one trading day.

    Returns (df, era, bytes). A None dataframe means no trading that
    day — a legitimate outcome, not a failure.
    """
    url, era = build_url(trade_date)
    resp = session.get(url, timeout=30)

    if resp.status_code == 404:
        return None, era, 0
    resp.raise_for_status()

    nbytes = len(resp.content)
    if nbytes < 500:
        return None, era, nbytes                    # placeholder/empty file

    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError("zip archive is empty")
        with archive.open(members[0]) as handle:
            raw = pd.read_csv(handle, low_memory=False)

    parsed = parse_udiff(raw) if era == "udiff" else parse_legacy(raw)
    return validate(parsed, trade_date), era, nbytes


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSE bhavcopy price history.")
    ap.add_argument("--start", type=str, help="start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="end date YYYY-MM-DD")
    ap.add_argument("--days", type=int, help="fetch only the last N days")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="seconds between requests (default 0.4)")
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
        name="bhavcopy",
        start=start,
        end=end,
        output_dir=BHAVCOPY_DIR,
        fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR,
        request_delay=args.delay,
    )


if __name__ == "__main__":
    main()