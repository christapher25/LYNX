"""
Stage 3 — NSE sec_bhavdata_full ingestion. Delivery data, 2019 to present.

This file carries OHLCV *and* delivery percentage in a single format
that survived the July 2024 UDiFF cutover unchanged — verified identical
on both 2024-06-03 and 2024-07-10. One parser, no era branching.

Delivery percentage is the strongest India-specific feature in the LYNX
design and exists in no other free source. It is only available from
2019-09-30 (measured, see data/raw/DATA_NOTES.md), which is why price
history comes from bhavcopy instead and delivery is treated as a
per-fold-optional feature.

Usage
-----
    python data/ingest_bhavdata.py                     # full backfill
    python data/ingest_bhavdata.py --start 2024-01-01  # from a date
    python data/ingest_bhavdata.py --days 30           # last 30 days
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DELIVERY_DIR, PROJECT_ROOT
from data.ingest_common import run_ingestion

# ─────────────────────────────────────────────────────────────────
# Constants — measured in Stage 3, see data/raw/DATA_NOTES.md
# ─────────────────────────────────────────────────────────────────

FIRST_AVAILABLE = date(2019, 9, 30)      # verified: 27 Sep → 404, 30 Sep → 200

BASE_URL = (
    "https://nsearchives.nseindia.com/products/content/"
    "sec_bhavdata_full_{ddmmyyyy}.csv"
)

OUTPUT_COLS = [
    "SYMBOL", "SERIES", "DATE", "PREV_CLOSE", "OPEN", "HIGH", "LOW",
    "LAST", "CLOSE", "VWAP", "VOLUME", "TURNOVER_LACS", "NUM_TRADES",
    "DELIV_QTY", "DELIV_PER",
]

NUMERIC_COLS = [
    "PREV_CLOSE", "OPEN", "HIGH", "LOW", "LAST", "CLOSE", "VWAP",
    "VOLUME", "TURNOVER_LACS", "NUM_TRADES", "DELIV_QTY", "DELIV_PER",
]

# Sentinels NSE uses for "no data". Each must become NaN, never zero.
MISSING_SENTINELS = {"-", "", "NA", "N/A", "nan", "None"}

MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifests" / "bhavdata.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to numeric, mapping NSE's missing-value sentinels to NaN.

    The '-' sentinel appears in DELIV_QTY and DELIV_PER for securities
    where delivery is not reported — commonly BE series and other
    restricted categories. Zero-filling these would assert that no
    shares were delivered, which is a different and false claim.
    """
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.where(~cleaned.isin(MISSING_SENTINELS), other=None)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_bhavdata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse one day's sec_bhavdata_full file.

    Three documented quirks, all handled here:
      1. Every column except SYMBOL carries a leading space
      2. '-' denotes missing in delivery columns, not zero
      3. Date format is '03-Jun-2024', requiring explicit parsing
    """
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "SYMBOL": df["SYMBOL"].astype(str).str.strip(),
        "SERIES": df["SERIES"].astype(str).str.strip(),
        "DATE": pd.to_datetime(
            df["DATE1"].astype(str).str.strip(),
            format="%d-%b-%Y",
            errors="coerce",
        ),
        "PREV_CLOSE": _clean_numeric(df["PREV_CLOSE"]),
        "OPEN": _clean_numeric(df["OPEN_PRICE"]),
        "HIGH": _clean_numeric(df["HIGH_PRICE"]),
        "LOW": _clean_numeric(df["LOW_PRICE"]),
        "LAST": _clean_numeric(df["LAST_PRICE"]),
        "CLOSE": _clean_numeric(df["CLOSE_PRICE"]),
        "VWAP": _clean_numeric(df["AVG_PRICE"]),
        "VOLUME": _clean_numeric(df["TTL_TRD_QNTY"]),
        "TURNOVER_LACS": _clean_numeric(df["TURNOVER_LACS"]),
        "NUM_TRADES": _clean_numeric(df["NO_OF_TRADES"]),
        "DELIV_QTY": _clean_numeric(df["DELIV_QTY"]),
        "DELIV_PER": _clean_numeric(df["DELIV_PER"]),
    })
    return out[OUTPUT_COLS]


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Structural checks before a file is accepted.

    The delivery-specific checks matter because a silent parser failure
    on DELIV_PER would degrade LYNX's strongest feature without any
    visible symptom until model performance quietly underperformed.
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

    # Delivery percentage is a percentage: it cannot leave [0, 100].
    delivered = df["DELIV_PER"].dropna()
    if not delivered.empty:
        out_of_range = delivered[(delivered < 0) | (delivered > 100)]
        if not out_of_range.empty:
            raise ValueError(
                f"{len(out_of_range)} DELIV_PER values outside [0, 100] "
                f"(min={delivered.min()}, max={delivered.max()})"
            )

    # Delivered quantity cannot exceed traded quantity.
    both = df[["DELIV_QTY", "VOLUME"]].dropna()
    if not both.empty:
        impossible = both[both["DELIV_QTY"] > both["VOLUME"]]
        if not impossible.empty:
            raise ValueError(
                f"{len(impossible)} rows have DELIV_QTY > VOLUME — impossible"
            )

    # A total absence of delivery data means the parser broke, not that
    # the market delivered nothing.
    if df["DELIV_PER"].isna().all():
        raise ValueError("every DELIV_PER is null — parser or format change")

    return df


# ─────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────

def fetch_day(
    session: requests.Session, trade_date: date
) -> tuple[pd.DataFrame | None, str, int]:
    """
    Fetch one trading day.

    Returns (df, era, bytes). A None dataframe means no trading that
    day — a legitimate outcome, not a failure.
    """
    url = BASE_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))
    resp = session.get(url, timeout=30)

    if resp.status_code == 404:
        return None, "standard", 0
    resp.raise_for_status()

    nbytes = len(resp.content)
    if nbytes < 1000:
        return None, "standard", nbytes             # placeholder/empty file

    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    parsed = parse_bhavdata(raw)
    return validate(parsed, trade_date), "standard", nbytes


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest NSE sec_bhavdata_full (OHLCV + delivery)."
    )
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
        name="bhavdata",
        start=start,
        end=end,
        output_dir=DELIVERY_DIR,
        fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR,
        request_delay=args.delay,
    )


if __name__ == "__main__":
    main()