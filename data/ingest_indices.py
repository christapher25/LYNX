"""
Stage 8 — index data ingestion (NIFTY 50, NIFTY BANK, INDIA VIX, and
159+ other indices in the same daily file).

Source confirmed via direct probing: static archive, same domain family
as bhavcopy/delivery. First available date measured: 2012-02-21.

NSE renamed several indices over this history — confirmed empirically,
not assumed:
    Nifty 50 equivalent: "S&P CNX Nifty" (2012-2013) -> "CNX Nifty"
                          (2013-2015) -> "Nifty 50" (late 2015 onward)
    Nifty Bank equivalent: "CNX Bank" (2012-2015) -> "Nifty Bank"
                            (2015 onward)
    India VIX: absent entirely before ~2014, present from ~2015 onward
               (a genuinely new index, not a rename)

The original version hard-failed any file missing the CURRENT names,
which meant every single day before late 2015 failed ingestion. Fixed
two ways: names are normalized to a canonical form on parse, and the
validation no longer hard-fails when VIX is legitimately absent from
early history — that's a real historical fact, not a parser error.
"""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT
from data.ingest_common import run_ingestion

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "indices"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifests" / "indices.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

FIRST_AVAILABLE = date(2012, 2, 21)   # measured via binary search
BASE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{}.csv"

# Confirmed via direct probing across 2012-2017 files — every historical
# name each index has actually used in this archive, mapped to one
# canonical name so downstream code never has to know the file's era.
NAME_ALIASES = {
    "S&P CNX Nifty": "Nifty 50",
    "CNX Nifty": "Nifty 50",
    "Nifty 50": "Nifty 50",
    "CNX Bank": "Nifty Bank",
    "Nifty Bank": "Nifty Bank",
}

NUMERIC_COLS = [
    "OPEN", "HIGH", "LOW", "CLOSE", "POINTS_CHANGE", "PCT_CHANGE",
    "VOLUME", "TURNOVER_CR", "PE", "PB", "DIV_YIELD",
]


def _parse_index_date(raw: str, requested: date) -> pd.Timestamp | None:
    """
    Two independently-confirmed quirks in this archive's date field,
    handled generically rather than as one-off patches:

    1. ~June 2014 - June 2015 uses '/' as the delimiter instead of '-'
       (e.g. "09/06/2014" vs the expected "09-06-2014") — confirmed via
       59 failures clustered exactly in that window.
    2. A small number of isolated dates (confirmed: 3 in April 2023)
       have day and month transposed in NSE's own source file — a data
       error at the source, not a parsing bug. Detected by checking
       whether the primary parse matches the date we actually requested;
       if not, a day/month swap is tried before giving up.
    """
    normalized = raw.replace("/", "-")
    try:
        parsed = pd.to_datetime(normalized, format="%d-%m-%Y")
        if parsed.date() == requested:
            return parsed
    except (ValueError, TypeError):
        parsed = None

    # Primary parse either failed or didn't match — try day/month swapped
    try:
        swapped = pd.to_datetime(normalized, format="%m-%d-%Y")
        if swapped.date() == requested:
            return swapped
    except (ValueError, TypeError):
        pass

    return parsed   # return whatever we got (possibly None or mismatched)
                     # — validate() catches a real mismatch and raises


def parse_index_file(raw_text: str, requested_date: date) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(raw_text))
    df.columns = [c.strip() for c in df.columns]

    raw_names = df["Index Name"].astype(str).str.strip()
    canonical_names = raw_names.replace(NAME_ALIASES)

    parsed_dates = df["Index Date"].astype(str).str.strip().apply(
        lambda s: _parse_index_date(s, requested_date)
    )

    out = pd.DataFrame({
        "INDEX_NAME": canonical_names,
        "RAW_INDEX_NAME": raw_names,
        "DATE": parsed_dates,
        "OPEN": df["Open Index Value"],
        "HIGH": df["High Index Value"],
        "LOW": df["Low Index Value"],
        "CLOSE": df["Closing Index Value"],
        "POINTS_CHANGE": df["Points Change"],
        "PCT_CHANGE": df["Change(%)"],
        "VOLUME": df["Volume"],
        "TURNOVER_CR": df["Turnover (Rs. Cr.)"],
        "PE": df["P/E"],
        "PB": df["P/B"],
        "DIV_YIELD": df["Div Yield"],
    })

    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(
            out[col].astype(str).str.strip().replace("-", None), errors="coerce"
        )

    return out


def validate(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    if df.empty:
        raise ValueError("parsed to zero rows")
    if df["DATE"].isna().any():
        raise ValueError(f"{df['DATE'].isna().sum()} rows failed date parsing")

    parsed_dates = df["DATE"].dt.date.unique()
    if len(parsed_dates) != 1 or parsed_dates[0] != trade_date:
        raise ValueError(f"date mismatch: file has {parsed_dates}, requested {trade_date}")

    if df["CLOSE"].isna().all():
        raise ValueError("every CLOSE is null — likely a parser mismatch")

    # Nifty 50 and Nifty Bank (post-alias-normalization) must be present
    # in every file across the whole history — these ARE always there,
    # just under different names pre-2015, which normalization handles.
    # If either is missing even after normalization, that's a genuine
    # parser problem worth failing loudly on.
    present = set(df["INDEX_NAME"])
    hard_required = {"Nifty 50", "Nifty Bank"}
    missing_hard = hard_required - present
    if missing_hard:
        raise ValueError(f"required indices missing even after alias normalization: {missing_hard}")

    # India VIX is NOT hard-required — confirmed absent from the archive
    # before ~2014 as a real historical fact, not a parsing failure.
    # No exception here; downstream feature code must handle VIX being
    # unavailable for folds before it existed.

    return df


def fetch_day(
    session: requests.Session, trade_date: date
) -> tuple[pd.DataFrame | None, str, int]:
    url = BASE_URL.format(trade_date.strftime("%d%m%Y"))
    resp = session.get(url, timeout=20)

    if resp.status_code == 404:
        return None, "standard", 0
    resp.raise_for_status()

    nbytes = len(resp.content)
    if nbytes < 200:
        return None, "standard", nbytes

    parsed = parse_index_file(resp.text, trade_date)
    return validate(parsed, trade_date), "standard", nbytes


def main() -> None:
    run_ingestion(
        name="indices",
        start=FIRST_AVAILABLE,
        end=date.today(),
        output_dir=OUTPUT_DIR,
        fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR,
        request_delay=0.4,
    )


if __name__ == "__main__":
    main()