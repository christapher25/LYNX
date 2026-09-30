"""
Stage 8 — USD-INR ingestion.

Source: RBI Forex Reference Rates Archive (Zenodo, DOI 10.5281/zenodo.22228722).
CC0 1.0 public domain — no licensing conflict with commercial use, unlike
the FII/DII third-party aggregator that was ruled out for that reason.

Methodology is transparent and documented by the dataset itself: 1998-2018
from RBI's own archive, post-2018 from FBIL (the official benchmark
administrator RBI handed this over to), with a documented Jul2018-Mar2022
gap-fill cross-checked against two independent sources.

Not a per-day archive like everything else tonight — this is a single
bulk file, downloaded once. Filtered to USD only; the source dataset
covers 6 currencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT
from data.ingest_common import write_parquet_atomic, get_logger

SOURCE_URL = "https://zenodo.org/records/22228722/files/rbi_forex_reference_rates.parquet?download=1"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "usdinr.parquet"
LOG_DIR = PROJECT_ROOT / "logs"


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("USD-INR data is empty after filtering")
    if df["rate"].isna().any():
        raise ValueError(f"{df['rate'].isna().sum()} null rates — should not happen post-filter")
    if (df["rate"] <= 0).any():
        raise ValueError("non-positive exchange rate present — impossible")
    # Sanity band: USD-INR has never been outside roughly 30-100 across
    # this dataset's range. A value outside this is worth a manual look,
    # not silent acceptance.
    out_of_band = df[(df["rate"] < 30) | (df["rate"] > 100)]
    if not out_of_band.empty:
        raise ValueError(
            f"{len(out_of_band)} rates outside the [30,100] sanity band — "
            f"check for a unit error (paise vs rupees) before trusting this data"
        )


def main() -> None:
    logger = get_logger("usdinr", LOG_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from {SOURCE_URL}")
    resp = requests.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()

    tmp_path = OUTPUT_PATH.with_suffix(".raw.parquet")
    tmp_path.write_bytes(resp.content)
    full = pd.read_parquet(tmp_path)
    tmp_path.unlink()

    logger.info(f"Downloaded {len(full):,} rows across all currencies")
    logger.info(f"Columns: {list(full.columns)}")

    usd = full[full["currency"] == "USD"].copy()
    usd["date"] = pd.to_datetime(usd["date"])
    usd = usd.sort_values("date").reset_index(drop=True)

    validate(usd)

    write_parquet_atomic(usd, OUTPUT_PATH)
    logger.info(
        f"USD-INR: {len(usd):,} rows, {usd['date'].min().date()} to "
        f"{usd['date'].max().date()}, written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()