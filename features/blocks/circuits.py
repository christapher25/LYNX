"""
features/blocks/circuits.py

Stage 10 — circuit limits, per symbol per session. Two blocks, each with its own shared missing flag (their availability differs
by ten years, exactly as FII and DII flows do; one shared flag would mark the deep series missing wherever the shallow one is):

  BAND   price_band_pct        the price band in force on that session, in percent (2, 5, 10, 20, 40); NaN for "No Band"
         is_no_band            1.0 if the security is listed with no static band, 0.0 if it has one, NaN if unknown
         band_is_inferred      0.0 for a band read from NSE's file, 1.0 for one inferred (see below), NaN if unknown
         is_missing_circuit_band
  HITS   hit_upper_circuit     1.0 if the stock touched its upper price band that session, else 0.0
         hit_lower_circuit     likewise for the lower band; a stock can hit both
         is_missing_circuit_hits

Band. The price band in force on session t is the one in the sec_list file dated the PREVIOUS SESSION (published the evening before).
Measured 2026-09-24 on 7,045 real hits in 60 sampled sessions: the band that reproduces each hit's limit price agrees with the
previous session's file 97.2% of the time and with the same-day file 92.8%. Band limits are floor-to-tick for the upper band and
ceil for the lower. Bands change often (the 5% bucket ranged from 295 to 704 securities across 28 sampled dates), so a band is only
meaningful for its own date: never apply today's band to history. sec_list starts 2019-12-30; before that a band exists only if an
inference method has been measured against 2020+ truth and is passed in (`inferred`); otherwise it is NaN and flagged.
A symbol absent from the previous session's file, a missing file, or the first session gives NaN + flag: nothing is carried
forward from an older file.

Hits. The bh file dated t (the PR zip's list of securities that hit their price bands during the day) is public the evening of t
(PR zips carried real same-day timestamps of 16:35-17:45 IST from 2020; earlier stamps were rewritten in 2023, so for 2010-2019
publication before the 19:00 run is an assumption, not a measurement). A published list that does not name the stock is a real 0;
a session with no published list (before 2010-01-04, or a failed day) is NaN + flag, never 0.

Circuit-band feature semantics for a "No Band" security: its only price limits are dynamic ones NSE does not publish in this file,
so price_band_pct is NaN and is_no_band is 1.0, a KNOWN state (the flag stays False).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Collection, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ADJUSTED_PRICES_DIR, CIRCUIT_FEATURES_CHECKPOINT, CIRCUIT_FEATURES_DIR, PRICE_BAND_HITS_DIR, SEC_LIST_DIR
from data.trading_calendar import trading_sessions
from features.blocks.common import validated_sessions

CIRCUIT_SERIES = "EQ"                # Stage 4 (adjust_prices.py) keeps SERIES == "EQ" only
BAND_COLUMNS = ["price_band_pct", "is_no_band", "band_is_inferred"]
HIT_COLUMNS = ["hit_upper_circuit", "hit_lower_circuit"]
FEATURE_COLUMNS = ["date", *BAND_COLUMNS, "is_missing_circuit_band", *HIT_COLUMNS, "is_missing_circuit_hits"]
OUTPUT_COLUMNS = ["symbol", *FEATURE_COLUMNS]


def compute_circuit_features(dates: Sequence, bands: pd.DataFrame, hits: pd.DataFrame, hit_file_dates: Collection,
                             sessions: Sequence[date], inferred: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    dates          : the symbol's ascending row dates (its Stage 4 date spine).
    bands          : this symbol's EQ rows of the sec_list files: {DATE (the FILE's date), BAND_PCT, IS_NO_BAND}, one row per file date.
    hits           : this symbol's EQ rows of the bh files: {DATE, HIT in {'H', 'L'}}.
    hit_file_dates : every date a bh list was published (including days on which nobody hit a band).
    sessions       : ascending trading-session dates (the global calendar): 'previous session' means the previous one of these.
    inferred       : optional {DATE (a ROW date), BAND_PCT} from a measured inference method, used only where no exact band exists.

    Causal: a row depends only on the previous session's band file and that session's own hit list.
    """
    sess = validated_sessions(sessions)
    rows = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    if bands["DATE"].duplicated().any():
        raise ValueError("more than one band row for the same file date")
    published = rows.isin(pd.DatetimeIndex(pd.to_datetime(list(hit_file_dates))).normalize())

    # BAND: the file dated the previous SESSION.
    position = sess.get_indexer(rows)
    previous = pd.DatetimeIndex([sess[p - 1] if p >= 1 else pd.NaT for p in position])
    by_file = bands.assign(DATE=pd.to_datetime(bands["DATE"]).dt.normalize()).set_index("DATE")
    listed = previous.isin(by_file.index)
    exact_pct = by_file["BAND_PCT"].reindex(previous).to_numpy(float)
    exact_nb = by_file["IS_NO_BAND"].reindex(previous).astype("float64").to_numpy()
    pct = np.where(listed, exact_pct, np.nan)
    no_band = np.where(listed, exact_nb, np.nan)
    inferred_flag = np.where(listed, 0.0, np.nan)
    if inferred is not None and len(inferred):
        guess = inferred.assign(DATE=pd.to_datetime(inferred["DATE"]).dt.normalize()).set_index("DATE")["BAND_PCT"].reindex(rows).to_numpy(float)
        fill = ~listed & ~np.isnan(guess)
        pct, no_band, inferred_flag = np.where(fill, guess, pct), np.where(fill, 0.0, no_band), np.where(fill, 1.0, inferred_flag)
    known_band = ~np.isnan(no_band)

    # HITS: the bh file dated t. A hit on a date with no published list is not trusted.
    hit_days = hits.assign(DATE=pd.to_datetime(hits["DATE"]).dt.normalize())
    upper = rows.isin(pd.DatetimeIndex(hit_days.loc[hit_days["HIT"] == "H", "DATE"]))
    lower = rows.isin(pd.DatetimeIndex(hit_days.loc[hit_days["HIT"] == "L", "DATE"]))

    out = pd.DataFrame({
        "date": rows, "price_band_pct": pct, "is_no_band": no_band, "band_is_inferred": inferred_flag,
        "is_missing_circuit_band": ~known_band,
        "hit_upper_circuit": np.where(published, upper.astype(float), np.nan),
        "hit_lower_circuit": np.where(published, lower.astype(float), np.nan),
        "is_missing_circuit_hits": ~published,
    })
    return out[FEATURE_COLUMNS]


# ─── loaders and driver ──────────────────────────────────────────────────────────────────────────────────────────

def load_band_files(directory: Path = SEC_LIST_DIR) -> pd.DataFrame:
    """Every EQ row of every sec_list file: {SYMBOL, DATE (file date), BAND_PCT, IS_NO_BAND}."""
    frames = [pd.read_parquet(p, columns=["DATE", "SYMBOL", "SERIES", "BAND_PCT", "IS_NO_BAND"]) for p in sorted(Path(directory).glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=["SYMBOL", "DATE", "BAND_PCT", "IS_NO_BAND"])
    both = pd.concat(frames, ignore_index=True)
    return both[both["SERIES"] == CIRCUIT_SERIES].drop(columns="SERIES")


def load_hit_files(directory: Path = PRICE_BAND_HITS_DIR) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """(EQ hit rows {SYMBOL, DATE, HIT}, the set of every date a bh list was published, NIL days included)."""
    frames = [pd.read_parquet(p, columns=["DATE", "SYMBOL", "SERIES", "HIT"]) for p in sorted(Path(directory).glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=["SYMBOL", "DATE", "HIT"]), set()
    both = pd.concat(frames, ignore_index=True)
    published = set(pd.to_datetime(both["DATE"]).dt.normalize().unique())
    named = both[(both["SERIES"] == CIRCUIT_SERIES) & both["SYMBOL"].notna()].drop(columns="SERIES")
    return named, published


def load_date_spine(symbol: str) -> pd.DatetimeIndex:
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE"])
    return pd.DatetimeIndex(raw["DATE"].sort_values().reset_index(drop=True))


def _load_checkpoint() -> set[str]:
    if CIRCUIT_FEATURES_CHECKPOINT.exists():
        return set(json.loads(CIRCUIT_FEATURES_CHECKPOINT.read_text())["completed_symbols"])
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    CIRCUIT_FEATURES_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CIRCUIT_FEATURES_CHECKPOINT.write_text(json.dumps({"completed_symbols": sorted(completed),
                                                       "updated_at": datetime.now(timezone.utc).isoformat()}))


def run_circuits_pipeline(resume: bool = True, inferred_by_symbol: dict[str, pd.DataFrame] | None = None) -> dict:
    """One Parquet per symbol, row-aligned to its Stage 4 rows, checkpointed after each symbol. Rerun with resume=False whenever the
    band or hit files change."""
    from data.ingest_common import write_parquet_atomic
    from signals.indicators import list_universe_symbols

    sessions = trading_sessions()
    symbols = list_universe_symbols()
    all_bands = load_band_files()
    all_hits, hit_days = load_hit_files()
    bands_by, hits_by = ({s: g for s, g in all_bands.groupby("SYMBOL", sort=False)}, {s: g for s, g in all_hits.groupby("SYMBOL", sort=False)})
    empty_bands = pd.DataFrame(columns=["DATE", "BAND_PCT", "IS_NO_BAND"])
    empty_hits = pd.DataFrame(columns=["DATE", "HIT"])
    completed = _load_checkpoint() if resume else set()
    remaining = [s for s in symbols if s not in completed]
    logger.info(f"circuits pipeline: {len(symbols):,} symbols, {len(completed):,} done, {len(remaining):,} remaining; "
                f"{len(all_bands):,} band rows over {all_bands['DATE'].nunique():,} files, {len(all_hits):,} hit rows over {len(hit_days):,} lists")
    rows = with_band = with_hits = 0
    for symbol in tqdm(remaining, desc="circuit features"):
        features = compute_circuit_features(load_date_spine(symbol), bands_by.get(symbol, empty_bands), hits_by.get(symbol, empty_hits),
                                            hit_days, sessions, (inferred_by_symbol or {}).get(symbol))
        features.insert(0, "symbol", symbol)
        write_parquet_atomic(features[OUTPUT_COLUMNS], CIRCUIT_FEATURES_DIR / f"{symbol}.parquet")
        rows += len(features)
        with_band += int((~features["is_missing_circuit_band"]).sum())
        with_hits += int((~features["is_missing_circuit_hits"]).sum())
        completed.add(symbol)
        _save_checkpoint(completed)
    summary = {"symbols_written": len(remaining), "rows": rows, "rows_with_band": with_band, "rows_with_hits": with_hits}
    logger.success(f"circuits pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LYNX Stage 10 — circuits feature block")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint, recompute all")
    run_circuits_pipeline(resume=not parser.parse_args().no_resume)
