"""
features/blocks/pledging.py

Stage 10 — promoter pledging (encumbrance), per symbol per session.

    pledged_pct_of_promoter   100 * encumbered shares / promoter-group shares (from share COUNTS), latest public filing
    pledged_pct_chg           that ratio minus the previous PERIOD's ratio, percentage points
    pledge_rising             1.0 if pledged_pct_chg > 0, 0.0 if not, NaN when there is no change to compute
    pledging_age_days         session date minus the filing's period end: how stale the disclosure is
    source_period_end         the period the value describes
    is_missing_pledging       True exactly where no valid ratio is public

Source: NSE shareholding-pattern XBRL filings (data/ingest_shp_pledge.py). "Encumbered" is pledge + non-disposal
undertaking + other encumbrances (the 2025-10-31 schema) or the older schema's combined "pledged or otherwise encumbered": the
definition that is comparable across both. Disclosures are quarterly, so between filings the value is the LAST DISCLOSED one,
labelled by pledging_age_days; that is persistence of a real disclosure, not imputation.

Point-in-time. A filing becomes usable at the first session whose 19:00 IST nightly run is at or after its BROADCAST timestamp
(data/publication.py), never on the quarter it describes, its submission date, or the date a pledge was created. Intra-quarter
pledge events are invisible until the next filing. A filing revised later REPLACES the original in NSE's list, so the original
values are unrecoverable and the revised ones appear only from the revision's broadcast. A late revision of an older period is
kept in that period's slot and never displaces a newer period as the current filing. The oldest filings (period Sep-2021 and
Dec-2021) were re-broadcast in one batch on 2022-01-06/07, which is the conservative key for them.

Missing means missing. NaN + flag where: no filing is public yet; the filing has no promoter group (or 0 promoter shares); the
filing failed to download or parse (its row is NOT filled from the previous quarter); the broadcast time is unknown (such a
filing is never used). A real 0% stays 0.0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ADJUSTED_PRICES_DIR, PLEDGING_FEATURES_CHECKPOINT, PLEDGING_FEATURES_DIR, SHP_DIR
from data.publication import usable_from_series
from data.trading_calendar import trading_sessions
from features.blocks.common import validated_sessions

PLEDGING_COLUMNS = ["pledged_pct_of_promoter", "pledged_pct_chg", "pledge_rising", "pledging_age_days"]
OUTPUT_COLUMNS = ["symbol", "date", *PLEDGING_COLUMNS, "source_period_end", "is_missing_pledging"]
FILING_COLUMNS = ["PERIOD_END", "BROADCAST_IST", "PROMOTER_SHARES", "ENCUMBERED_SHARES", "HAS_PROMOTER_GROUP", "PARSE_ERROR"]
RISING_TOLERANCE_PP = 1e-9
_OFFSET = re.compile(r"(?:[+-]\d{2}:\d{2}|Z)$")


def _filing_ratio(filings: pd.DataFrame) -> np.ndarray:
    """Encumbered / promoter shares as a percentage; NaN unless the filing parsed and has a promoter group with shares."""
    promoter = pd.to_numeric(filings["PROMOTER_SHARES"], errors="coerce").to_numpy(float)
    encumbered = pd.to_numeric(filings["ENCUMBERED_SHARES"], errors="coerce").to_numpy(float)
    has_group = filings["HAS_PROMOTER_GROUP"].fillna(False).astype(bool).to_numpy()
    clean = filings["PARSE_ERROR"].isna().to_numpy()
    valid = has_group & clean & (promoter > 0) & ~np.isnan(encumbered)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, 100.0 * encumbered / promoter, np.nan)


def compute_pledging_features(dates: Sequence, filings: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    """
    Parameters
    ----------
    dates    : the symbol's ascending row dates (its Stage 4 date spine); the output has exactly one row per date.
    filings  : one symbol's filings, columns FILING_COLUMNS. BROADCAST_IST is an ISO timestamp WITH offset (IST); null = never used.
    sessions : ascending trading-session dates (the global calendar), used to place each broadcast on a session.

    Causal: a row depends only on filings broadcast before that session's nightly run.
    """
    if not set(FILING_COLUMNS) <= set(filings.columns):
        raise ValueError(f"filings needs columns {FILING_COLUMNS}")
    sess = validated_sessions(sessions)
    rows = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    out = pd.DataFrame({"date": rows, **{c: np.nan for c in PLEDGING_COLUMNS}, "source_period_end": pd.NaT})

    stamps = filings["BROADCAST_IST"]
    given = stamps.dropna().astype(str)
    if (~given.map(lambda s: bool(_OFFSET.search(s.strip())))).any():
        raise ValueError("a broadcast timestamp has no timezone; refusing to assume one")
    keep = stamps.notna().to_numpy()
    if not keep.any():
        out["is_missing_pledging"] = True
        return out[["date", *PLEDGING_COLUMNS, "source_period_end", "is_missing_pledging"]]

    used = filings[keep].reset_index(drop=True)
    published = pd.to_datetime(used["BROADCAST_IST"], utc=True)
    usable = usable_from_series(pd.Series(published.to_numpy(), index=used.index), [d.date() for d in sess])
    used = used.assign(usable_at=pd.to_datetime(usable.to_numpy()), published_at=published.to_numpy(),
                       ratio=_filing_ratio(used), period=pd.to_datetime(used["PERIOD_END"]).dt.normalize())
    used = used.dropna(subset=["usable_at"]).sort_values(["usable_at", "published_at"], kind="mergesort")

    # State after each distinct usable date: the newest PERIOD is current, the one before it is the comparison. A later filing
    # for the same period (a revision) replaces that period's slot, so a restated old quarter cannot displace a newer one.
    by_period: dict[pd.Timestamp, float] = {}
    event_dates, current, current_ratio, previous_ratio = [], [], [], []
    for usable_date, group in used.groupby("usable_at", sort=True):
        for rec in group.itertuples():
            by_period[rec.period] = rec.ratio
        periods = sorted(by_period)
        newest = periods[-1]
        event_dates.append(usable_date)
        current.append(newest)
        current_ratio.append(by_period[newest])
        previous_ratio.append(by_period[periods[-2]] if len(periods) > 1 else np.nan)

    if event_dates:
        at = np.searchsorted(np.array(event_dates, dtype="datetime64[ns]"), rows.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        seen = at >= 0
        ratio = np.where(seen, np.asarray(current_ratio, float)[np.clip(at, 0, None)], np.nan)
        prior = np.where(seen, np.asarray(previous_ratio, float)[np.clip(at, 0, None)], np.nan)
        period = pd.DatetimeIndex(np.asarray(current, dtype="datetime64[ns]")[np.clip(at, 0, None)])
        change = ratio - prior
        out["pledged_pct_of_promoter"] = ratio
        out["pledged_pct_chg"] = change
        out["pledge_rising"] = np.where(np.isnan(change), np.nan, (change > RISING_TOLERANCE_PP).astype(float))
        known = ~np.isnan(ratio)
        out["source_period_end"] = pd.Series(np.where(seen, period.to_numpy(dtype="datetime64[ns]"), np.datetime64("NaT", "ns")), index=out.index)
        age = (rows - pd.DatetimeIndex(out["source_period_end"])).days.to_numpy(float)
        out["pledging_age_days"] = np.where(seen, age, np.nan)
        # a row whose current filing is unknown has no usable value; keep its period label out of the output as well
        out.loc[~known, ["pledging_age_days"]] = np.nan
        out.loc[~known, "source_period_end"] = pd.NaT
    out["is_missing_pledging"] = out["pledged_pct_of_promoter"].isna()
    return out[["date", *PLEDGING_COLUMNS, "source_period_end", "is_missing_pledging"]]


# ─── loaders and driver ──────────────────────────────────────────────────────────────────────────────────────────

def load_filings(symbol: str, directory: Path = SHP_DIR) -> pd.DataFrame:
    path = Path(directory) / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=FILING_COLUMNS)
    return pd.read_parquet(path, columns=FILING_COLUMNS)


def load_date_spine(symbol: str) -> pd.DatetimeIndex:
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE"])
    return pd.DatetimeIndex(raw["DATE"].sort_values().reset_index(drop=True))


def build_symbol_features(symbol: str, spine: pd.DatetimeIndex, filings: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    features = compute_pledging_features(spine, filings, sessions)
    features.insert(0, "symbol", symbol)
    return features[OUTPUT_COLUMNS]


def _load_checkpoint() -> set[str]:
    if PLEDGING_FEATURES_CHECKPOINT.exists():
        return set(json.loads(PLEDGING_FEATURES_CHECKPOINT.read_text())["completed_symbols"])
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    PLEDGING_FEATURES_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    PLEDGING_FEATURES_CHECKPOINT.write_text(json.dumps({"completed_symbols": sorted(completed),
                                                        "updated_at": datetime.now(timezone.utc).isoformat()}))


def run_pledging_pipeline(resume: bool = True) -> dict:
    """One Parquet per symbol, row-aligned to its Stage 4 rows, checkpointed after each symbol. Rerun with resume=False whenever
    the SHP filings change (a resumed run keeps symbols written from older inputs)."""
    from data.ingest_common import write_parquet_atomic
    from signals.indicators import list_universe_symbols

    sessions = trading_sessions()
    symbols = list_universe_symbols()
    completed = _load_checkpoint() if resume else set()
    remaining = [s for s in symbols if s not in completed]
    logger.info(f"pledging pipeline: {len(symbols):,} symbols, {len(completed):,} done, {len(remaining):,} remaining")
    rows = with_value = no_filings = 0
    for symbol in tqdm(remaining, desc="pledging features"):
        filings = load_filings(symbol)
        features = build_symbol_features(symbol, load_date_spine(symbol), filings, sessions)
        write_parquet_atomic(features, PLEDGING_FEATURES_DIR / f"{symbol}.parquet")
        rows += len(features)
        with_value += int((~features["is_missing_pledging"]).sum())
        no_filings += int(filings.empty)
        completed.add(symbol)
        _save_checkpoint(completed)
    summary = {"symbols_written": len(remaining), "rows": rows, "rows_with_value": with_value, "symbols_without_filings": no_filings}
    logger.success(f"pledging pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LYNX Stage 10 — pledging feature block")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint, recompute all")
    run_pledging_pipeline(resume=not parser.parse_args().no_resume)
