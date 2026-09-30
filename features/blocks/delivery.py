"""
features/blocks/delivery.py

Stage 10 — India block, delivery sub-block.

Delivery percentage (the share of a day's traded quantity that was actually taken into demat
accounts rather than squared off intraday) exists only in NSE's sec_bhavdata_full files and is
LYNX's strongest India-specific feature. This block turns those raw files into four per-symbol,
per-day features plus one shared missingness flag:

    delivery_pct             NSE's reported delivery %, 0-100
    delivery_pct_chg_5d      delivery_pct minus its value 5 rows earlier   (percentage points)
    delivery_pct_chg_20d     delivery_pct minus its value 20 rows earlier  (percentage points)
    delivery_qty_ratio_20d   delivered qty / trailing-20-row mean of delivered qty (NOT total volume)
    is_missing_delivery      True when the day's delivery observation is absent or unusable

Missing is NaN, never 0 (same convention as signals/indicators.py). One shared flag rather than
per-feature masks: a row is usable for all four features or for none of them, so all four are NaN
exactly where the flag is True. NaN with the flag False is warm-up (not enough history yet), not
missing data. A real zero (a day that traded but delivered nothing) is a fact and stays 0.

Output is row-aligned 1:1 with the symbol's Stage 4 adjusted-price rows, so history before delivery
data exists (it starts mid-2019; prices start 1994) appears as flagged-missing rows rather than
absent ones. That is what lets the walk-forward harness treat delivery as per-fold-optional.

Decisions forced by what the raw archive actually contains (measured, not assumed):

1. KEY ON THE ROW DATE, NEVER THE FILENAME. Five files are named for one day and hold another
   day's rows (NSE re-served stale content). See tests/invariants/test_delivery_row_date.py.
2. EQ SERIES ONLY. Stage 4's adjusted-price panel is EQ-only, so other series cannot join to it.
3. QUANTITY IS SPLIT/BONUS-ADJUSTED. Raw DELIV_QTY is in that day's share count, so a 2:1 split
   doubles it overnight with no change in real activity. Stage 4 rescales VOLUME -> ADJ_VOLUME for
   exactly this reason; delivered quantity gets the same per-row factor (ADJ_VOLUME / VOLUME) or
   delivery_qty_ratio_20d would spike on every corporate action in history. (Same reason
   indicators.py builds volume_ratio from ADJ_VOLUME.)
4. INCONSISTENT NSE ROWS ARE QUARANTINED. On five dates in the raw archive (2020-04-13, 2020-07-13,
   2023-09-04, 2025-10-21, 2026-09-11) 4,191 EQ rows have a DELIV_PER that contradicts the same
   row's DELIV_QTY / VOLUME, always upward (e.g. 41 traded, 1 delivered, reported "100%"). Which
   field is wrong cannot be known, so those observations are treated as missing rather than
   trusted. Today 1,169 of them (3 dates) fall inside the price panel; 2026-09-11 lies beyond
   bhavcopy's last date and 2020-07-13 is a day Stage 3's bhavcopy ingest failed, so neither joins
   yet. The rule needs no update when they do.
5. TRAILING-20 WINDOW NEEDS 15 OBSERVATIONS (MIN_OBS_20D). indicators.py uses strict windows, but
   there a gap is essentially absent; here one quarantined date would otherwise blank the ratio
   for every stock for 20 trading days, including at the live edge. A mean over 15+ genuinely
   observed days is a real statistic, not a fabricated one.

Pure computation (compute_delivery_features) has no I/O and is unit-testable without any files;
the I/O layer below reads raw Parquet, joins to Stage 4, and writes one Parquet per symbol.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from config import (
    ADJUSTED_PRICES_DIR,
    DELIVERY_DIR,
    DELIVERY_FEATURES_CHECKPOINT,
    DELIVERY_FEATURES_DIR,
)
from data.ingest_common import write_parquet_atomic
from signals.indicators import list_universe_symbols

DELIVERY_SERIES = "EQ"                # Stage 4 (adjust_prices.py) keeps SERIES == "EQ" only
DELIVERY_PCT_TOLERANCE_PP = 0.05      # DELIV_PER has 2 dp (max rounding error 0.005); 10x margin
MIN_OBS_20D = 15                      # observed days required inside the trailing-20 window

FEATURE_COLUMNS = [
    "delivery_pct", "delivery_pct_chg_5d", "delivery_pct_chg_20d", "delivery_qty_ratio_20d",
]
OUTPUT_COLUMNS = ["symbol", "date", *FEATURE_COLUMNS, "is_missing_delivery"]

# canonical column -> raw names it has carried in the archive. The 24 files written before the
# ingester was hardened (2019-09-30 .. 2019-10-31) kept NSE's own column names.
RAW_COLUMN_NAMES = {
    "SYMBOL": ("SYMBOL",),
    "SERIES": ("SERIES",),
    "DATE": ("DATE", "DATE1"),
    "DELIV_QTY": ("DELIV_QTY",),
    "DELIV_PER": ("DELIV_PER",),
    "VOLUME": ("VOLUME", "TTL_TRD_QNTY"),
}
NUMERIC_COLUMNS = ["DELIV_QTY", "DELIV_PER", "VOLUME"]
REQUIRED_OBS_COLUMNS = {"date", "deliv_qty", "deliv_per", "volume", "volume_adj_factor"}


# --------------------------------------------------------------------------- #
# Pure computation — no I/O. See tests/test_delivery_features.py.
# --------------------------------------------------------------------------- #

def usable_observation_mask(obs: pd.DataFrame) -> pd.Series:
    """
    True where a row's delivery observation can be trusted and used.

    Usable means: delivered qty and delivery % both present, the volume needed to validate them is
    positive, the reported % agrees with delivered qty / volume * 100 to within rounding, and the
    Stage 4 volume-adjustment factor is present. Every NaN comparison is False, so an absent or
    unvalidatable input can only ever make a row unusable, never usable by accident.
    """
    traded = obs["volume"].where(obs["volume"] > 0)
    implied_pct = obs["deliv_qty"] / traded * 100
    consistent = (implied_pct - obs["deliv_per"]).abs() <= DELIVERY_PCT_TOLERANCE_PP
    return consistent & (obs["volume_adj_factor"] > 0)


def compute_delivery_features(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Delivery features for ONE symbol.

    Parameters
    ----------
    obs : one row per price-panel date, ascending, columns
          {date, deliv_qty, deliv_per, volume, volume_adj_factor}. NaN in deliv_qty / deliv_per
          means NSE gave no observation that day. volume_adj_factor is Stage 4's
          ADJ_VOLUME / VOLUME for that row (1.0 when no corporate action follows it).

    Returns
    -------
    DataFrame with `date`, the FEATURE_COLUMNS and `is_missing_delivery`, same row count and order
    as the input. Causal: row t depends only on rows <= t.
    """
    missing = REQUIRED_OBS_COLUMNS - set(obs.columns)
    if missing:
        raise ValueError(f"compute_delivery_features: missing required columns {missing}")
    if len(obs) > 1 and not obs["date"].is_monotonic_increasing:
        raise ValueError("compute_delivery_features: input must be sorted ascending by date")

    df = obs.reset_index(drop=True)
    usable = usable_observation_mask(df)

    pct = df["deliv_per"].where(usable)
    adjusted_qty = (df["deliv_qty"] * df["volume_adj_factor"]).where(usable)

    out = pd.DataFrame({"date": df["date"]})
    out["delivery_pct"] = pct
    out["delivery_pct_chg_5d"] = pct - pct.shift(5)
    out["delivery_pct_chg_20d"] = pct - pct.shift(20)
    mean_20d = adjusted_qty.rolling(20, min_periods=MIN_OBS_20D).mean()
    out["delivery_qty_ratio_20d"] = adjusted_qty / mean_20d.where(mean_20d > 0)  # all-zero window: undefined
    out["is_missing_delivery"] = ~usable
    return out


# --------------------------------------------------------------------------- #
# I/O layer — raw delivery Parquet -> per-symbol feature Parquet.
# --------------------------------------------------------------------------- #

def read_delivery_file(path: Path) -> pd.DataFrame:
    """
    Read one raw delivery file into canonical lower-case columns
    {symbol, series, date, deliv_qty, deliv_per, volume, source_file}.

    The date of every row is the date EMBEDDED in the file. The filename is never parsed as a date
    here: five archive files are named for one day and hold another's rows. A file whose rows carry
    no parseable date, or more than one date, is rejected rather than dated by its name.
    """
    path = Path(path)
    on_disk = set(pq.read_schema(path).names)
    chosen: dict[str, str] = {}
    for canonical, candidates in RAW_COLUMN_NAMES.items():
        found = next((c for c in candidates if c in on_disk), None)
        if found is None:
            raise ValueError(f"{path.name}: no column for {canonical} (tried {candidates}, file has {sorted(on_disk)})")
        chosen[canonical] = found

    raw = pd.read_parquet(path, columns=list(chosen.values()))
    frame = raw.rename(columns={v: k for k, v in chosen.items()})

    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
    if frame["DATE"].isna().any():
        raise ValueError(f"{path.name}: {int(frame['DATE'].isna().sum())} rows have an unparseable date")
    distinct = frame["DATE"].dt.normalize().unique()
    if len(distinct) > 1:
        raise ValueError(f"{path.name}: rows carry more than one date ({sorted(str(d)[:10] for d in distinct)})")

    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.strip()
    frame["SERIES"] = frame["SERIES"].astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)  # unparseable -> NaN, never 0

    frame = frame.rename(columns=str.lower)
    frame["source_file"] = path.name
    return frame[["symbol", "series", "date", "deliv_qty", "deliv_per", "volume", "source_file"]]


def _collapse_duplicate_observations(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Stale re-served files repeat an earlier day's rows verbatim, so the same (symbol, date)
    arrives from two files. Identical copies collapse to one; copies that disagree raise, because
    silently picking one would be a guess about which file is right.
    """
    key = ["symbol", "date"]
    duplicated = rows.duplicated(key, keep=False)
    if not duplicated.any():
        return rows
    values = ["deliv_qty", "deliv_per", "volume"]
    distinct_values = rows[duplicated].groupby(key)[values].nunique(dropna=False)
    conflicting = distinct_values[(distinct_values > 1).any(axis=1)]
    if len(conflicting):
        raise ValueError(
            f"conflicting delivery rows for {len(conflicting)} (symbol, date) keys "
            f"(e.g. {[tuple(map(str, k)) for k in conflicting.index[:3]]}); two files disagree about the same day"
        )
    logger.info(f"collapsed {int(duplicated.sum() - rows[duplicated].drop_duplicates(key).shape[0]):,} identical duplicate rows")
    return rows.drop_duplicates(key, keep="first")


def load_delivery(paths: Iterable[Path] | None = None) -> pd.DataFrame:
    """
    Load EQ-series delivery observations from raw files (default: every file in DELIVERY_DIR),
    keyed on each file's embedded row date. One row per (symbol, date).
    """
    paths = sorted(DELIVERY_DIR.glob("*.parquet")) if paths is None else [Path(p) for p in paths]
    frames, renamed_by_nse = [], []
    for path in paths:
        frame = read_delivery_file(path)
        if len(frame) and frame["date"].iloc[0].strftime("%Y-%m-%d") != path.stem:
            renamed_by_nse.append((path.stem, frame["date"].iloc[0].strftime("%Y-%m-%d")))
        frames.append(frame)
    if renamed_by_nse:
        logger.warning(
            f"{len(renamed_by_nse)} delivery file(s) hold rows for a different date than their name "
            f"(dated by their rows, as always): {renamed_by_nse[:6]}"
        )
    if not frames:
        return pd.DataFrame(columns=["symbol", "series", "date", "deliv_qty", "deliv_per", "volume"])

    rows = pd.concat(frames, ignore_index=True).drop(columns="source_file")
    rows = rows[rows["series"] == DELIVERY_SERIES]
    rows = _collapse_duplicate_observations(rows)
    return rows.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_price_panel(symbol: str) -> pd.DataFrame:
    """
    The symbol's Stage 4 date spine plus its volume-adjustment factor: {date, volume_adj_factor}.
    factor = ADJ_VOLUME / VOLUME (NaN where VOLUME is not positive, so it can never be used).
    """
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE", "VOLUME", "ADJ_VOLUME"])
    raw = raw.sort_values("DATE").reset_index(drop=True)
    panel = pd.DataFrame({"date": raw["DATE"]})
    panel["volume_adj_factor"] = raw["ADJ_VOLUME"] / raw["VOLUME"].where(raw["VOLUME"] > 0)
    return panel


def build_symbol_features(symbol: str, panel: pd.DataFrame, delivery: pd.DataFrame | None) -> pd.DataFrame:
    """
    Left-join one symbol's delivery observations onto its price-panel dates and compute the
    features. Panel dates with no delivery row become flagged-missing rows; delivery rows on dates
    the panel lacks (e.g. a Sunday Muhurat session) are not joined.
    """
    if delivery is None or delivery.empty:
        obs = panel.assign(deliv_qty=np.nan, deliv_per=np.nan, volume=np.nan)
    else:
        obs = panel.merge(delivery[["date", "deliv_qty", "deliv_per", "volume"]],
                          on="date", how="left", validate="one_to_one")
    features = compute_delivery_features(obs)
    features.insert(0, "symbol", symbol)
    return features[OUTPUT_COLUMNS]


def _load_checkpoint() -> set[str]:
    if DELIVERY_FEATURES_CHECKPOINT.exists():
        return set(json.loads(DELIVERY_FEATURES_CHECKPOINT.read_text())["completed_symbols"])
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    DELIVERY_FEATURES_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    DELIVERY_FEATURES_CHECKPOINT.write_text(json.dumps({
        "completed_symbols": sorted(completed),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))


def run_delivery_pipeline(resume: bool = True) -> dict:
    """
    Compute and write delivery features for every symbol in the historical universe, one Parquet
    per symbol, checkpointing after each. Rerun with resume=False whenever raw delivery files change
    (a resumed run reuses symbols already written from the older inputs).
    """
    symbols = list_universe_symbols()
    delivery = load_delivery()
    by_symbol = {s: g for s, g in delivery.groupby("symbol", sort=False)}
    completed = _load_checkpoint() if resume else set()
    remaining = [s for s in symbols if s not in completed]
    logger.info(
        f"delivery pipeline: {len(symbols):,} symbols, {len(completed):,} already done, {len(remaining):,} remaining; "
        f"{len(delivery):,} EQ delivery rows across {len(by_symbol):,} symbols"
    )

    rows = rows_with_delivery = rows_quarantined = 0
    unjoined_dates: Counter = Counter()
    quarantined_dates: Counter = Counter()
    no_delivery_symbols: list[str] = []

    for symbol in tqdm(remaining, desc="delivery features"):
        panel = load_price_panel(symbol)
        sym_delivery = by_symbol.get(symbol)
        features = build_symbol_features(symbol, panel, sym_delivery)
        write_parquet_atomic(features, DELIVERY_FEATURES_DIR / f"{symbol}.parquet")

        rows += len(features)
        rows_with_delivery += int((~features["is_missing_delivery"]).sum())
        if sym_delivery is None:
            no_delivery_symbols.append(symbol)
        else:
            on_panel = sym_delivery["date"].isin(panel["date"])
            unjoined_dates.update(sym_delivery.loc[~on_panel, "date"].dt.strftime("%Y-%m-%d"))
            reported = sym_delivery.loc[on_panel].merge(features[["date", "is_missing_delivery"]], on="date")
            quarantined = reported[reported["deliv_per"].notna() & reported["deliv_qty"].notna()
                                   & reported["is_missing_delivery"]]
            rows_quarantined += len(quarantined)
            quarantined_dates.update(quarantined["date"].dt.strftime("%Y-%m-%d"))

        completed.add(symbol)
        _save_checkpoint(completed)

    summary = {
        "symbols_written": len(remaining), "rows": rows, "rows_with_delivery": rows_with_delivery,
        "rows_quarantined": rows_quarantined, "quarantined_by_date": dict(quarantined_dates.most_common(10)),
        "delivery_rows_off_panel_by_date": dict(unjoined_dates.most_common(12)),
        "symbols_without_any_delivery": len(no_delivery_symbols),
    }
    logger.success(f"delivery pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LYNX Stage 10 — delivery feature block")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint, recompute all")
    run_delivery_pipeline(resume=not parser.parse_args().no_resume)
