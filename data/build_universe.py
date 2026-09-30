"""
Stage 5 — point-in-time universe construction.

Per-year constituent lists, ranked by trailing liquidity (avg daily
turnover), computed using only data available as of that point — never
today's universe projected backward (the survivorship-bias bug the
architecture doc explicitly forbids under "Walk-forward on historical
constituents only").

Delisted names are retained with a terminal value (last traded ADJ_CLOSE
before they stop appearing in the archive) rather than dropped — dropping
them would make historical years look artificially survivor-biased.

Output: data/processed/universe/{YEAR}.parquet — one file per year,
columns: SYMBOL, RANK, AVG_TURNOVER, FIRST_DATE, LAST_DATE, IS_DELISTED,
TERMINAL_VALUE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BHAVCOPY_DIR, PROJECT_ROOT
from data.ingest_common import write_parquet_atomic, get_logger

ADJUSTED_DIR = PROJECT_ROOT / "data" / "processed" / "adjusted"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "universe"
LOG_DIR = PROJECT_ROOT / "logs"

UNIVERSE_SIZE = 700          # matches the project's stated target universe
TRAILING_DAYS = 252          # ~1 trading year of turnover history


def load_all_adjusted() -> pd.DataFrame:
    """Concatenate every symbol's adjusted series into one long frame."""
    frames = []
    for f in sorted(ADJUSTED_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["SYMBOL", "DATE", "ADJ_CLOSE", "ADJ_VOLUME"])
        df["TURNOVER"] = df["ADJ_CLOSE"] * df["ADJ_VOLUME"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_year_universe(
    panel: pd.DataFrame, year: int, universe_size: int, trailing_days: int
) -> pd.DataFrame:
    """
    Rank symbols by trailing turnover using ONLY data strictly before
    Jan 1 of the target year — this is the point-in-time boundary. A
    symbol's inclusion in year Y's universe never depends on anything
    that happened in or after year Y.
    """
    cutoff = pd.Timestamp(f"{year}-01-01")
    window_start = cutoff - pd.Timedelta(days=trailing_days * 1.5)  # calendar buffer

    trailing = panel[(panel["DATE"] >= window_start) & (panel["DATE"] < cutoff)]
    if trailing.empty:
        return pd.DataFrame()

    liquidity = (
        trailing.groupby("SYMBOL")["TURNOVER"]
        .mean()
        .rename("AVG_TURNOVER")
        .reset_index()
    )
    ranked = liquidity.sort_values("AVG_TURNOVER", ascending=False).reset_index(drop=True)
    ranked["RANK"] = ranked.index + 1
    top = ranked.head(universe_size).copy()

    # First/last trade date for each selected symbol, across its FULL
    # history — this is what identifies delisting, not just the trailing
    # window used for ranking.
    full_dates = panel.groupby("SYMBOL")["DATE"].agg(["min", "max"])
    top = top.merge(full_dates, on="SYMBOL", how="left")
    top = top.rename(columns={"min": "FIRST_DATE", "max": "LAST_DATE"})

    # A symbol is "delisted" for this snapshot's purposes if its last
    # trade predates the year we're building the universe for.
    top["IS_DELISTED"] = top["LAST_DATE"] < cutoff

    terminal = panel.sort_values("DATE").groupby("SYMBOL")["ADJ_CLOSE"].last()
    top["TERMINAL_VALUE"] = top["SYMBOL"].map(terminal)
    top.loc[~top["IS_DELISTED"], "TERMINAL_VALUE"] = None  # only meaningful if delisted

    return top[["SYMBOL", "RANK", "AVG_TURNOVER", "FIRST_DATE", "LAST_DATE",
                "IS_DELISTED", "TERMINAL_VALUE"]]


def main() -> None:
    logger = get_logger("universe", LOG_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading adjusted price history...")
    panel = load_all_adjusted()
    logger.info(f"Loaded {len(panel):,} rows across {panel['SYMBOL'].nunique()} symbols")

    years = sorted(panel["DATE"].dt.year.unique())
    years = [y for y in years if y >= 1996]   # need >=1yr trailing history first

    for year in years:
        out_path = OUTPUT_DIR / f"{year}.parquet"
        if out_path.exists():
            continue

        universe = build_year_universe(panel, year, UNIVERSE_SIZE, TRAILING_DAYS)
        if universe.empty:
            logger.warning(f"{year}: no trailing data, skipped")
            continue

        write_parquet_atomic(universe, out_path)
        n_delisted = int(universe["IS_DELISTED"].sum())
        logger.info(f"{year}: {len(universe)} symbols, {n_delisted} delisted-by-then")

    logger.info(f"Complete. {len(years)} yearly universe snapshots written.")


if __name__ == "__main__":
    main()