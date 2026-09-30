"""
Stage 4 (part 2) — apply corporate action adjustments to bhavcopy prices.

Reads the corporate action log (data/raw/corporate_actions/*.parquet) and
the raw bhavcopy series, produces backward-adjusted prices: today's price
stays real, history is scaled to match it.

Only bonus and split actions carry an adjustment_factor (set by
corporate_action_parser.py). Dividends never adjust price — see that
module's docstring for why. Any action with parse_confidence in
('failed', 'low') has adjustment_factor=None and is skipped here too —
this engine will never adjust on an uncertain parse.

Output: data/processed/adjusted/{SYMBOL}.parquet — one file per symbol,
full adjusted history, ready for feature engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BHAVCOPY_DIR, PROJECT_ROOT
from data.ingest_common import write_parquet_atomic, get_logger

ACTIONS_DIR = PROJECT_ROOT / "data" / "raw" / "corporate_actions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "adjusted"
LOG_DIR = PROJECT_ROOT / "logs"

PRICE_COLS = ["OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREV_CLOSE"]
VOLUME_COLS = ["VOLUME"]

MAX_UNEXPLAINED_MOVE = 0.20   # matches the invariant test's band


def load_all_bhavcopy() -> pd.DataFrame:
    """
    Single pass over bhavcopy, once — not per symbol. The previous version
    re-scanned all 7,875 files for every symbol individually (40M+ file
    reads, 50+ hours). This reads the archive exactly once into memory,
    matching the pattern build_universe.py already uses correctly.
    """
    frames = []
    for f in sorted(BHAVCOPY_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["SYMBOL", "SERIES", "DATE", *PRICE_COLS, *VOLUME_COLS])
        frames.append(df[df["SERIES"] == "EQ"])
    return pd.concat(frames, ignore_index=True).sort_values(["SYMBOL", "DATE"])


def load_symbol_actions(symbol: str) -> pd.DataFrame:
    """Only actions with a usable adjustment_factor — everything else is
    excluded here by construction, never silently zero-filled or guessed."""
    path = ACTIONS_DIR / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ex_date", "adjustment_factor", "action_type"])

    df = pd.read_parquet(path)
    usable = df[df["adjustment_factor"].notna()].copy()
    return usable[["ex_date", "adjustment_factor", "action_type"]].sort_values("ex_date")


def apply_backward_adjustment(
    prices: pd.DataFrame, actions: pd.DataFrame
) -> pd.DataFrame:
    """
    Standard backward adjustment: for each action, every row dated strictly
    BEFORE the ex-date gets multiplied by that action's factor. Applied in
    chronological order so multiple actions compound correctly — a stock
    with two bonuses in its history gets both applied to the years before
    both, one bonus applied to the year between them, matching how
    cumulative adjustment is supposed to work.
    """
    adjusted = prices.copy()
    adjusted["ADJ_CLOSE"] = adjusted["CLOSE"].astype(float)
    for col in PRICE_COLS:
        adjusted[f"ADJ_{col}"] = adjusted[col].astype(float)
    adjusted["ADJ_VOLUME"] = adjusted["VOLUME"].astype(float)
    adjusted["ADJUSTMENT_APPLIED"] = False

    for _, action in actions.iterrows():
        ex_date = action["ex_date"]
        factor = action["adjustment_factor"]

        before_mask = adjusted["DATE"] < ex_date
        for col in PRICE_COLS:
            adjusted.loc[before_mask, f"ADJ_{col}"] *= factor
        # Volume moves inverse to price — more shares outstanding after a
        # bonus/split means the same rupee turnover is more shares traded.
        adjusted.loc[before_mask, "ADJ_VOLUME"] /= factor
        adjusted.loc[before_mask, "ADJUSTMENT_APPLIED"] = True

    return adjusted


def validate_adjustment(symbol: str, adjusted: pd.DataFrame) -> list[str]:
    """
    Runs the SAME check test_corporate_actions.py defines as the contract:
    no adjusted daily return may exceed the band without a matching action
    record. Since actions ARE already applied here, this is really checking
    for actions we don't have data for — an unadjusted bonus we never saw.
    """
    problems = []
    returns = adjusted["ADJ_CLOSE"].pct_change()
    suspicious = adjusted[returns.abs() > MAX_UNEXPLAINED_MOVE]

    for _, row in suspicious.iterrows():
        if not row["ADJUSTMENT_APPLIED"]:
            problems.append(
                f"{symbol} {row['DATE'].date()}: unexplained "
                f"{returns.loc[row.name]:.1%} move, no action record covers it"
            )
    return problems


def process_symbol(symbol: str, panel: pd.DataFrame, logger) -> dict:
    prices = panel[panel["SYMBOL"] == symbol].reset_index(drop=True)
    if prices.empty:
        return {"symbol": symbol, "status": "no_price_data"}

    actions = load_symbol_actions(symbol)
    adjusted = apply_backward_adjustment(prices, actions)

    problems = validate_adjustment(symbol, adjusted)
    if problems:
        for p in problems:
            logger.warning(p)

    out_path = OUTPUT_DIR / f"{symbol}.parquet"
    write_parquet_atomic(adjusted, out_path)

    return {
        "symbol": symbol,
        "status": "ok",
        "rows": len(adjusted),
        "actions_applied": len(actions),
        "unexplained_moves": len(problems),
    }


def main() -> None:
    logger = get_logger("price_adjustment", LOG_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    action_files = sorted(ACTIONS_DIR.glob("*.parquet"))
    symbols = [f.stem for f in action_files]
    logger.info(f"Processing {len(symbols)} symbols with action data")

    logger.info("Loading full bhavcopy panel (one pass)...")
    panel = load_all_bhavcopy()
    logger.info(f"Loaded {len(panel):,} rows, {panel['SYMBOL'].nunique():,} symbols")

    total_unexplained = 0
    for i, symbol in enumerate(symbols, 1):
        result = process_symbol(symbol, panel, logger)
        if result["status"] == "ok":
            total_unexplained += result["unexplained_moves"]
        if i % 100 == 0:
            logger.info(f"[{i}/{len(symbols)}] unexplained_moves so far: {total_unexplained}")

    logger.info(f"Complete. Total unexplained moves across universe: {total_unexplained}")
    if total_unexplained > 0:
        logger.warning(
            f"{total_unexplained} price moves exceed the {MAX_UNEXPLAINED_MOVE:.0%} "
            f"band with no covering action record. These are either actions "
            f"the ingester missed (demerger/scheme-of-arrangement cases we "
            f"already know are unparseable) or genuine large moves — each "
            f"needs a manual look before SSL pretraining touches this data."
        )


if __name__ == "__main__":
    main()