"""
signals/indicators.py

Stage 9 — Technical indicators engine for LYNX.

Computes a fixed technical-indicator panel per (symbol, date) from adjusted
OHLCV data. The indicator set is carried over from SignalForge's
signals/indicators.py (RSI, MACD, Bollinger, ATR, ADX, OBV, EMA(200),
volume ratio, 52-week distance) — that part travels unchanged; it's generic
technical analysis, not something India-specific about it.

What's deliberately different from SignalForge's version:
  - Pure computation (compute_indicators) has zero DB/IO dependency and zero
    third-party TA library dependency — hand-rolled with pandas/numpy only,
    so it's fully unit-testable without a database and doesn't add a
    dependency (pandas-ta) the rest of the LYNX stack doesn't already use.
  - Missing values are never zero-filled. A stock with 90 days of history has
    NaN for RSI(14) until day 14, NaN for EMA(200) until day 200 — a zero
    would silently read as "maximally oversold," which is worse than absent.

This module handles ONE symbol's price history per call, by design — batching
across the 700-symbol universe is the driver's job (run_indicators_pipeline),
so the core logic stays trivial to unit test against a hand-built DataFrame.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from config import (
    ADJUSTED_PRICES_DIR,
    INDICATORS_CHECKPOINT,
    UNIVERSE_DIR,
    get_db_connection,
)

# Inputs are Parquet, not Postgres: Stage 4 writes data/processed/adjusted/{SYMBOL}.parquet
# and Stage 5 writes data/processed/universe/{YEAR}.parquet (see config.py). Only the
# indicator OUTPUT lives in Postgres.
INDICATORS_TABLE = "indicators"
MIN_ROWS_REQUIRED = 200  # below this, EMA(200) and the 52w window are meaningless
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

# Stage 4 adjusted-price column -> compute_indicators input column. Volume MUST come
# from ADJ_VOLUME: raw VOLUME isn't split/bonus-adjusted, so volume_ratio and OBV would
# jump at every corporate action while the prices next to them didn't.
ADJUSTED_COLUMN_MAP = {
    "DATE": "date",
    "ADJ_OPEN": "open",
    "ADJ_HIGH": "high",
    "ADJ_LOW": "low",
    "ADJ_CLOSE": "close",
    "ADJ_VOLUME": "volume",
}
PRICE_COLUMNS = ["open", "high", "low", "close"]

INDICATOR_COLUMNS = [
    "rsi_14", "macd_delta", "bb_position", "atr_14", "atr_normalized",
    "adx_14", "obv", "obv_trend", "ema_200", "volume_ratio",
    "dist_from_52w_high", "dist_from_52w_low",
]


# --------------------------------------------------------------------------- #
# Pure computation — no DB, no I/O, no third-party TA library. Unit test this
# directly against a hand-built DataFrame; see tests/test_indicators.py.
# --------------------------------------------------------------------------- #

def compute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full indicator panel for a single symbol's adjusted OHLCV history.

    Parameters
    ----------
    ohlcv : DataFrame sorted ascending by date, columns
            {date, open, high, low, close, volume}. Must already be
            corporate-action-adjusted (Stage 4 output) — this function does
            not adjust prices itself, and does not re-derive adjustment.

    Returns
    -------
    DataFrame with a `date` column plus one column per indicator in
    INDICATOR_COLUMNS, same row count and order as the input. Rows before an
    indicator's lookback window has filled are NaN.

    Raises
    ------
    ValueError if required columns are missing or dates aren't ascending —
    a silently mis-sorted input corrupts every rolling/ewm computation below
    without raising, which is exactly the kind of thing this guards against.
    """
    missing = REQUIRED_COLUMNS - set(ohlcv.columns)
    if missing:
        raise ValueError(f"compute_indicators: missing required columns {missing}")
    if len(ohlcv) > 1 and not ohlcv["date"].is_monotonic_increasing:
        raise ValueError("compute_indicators: input must be sorted ascending by date")

    df = ohlcv.reset_index(drop=True)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]

    # --- RSI(14), Wilder's smoothing (alpha = 1/14, not a plain 14-day SMA) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    # avg_loss == 0 with gains is RS -> inf, i.e. RSI 100, not "undefined". Only a flat window
    # (avg_gain == avg_loss == 0, 0/0) is genuinely undefined and stays NaN.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_14"] = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)

    # --- MACD(12,26,9) — store only the delta (line minus signal); the
    # scorer needs "is momentum turning," not the raw line/signal pair ---
    ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    out["macd_delta"] = macd_line - macd_signal

    # --- Bollinger Bands(20, 2) — store position within the band, not the
    # raw bounds; 0 = at lower band, 1 = at upper band, clipped ---
    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std(ddof=0)
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    # Flat-close edge case: 20 identical closes -> zero band width -> position is undefined, so NaN (not 0 or 0.5).
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_position"] = ((close - bb_lower) / band_width).clip(0, 1)

    # --- ATR(14), Wilder's smoothing, plus price-normalized version ---
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["atr_14"] = atr_14
    out["atr_normalized"] = atr_14 / close.replace(0, np.nan)

    # --- ADX(14), via +DI/-DI from directional movement ---
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    atr_for_di = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_for_di
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_for_di
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # --- OBV + 5-day slope (the slope is what the scorer actually uses;
    # raw cumulative OBV has no natural scale to threshold against) ---
    # fillna(0) covers row 0 (no prior close); it assumes close is never NaN afterwards, else a NaN close would read as "unchanged".
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    out["obv"] = obv
    out["obv_trend"] = obv.diff(5) / 5

    # --- EMA(200) ---
    out["ema_200"] = close.ewm(span=200, min_periods=200, adjust=False).mean()

    # --- Volume ratio (today vs 20-day average) ---
    vol_avg_20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio"] = volume / vol_avg_20.replace(0, np.nan)

    # --- Distance from 52-week high/low (~252 trading days; 200-day
    # min_periods so this activates before a full year of history exists) ---
    # Thin-history edge case: min_periods counts non-NaN values, so a masked (missing) high/low in the first 200 rows leaves this NaN.
    roll_high = high.rolling(252, min_periods=200).max()
    roll_low = low.rolling(252, min_periods=200).min()
    out["dist_from_52w_high"] = (close - roll_high) / roll_high.replace(0, np.nan)
    out["dist_from_52w_low"] = (close - roll_low) / roll_low.replace(0, np.nan)

    return out


# --------------------------------------------------------------------------- #
# I/O layer — reads Stage 4/5 Parquet, writes the Postgres `indicators` table,
# batches over the universe, checkpoints, resumes.
# --------------------------------------------------------------------------- #

def prepare_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map a Stage 4 adjusted-price frame onto compute_indicators' input schema.

    Non-positive OHLC values become NaN. The raw NSE archive records a missing
    open/high/low as 0 (34-40 such cells across the universe, all pre-2000s and never in
    CLOSE); left as 0 they'd read as real prices, e.g. a 0 LOW inflates true range and
    pins the 52-week low at 0. Missing means missing.
    """
    df = raw[list(ADJUSTED_COLUMN_MAP)].rename(columns=ADJUSTED_COLUMN_MAP)
    df = df.sort_values("date").reset_index(drop=True)
    prices = df[PRICE_COLUMNS]
    df[PRICE_COLUMNS] = prices.where(prices > 0)
    return df


def load_adjusted_prices(symbol: str) -> pd.DataFrame:
    """Load one symbol's full adjusted OHLCV history, ascending by date."""
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=list(ADJUSTED_COLUMN_MAP))
    n_masked = int((raw[["ADJ_OPEN", "ADJ_HIGH", "ADJ_LOW", "ADJ_CLOSE"]] <= 0).sum().sum())
    if n_masked:
        logger.warning(f"{symbol}: {n_masked} non-positive OHLC value(s) treated as missing")
    return prepare_ohlcv(raw)


def list_universe_symbols() -> list[str]:
    """
    All symbols that have ever appeared in a Stage 5 point-in-time universe
    snapshot — the indicator panel is computed for the full historical
    universe, not just today's constituents, so walk-forward folds over any
    past year have indicators available for the symbols that were live then.
    """
    symbols: set[str] = set()
    for f in sorted(UNIVERSE_DIR.glob("*.parquet")):
        symbols |= set(pd.read_parquet(f, columns=["SYMBOL"])["SYMBOL"])

    available = {f.stem for f in ADJUSTED_PRICES_DIR.glob("*.parquet")}
    missing = sorted(symbols - available)
    if missing:
        logger.warning(f"{len(missing)} universe symbols have no adjusted-price file: {missing[:10]}")
    return sorted(symbols & available)


def ensure_indicators_table(conn) -> None:
    """
    Create the indicators table from INDICATOR_COLUMNS if absent. If it already
    exists, its columns must match exactly — a stale table silently missing a new
    indicator (or carrying a dropped one) is worse than a loud failure here.
    """
    expected = ["symbol", "date", *INDICATOR_COLUMNS]
    ddl_cols = ",\n            ".join(f"{c} DOUBLE PRECISION" for c in INDICATOR_COLUMNS)
    with conn, conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {INDICATORS_TABLE} (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                {ddl_cols},
                PRIMARY KEY (symbol, date)
            )
        """)
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
            (INDICATORS_TABLE,),
        )
        actual = [row[0] for row in cur.fetchall()]
    if actual != expected:
        raise RuntimeError(
            f"table {INDICATORS_TABLE} columns {actual} do not match INDICATOR_COLUMNS {expected}; "
            f"drop or migrate the table before running"
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def write_indicators(conn, symbol: str, indicators: pd.DataFrame) -> None:
    """
    Replace one symbol's indicator panel, atomically. Delete-then-COPY in a single
    transaction makes reruns and retries idempotent. NaN is written as SQL NULL, not
    float NaN — Postgres would happily store NaN, and `IS NULL` checks and aggregates
    downstream would then mishandle it.
    """
    frame = indicators[["date", *INDICATOR_COLUMNS]].copy()
    frame.insert(0, "symbol", symbol)
    buf = io.StringIO()
    frame.to_csv(buf, index=False, header=False, na_rep="", date_format="%Y-%m-%d")
    buf.seek(0)

    columns = ", ".join(["symbol", "date", *INDICATOR_COLUMNS])
    with conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM {INDICATORS_TABLE} WHERE symbol = %s", (symbol,))
        cur.copy_expert(f"COPY {INDICATORS_TABLE} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)


def _load_checkpoint() -> set[str]:
    if INDICATORS_CHECKPOINT.exists():
        return set(json.loads(INDICATORS_CHECKPOINT.read_text())["completed_symbols"])
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    INDICATORS_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    INDICATORS_CHECKPOINT.write_text(json.dumps({
        "completed_symbols": sorted(completed),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))


def run_indicators_pipeline(resume: bool = True) -> None:
    """
    Batch driver: compute and write indicators for every symbol in the
    historical universe. Checkpoints after every symbol so an interrupted run
    resumes rather than restarting.
    """
    symbols = list_universe_symbols()
    completed = _load_checkpoint() if resume else set()
    remaining = [s for s in symbols if s not in completed]

    logger.info(
        f"indicators pipeline: {len(symbols)} symbols total, "
        f"{len(completed)} already done, {len(remaining)} remaining"
    )

    skipped_short_history: list[str] = []

    conn = get_db_connection()
    try:
        ensure_indicators_table(conn)
        for symbol in tqdm(remaining, desc="computing indicators"):
            try:
                prices = load_adjusted_prices(symbol)
            except Exception:
                logger.exception(f"failed to load prices for {symbol}, skipping this run")
                continue

            if len(prices) < MIN_ROWS_REQUIRED:
                skipped_short_history.append(symbol)
                completed.add(symbol)  # don't retry every run — short history won't change
                continue

            indicators = compute_indicators(prices)
            write_indicators(conn, symbol, indicators)
            completed.add(symbol)
            _save_checkpoint(completed)
    finally:
        conn.close()
    _save_checkpoint(completed)

    if skipped_short_history:
        logger.warning(
            f"{len(skipped_short_history)} symbols skipped, fewer than "
            f"{MIN_ROWS_REQUIRED} rows of history: {skipped_short_history[:10]}"
            f"{'...' if len(skipped_short_history) > 10 else ''}"
        )
    logger.success(f"indicators pipeline complete: {len(completed)}/{len(symbols)} symbols")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LYNX Stage 9 — technical indicators engine")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint, recompute all")
    args = parser.parse_args()
    run_indicators_pipeline(resume=not args.no_resume)