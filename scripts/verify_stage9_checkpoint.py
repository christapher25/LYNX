"""
Stage 9 checkpoint verification.

Checks, against the live Postgres `indicators` table and the Stage 4/5 Parquet inputs:
  1. The connection reaches the LYNX container's database (not some other Postgres).
  2. Indicator row count == OHLCV row count, for every universe symbol and for a sample ticker.
  3. Warm-up rows are NULL, never 0 (and never float NaN / inf) — universe-wide.
  4. DB values round-trip exactly against a fresh compute_indicators() run for the sample ticker.
  5. score_signal() returns the full audit dict for the sample ticker's latest row.

Usage: python -m scripts.verify_stage9_checkpoint [--ticker RELIANCE]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from config import ADJUSTED_PRICES_DIR, PROJECT_ROOT, get_db_connection
from signals.indicators import (
    INDICATOR_COLUMNS,
    INDICATORS_TABLE,
    MIN_ROWS_REQUIRED,
    compute_indicators,
    list_universe_symbols,
    load_adjusted_prices,
)
from signals.scorer import score_signal

# Rows that must be NULL at the start of every symbol's history, from each indicator's
# hard lookback (min_periods) in compute_indicators. A value before this is a fabricated one.
MIN_WARMUP_NULL_ROWS = {
    "rsi_14": 14, "atr_14": 13, "atr_normalized": 13, "adx_14": 26,
    "macd_delta": 33, "bb_position": 19, "volume_ratio": 19, "obv_trend": 5,
    "ema_200": 199, "dist_from_52w_high": 199, "dist_from_52w_low": 199,
}
MACRO_DIR = PROJECT_ROOT / "data" / "raw"

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


def check_connection(cur) -> None:
    cur.execute("SELECT current_setting('data_directory'), split_part(version(), ' ', 2)")
    data_dir, version = cur.fetchone()
    check(data_dir.startswith("/var/lib/postgresql"), "connected to the LYNX container DB",
          f"(data_directory={data_dir}, PG {version})")


def check_universe_row_counts(cur) -> None:
    cur.execute(f"SELECT symbol, count(*) FROM {INDICATORS_TABLE} GROUP BY symbol")
    actual = dict(cur.fetchall())
    symbols = list_universe_symbols()
    expected = {}
    for s in symbols:
        n = pq.ParquetFile(ADJUSTED_PRICES_DIR / f"{s}.parquet").metadata.num_rows
        expected[s] = n if n >= MIN_ROWS_REQUIRED else 0
    mismatched = [s for s in symbols if actual.get(s, 0) != expected[s]]
    extra = sorted(set(actual) - set(symbols))
    check(not mismatched and not extra, "indicator rows == OHLCV rows for every universe symbol",
          f"({len(symbols)} symbols, {sum(expected.values()):,} rows expected, {sum(actual.values()):,} found; "
          f"mismatched={mismatched[:5]}, extra={extra[:5]})")


def check_no_fake_values(cur) -> None:
    cols = ", ".join(INDICATOR_COLUMNS)
    nan_or_inf = " + ".join(
        f"count(*) FILTER (WHERE {c} = 'NaN'::float8 OR {c} = 'Infinity'::float8 OR {c} = '-Infinity'::float8)"
        for c in INDICATOR_COLUMNS
    )
    cur.execute(f"SELECT {nan_or_inf} FROM {INDICATORS_TABLE}")
    check(cur.fetchone()[0] == 0, "no float NaN / inf stored (missing is NULL)")

    warm = ", ".join(
        f"count(*) FILTER (WHERE rn <= {n} AND {c} IS NOT NULL) AS {c}" for c, n in MIN_WARMUP_NULL_ROWS.items()
    )
    cur.execute(f"""
        SELECT {warm} FROM (
            SELECT {cols}, row_number() OVER (PARTITION BY symbol ORDER BY date) AS rn FROM {INDICATORS_TABLE}
        ) t
    """)
    early_non_null = dict(zip(MIN_WARMUP_NULL_ROWS, cur.fetchone()))
    bad = {c: n for c, n in early_non_null.items() if n}
    check(not bad, "every indicator is NULL (not 0) before its lookback fills, all symbols", f"(violations: {bad})")

    unexpected = ", ".join(
        f"count(*) FILTER (WHERE rn > {n} AND {c} IS NULL) AS {c}" for c, n in MIN_WARMUP_NULL_ROWS.items()
    )
    cur.execute(f"""
        SELECT {unexpected} FROM (
            SELECT {cols}, row_number() OVER (PARTITION BY symbol ORDER BY date) AS rn FROM {INDICATORS_TABLE}
        ) t
    """)
    late_nulls = {c: n for c, n in zip(MIN_WARMUP_NULL_ROWS, cur.fetchone()) if n}
    logger.info(f"INFO  NULLs after warm-up (undefined by construction, e.g. flat prices): {late_nulls or 'none'}")


def check_sample_ticker(cur, ticker: str) -> pd.DataFrame:
    prices = load_adjusted_prices(ticker)
    cur.execute(f"SELECT date, {', '.join(INDICATOR_COLUMNS)} FROM {INDICATORS_TABLE} WHERE symbol = %s ORDER BY date",
                (ticker,))
    db = pd.DataFrame(cur.fetchall(), columns=["date", *INDICATOR_COLUMNS])
    db["date"] = pd.to_datetime(db["date"])
    db[INDICATOR_COLUMNS] = db[INDICATOR_COLUMNS].astype(float)  # NULL -> NaN for comparison only

    check(len(db) == len(prices), f"{ticker}: indicator count == OHLCV row count", f"({len(db):,} vs {len(prices):,})")
    check(db["date"].tolist() == prices["date"].tolist(), f"{ticker}: dates line up 1:1")

    head = db.head(MIN_WARMUP_NULL_ROWS["ema_200"])
    check(head["ema_200"].isna().all() and db["ema_200"].iloc[199:].notna().all(),
          f"{ticker}: ema_200 NULL for first 199 rows, populated from row 200")
    check(db["rsi_14"].iloc[:14].isna().all() and db["rsi_14"].iloc[14:].notna().any(),
          f"{ticker}: rsi_14 NULL for first 14 rows")
    check(not (db[[c for c in MIN_WARMUP_NULL_ROWS]].iloc[:5] == 0).any().any(),
          f"{ticker}: no zero-filled values in first 5 rows")
    logger.info(f"first 3 rows for {ticker}:\n{db.head(3).to_string(index=False)}")
    logger.info(f"rows 198-200 (ema_200 wakes up) for {ticker}:\n{db.iloc[197:200].to_string(index=False)}")

    fresh = compute_indicators(prices)
    same_nulls = (fresh[INDICATOR_COLUMNS].isna().values == db[INDICATOR_COLUMNS].isna().values).all()
    close_vals = np.allclose(fresh[INDICATOR_COLUMNS].values.astype(float), db[INDICATOR_COLUMNS].values,
                             rtol=1e-9, atol=1e-9, equal_nan=True)
    check(same_nulls and close_vals, f"{ticker}: DB values == fresh compute_indicators() (NULL pattern + values)")
    return db


def market_context_asof(date: pd.Timestamp) -> pd.Series:
    """Real Stage 8 values where they exist as of `date`; NaN (missing) where they don't."""
    ctx = {"india_vix": np.nan, "fii_net_flow_cr": np.nan,
           "nifty500_pct_above_200dma": np.nan, "usdinr_chg_pct": np.nan}

    idx_path = MACRO_DIR / "indices" / f"{date:%Y-%m-%d}.parquet"
    if idx_path.exists():
        idx = pd.read_parquet(idx_path, columns=["INDEX_NAME", "CLOSE"])
        vix = idx.loc[idx["INDEX_NAME"] == "India VIX", "CLOSE"]
        if len(vix):
            ctx["india_vix"] = float(vix.iloc[0])

    fx = pd.read_parquet(MACRO_DIR / "macro" / "usdinr.parquet", columns=["date", "rate"])
    fx = fx[fx["date"] <= date].sort_values("date")
    if len(fx) >= 2:
        ctx["usdinr_chg_pct"] = float(fx["rate"].iloc[-1] / fx["rate"].iloc[-2] - 1)  # fraction, matching scorer thresholds

    fii_path = MACRO_DIR / "fii_dii" / f"{date:%Y-%m-%d}.parquet"
    if fii_path.exists():
        fii = pd.read_parquet(fii_path)
        net = fii.loc[fii["CATEGORY"] == "FII/FPI", "NET_VALUE_CR"]
        if len(net):
            ctx["fii_net_flow_cr"] = float(net.iloc[0])
    return pd.Series(ctx, dtype=float)


def check_score_signal(db: pd.DataFrame, ticker: str) -> None:
    prices = load_adjusted_prices(ticker)
    latest = db.iloc[-1]
    close = float(prices.loc[prices["date"] == latest["date"], "close"].iloc[0])
    indicator_row = latest[INDICATOR_COLUMNS].astype(float)
    indicator_row["close"] = close
    context_row = market_context_asof(latest["date"])
    logger.info(f"latest row for {ticker}: {latest['date']:%Y-%m-%d} close={close}\n{indicator_row.to_string()}")
    logger.info(f"market context as of {latest['date']:%Y-%m-%d} (NaN = no Stage 8 data that day):\n{context_row.to_string()}")

    audit = score_signal(indicator_row, context_row)
    logger.info(f"score_signal audit dict: {audit}")

    expected_keys = {"technical_score", "market_context_score", "fundamental_score", "sentiment_score",
                     "active_layers", "renormalized_weights", "single_row_composite_NOT_cross_sectional"}
    check(set(audit) == expected_keys, "score_signal returns the full audit dict", f"(keys={sorted(audit)})")
    check(0 <= audit["technical_score"] <= 100 and 0 <= audit["market_context_score"] <= 100, "layer scores within 0-100")
    check(audit["fundamental_score"] is None and audit["sentiment_score"] is None
          and audit["active_layers"] == ["technical", "market_context"], "Phase 4 layers absent, not faked")
    check(abs(sum(audit["renormalized_weights"].values()) - 1.0) < 1e-9, "renormalized weights sum to 1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Stage 9 checkpoint")
    parser.add_argument("--ticker", default="RELIANCE")
    args = parser.parse_args()

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            check_connection(cur)
            check_universe_row_counts(cur)
            check_no_fake_values(cur)
            db = check_sample_ticker(cur, args.ticker)
        check_score_signal(db, args.ticker)
    finally:
        conn.close()

    if failures:
        logger.error(f"Stage 9 checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 9 checkpoint PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
