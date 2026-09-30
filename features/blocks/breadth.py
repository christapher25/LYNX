"""
features/blocks/breadth.py

Stage 10 — market breadth: the share of stocks trading above their own 200-day simple moving average.

    universe_pct_above_200dma   100 * (# evaluable members with ADJ_CLOSE > SMA200) / (# evaluable members)
    universe_n_evaluable        the denominator: members with a full 200-row window on that session
    is_missing_breadth          True exactly where the percentage is NaN

The column is named for what it measures. It is NOT "% of Nifty 500 above the 200-DMA": no historical Nifty 500
constituent lists exist on disk, so the population is the Stage 5 point-in-time universe (the top 700 by trailing turnover,
ranked on data strictly before 1 January of each year and applied to that year's sessions). The scorer's market-context
layer expects a `nifty500_pct_above_200dma` input; this is a proxy for it, not the same quantity, and must not be renamed.

Definition choices, all recorded rather than implied:
  * SMA (not the EMA(200) Stage 9 uses for single-stock features): "200-day average" in breadth is the simple average.
  * The window is 200 ROWS of that stock's own adjusted-close history, current session included; a suspension leaves no row,
    so the window then spans more calendar days (the same row-based convention as the Stage 9 indicators).
  * Strict: a close equal to its average is not "above", and "equal" means within EQUALITY_TOLERANCE (1e-9, relative).
    Two correct ways to average a flat window differ in the last bit (np.mean of three 0.7s is below 0.7), so an exact
    comparison would make a tie depend on the implementation. Nothing real is that close: one 0.05 tick on Rs 1,000 is 5e-5.
    (Measured: 0 exactly-flat 200-row windows among 1.3M evaluable windows of the first 600 symbols.)
  * A stock is in the denominator only with a full, gap-free window of positive closes. A non-positive close is a missing
    price (never a real one) and blanks every window that contains it. Missing means missing, never 0.
  * A session is missing (NaN, flagged) when it has no price rows at all (e.g. the 2020-07-13 bhavcopy gap), or fewer than
    MIN_EVALUABLE stocks are evaluable: a thin sample is not the market. On the real data the count never falls below 204.

Point-in-time. The value for session t uses only closes up to and including t (the same-day bhavcopy, public well before
the 19:00 IST run) and the universe file for t's year, which was built from data before that year began. Adjusted closes are
rescaled retroactively by later corporate actions, but a future action rescales every close before it by one constant, so
"close > average" for any session before it is unchanged; the truncation invariant test exercises exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ADJUSTED_PRICES_DIR, BREADTH_FEATURES_DIR, UNIVERSE_DIR
from data.trading_calendar import trading_sessions
from features.blocks.common import validated_sessions

BREADTH_WINDOW = 200
MIN_EVALUABLE = 100
EQUALITY_TOLERANCE = 1e-9          # relative: a close this near its average is a tie (float noise), not "above"
BREADTH_COLUMNS = ["universe_pct_above_200dma", "universe_n_evaluable"]


def compute_breadth_features(panel: pd.DataFrame, members_by_year: Mapping[int, Collection[str]], sessions: Sequence,
                             window: int = BREADTH_WINDOW, min_evaluable: int = MIN_EVALUABLE) -> pd.DataFrame:
    """
    Parameters
    ----------
    panel           : {SYMBOL, DATE, ADJ_CLOSE}, one row per symbol per trading day; NaN / non-positive close = missing.
    members_by_year : {year: symbols in that year's point-in-time universe}.
    sessions        : ascending trading-session dates; the output has exactly one row per session.

    Returns {date, BREADTH_COLUMNS, is_missing_breadth}. Causal: a row depends only on panel rows dated on or before it.
    """
    if not {"SYMBOL", "DATE", "ADJ_CLOSE"} <= set(panel.columns):
        raise ValueError("panel needs columns {SYMBOL, DATE, ADJ_CLOSE}")
    sess = validated_sessions(sessions)
    work = pd.DataFrame({"SYMBOL": panel["SYMBOL"].to_numpy(), "DATE": pd.to_datetime(panel["DATE"]).dt.normalize().to_numpy()})
    if work.duplicated(["SYMBOL", "DATE"]).any():
        raise ValueError("duplicate (SYMBOL, DATE) rows in panel")
    close = panel["ADJ_CLOSE"].to_numpy(float)
    work["close"] = np.where(close > 0, close, np.nan)                      # a non-positive close is a missing price
    work = work.sort_values(["SYMBOL", "DATE"], kind="mergesort")
    work["sma"] = work.groupby("SYMBOL", sort=False)["close"].transform(lambda s: s.rolling(window, min_periods=window).mean())

    year = work["DATE"].dt.year.to_numpy()
    member = np.zeros(len(work), dtype=bool)
    for y, symbols in members_by_year.items():
        rows = year == int(y)                                               # a symbol is a member only in the years it is listed for
        member[rows] = work.loc[rows, "SYMBOL"].isin(set(symbols)).to_numpy()
    evaluable = member & work["sma"].notna().to_numpy()                     # sma is defined only from a full window of real closes
    above = evaluable & (work["close"].to_numpy() > work["sma"].to_numpy() * (1.0 + EQUALITY_TOLERANCE))

    daily = pd.DataFrame({"DATE": work["DATE"].to_numpy(), "e": evaluable, "a": above}).groupby("DATE").agg(n_eval=("e", "sum"), n_above=("a", "sum"))
    daily = daily.reindex(sess)                                             # a session with no price rows at all -> NaN, not 0
    enough = daily["n_eval"] >= min_evaluable
    pct = (100.0 * daily["n_above"] / daily["n_eval"]).where(enough)
    out = pd.DataFrame({"date": sess, "universe_pct_above_200dma": pct.to_numpy(float),
                        "universe_n_evaluable": daily["n_eval"].to_numpy(float)})
    out["is_missing_breadth"] = out["universe_pct_above_200dma"].isna()
    return out[["date", *BREADTH_COLUMNS, "is_missing_breadth"]]


# ─── loaders and driver ──────────────────────────────────────────────────────────────────────────────────────────

def load_universe_members(directory: Path = UNIVERSE_DIR) -> dict[int, set[str]]:
    return {int(p.stem): set(pd.read_parquet(p, columns=["SYMBOL"])["SYMBOL"]) for p in sorted(Path(directory).glob("*.parquet"))}


def load_adjusted_close_panel(directory: Path = ADJUSTED_PRICES_DIR) -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=["SYMBOL", "DATE", "ADJ_CLOSE"]) for p in sorted(Path(directory).glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def run_breadth_pipeline() -> dict:
    """Build the market-level breadth block over the trading-session calendar and write it."""
    from data.ingest_common import write_parquet_atomic

    sessions = trading_sessions()
    out = compute_breadth_features(load_adjusted_close_panel(), load_universe_members(), sessions)
    write_parquet_atomic(out, BREADTH_FEATURES_DIR / "universe_breadth.parquet")
    return {"sessions": len(out), "with_data": int((~out["is_missing_breadth"]).sum())}


if __name__ == "__main__":
    print(run_breadth_pipeline())
