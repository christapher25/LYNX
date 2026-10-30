"""
gates/liquidity.py

Stage 11 -- point-in-time liquidity rank for the liquidity-cliff gate (v6 section 6.2: "the liquidity cliff below the top 200").

    liquidity_rank   1 = most liquid. Rank of the symbol's trailing mean TURNOVER (ADJ_CLOSE * ADJ_VOLUME, which a split leaves
                     unchanged) among all symbols with a rank that session, over the TRAILING_SESSIONS sessions strictly BEFORE t.
    is_missing_liquidity_rank   True exactly where the rank is NaN.

Point-in-time: session t's rank uses sessions up to t-1 only (the rolling mean is shifted one row), so it is knowable at the t close
when the signal is made. A rank needs MIN_HISTORY observed prior sessions and at least one traded session in the last ACTIVE_SESSIONS
(a name that has stopped trading does not keep holding a rank slot against live names). Missing turnover is missing, never 0: such a
row gets NaN and a flag, and does not take a slot. Ties share the better (lower) rank.

Window length and minimum history are CHOSEN (the v6 document names the top-200 cut but no window); TRAILING_SESSIONS matches Stage 5's
252-day trailing turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAILING_SESSIONS = 252
MIN_HISTORY = 126
ACTIVE_SESSIONS = 20
_BUFFER_DAYS = 450                    # calendar days of history loaded before `start` so the first rank has a full window


def trailing_turnover_rank(panel: pd.DataFrame, start=None, window: int = TRAILING_SESSIONS, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """
    panel : {SYMBOL, DATE, ADJ_CLOSE, ADJ_VOLUME}, one row per symbol per trading day it has a bar. NaN / non-positive close or NaN
            volume make that day's turnover missing.
    start : first session to return (history before it is still used for the window).
    Returns {symbol, date, liquidity_rank, is_missing_liquidity_rank} for every (symbol, date) row of the panel on or after `start`.
    """
    need = {"SYMBOL", "DATE", "ADJ_CLOSE", "ADJ_VOLUME"}
    if not need <= set(panel.columns):
        raise ValueError(f"panel needs columns {sorted(need)}")
    work = pd.DataFrame({"symbol": panel["SYMBOL"].to_numpy(), "date": pd.to_datetime(panel["DATE"]).dt.normalize().to_numpy()})
    if work.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate (SYMBOL, DATE) rows in panel")
    close = panel["ADJ_CLOSE"].to_numpy(float)
    volume = panel["ADJ_VOLUME"].to_numpy(float)
    work["turnover"] = np.where(close > 0, close * volume, np.nan)
    if start is not None:
        start = pd.Timestamp(start)
        work = work[work["date"] >= start - pd.Timedelta(days=_BUFFER_DAYS)]

    wide = work.pivot(index="date", columns="symbol", values="turnover").sort_index()
    mean_before = wide.rolling(window, min_periods=min_history).mean().shift(1)
    traded_recently = wide.notna().astype(float).rolling(ACTIVE_SESSIONS, min_periods=1).sum().shift(1) > 0
    rankable = mean_before.where(traded_recently)
    rank = rankable.rank(axis=1, ascending=False, method="min")

    out = rank.reset_index().melt(id_vars="date", var_name="symbol", value_name="liquidity_rank")
    keep = work[["symbol", "date"]]
    out = keep.merge(out, on=["symbol", "date"], how="left")                 # only (symbol, date) rows that exist in the panel
    if start is not None:
        out = out[out["date"] >= start]
    out["is_missing_liquidity_rank"] = out["liquidity_rank"].isna()
    return out.sort_values(["symbol", "date"], ignore_index=True)
