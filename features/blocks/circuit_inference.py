"""
features/blocks/circuit_inference.py

Price-band inference for the sessions before NSE's security list exists (sec_list starts 2019-12-30). PIT-safe by construction: the
estimate for session t uses only prices and hit lists dated BEFORE t. (The band for t is published in the file of the previous
session, so that session's prices are the newest evidence that could have existed. Backfilling from a later file would let future
data change past values: it fails the truncation invariant and is not offered.)

Two kinds of evidence, in order of strength:

  hit       a stock that touched its band reveals it exactly. The limit is the previous close times (1 +/- band), floored to the
            0.05 tick for the upper band and ceiled for the lower (measured 2026-09-24: 84-87% of 7,045 real hits reproduce under
            floor/ceil against 48% under nearest-tick). A hit names its band only if exactly one static band reproduces the
            extreme price; a tick-collapsed price that fits several bands is ambiguous and is not used. A hit expires after
            `hit_ttl` sessions and is dropped as soon as a later excursion exceeds the band it named (the band was widened).
  envelope  a stock never moves beyond its band (limits are floored/ceiled INSIDE it), so the tightest static band containing the
            trailing maximum excursion is a LOWER BOUND on the true band. It understates the band of a calm stock.

An excursion beyond the widest static band (20%) gives no estimate: it is a "No Band" stock, a 40% band, or a corporate-action
day the exchange did not adjust for. How accurate these are is MEASURED against 2020+ truth by scripts/measure_band_inference.py
before anything relies on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = (2.0, 5.0, 10.0, 20.0)
TICK = 0.05
TOLERANCE = 1e-6                  # rounding slack on prices quoted to 2 decimals


def upper_limit(prev_close, band):
    return np.floor(np.asarray(prev_close, float) * (1 + band / 100.0) / TICK + 1e-9) * TICK


def lower_limit(prev_close, band):
    return np.ceil(np.asarray(prev_close, float) * (1 - band / 100.0) / TICK - 1e-9) * TICK


def excursion_pct(prev_close, high, low):
    """The largest move away from the previous close during the day, in percent; NaN for a missing or non-positive price."""
    prev, hi, lo = (np.asarray(x, float) for x in (prev_close, high, low))
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.maximum(np.abs(hi - prev), np.abs(lo - prev)) / prev * 100.0
    return np.where((prev > 0) & (hi > 0) & (lo > 0), move, np.nan) if move.ndim else (float(move) if prev > 0 and hi > 0 and lo > 0 else np.nan)


def implied_band(prev_close: float, extreme: float, hit: str) -> float:
    """The unique static band whose limit price equals `extreme` (the day's high for 'H', low for 'L'); NaN if none or several."""
    if not (prev_close > 0 and extreme > 0):
        return float("nan")
    limit = upper_limit if hit == "H" else lower_limit
    matches = [b for b in BANDS if abs(float(limit(prev_close, b)) - extreme) < 0.006]
    return matches[0] if len(matches) == 1 else float("nan")


def _tightest_band(excursion: np.ndarray) -> np.ndarray:
    out = np.full(len(excursion), np.nan)
    for band in reversed(BANDS):                                   # from the widest down, so the tightest fitting band wins
        out = np.where(excursion <= band + TOLERANCE, band, out)
    return np.where(np.isnan(excursion), np.nan, out)


def infer_bands(rows: pd.DataFrame, hits: pd.DataFrame, window: int = 250, min_history: int = 60, hit_ttl: int = 250) -> pd.DataFrame:
    """
    Parameters
    ----------
    rows : one symbol's ascending daily rows {date, prev_close, high, low} (the exchange's own previous close, unadjusted).
    hits : {date, HIT in {'H', 'L'}} for that symbol.

    Returns {date, BAND_PCT, METHOD} on rows' index: METHOD is 'hit', 'envelope' or None (NaN estimate). Row i uses rows[:i] only.
    """
    n = len(rows)
    dates = pd.to_datetime(rows["date"]).to_numpy()
    prev, high, low = (rows[c].to_numpy(float) for c in ("prev_close", "high", "low"))
    exc = excursion_pct(prev, high, low)

    valid = pd.Series(~np.isnan(exc))
    history = valid.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy()
    trailing = pd.Series(exc).shift(1).rolling(window, min_periods=1).max().to_numpy()
    envelope = np.where(history >= min_history, _tightest_band(trailing), np.nan)

    hit_band = np.full(n, np.nan)
    if len(hits):
        by_date = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(rows["date"]))}
        named: dict[int, set] = {}
        for rec in hits.itertuples():
            i = by_date.get(pd.Timestamp(rec.date))
            if i is None:
                continue
            extreme = high[i] if rec.HIT == "H" else low[i]
            named.setdefault(i, set()).add(implied_band(prev[i], extreme, rec.HIT))
        for i, bands in named.items():
            hit_band[i] = next(iter(bands)) if len(bands) == 1 and not np.isnan(next(iter(bands))) else np.nan

    band, method = np.full(n, np.nan), np.full(n, None, dtype=object)
    current, current_at, since = np.nan, -1, np.nan
    for i in range(n):
        if i > 0:
            if not np.isnan(hit_band[i - 1]):
                current, current_at, since = hit_band[i - 1], i - 1, exc[i - 1]
            elif current_at >= 0 and not np.isnan(exc[i - 1]):
                since = exc[i - 1] if np.isnan(since) else max(since, exc[i - 1])
        if current_at >= 0 and (i - current_at > hit_ttl or (not np.isnan(since) and since > current + TOLERANCE)):
            current, current_at = np.nan, -1                       # expired, or a later excursion shows the band was widened
        if current_at >= 0:
            band[i], method[i] = current, "hit"
        elif not np.isnan(envelope[i]):
            band[i], method[i] = envelope[i], "envelope"
    return pd.DataFrame({"date": rows["date"].to_numpy(), "BAND_PCT": band, "METHOD": method}, index=rows.index)
