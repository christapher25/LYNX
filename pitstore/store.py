"""
pitstore/store.py

Stage 12 -- the point-in-time read API over the Stage 10 feature blocks.

What "knowable as of t" means here
----------------------------------
Every Stage 10 block already places a value on the first session whose 19:00 IST nightly run (data/publication.py) can use it,
so a row dated s is built by, and first knowable at, session s's nightly run. A read with `as_of_date = t` is a decision made
AFTER t's nightly run (the entry is at t+1 or later): it admits rows whose AVAILABILITY DATE is <= t. With
`before_nightly_run=True` the decision is made during t, before its run: availability < t.

Availability date = the row date, pushed LATER (never earlier) if the row names a source stamp that is not strictly before it:
  flows    `source_date`        the reporting day; NSDL publishes on the next session, so a row placed on its own source day
                                is not knowable until the session after that day.
  pledging `source_period_end`  the quarter the filing describes; a filing cannot be public before its period has ended.
A row whose source stamp is missing keeps its row date. Nothing is ever made available EARLIER than its row date.
Blocks whose publication was measured AFTER the 19:00 cutoff carry `lag_sessions=1` and a `reason` (delivery, F&O open interest:
data/raw/_publication/first_seen.jsonl): a row dated s is knowable from the next session's run, so as_of=s returns that row as
absent and as_of=next session returns it. The F&O file is read as two blocks, `fno_oi` (lag 1) and `fno_ban` (ban list, publication
unmeasured, row-date availability by decision).

Missing stays missing
---------------------
The read returns one row per (requested symbol, session in the window). A row that is absent from a block, or present but not
yet knowable, comes back as NaN values + every missing flag True + `<block>_row_absent` True. It is never carried forward from
an older row and never zero-filled. Rows that exist keep their own NaN + flags exactly as the block wrote them.

`select_as_of` is the pure core (frames in, frame out); `read_as_of` loads Parquet (date-filtered, per symbol) and calls it.
`sessions` is always an explicit argument: the store never infers a calendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config import (BREADTH_FEATURES_DIR, CIRCUIT_FEATURES_DIR, DELIVERY_FEATURES_DIR, FLOW_FEATURES_DIR, FNO_FEATURES_DIR,
                    PLEDGING_FEATURES_DIR)


@dataclass(frozen=True)
class BlockSpec:
    name: str
    scope: str                                   # "symbol" (frame with a symbol column) or "market" (date-only)
    value_columns: tuple[str, ...]               # masked to NaN when a row is absent / not yet knowable
    flag_columns: tuple[str, ...]                # the block's missing flags; forced True on an absent/masked row
    source_column: str | None = None             # a stamp that must be strictly before the availability date
    path: Path | None = None                     # directory (symbol scope) or file (market scope)
    lag_sessions: int = 0                        # row dated s is first knowable at the nightly run of the s+lag-th session
    reason: str = ""                             # why: the evidence for the lag (required whenever lag_sessions > 0)

    def __post_init__(self):
        if self.lag_sessions < 0 or (self.lag_sessions > 0 and not self.reason):
            raise ValueError(f"{self.name}: a lag needs a non-negative session count and a documented reason")


_FIRST_SEEN = ("data/raw/_publication/first_seen.jsonl, target 2026-09-24, the only publication measurement on disk: ")
_DELIVERY_REASON = (_FIRST_SEEN + "nse_sec_bhavdata_full last_modified 19:21 IST, after the 19:00 IST nightly cutoff "
                    "(data/publication.py), so the figure for s is first usable at the next session's run")
_FNO_OI_REASON = (_FIRST_SEEN + "nse_fo_bhavcopy_udiff last_modified 19:08 IST, after the 19:00 IST cutoff, so the F&O "
                  "bhavcopy (open interest, PCR, roll, expiry) for s is first usable at the next session's run")

BLOCK_SPECS: dict[str, BlockSpec] = {
    "delivery": BlockSpec("delivery", "symbol", ("delivery_pct", "delivery_pct_chg_5d", "delivery_pct_chg_20d", "delivery_qty_ratio_20d"),
                          ("is_missing_delivery",), None, DELIVERY_FEATURES_DIR, 1, _DELIVERY_REASON),
    # The F&O file mixes two publications, so it is read as two blocks. The ban list for target D was last modified at
    # 18:07 IST (D=2026-09-24) and 18:17 IST (D=2026-09-25) on the evening BEFORE D (first_seen.jsonl, item nse_fo_ban):
    # before the 19:00 cutoff, so it keeps row-date availability (conservative; Stage 11: a verdict at t applies to t+1).
    "fno_oi": BlockSpec("fno_oi", "symbol", ("fut_oi", "fut_oi_chg_1d", "fut_oi_chg_5d", "fut_oi_ratio_20d", "fut_roll_next_share",
                                             "fut_days_to_expiry", "opt_pcr_oi"),
                        ("is_missing_fno", "fno_day_unavailable", "is_missing_opt"), None, FNO_FEATURES_DIR, 1, _FNO_OI_REASON),
    "fno_ban": BlockSpec("fno_ban", "symbol", ("fno_in_ban", "fno_ban_new_entry", "fno_ban_days_20d"), ("is_missing_ban",),
                         None, FNO_FEATURES_DIR),
    "pledging": BlockSpec("pledging", "symbol", ("pledged_pct_of_promoter", "pledged_pct_chg", "pledge_rising", "pledging_age_days"),
                          ("is_missing_pledging",), "source_period_end", PLEDGING_FEATURES_DIR),
    "circuits": BlockSpec("circuits", "symbol", ("price_band_pct", "is_no_band", "band_is_inferred", "hit_upper_circuit", "hit_lower_circuit"),
                          ("is_missing_circuit_band", "is_missing_circuit_hits"), None, CIRCUIT_FEATURES_DIR),
    "flows_fii": BlockSpec("flows_fii", "market", ("fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"),
                           ("is_missing_fii_flows",), "source_date", FLOW_FEATURES_DIR / "fii_flows.parquet"),
    "flows_dii": BlockSpec("flows_dii", "market", ("dii_net_cr",), ("is_missing_dii_flows",), "source_date",
                           FLOW_FEATURES_DIR / "dii_flows.parquet"),
    "breadth": BlockSpec("breadth", "market", ("universe_pct_above_200dma", "universe_n_evaluable"), ("is_missing_breadth",),
                         None, BREADTH_FEATURES_DIR / "universe_breadth.parquet"),
}

Loader = Callable[[BlockSpec, "str | None", pd.Timestamp], pd.DataFrame]


# ------------------------------------------------------------------------------------------------ pure core

def knowable_cutoff(as_of: pd.Timestamp, before_nightly_run: bool) -> pd.Timestamp:
    """Latest availability date a read may admit: `as_of` (after t's nightly run) or the day before it (during t)."""
    day = pd.Timestamp(as_of).normalize()
    return day - pd.Timedelta(days=1) if before_nightly_run else day


def compute_availability(spec: BlockSpec, frame: pd.DataFrame, sessions: Sequence) -> pd.Series:
    """Availability date per row (NaT = not knowable within the calendar). Never earlier than the row date."""
    row_date = pd.to_datetime(frame["date"]).dt.normalize().astype("datetime64[ns]")
    cal = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions]).normalize()
    if spec.lag_sessions:
        # the lag-th session strictly after the row date; NaT when the calendar does not reach it
        pos = cal.searchsorted(pd.DatetimeIndex(row_date), side="right") + spec.lag_sessions - 1
        reach = pos < len(cal)
        lagged = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        if len(cal):
            lagged[reach] = cal[pos[reach]].to_numpy()
        row_date = lagged
    if spec.source_column is None or spec.source_column not in frame.columns:
        return row_date
    source = pd.to_datetime(frame[spec.source_column])
    position = cal.searchsorted(pd.DatetimeIndex(source.fillna(pd.Timestamp("1900-01-01"))), side="right")
    has_next = position < len(cal)
    next_session = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if len(cal):
        next_session[has_next] = cal[position[has_next]].to_numpy()
    later = pd.concat([row_date, next_session], axis=1).max(axis=1, skipna=False)   # NaT: no later session
    return later.where(source.notna(), row_date)


def _mask_rows(frame: pd.DataFrame, spec: BlockSpec, mask: pd.Series) -> pd.DataFrame:
    """Set value columns NaN (and any source stamp NaT) and every flag True where `mask`. Never a default value."""
    out = frame.copy()
    for column in spec.value_columns:
        out.loc[mask, column] = np.nan
    if spec.source_column and spec.source_column in out.columns:
        out.loc[mask, spec.source_column] = pd.NaT
    for column in spec.flag_columns:
        out[column] = out[column].astype(bool) | mask
    return out


def _window(sessions: Sequence, cutoff: pd.Timestamp, lookback: int) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions]).normalize()
    if any(b <= a for a, b in zip(cal[:-1], cal[1:])):
        raise ValueError("sessions must be sorted ascending with no duplicates")
    eligible = cal[cal <= cutoff]
    if len(eligible) == 0:
        raise ValueError(f"no session on or before {cutoff.date()} in the supplied calendar")
    return eligible[-lookback:]


def select_as_of(frames: Mapping[str, pd.DataFrame], specs: Mapping[str, BlockSpec], symbols: Sequence[str], as_of_date,
                 sessions: Sequence, *, before_nightly_run: bool, lookback_sessions: int = 1) -> pd.DataFrame:
    """
    Pure as-of selection. `frames[name]` holds the block's rows (symbol blocks carry a `symbol` column; market blocks do not).
    Returns one row per (symbol, session) in the last `lookback_sessions` sessions dated <= the knowable cutoff, with every
    block's columns, `<block>_row_absent` and `<block>_available_on`.
    """
    if lookback_sessions < 1:
        raise ValueError("lookback_sessions must be >= 1")
    cutoff = knowable_cutoff(pd.Timestamp(as_of_date), before_nightly_run)
    window = _window(sessions, cutoff, lookback_sessions)
    out = pd.MultiIndex.from_product([list(symbols), window], names=["symbol", "date"]).to_frame(index=False)
    for name, frame in frames.items():
        spec = specs[name]
        keys = ["symbol", "date"] if spec.scope == "symbol" else ["date"]
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize().astype("datetime64[ns]")
        frame = frame[frame["date"] <= cutoff]                                       # nothing dated after t is even looked at
        if spec.scope == "symbol":
            frame = frame[frame["symbol"].isin(list(symbols))]
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate {keys} rows; refusing to pick one")
        avail_col = f"{name}_available_on"
        frame[avail_col] = compute_availability(spec, frame, sessions)
        keep = keys + [c for c in (*spec.value_columns, *spec.flag_columns) if c in frame.columns]
        if spec.source_column and spec.source_column in frame.columns:
            keep.append(spec.source_column)
        merged = out[keys].merge(frame[keep + [avail_col]], on=keys, how="left")
        for column in spec.value_columns:
            if column not in merged.columns:
                merged[column] = np.nan
        for column in spec.flag_columns:
            if column not in merged.columns:
                merged[column] = True
            merged[column] = merged[column].where(merged[column].notna(), True).astype(bool)
        absent = merged[avail_col].isna() | (merged[avail_col] > cutoff)
        masked_future = absent & merged[avail_col].notna()
        if masked_future.any() and not spec.lag_sessions:
            logger.warning(f"{name}: {int(masked_future.sum())} row(s) dated <= {cutoff.date()} are not knowable yet and were masked")
        merged = _mask_rows(merged, spec, absent)
        merged.loc[absent, avail_col] = pd.NaT                                     # a masked row has no availability to report
        merged[f"{name}_row_absent"] = absent.to_numpy()
        if spec.source_column and spec.source_column in merged.columns:        # prefixed: two blocks may share a stamp name
            merged = merged.rename(columns={spec.source_column: f"{name}_{spec.source_column}"})
        out = pd.concat([out.reset_index(drop=True), merged.drop(columns=keys).reset_index(drop=True)], axis=1)
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ I/O

def default_loader(spec: BlockSpec, symbol: str | None, upto: pd.Timestamp) -> pd.DataFrame:
    """Rows dated <= `upto` for one symbol (symbol blocks) or the whole market file. Empty frame if the file is absent."""
    path = spec.path / f"{symbol}.parquet" if spec.scope == "symbol" else spec.path
    if not path.exists():
        return pd.DataFrame(columns=(["symbol"] if spec.scope == "symbol" else []) + ["date"])
    return pd.read_parquet(path, filters=[("date", "<=", pd.Timestamp(upto))])


def truncated_loader(inner: Loader, last_date) -> Loader:
    """Test helper: a loader that deletes everything after `last_date` before the store sees it."""
    last = pd.Timestamp(last_date)

    def load(spec: BlockSpec, symbol: str | None, upto: pd.Timestamp) -> pd.DataFrame:
        frame = inner(spec, symbol, upto)
        return frame[pd.to_datetime(frame["date"]) <= last]
    return load


def read_as_of(symbols: Sequence[str], as_of_date, blocks: Sequence[str], sessions: Sequence, *, before_nightly_run: bool,
               lookback_sessions: int = 1, loader: Loader = default_loader,
               specs: Mapping[str, BlockSpec] = BLOCK_SPECS) -> pd.DataFrame:
    """
    Features knowable as of `as_of_date` for `symbols`, from `blocks` (keys of BLOCK_SPECS). See the module docstring.

    as_of_date          date of the nightly run the decision follows (entry at the next session or later).
    sessions            ascending trading-session dates (explicit; e.g. data.trading_calendar.trading_sessions()).
    lookback_sessions   1 = the as-of session's row only; N = the last N sessions' rows, for rolling use.
    before_nightly_run  REQUIRED. False = the decision follows as_of_date's 19:00 run: rows available on or before it. True = the
                        decision is made during as_of_date, before that run: only rows available before it.
    """
    unknown = [b for b in blocks if b not in specs]
    if unknown:
        raise KeyError(f"unknown block(s) {unknown}; known: {sorted(specs)}")
    cutoff = knowable_cutoff(pd.Timestamp(as_of_date), before_nightly_run)
    frames: dict[str, pd.DataFrame] = {}
    for name in blocks:
        spec = specs[name]
        if spec.scope == "symbol":
            parts = [p for p in (loader(spec, s, cutoff) for s in symbols) if len(p)]
            frames[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"symbol": [], "date": []})
        else:
            frames[name] = loader(spec, None, cutoff)
    return select_as_of(frames, {b: specs[b] for b in blocks}, symbols, as_of_date, sessions,
                        lookback_sessions=lookback_sessions, before_nightly_run=before_nightly_run)
