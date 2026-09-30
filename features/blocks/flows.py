"""
features/blocks/flows.py

Stage 10 — India block, flows sub-block. Two separate blocks, each with its own shared missing flag:

  FII  fpi_equity_net_cr, fpi_equity_net_5d_cr, fpi_equity_net_20d_cr, is_missing_fii_flows
       Source: NSDL daily FPI equity net investment (data/ingest_nsdl_fpi.py), 1999-01-01 -> present.
  DII  dii_net_cr, is_missing_dii_flows
       Source: NSE FII/DII endpoint, forward capture only (data/ingest_fii_dii.py); NSE serves no history.

They are separate because their availability differs by 27 years. One shared flag would mark the deep FII series missing
everywhere the DII series has not started, which is nearly everywhere.

Point-in-time. Every value is keyed on when it became PUBLIC, never on the day it describes:

  * FII: NSDL states its confirmed figures come from custodian reports covering trades "on and up to the previous
    trading day(s)", so the figure for reporting day t is public no earlier than the next session. It is placed in the
    row for the first session after t. FPI_PUBLICATION_TIME_IST is the ASSUMED time of day it appears on that session;
    the assumption is recorded, not hidden, and any time before the 19:00 IST nightly run gives the same placement
    (observed once: the 23-Sep-2026 figure was on NSDL's page at 15:31 IST on 24-Sep; data/publication_monitor.py
    records more each evening). Change the constant and the placement follows via data/publication.py.
  * DII: the row is placed at the first session whose nightly run is at or after the capture timestamp stored with the
    value. A value with no capture timestamp is never used: there is no evidence of when it was public.

NaN is the missing marker, never 0 (a real zero flow stays 0). The flag is True exactly where no figure landed on that
session (a stale figure is never carried forward to fill it). The 5- and 20-day sums are over the last 5 / 20 REPORTED
days, so a day NSDL did not report is skipped, not counted as a zero; they are NaN only during warm-up.
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FII_DII_DIR, FLOW_FEATURES_DIR, NSDL_FPI_DIR
from data.trading_calendar import trading_sessions
from data.publication import IST, usable_from_series
from features.blocks.common import validated_sessions as _validated_sessions

FPI_PUBLICATION_TIME_IST = time(10, 0)      # ASSUMED time on the next session that NSDL's figure is public (see docstring)
FII_COLUMNS = ["fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"]
DII_COLUMNS = ["dii_net_cr"]
_OFFSET = re.compile(r"(?:[+-]\d{2}:\d{2}|Z)$")


def _place(usable: pd.Series, source_dates: pd.DatetimeIndex, values: dict[str, np.ndarray], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Put each value on the session it becomes usable; if several land on one session the latest source date wins."""
    frame = pd.DataFrame({"usable": usable.to_numpy(), "source_date": source_dates, **values}).dropna(subset=["usable"])
    frame = frame.sort_values("source_date").drop_duplicates("usable", keep="last")
    return pd.DataFrame({"date": sessions}).merge(frame, how="left", left_on="date", right_on="usable").drop(columns="usable")


def compute_fii_flow_features(daily: pd.DataFrame, sessions: Sequence[date],
                              publication_time: time = FPI_PUBLICATION_TIME_IST) -> pd.DataFrame:
    """
    Parameters
    ----------
    daily    : {date, net_cr}: NSDL FPI equity net investment (Rs crore) by REPORTING date; NaN/absent = no figure.
    sessions : ascending trading-session dates; the output has exactly one row per session.

    Returns {date, FII_COLUMNS, source_date, is_missing_fii_flows}. Row `date` carries the latest figure that was public
    at that session's nightly run; `source_date` (always earlier than `date`) is the reporting day it describes.
    Causal: a row depends only on reporting days whose figures were public by then.
    """
    if not {"date", "net_cr"} <= set(daily.columns):
        raise ValueError("daily needs columns {date, net_cr}")
    sess = _validated_sessions(sessions)
    dates = pd.to_datetime(daily["date"]).dt.normalize()
    if dates.duplicated().any():
        raise ValueError("duplicate reporting dates in daily")
    net = pd.Series(daily["net_cr"].to_numpy(float), index=pd.DatetimeIndex(dates)).sort_index()
    observed = net.dropna()
    if observed.empty:
        out = pd.DataFrame({"date": sess, **{c: np.nan for c in FII_COLUMNS}, "source_date": pd.NaT})
        out["is_missing_fii_flows"] = True
        return out[["date", *FII_COLUMNS, "source_date", "is_missing_fii_flows"]]

    # Windows span the last k REPORTED days. A session with no NSDL figure (custodians closed: a weekend Budget session,
    # Muhurat trading, a bank holiday) has no flow to add, so it is skipped rather than held open as a NaN slot; leaving
    # it in blanked the next 20 rows of the 20-day sum (47% of them since 2010). Only observed values are ever summed:
    # nothing is filled, and a window is NaN only until k reported days exist. A reporting day the session calendar lacks
    # (a weekend session) is a reported day and counts.
    sum5 = observed.rolling(5, min_periods=5).sum()
    sum20 = observed.rolling(20, min_periods=20).sum()

    # Each figure becomes public on the first session strictly after its reporting day, at `publication_time`.
    position = sess.searchsorted(observed.index, side="right")
    published = pd.Series(
        [datetime.combine(sess[p].date(), publication_time, tzinfo=IST) if p < len(sess) else None for p in position],
        index=observed.index, dtype=object)
    usable = usable_from_series(published, [d.date() for d in sess])

    placed = _place(usable, observed.index, {
        "fpi_equity_net_cr": observed.to_numpy(), "fpi_equity_net_5d_cr": sum5.to_numpy(),
        "fpi_equity_net_20d_cr": sum20.to_numpy()}, sess)
    placed["is_missing_fii_flows"] = placed["fpi_equity_net_cr"].isna()
    return placed[["date", *FII_COLUMNS, "source_date", "is_missing_fii_flows"]]


def compute_dii_flow_features(captures: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    """
    Parameters
    ----------
    captures : {DATE, NET_VALUE_CR, CAPTURED_AT_IST}: DII rows of NSE's forward capture. CAPTURED_AT_IST is an ISO
               timestamp WITH offset; a null one means the value cannot be placed on any date and is not used.
    sessions : ascending trading-session dates.
    """
    if not {"DATE", "NET_VALUE_CR", "CAPTURED_AT_IST"} <= set(captures.columns):
        raise ValueError("captures needs columns {DATE, NET_VALUE_CR, CAPTURED_AT_IST}")
    sess = _validated_sessions(sessions)
    stamps = captures["CAPTURED_AT_IST"]
    naive = stamps.dropna().astype(str).map(lambda s: not _OFFSET.search(s.strip()))
    if naive.any():
        raise ValueError("a capture timestamp has no timezone; refusing to assume one")
    published = pd.to_datetime(stamps, utc=True)
    work = pd.DataFrame({"date": pd.to_datetime(captures["DATE"]).dt.normalize(), "net": captures["NET_VALUE_CR"].to_numpy(float),
                         "published": published}).dropna(subset=["published"]).sort_values("published")
    work = work.drop_duplicates("date", keep="first")           # the EARLIEST capture of a day is when it became public
    if work.empty:
        out = pd.DataFrame({"date": sess, "dii_net_cr": np.nan, "source_date": pd.NaT})
        out["is_missing_dii_flows"] = True
        return out[["date", *DII_COLUMNS, "source_date", "is_missing_dii_flows"]]
    usable = usable_from_series(pd.Series(work["published"].to_numpy(), index=work["date"].to_numpy()), [d.date() for d in sess])
    placed = _place(usable, pd.DatetimeIndex(work["date"]), {"dii_net_cr": work["net"].to_numpy()}, sess)
    placed["is_missing_dii_flows"] = placed["dii_net_cr"].isna()
    return placed[["date", *DII_COLUMNS, "source_date", "is_missing_dii_flows"]]


# ─── loaders and driver ──────────────────────────────────────────────────────────────────────────────────────────

def load_fpi_daily(directory: Path = NSDL_FPI_DIR) -> pd.DataFrame:
    """Equity TOTAL net investment per reporting day from the NSDL monthly files: {date, net_cr}."""
    frames = [pd.read_parquet(p) for p in sorted(Path(directory).glob("*.parquet"))]
    if not frames:
        return pd.DataFrame({"date": pd.to_datetime([]), "net_cr": np.array([], dtype=float)})
    raw = pd.concat(frames, ignore_index=True)
    equity = raw[(raw["ASSET"] == "Equity") & (raw["ROUTE"].isna() | (raw["ROUTE"] == "Sub-total"))]
    return equity.rename(columns={"DATE": "date", "NET_CR": "net_cr"})[["date", "net_cr"]].sort_values("date").reset_index(drop=True)


def load_dii_captures(directory: Path = FII_DII_DIR) -> pd.DataFrame:
    """
    DII rows of the forward capture with their capture timestamps. Captures written before the timestamp column existed
    (2026-09-21) fall back to the file's modification time, which is when the value was demonstrably in our hands: a
    valid, conservative upper bound on when it was public.
    """
    frames = []
    for path in sorted(Path(directory).glob("*.parquet")):
        df = pd.read_parquet(path)
        if "CAPTURED_AT_IST" not in df.columns:
            df = df.assign(CAPTURED_AT_IST=datetime.fromtimestamp(path.stat().st_mtime, IST).isoformat())
        frames.append(df[df["CATEGORY"] == "DII"][["DATE", "NET_VALUE_CR", "CAPTURED_AT_IST"]])
    if not frames:
        return pd.DataFrame({"DATE": pd.to_datetime([]), "NET_VALUE_CR": np.array([], dtype=float), "CAPTURED_AT_IST": []})
    return pd.concat(frames, ignore_index=True)


def run_flows_pipeline() -> dict:
    """Build both flow blocks over the trading-session calendar and write them (market-level: one file per block)."""
    from data.ingest_common import write_parquet_atomic

    sessions = trading_sessions()
    fii = compute_fii_flow_features(load_fpi_daily(), sessions)
    dii = compute_dii_flow_features(load_dii_captures(), sessions)
    write_parquet_atomic(fii, FLOW_FEATURES_DIR / "fii_flows.parquet")
    write_parquet_atomic(dii, FLOW_FEATURES_DIR / "dii_flows.parquet")
    return {"sessions": len(sessions), "fii_rows_with_data": int((~fii["is_missing_fii_flows"]).sum()),
            "dii_rows_with_data": int((~dii["is_missing_dii_flows"]).sum())}


if __name__ == "__main__":
    print(run_flows_pipeline())
