"""
data/publication.py

The point-in-time primitive for every Stage 10 India-block feature: given WHEN a value became public, which nightly
feature run is the first allowed to use it?

Why this exists. A feature keyed on the date a figure DESCRIBES leaks the future: NSDL publishes an FII figure for day t
on t+1, so a row for day t that carries it was built from information nobody had at the time. A pledge disclosed a week
after it was created must be invisible for that week. Every block therefore stores, per value, the timestamp it became
public, and asks this module for the first session whose nightly run happens at or after that timestamp.

The nightly feature run is 19:00 IST (ARCHITECTURE.md nightly timeline: 18:00 bhavcopy lands, 19:00 feature blocks
-> point-in-time store). A value published at or before 19:00 IST on session s is usable by s's run; anything later
waits for the next session, so weekend and holiday publications wait for the next trading day.

Timestamps must be timezone-aware. A naive timestamp is rejected, not assumed to be IST or UTC: the NSE archive's
Last-Modified header is GMT and a silent 5.5-hour error would move values across the cutoff.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
FEATURE_CUTOFF_IST = time(19, 0)


def to_ist(ts: datetime) -> datetime:
    """Convert an aware timestamp to IST; raise on a naive one."""
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {ts!r} has no timezone; refusing to assume one")
    return ts.astimezone(IST)


def _check_sessions(sessions: Sequence[date]) -> list[date]:
    days = [pd.Timestamp(s).date() for s in sessions]
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("sessions must be sorted ascending with no duplicates")
    return days


def _cutoffs_utc(days: list[date], cutoff: time) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([datetime.combine(d, cutoff, tzinfo=IST) for d in days]).tz_convert("UTC")


def usable_from(published_at: datetime, sessions: Sequence[date], cutoff: time = FEATURE_CUTOFF_IST) -> date | None:
    """
    First session date whose nightly run (at `cutoff` IST on that date) is at or after `published_at`, or None if the
    value was published after the last known session's run (it is not usable within the known calendar).
    """
    published = to_ist(published_at)
    days = _check_sessions(sessions)
    if not days:
        return None
    position = _cutoffs_utc(days, cutoff).searchsorted(pd.Timestamp(published).tz_convert("UTC"), side="left")
    return days[position] if position < len(days) else None


def usable_from_series(published_at: pd.Series, sessions: Sequence[date], cutoff: time = FEATURE_CUTOFF_IST) -> pd.Series:
    """
    Vectorised `usable_from`. Returns a datetime64[ns] Series of session dates (midnight, naive) on the input's index.
    A missing publication time, or one after the last known session, gives NaT: never a guess, never a default date.
    """
    stamps = pd.to_datetime(published_at)
    if stamps.notna().any() and stamps.dt.tz is None:
        raise ValueError("publication timestamps have no timezone; refusing to assume one")
    out = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    days = _check_sessions(sessions)
    valid = stamps.notna().to_numpy()
    if not days or not valid.any():
        return out
    utc = stamps[valid].dt.tz_convert("UTC")
    position = _cutoffs_utc(days, cutoff).searchsorted(pd.DatetimeIndex(utc), side="left")
    inside = position < len(days)
    session_index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    positions = out.index[valid][inside]
    out.loc[positions] = session_index[position[inside]].to_numpy()
    return out
