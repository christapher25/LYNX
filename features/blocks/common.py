"""
features/blocks/common.py

Small helpers shared by the market-level Stage 10 blocks (flows, breadth, and the symbol blocks' calendars).
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd


def validated_sessions(sessions: Sequence[date]) -> pd.DatetimeIndex:
    """Trading-session dates as a normalised DatetimeIndex; sorted ascending with no duplicates or it raises."""
    index = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions]).normalize()
    if any(b <= a for a, b in zip(index[:-1], index[1:])):
        raise ValueError("sessions must be sorted ascending with no duplicates")
    return index
