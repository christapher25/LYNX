"""
Walk-forward helpers (Stage 13). Currently only label purging (N11); fold construction and fold-only fitting are added in a later row.

purge_training_rows: drop every training row whose label_end_date >= the cutoff, where the cutoff is the session `embargo`
positions before test_start on the trading calendar (VANTA_SPEC.md line 77: label_end_date >= test_start - embargo is dropped,
so a row ending exactly on the cutoff session is dropped). The embargo is counted in trading SESSIONS, never calendar days.
"""

from __future__ import annotations

import pandas as pd


def purge_training_rows(labels_df: pd.DataFrame, test_start, embargo: int, sessions) -> pd.DataFrame:
    """Return the rows of labels_df with label_end_date strictly before the session `embargo` sessions before test_start.

    labels_df is not mutated; kept rows keep their order and columns. `sessions` must be a sorted calendar containing
    test_start and at least `embargo` earlier sessions, otherwise ValueError (a silent shorter embargo would leak)."""
    if isinstance(embargo, bool) or int(embargo) != embargo or embargo < 0:
        raise ValueError(f"embargo must be a non-negative integer number of sessions, got {embargo!r}")
    sess = pd.DatetimeIndex(sessions)
    if not sess.is_monotonic_increasing:
        raise ValueError("sessions must be sorted ascending")
    ts = pd.Timestamp(test_start)
    if ts not in sess:
        raise ValueError(f"test_start {ts.date()} is not in the session calendar")
    pos = int(sess.get_loc(ts)) - int(embargo)
    if pos < 0:
        raise ValueError(f"calendar has fewer than {embargo} sessions before test_start {ts.date()}")
    cutoff = sess[pos]
    return labels_df[pd.to_datetime(labels_df["label_end_date"]) < cutoff]
