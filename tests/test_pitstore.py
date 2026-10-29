"""Unit tests for pitstore.store: availability rule, window, argument validation."""

import numpy as np
import pandas as pd
import pytest

from pitstore.store import BlockSpec, compute_availability, read_as_of, select_as_of

SESSIONS = list(pd.bdate_range("2024-03-04", periods=8).date)
SPEC = BlockSpec("m", "market", ("x",), ("is_missing_m",), "source_date")


def test_availability_is_next_session_after_source_and_never_earlier_than_row_date():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-03-06", "2024-03-06", "2024-03-06"]),
                          "source_date": pd.to_datetime(["2024-03-05", "2024-03-06", None])})
    got = compute_availability(SPEC, frame, SESSIONS).tolist()
    assert got[0] == pd.Timestamp("2024-03-06")      # normal placement: source the day before
    assert got[1] == pd.Timestamp("2024-03-07")      # misplaced on its own source day: pushed to the next session
    assert got[2] == pd.Timestamp("2024-03-06")      # no source stamp keeps the row date


def test_source_with_no_later_session_is_unknowable():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-03-13"]), "source_date": pd.to_datetime(["2024-03-13"])})
    assert compute_availability(SPEC, frame, SESSIONS).isna().all()


def test_unknown_block_and_bad_lookback_raise():
    with pytest.raises(KeyError):
        read_as_of(["AAA"], "2024-03-08", ["nope"], SESSIONS, before_nightly_run=False)
    with pytest.raises(ValueError):
        select_as_of({}, {}, ["AAA"], pd.Timestamp("2024-03-08"), SESSIONS, lookback_sessions=0, before_nightly_run=False)


def test_as_of_before_calendar_raises():
    with pytest.raises(ValueError):
        select_as_of({}, {}, ["AAA"], pd.Timestamp("2024-01-01"), SESSIONS, before_nightly_run=False)


def test_no_blocks_returns_just_the_symbol_session_spine():
    out = select_as_of({}, {}, ["AAA", "BBB"], pd.Timestamp("2024-03-08"), SESSIONS, lookback_sessions=2, before_nightly_run=False)
    assert len(out) == 4 and out["date"].max() == pd.Timestamp("2024-03-08")
