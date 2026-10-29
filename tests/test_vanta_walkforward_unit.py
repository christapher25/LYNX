"""Extra unit tests for vanta.walkforward.purge_training_rows (N11)."""

import pandas as pd
import pytest

from vanta.walkforward import purge_training_rows

D = pd.Timestamp
SESSIONS = pd.bdate_range("2019-05-01", "2020-03-31").drop(D("2019-12-25"))
TEST_START = D("2020-01-02")
CUTOFF = D("2019-12-26")                      # 5 sessions before TEST_START across the 12-25 holiday


def df(*ends):
    return pd.DataFrame({"decision_date": [e - pd.Timedelta(days=10) for e in ends], "label_end_date": list(ends)})


def test_row_ending_exactly_on_cutoff_is_dropped_and_day_before_kept():
    kept = purge_training_rows(df(CUTOFF, D("2019-12-24")), TEST_START, 5, SESSIONS)
    assert list(kept["label_end_date"]) == [D("2019-12-24")]


def test_calendar_day_arithmetic_would_keep_a_row_the_session_embargo_drops():
    row = df(D("2019-12-27"))                                  # calendar-day cutoff (12-28) would keep it
    assert purge_training_rows(row, TEST_START, 5, SESSIONS).empty
    wrong = row[row["label_end_date"] < TEST_START - pd.Timedelta(days=5)]
    assert len(wrong) == 1                                     # the mutant keeps it, so the real function differs from it


def test_input_not_mutated_and_order_columns_preserved():
    src = df(D("2019-06-03"), D("2019-12-30"), D("2019-07-01"))
    before = src.copy()
    out = purge_training_rows(src, TEST_START, 5, SESSIONS)
    pd.testing.assert_frame_equal(src, before)
    assert list(out.columns) == list(src.columns) and list(out.index) == [0, 2]


def test_zero_embargo_cutoff_is_test_start():
    out = purge_training_rows(df(D("2019-12-31"), TEST_START), TEST_START, 0, SESSIONS)
    assert list(out["label_end_date"]) == [D("2019-12-31")]


def test_bad_inputs_raise_rather_than_shorten_the_embargo():
    with pytest.raises(ValueError):
        purge_training_rows(df(D("2019-06-03")), D("2019-05-03"), 20, SESSIONS)      # not enough history
    with pytest.raises(ValueError):
        purge_training_rows(df(D("2019-06-03")), D("2019-12-25"), 5, SESSIONS)       # test_start not a session
    with pytest.raises(ValueError):
        purge_training_rows(df(D("2019-06-03")), TEST_START, -1, SESSIONS)
