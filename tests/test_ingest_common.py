"""
tests/test_ingest_common.py

Tests for the shared candidate-day generator in data/ingest_common.py.

Stage 3 requested weekdays only. That silently skips every weekend trading session: Budget-day Saturdays
(2015-02-28, 2020-02-01, 2025-02-01), special Saturday sessions (2024-01-20, 2024-03-02, 2024-05-18) and
Sunday/Saturday Muhurat sessions (2009-10-17, 2016-10-30, 2019-10-27, 2023-11-12). Probing showed the F&O
archive holds all nine, and a missing session breaks open-interest continuity: the next file's CHG_IN_OI is
measured against the weekend session, so it cannot be reconciled with the last file we have.
"""

from datetime import date, timedelta

import pytest

from data.ingest_common import trading_day_candidates

MON, SUN = date(2025, 1, 27), date(2025, 2, 9)          # exactly two calendar weeks, Monday to Sunday


def test_default_is_weekdays_only_exactly_as_before():
    days = list(trading_day_candidates(MON, SUN))
    assert len(days) == 10 and all(d.weekday() < 5 for d in days)


def test_weekends_mode_yields_only_saturdays_and_sundays():
    days = list(trading_day_candidates(MON, SUN, days="weekends"))
    assert days == [date(2025, 2, 1), date(2025, 2, 2), date(2025, 2, 8), date(2025, 2, 9)]


def test_all_mode_is_every_calendar_day():
    days = list(trading_day_candidates(MON, SUN, days="all"))
    assert days == [MON + timedelta(days=i) for i in range(14)]


def test_weekdays_and_weekends_partition_the_range_with_no_overlap_and_no_gap():
    weekdays = set(trading_day_candidates(MON, SUN, days="weekdays"))
    weekends = set(trading_day_candidates(MON, SUN, days="weekends"))
    assert not weekdays & weekends
    assert weekdays | weekends == set(trading_day_candidates(MON, SUN, days="all"))


def test_the_budget_saturday_is_a_candidate_in_weekends_mode():
    assert date(2025, 2, 1) in set(trading_day_candidates(date(2025, 1, 1), date(2025, 2, 28), days="weekends"))
    assert date(2025, 2, 1) not in set(trading_day_candidates(date(2025, 1, 1), date(2025, 2, 28)))


def test_days_come_out_in_ascending_order():
    for mode in ("weekdays", "weekends", "all"):
        days = list(trading_day_candidates(date(2024, 1, 1), date(2024, 3, 31), days=mode))
        assert days == sorted(days)


def test_an_empty_range_is_empty_and_a_single_day_range_works():
    assert list(trading_day_candidates(SUN, MON)) == []
    assert list(trading_day_candidates(date(2025, 2, 1), date(2025, 2, 1), days="weekends")) == [date(2025, 2, 1)]


def test_an_unknown_mode_is_rejected_rather_than_treated_as_weekdays():
    with pytest.raises(ValueError, match="days"):
        list(trading_day_candidates(MON, SUN, days="weekend"))          # typo for "weekends"
