"""
tests/invariants/test_publication_clock.py

Invariant: a value is never usable by a nightly run that started before the value was public.

Every Stage 10 India-block feature (flows, pledging, circuits, breadth) is keyed on WHEN IT BECAME PUBLIC, not on
the date it describes: an FII figure for day t that NSDL publishes on t+1 must not appear in day t's feature row; a
pledge disclosed a week after it was created is invisible for that week. data/publication.py is the one place that
turns a publication timestamp into "the first nightly run that may use it", so the rule is tested once, here.

The nightly feature run is at 19:00 IST (ARCHITECTURE.md timeline: 18:00 bhavcopy lands, 19:00 feature blocks).
"""

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.publication import FEATURE_CUTOFF_IST, IST, to_ist, usable_from, usable_from_series

# Mon 2026-09-21 .. Fri 2026-09-25, then a weekend, then Mon 2026-09-28. 2026-09-23 is treated as a holiday (absent).
SESSIONS = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)]


def ist(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST)


class TestUsableFrom:
    def test_published_before_the_cutoff_is_usable_the_same_session(self):
        assert usable_from(ist(2026, 9, 22, 17, 50), SESSIONS) == date(2026, 9, 22)

    def test_published_after_the_cutoff_waits_for_the_next_session(self):
        assert usable_from(ist(2026, 9, 22, 19, 7), SESSIONS) == date(2026, 9, 24)   # 23rd is a holiday: skipped

    def test_exactly_at_the_cutoff_is_usable_that_night(self):
        assert usable_from(ist(2026, 9, 22, 19, 0, 0), SESSIONS) == date(2026, 9, 22)
        assert usable_from(ist(2026, 9, 22, 19, 0, 1), SESSIONS) == date(2026, 9, 24)

    def test_the_cutoff_is_nineteen_hundred_ist(self):
        assert FEATURE_CUTOFF_IST == time(19, 0)

    def test_weekend_publication_waits_for_monday(self):
        assert usable_from(ist(2026, 9, 26, 11, 0), SESSIONS) == date(2026, 9, 28)   # Saturday morning
        assert usable_from(ist(2026, 9, 27, 23, 59), SESSIONS) == date(2026, 9, 28)  # Sunday night

    def test_holiday_publication_waits_for_the_next_real_session(self):
        assert usable_from(ist(2026, 9, 23, 9, 0), SESSIONS) == date(2026, 9, 24)

    def test_published_long_before_the_first_session_is_usable_from_the_first_session(self):
        assert usable_from(ist(2020, 1, 1), SESSIONS) == date(2026, 9, 21)

    def test_published_after_the_last_known_session_is_not_usable_at_all(self):
        assert usable_from(ist(2026, 9, 28, 19, 30), SESSIONS) is None
        assert usable_from(ist(2027, 1, 1), SESSIONS) is None

    def test_a_value_is_never_usable_before_it_was_published(self):
        """The invariant itself, checked over every minute of a day."""
        for minutes in range(0, 24 * 60, 7):
            published = ist(2026, 9, 22) + timedelta(minutes=minutes)
            session = usable_from(published, SESSIONS)
            assert session is not None
            run_time = datetime.combine(session, FEATURE_CUTOFF_IST, tzinfo=IST)
            assert run_time >= published, (published, session)

    def test_it_is_also_the_earliest_such_session(self):
        for minutes in range(0, 24 * 60, 7):
            published = ist(2026, 9, 22) + timedelta(minutes=minutes)
            session = usable_from(published, SESSIONS)
            earlier = [s for s in SESSIONS if s < session]
            assert all(datetime.combine(s, FEATURE_CUTOFF_IST, tzinfo=IST) < published for s in earlier)

    def test_a_naive_timestamp_is_rejected_rather_than_guessed_at(self):
        with pytest.raises(ValueError, match="timezone"):
            usable_from(datetime(2026, 9, 22, 17, 50), SESSIONS)

    def test_utc_timestamps_are_converted_to_ist_before_comparing(self):
        # 12:20 GMT is 17:50 IST (an NSE archive Last-Modified) -> same session; 13:37 GMT is 19:07 IST -> next
        assert usable_from(datetime(2026, 9, 22, 12, 20, tzinfo=timezone.utc), SESSIONS) == date(2026, 9, 22)
        assert usable_from(datetime(2026, 9, 22, 13, 37, tzinfo=timezone.utc), SESSIONS) == date(2026, 9, 24)

    def test_unsorted_or_duplicate_sessions_are_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            usable_from(ist(2026, 9, 22, 12), [date(2026, 9, 24), date(2026, 9, 22)])
        with pytest.raises(ValueError, match="sorted"):
            usable_from(ist(2026, 9, 22, 12), [date(2026, 9, 22), date(2026, 9, 22)])

    def test_no_sessions_means_nothing_is_usable(self):
        assert usable_from(ist(2026, 9, 22, 12), []) is None


class TestUsableFromSeries:
    def test_matches_the_scalar_rule_row_by_row(self):
        stamps = [ist(2026, 9, 22, 17, 50), ist(2026, 9, 22, 19, 7), ist(2026, 9, 26, 11), ist(2026, 9, 28, 19, 30),
                  ist(2020, 1, 1)]
        out = usable_from_series(pd.Series(stamps), SESSIONS)
        expected = [usable_from(s, SESSIONS) for s in stamps]
        assert [None if pd.isna(v) else v.date() for v in out] == expected

    def test_missing_publication_time_gives_not_a_time_never_a_guess(self):
        out = usable_from_series(pd.Series([ist(2026, 9, 22, 12), pd.NaT, None]), SESSIONS)
        assert out.iloc[0] == pd.Timestamp("2026-09-22")
        assert pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])

    def test_empty_series(self):
        assert len(usable_from_series(pd.Series([], dtype="datetime64[ns, Asia/Kolkata]"), SESSIONS)) == 0

    def test_a_naive_series_is_rejected(self):
        with pytest.raises(ValueError, match="timezone"):
            usable_from_series(pd.Series([datetime(2026, 9, 22, 12)]), SESSIONS)


class TestToIst:
    def test_aware_timestamps_convert(self):
        assert to_ist(datetime(2026, 9, 22, 12, 20, tzinfo=timezone.utc)) == ist(2026, 9, 22, 17, 50)

    def test_naive_timestamps_raise(self):
        with pytest.raises(ValueError, match="timezone"):
            to_ist(datetime(2026, 9, 22, 12, 20))
