"""
tests/test_flows_block.py

Unit tests for features/blocks/flows.py — Stage 10 flows. Two blocks, each with its own shared missing flag:

  FII  (NSDL daily FPI equity net investment, 1999 ->): fpi_equity_net_cr, its 5- and 20-session sums, is_missing_fii_flows
  DII  (NSE FII/DII endpoint, forward capture only):    dii_net_cr, is_missing_dii_flows

They are separate blocks because their availability differs by a quarter century: one shared flag would mark the deep
FII series missing everywhere the (barely started) DII series is.

Every value is keyed on when it became public. An FII figure for reporting day t is public no earlier than the next
session (NSDL compiles it from custodian reports), so it appears in the row for the session AFTER t.
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from data.publication import IST
from features.blocks.flows import (
    DII_COLUMNS,
    FII_COLUMNS,
    FPI_PUBLICATION_TIME_IST,
    compute_dii_flow_features,
    compute_fii_flow_features,
)

# Mon 21, Tue 22, (Wed 23 holiday), Thu 24, Fri 25, Mon 28, Tue 29, Wed 30 September 2026
SESSIONS = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28),
            date(2026, 9, 29), date(2026, 9, 30)]


def _daily(pairs):
    return pd.DataFrame({"date": pd.to_datetime([d for d, _ in pairs]), "net_cr": [v for _, v in pairs]})


def _row(out, day):
    return out[out["date"] == pd.Timestamp(day)].iloc[0]


class TestFiiLag:
    def test_a_figure_is_visible_only_from_the_next_session(self):
        out = compute_fii_flow_features(_daily([("2026-09-21", 100.0)]), SESSIONS)
        assert np.isnan(_row(out, "2026-09-21")["fpi_equity_net_cr"])           # same day: not yet public
        assert _row(out, "2026-09-22")["fpi_equity_net_cr"] == 100.0
        assert _row(out, "2026-09-22")["source_date"] == pd.Timestamp("2026-09-21")

    def test_the_next_session_skips_a_holiday(self):
        out = compute_fii_flow_features(_daily([("2026-09-22", 7.0)]), SESSIONS)
        assert _row(out, "2026-09-24")["fpi_equity_net_cr"] == 7.0              # 23rd is not a session

    def test_a_friday_figure_lands_on_monday(self):
        out = compute_fii_flow_features(_daily([("2026-09-25", -55.5)]), SESSIONS)
        assert _row(out, "2026-09-28")["fpi_equity_net_cr"] == -55.5

    def test_the_last_sessions_figure_has_no_row_yet(self):
        out = compute_fii_flow_features(_daily([("2026-09-30", 1.0)]), SESSIONS)
        assert out["fpi_equity_net_cr"].isna().all()

    def test_source_date_is_always_strictly_before_the_row_date(self):
        daily = _daily([(d, float(i)) for i, d in enumerate(SESSIONS)])
        out = compute_fii_flow_features(daily, SESSIONS).dropna(subset=["source_date"])
        assert (out["source_date"] < out["date"]).all()

    def test_a_later_publication_time_pushes_the_figure_a_session_later(self):
        """The publication time is a parameter of the rule, not decoration: publish after 19:00 and the value waits."""
        daily = _daily([("2026-09-21", 100.0)])
        late = compute_fii_flow_features(daily, SESSIONS, publication_time=time(19, 30))
        assert np.isnan(_row(late, "2026-09-22")["fpi_equity_net_cr"])
        assert _row(late, "2026-09-24")["fpi_equity_net_cr"] == 100.0

    def test_the_assumed_publication_time_is_recorded_as_an_assumption_before_the_cutoff(self):
        assert FPI_PUBLICATION_TIME_IST < time(19, 0)


class TestFiiRollingSums:
    def _series(self, n=30):
        sessions = [date(2026, 8, 3) + timedelta(days=i) for i in range(60) if (date(2026, 8, 3) + timedelta(days=i)).weekday() < 5][:n]
        return sessions, _daily([(d, float(i + 1)) for i, d in enumerate(sessions)])

    def test_five_and_twenty_session_sums_are_trailing_and_include_the_reporting_day(self):
        sessions, daily = self._series()
        out = compute_fii_flow_features(daily, sessions)
        row = out.iloc[21]                        # published for reporting day index 20 (value 21)
        assert row["fpi_equity_net_cr"] == 21.0
        assert row["fpi_equity_net_5d_cr"] == sum(range(17, 22))
        assert row["fpi_equity_net_20d_cr"] == sum(range(2, 22))

    def test_warm_up_rows_are_nan_not_zero(self):
        sessions, daily = self._series()
        out = compute_fii_flow_features(daily, sessions)
        assert out["fpi_equity_net_5d_cr"].iloc[:5].isna().all()               # first published row is index 1 (reporting day 0)
        assert out["fpi_equity_net_20d_cr"].iloc[:20].isna().all()
        assert out["fpi_equity_net_20d_cr"].iloc[20:].notna().all()

    def test_windows_span_the_last_k_reported_days_and_skip_a_day_with_no_report(self):
        """
        A session with no NSDL figure (custodians closed: a weekend Budget session, Muhurat trading, a bank holiday) has no
        flow to add. The windows sum the last k days that HAD a report, so one such day does not blank the next 20 rows
        (measured on the real series: it blanked 47% of the 20-day sums since 2010). Only observed values are summed.
        """
        sessions, daily = self._series()
        daily = daily[daily["date"] != pd.Timestamp(sessions[10])]              # reporting day 10 (value 11) has no report
        out = compute_fii_flow_features(daily, sessions)
        gap_row = out.iloc[11]                                                 # would carry reporting day 10
        assert np.isnan(gap_row["fpi_equity_net_cr"]) and gap_row["is_missing_fii_flows"]
        assert np.isnan(gap_row["fpi_equity_net_5d_cr"])                       # no figure landed here: nothing to place
        assert out.iloc[12]["fpi_equity_net_cr"] == 12.0                       # reporting day 11
        assert out.iloc[12]["fpi_equity_net_5d_cr"] == 12 + 10 + 9 + 8 + 7     # last 5 REPORTED days: 11, 9, 8, 7, 6
        assert out.iloc[16]["fpi_equity_net_5d_cr"] == sum(range(12, 17))      # window 11..15 has no gap at all
        assert out.iloc[21]["fpi_equity_net_20d_cr"] == sum(range(1, 22)) - 11  # reporting days 0..20 are 21 days, 20 of them reported

    def test_the_twenty_day_sum_needs_twenty_reported_days_not_twenty_sessions(self):
        sessions, daily = self._series()
        daily = daily[daily["date"] != pd.Timestamp(sessions[10])]
        out = compute_fii_flow_features(daily, sessions)
        first = out["fpi_equity_net_20d_cr"].first_valid_index()
        assert first == 21                                                     # 20th reported day is index 20 (0..9 and 11..20), one later than with no gap
        assert out["fpi_equity_net_20d_cr"].iloc[first:].notna().all()          # and it never blanks again

    def test_a_gap_contributes_nothing_it_is_not_zero_filled(self):
        sessions, daily = self._series()
        with_gap = daily[daily["date"] != pd.Timestamp(sessions[10])]
        as_zero = daily.assign(net_cr=np.where(daily["date"] == pd.Timestamp(sessions[10]), 0.0, daily["net_cr"]))
        a = compute_fii_flow_features(with_gap, sessions)["fpi_equity_net_5d_cr"].iloc[12]
        b = compute_fii_flow_features(as_zero, sessions)["fpi_equity_net_5d_cr"].iloc[12]
        assert a == 12 + 10 + 9 + 8 + 7 and b == 12 + 0 + 10 + 9 + 8           # a zero would occupy a window slot; a gap does not

    def test_an_off_calendar_reporting_day_still_counts_toward_the_windows_and_maps_to_the_next_session(self):
        """A reporting day the session calendar lacks (say a weekend session) is a real day of flow."""
        sessions = [d for d in SESSIONS]
        daily = _daily([("2026-09-26", 10.0), ("2026-09-25", 3.0)])            # Saturday + Friday
        out = compute_fii_flow_features(daily, sessions)
        assert _row(out, "2026-09-28")["fpi_equity_net_cr"] == 10.0            # the later reporting day wins the session
        assert _row(out, "2026-09-28")["source_date"] == pd.Timestamp("2026-09-26")
        assert out["fpi_equity_net_cr"].dropna().tolist() == [10.0]            # the Friday figure is superseded, never shown alongside

    def test_a_superseded_off_calendar_day_still_counts_in_the_sums(self):
        sessions = [date(2026, 9, 21) + timedelta(days=i) for i in range(20) if (date(2026, 9, 21) + timedelta(days=i)).weekday() < 5]
        weekend = date(2026, 9, 26)                                            # a Saturday session the calendar lacks
        pairs = [(d, 1.0) for d in sessions[:5]] + [(weekend, 100.0)]
        out = compute_fii_flow_features(_daily(pairs), sessions)
        landing = out[out["date"] == pd.Timestamp(sessions[5])].iloc[0]         # first session after Friday the 25th and Saturday
        assert landing["source_date"] == pd.Timestamp(weekend) and landing["fpi_equity_net_cr"] == 100.0
        assert landing["fpi_equity_net_5d_cr"] == 100.0 + 4 * 1.0               # Friday's 1.0 is in the sum though not shown as the value


class TestFiiMissingness:
    def test_a_real_zero_stays_zero_and_is_not_missing(self):
        out = compute_fii_flow_features(_daily([("2026-09-21", 0.0)]), SESSIONS)
        row = _row(out, "2026-09-22")
        assert row["fpi_equity_net_cr"] == 0.0 and not row["is_missing_fii_flows"]

    def test_sessions_with_no_figure_are_nan_and_flagged_never_zero(self):
        out = compute_fii_flow_features(_daily([("2026-09-21", 5.0)]), SESSIONS)
        blank = out[out["date"] != pd.Timestamp("2026-09-22")]
        assert blank["fpi_equity_net_cr"].isna().all() and blank["is_missing_fii_flows"].all()
        assert not (out[FII_COLUMNS] == 0).any().any()

    def test_the_flag_is_a_plain_bool_and_features_are_nan_exactly_where_it_is_true(self):
        daily = _daily([(d, 1.0) for d in SESSIONS[:4]])
        out = compute_fii_flow_features(daily, SESSIONS)
        assert out["is_missing_fii_flows"].dtype == bool
        assert out.loc[out["is_missing_fii_flows"], "fpi_equity_net_cr"].isna().all()
        assert out.loc[~out["is_missing_fii_flows"], "fpi_equity_net_cr"].notna().all()

    def test_no_data_at_all_is_all_missing(self):
        out = compute_fii_flow_features(_daily([]), SESSIONS)
        assert out["is_missing_fii_flows"].all() and list(out["date"].dt.date) == SESSIONS


class TestFiiValidation:
    def test_output_has_one_row_per_session_in_order_with_the_declared_columns(self):
        out = compute_fii_flow_features(_daily([("2026-09-21", 1.0)]), SESSIONS)
        assert list(out["date"].dt.date) == SESSIONS
        assert list(out.columns) == ["date", *FII_COLUMNS, "source_date", "is_missing_fii_flows"]

    def test_duplicate_reporting_dates_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            compute_fii_flow_features(_daily([("2026-09-21", 1.0), ("2026-09-21", 2.0)]), SESSIONS)

    def test_missing_columns_are_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            compute_fii_flow_features(pd.DataFrame({"date": []}), SESSIONS)

    def test_unsorted_sessions_are_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            compute_fii_flow_features(_daily([("2026-09-21", 1.0)]), list(reversed(SESSIONS)))


class TestDii:
    def _captures(self, rows):
        return pd.DataFrame({
            "DATE": pd.to_datetime([r[0] for r in rows]), "NET_VALUE_CR": [r[1] for r in rows],
            "CAPTURED_AT_IST": [r[2] for r in rows],
        })

    def test_a_capture_after_the_cutoff_is_usable_only_the_next_session(self):
        cap = self._captures([("2026-09-24", 4301.18, datetime(2026, 9, 24, 19, 34, tzinfo=IST).isoformat())])
        out = compute_dii_flow_features(cap, SESSIONS)
        assert np.isnan(_row(out, "2026-09-24")["dii_net_cr"])                  # 19:34 misses tonight's 19:00 run
        assert _row(out, "2026-09-25")["dii_net_cr"] == 4301.18

    def test_a_capture_the_next_morning_is_usable_that_session(self):
        cap = self._captures([("2026-09-21", 2797.27, datetime(2026, 9, 22, 7, 38, tzinfo=IST).isoformat())])
        assert _row(compute_dii_flow_features(cap, SESSIONS), "2026-09-22")["dii_net_cr"] == 2797.27

    def test_the_earliest_capture_of_a_day_defines_when_it_became_public(self):
        cap = self._captures([("2026-09-21", 1.0, datetime(2026, 9, 21, 20, 0, tzinfo=IST).isoformat()),
                              ("2026-09-21", 1.0, datetime(2026, 9, 21, 18, 0, tzinfo=IST).isoformat())])
        out = compute_dii_flow_features(cap, SESSIONS)
        assert _row(out, "2026-09-21")["dii_net_cr"] == 1.0                      # 18:00 capture: same night

    def test_sessions_without_a_capture_are_nan_and_flagged(self):
        cap = self._captures([("2026-09-21", 1.0, datetime(2026, 9, 21, 18, 0, tzinfo=IST).isoformat())])
        out = compute_dii_flow_features(cap, SESSIONS)
        assert out["is_missing_dii_flows"].sum() == len(SESSIONS) - 1
        assert list(out.columns) == ["date", *DII_COLUMNS, "source_date", "is_missing_dii_flows"]

    def test_a_capture_with_no_timestamp_is_never_used(self):
        """No evidence of when it became public means it cannot be placed on any date."""
        cap = self._captures([("2026-09-21", 1.0, None)])
        assert compute_dii_flow_features(cap, SESSIONS)["is_missing_dii_flows"].all()

    def test_a_naive_capture_timestamp_is_rejected(self):
        cap = self._captures([("2026-09-21", 1.0, "2026-09-21T18:00:00")])
        with pytest.raises(ValueError, match="timezone"):
            compute_dii_flow_features(cap, SESSIONS)

    def test_zero_dii_is_a_value(self):
        cap = self._captures([("2026-09-21", 0.0, datetime(2026, 9, 21, 18, 0, tzinfo=IST).isoformat())])
        row = _row(compute_dii_flow_features(cap, SESSIONS), "2026-09-21")
        assert row["dii_net_cr"] == 0.0 and not row["is_missing_dii_flows"]
