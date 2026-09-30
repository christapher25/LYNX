"""
tests/test_pledging_block.py

Unit tests for features/blocks/pledging.py — Stage 10 pledging (promoter encumbrance):

  pledged_pct_of_promoter   100 * encumbered shares / promoter-group shares, from the latest filing public at that session
  pledged_pct_chg           that filing's ratio minus the previous PERIOD's ratio (percentage points)
  pledge_rising             1.0 if the change is positive, 0.0 if not, NaN if there is no change to compute
  pledging_age_days         days from the filing's period end to the session (staleness of the disclosure)
  is_missing_pledging       True exactly where no valid ratio is public

Every value is keyed on when its filing was BROADCAST, never on the quarter it describes: a 31-Mar filing broadcast on 21-Apr
is invisible until 21-Apr, and one broadcast after the 19:00 IST nightly run waits for the next session.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.blocks.pledging import PLEDGING_COLUMNS, compute_pledging_features

SESSIONS = [date(2025, 4, 1) + timedelta(days=i) for i in range(150) if (date(2025, 4, 1) + timedelta(days=i)).weekday() < 5]


def _filing(period_end, broadcast, promoter=1000.0, encumbered=0.0, has_promoter=True, error=None):
    return {"PERIOD_END": pd.Timestamp(period_end), "BROADCAST_IST": broadcast, "PROMOTER_SHARES": promoter,
            "ENCUMBERED_SHARES": encumbered, "HAS_PROMOTER_GROUP": has_promoter, "PARSE_ERROR": error}


def _run(filings, dates=SESSIONS):
    return compute_pledging_features(dates, pd.DataFrame(filings), SESSIONS)


def _row(out, day):
    return out[out["date"] == pd.Timestamp(day)].iloc[0]


class TestKeyedOnBroadcast:
    def test_a_filing_is_invisible_until_its_broadcast_session_not_its_period_end(self):
        out = _run([_filing("2025-03-31", "2025-04-21T16:10:33+05:30", encumbered=250.0)])
        assert np.isnan(_row(out, "2025-04-17")["pledged_pct_of_promoter"])          # the quarter had long ended; nothing was public
        assert _row(out, "2025-04-21")["pledged_pct_of_promoter"] == 25.0             # broadcast 16:10 IST: before that night's run

    def test_a_filing_broadcast_after_the_nightly_run_waits_for_the_next_session(self):
        out = _run([_filing("2025-03-31", "2025-04-21T19:24:44+05:30", encumbered=250.0)])
        assert np.isnan(_row(out, "2025-04-21")["pledged_pct_of_promoter"])
        assert _row(out, "2025-04-22")["pledged_pct_of_promoter"] == 25.0

    def test_a_broadcast_on_a_weekend_lands_on_the_next_session(self):
        out = _run([_filing("2025-03-31", "2025-04-19T12:14:17+05:30", encumbered=100.0)])     # a Saturday
        assert np.isnan(_row(out, "2025-04-18")["pledged_pct_of_promoter"]) and _row(out, "2025-04-21")["pledged_pct_of_promoter"] == 10.0

    def test_a_filing_with_no_broadcast_timestamp_is_never_used(self):
        out = _run([_filing("2025-03-31", None, encumbered=250.0)])
        assert out["pledged_pct_of_promoter"].isna().all() and out["is_missing_pledging"].all()

    def test_a_timestamp_without_a_timezone_is_rejected(self):
        with pytest.raises(ValueError):
            _run([_filing("2025-03-31", "2025-04-21T16:10:33", encumbered=250.0)])


class TestRatioAndChange:
    def test_the_ratio_is_computed_from_share_counts(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", promoter=2590349949.0, encumbered=2590189293.0)])
        assert _row(out, "2025-04-22")["pledged_pct_of_promoter"] == pytest.approx(99.993797, abs=1e-5)

    def test_a_real_zero_percent_is_zero_and_not_missing(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=0.0)])
        row = _row(out, "2025-04-22")
        assert row["pledged_pct_of_promoter"] == 0.0 and not row["is_missing_pledging"]

    def test_the_change_is_against_the_previous_period_and_the_first_filing_has_none(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=300.0)])
        first, second = _row(out, "2025-04-22"), _row(out, "2025-07-22")
        assert np.isnan(first["pledged_pct_chg"]) and np.isnan(first["pledge_rising"])
        assert second["pledged_pct_of_promoter"] == 30.0 and second["pledged_pct_chg"] == pytest.approx(20.0)
        assert second["pledge_rising"] == 1.0

    def test_unchanged_and_falling_are_not_rising(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=300.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=300.0)])
        assert _row(out, "2025-07-22")["pledge_rising"] == 0.0 and _row(out, "2025-07-22")["pledged_pct_chg"] == 0.0
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=300.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=100.0)])
        assert _row(out, "2025-07-22")["pledge_rising"] == 0.0 and _row(out, "2025-07-22")["pledged_pct_chg"] == pytest.approx(-20.0)

    def test_values_persist_between_filings_and_the_age_grows(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0)])
        assert _row(out, "2025-05-20")["pledged_pct_of_promoter"] == 10.0                      # last disclosed, still the latest
        assert _row(out, "2025-04-22")["pledging_age_days"] == 22 and _row(out, "2025-05-20")["pledging_age_days"] == 50

    def test_the_source_period_is_reported(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0)])
        assert _row(out, "2025-04-22")["source_period_end"] == pd.Timestamp("2025-03-31")


class TestMissingness:
    def test_before_any_filing_is_public_the_row_is_nan_and_flagged_never_zero(self):
        out = _run([_filing("2025-03-31", "2025-06-01T10:00:00+05:30", encumbered=100.0)])
        early = out[out["date"] < pd.Timestamp("2025-06-02")]
        assert early["pledged_pct_of_promoter"].isna().all() and early["is_missing_pledging"].all()
        assert early["pledging_age_days"].isna().all() and early["source_period_end"].isna().all()

    def test_no_promoter_group_is_missing_not_zero_percent(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", promoter=None, encumbered=None, has_promoter=False)])
        row = _row(out, "2025-04-22")
        assert np.isnan(row["pledged_pct_of_promoter"]) and row["is_missing_pledging"]

    def test_a_filing_with_zero_promoter_shares_is_missing_not_a_division_error(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", promoter=0.0, encumbered=0.0, has_promoter=False)])
        assert _row(out, "2025-04-22")["is_missing_pledging"]

    def test_a_filing_that_failed_to_parse_is_missing_and_does_not_fall_back_to_the_previous_quarter(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", promoter=None, encumbered=None, has_promoter=None,
                            error="404 Client Error")])
        row = _row(out, "2025-07-22")
        assert np.isnan(row["pledged_pct_of_promoter"]) and row["is_missing_pledging"]
        assert np.isnan(row["pledged_pct_chg"])                                                # its predecessor is the unknown one

    def test_a_good_filing_after_a_bad_one_restores_the_value(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", promoter=None, encumbered=None, has_promoter=None, error="x"),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=200.0)])
        assert np.isnan(_row(out, "2025-05-01")["pledged_pct_of_promoter"])                    # the bad quarter stays unknown
        assert _row(out, "2025-07-22")["pledged_pct_of_promoter"] == 20.0                      # the next good filing restores the value

    def test_no_filings_at_all_gives_all_missing(self):
        out = compute_pledging_features(SESSIONS, pd.DataFrame(columns=["PERIOD_END", "BROADCAST_IST", "PROMOTER_SHARES", "ENCUMBERED_SHARES",
                                                                      "HAS_PROMOTER_GROUP", "PARSE_ERROR"]), SESSIONS)
        assert out["is_missing_pledging"].all() and out["pledged_pct_of_promoter"].isna().all()

    def test_the_flag_is_a_plain_bool_and_the_value_is_nan_exactly_where_it_is_true(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0)])
        assert out["is_missing_pledging"].dtype == bool
        assert (out["is_missing_pledging"] == out["pledged_pct_of_promoter"].isna()).all()


class TestRevisions:
    def test_a_late_revision_of_an_older_period_does_not_become_the_current_filing(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=300.0),
                    _filing("2025-03-31", "2025-08-11T10:00:00+05:30", encumbered=200.0)])         # Q1 restated in August
        row = _row(out, "2025-08-12")
        assert row["source_period_end"] == pd.Timestamp("2025-06-30") and row["pledged_pct_of_promoter"] == 30.0
        assert row["pledged_pct_chg"] == pytest.approx(10.0)                                       # against the RESTATED Q1 (20%), not the original (10%)

    def test_the_original_is_what_was_public_before_the_revision(self):
        out = _run([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0),
                    _filing("2025-06-30", "2025-07-21T10:00:00+05:30", encumbered=300.0),
                    _filing("2025-03-31", "2025-08-11T10:00:00+05:30", encumbered=200.0)])
        assert _row(out, "2025-07-22")["pledged_pct_chg"] == pytest.approx(20.0)                   # 30% vs the original 10%


class TestShape:
    def test_one_row_per_date_in_order_with_the_declared_columns(self):
        dates = SESSIONS[10:40]
        out = compute_pledging_features(dates, pd.DataFrame([_filing("2025-03-31", "2025-04-21T10:00:00+05:30", encumbered=100.0)]), SESSIONS)
        assert list(out.columns) == ["date", *PLEDGING_COLUMNS, "source_period_end", "is_missing_pledging"]
        assert out["date"].tolist() == [pd.Timestamp(d) for d in dates]

    def test_a_filing_broadcast_after_the_last_session_is_not_yet_usable(self):
        out = _run([_filing("2025-06-30", "2026-12-01T10:00:00+05:30", encumbered=100.0)])
        assert out["is_missing_pledging"].all()
