"""
tests/test_circuits_block.py

Unit tests for features/blocks/circuits.py — Stage 10 circuits, two blocks each with its own shared missing flag (their
availability differs by ten years, like FII and DII flows):

  BAND   price_band_pct, is_no_band, band_is_inferred             flag is_missing_circuit_band
  HITS   hit_upper_circuit, hit_lower_circuit                     flag is_missing_circuit_hits

Measured 2026-09-24 on 7,045 real circuit hits in 60 sampled sessions: the band that reproduces a hit's limit price agrees with
the sec_list file dated the PREVIOUS SESSION 97.2% of the time and with the same-day file 92.8%, so the file dated d-1 gives the
band in force on session d (it is published the evening before). Limits are floor-to-tick for the upper band and ceil for the lower.
The hit flag for session t comes from the bh file dated t.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.blocks.circuits import BAND_COLUMNS, HIT_COLUMNS, compute_circuit_features

# Mon 2025-09-01 .. with a Saturday special session (09-06) and a weekday holiday gap (no 09-10)
SESSIONS = [date(2025, 9, 1), date(2025, 9, 2), date(2025, 9, 3), date(2025, 9, 4), date(2025, 9, 5), date(2025, 9, 6),
            date(2025, 9, 8), date(2025, 9, 9), date(2025, 9, 11), date(2025, 9, 12)]


def _bands(rows):
    return pd.DataFrame(rows, columns=["DATE", "BAND_PCT", "IS_NO_BAND"]).assign(DATE=lambda d: pd.to_datetime(d["DATE"]))


def _hits(rows):
    return pd.DataFrame(rows, columns=["DATE", "HIT"]).assign(DATE=lambda d: pd.to_datetime(d["DATE"]))


NO_HITS = _hits([])


def _run(bands=None, hits=NO_HITS, hit_days=SESSIONS, dates=SESSIONS):
    bands = bands if bands is not None else _bands([])
    return compute_circuit_features(dates, bands, hits, {pd.Timestamp(d) for d in hit_days}, SESSIONS)


def _row(out, day):
    return out[out["date"] == pd.Timestamp(day)].iloc[0]


class TestBandKeyedOnThePreviousSessionsFile:
    def test_the_band_for_a_session_is_read_from_the_file_dated_the_previous_session(self):
        out = _run(_bands([("2025-09-02", 5.0, False), ("2025-09-03", 10.0, False)]))
        assert _row(out, "2025-09-03")["price_band_pct"] == 5.0       # file of 09-02
        assert _row(out, "2025-09-04")["price_band_pct"] == 10.0      # file of 09-03
        assert np.isnan(_row(out, "2025-09-02")["price_band_pct"])    # the file of 09-01 does not exist

    def test_the_same_days_file_is_never_used_for_that_day(self):
        out = _run(_bands([("2025-09-03", 20.0, False)]))
        assert np.isnan(_row(out, "2025-09-03")["price_band_pct"]) and _row(out, "2025-09-03")["is_missing_circuit_band"]

    def test_a_band_change_takes_effect_on_the_session_after_its_file(self):
        out = _run(_bands([("2025-09-01", 5.0, False), ("2025-09-02", 5.0, False), ("2025-09-03", 20.0, False), ("2025-09-04", 20.0, False)]))
        assert [_row(out, d)["price_band_pct"] for d in ("2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05")] == [5.0, 5.0, 20.0, 20.0]

    def test_monday_uses_fridays_file_and_a_saturday_session_takes_its_place_in_the_chain(self):
        out = _run(_bands([("2025-09-05", 5.0, False), ("2025-09-06", 10.0, False)]))
        assert _row(out, "2025-09-06")["price_band_pct"] == 5.0        # Saturday session: Friday's file
        assert _row(out, "2025-09-08")["price_band_pct"] == 10.0       # Monday: the Saturday session's file, the previous SESSION

    def test_a_holiday_gap_is_bridged_by_the_previous_session_not_the_previous_calendar_day(self):
        out = _run(_bands([("2025-09-09", 2.0, False)]))
        assert _row(out, "2025-09-11")["price_band_pct"] == 2.0        # 09-10 was a holiday

    def test_a_symbol_absent_from_the_previous_file_is_missing_not_carried_from_earlier(self):
        out = _run(_bands([("2025-09-01", 5.0, False), ("2025-09-03", 5.0, False)]))     # nothing on 09-02: no band for 09-03's session
        row = _row(out, "2025-09-03")
        assert np.isnan(row["price_band_pct"]) and row["is_missing_circuit_band"]

    def test_the_first_session_has_no_previous_file(self):
        assert _row(_run(_bands([("2025-09-01", 5.0, False)])), "2025-09-01")["is_missing_circuit_band"]


class TestNoBand:
    def test_no_band_is_a_known_state_not_a_missing_value(self):
        out = _run(_bands([("2025-09-02", np.nan, True)]))
        row = _row(out, "2025-09-03")
        assert np.isnan(row["price_band_pct"]) and row["is_no_band"] == 1.0 and not row["is_missing_circuit_band"]

    def test_a_numeric_band_is_not_no_band(self):
        row = _row(_run(_bands([("2025-09-02", 5.0, False)])), "2025-09-03")
        assert row["is_no_band"] == 0.0 and not row["is_missing_circuit_band"]

    def test_an_unknown_band_leaves_is_no_band_unknown_too(self):
        row = _row(_run(_bands([])), "2025-09-03")
        assert np.isnan(row["is_no_band"]) and np.isnan(row["band_is_inferred"])

    def test_an_exact_band_is_marked_not_inferred(self):
        assert _row(_run(_bands([("2025-09-02", 5.0, False)])), "2025-09-03")["band_is_inferred"] == 0.0


class TestHits:
    def test_an_upper_and_a_lower_hit_on_the_same_session_date(self):
        out = _run(hits=_hits([("2025-09-03", "H")]))
        row = _row(out, "2025-09-03")
        assert row["hit_upper_circuit"] == 1.0 and row["hit_lower_circuit"] == 0.0 and not row["is_missing_circuit_hits"]
        assert _row(out, "2025-09-04")["hit_upper_circuit"] == 0.0     # a published list without this symbol: a real 0

    def test_a_stock_that_hit_both_bands_is_both(self):
        row = _row(_run(hits=_hits([("2025-09-03", "H"), ("2025-09-03", "L")])), "2025-09-03")
        assert row["hit_upper_circuit"] == 1.0 and row["hit_lower_circuit"] == 1.0

    def test_a_session_with_no_published_list_is_nan_and_flagged_never_zero(self):
        out = _run(hits=_hits([]), hit_days=[d for d in SESSIONS if d != date(2025, 9, 4)])
        row = _row(out, "2025-09-04")
        assert np.isnan(row["hit_upper_circuit"]) and np.isnan(row["hit_lower_circuit"]) and row["is_missing_circuit_hits"]

    def test_a_hit_is_visible_on_its_own_session_not_the_next(self):
        out = _run(hits=_hits([("2025-09-03", "L")]))
        assert _row(out, "2025-09-03")["hit_lower_circuit"] == 1.0 and _row(out, "2025-09-04")["hit_lower_circuit"] == 0.0

    def test_a_hit_on_a_date_without_a_published_list_is_ignored_not_trusted(self):
        out = _run(hits=_hits([("2025-09-04", "H")]), hit_days=[d for d in SESSIONS if d != date(2025, 9, 4)])
        assert _row(out, "2025-09-04")["is_missing_circuit_hits"]

    def test_the_two_blocks_have_independent_flags(self):
        out = _run(hits=_hits([("2025-09-03", "H")]))                  # hits known everywhere, bands nowhere
        assert out["is_missing_circuit_band"].all() and not out["is_missing_circuit_hits"].any()


class TestShape:
    def test_one_row_per_date_the_declared_columns_and_plain_bool_flags(self):
        out = _run(_bands([("2025-09-02", 5.0, False)]), _hits([("2025-09-03", "H")]), dates=SESSIONS[2:6])
        assert list(out.columns) == ["date", *BAND_COLUMNS, "is_missing_circuit_band", *HIT_COLUMNS, "is_missing_circuit_hits"]
        assert out["date"].tolist() == [pd.Timestamp(d) for d in SESSIONS[2:6]]
        assert out["is_missing_circuit_band"].dtype == bool and out["is_missing_circuit_hits"].dtype == bool

    def test_the_value_is_nan_exactly_where_the_flag_is_true(self):
        out = _run(_bands([("2025-09-02", 5.0, False), ("2025-09-03", np.nan, True)]), _hits([("2025-09-05", "H")]))
        assert (out["is_missing_circuit_band"] == out["is_no_band"].isna()).all()
        assert (out["is_missing_circuit_hits"] == out["hit_upper_circuit"].isna()).all()

    def test_a_row_date_that_is_not_a_session_is_missing_not_an_error(self):
        out = _run(_bands([("2025-09-02", 5.0, False)]), dates=[date(2025, 9, 7)])       # a Sunday the calendar lacks
        assert out["is_missing_circuit_band"].all()
