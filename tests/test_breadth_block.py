"""
tests/test_breadth_block.py

Unit tests for features/blocks/breadth.py — Stage 10 breadth: the share of stocks trading above their own 200-day simple
average, computed over the Stage 5 point-in-time universe (NOT Nifty 500: the column is named universe_pct_above_200dma
for what it is). The window is a parameter so hand-computable windows can be used here.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.blocks.breadth import BREADTH_COLUMNS, EQUALITY_TOLERANCE, MIN_EVALUABLE, compute_breadth_features

SESSIONS = [date(2025, 1, 1) + timedelta(days=i) for i in range(12)]


def _panel(closes: dict[str, list[float]], sessions=SESSIONS):
    rows = [(sym, pd.Timestamp(d), c) for sym, cs in closes.items() for d, c in zip(sessions, cs)]
    return pd.DataFrame(rows, columns=["SYMBOL", "DATE", "ADJ_CLOSE"])


def _run(closes, members=None, window=3, min_evaluable=1, sessions=SESSIONS):
    members = members if members is not None else {2025: set(closes)}
    return compute_breadth_features(_panel(closes, sessions), members, sessions, window=window, min_evaluable=min_evaluable)


class TestDefinition:
    def test_share_of_stocks_above_their_own_simple_average_as_a_percentage(self):
        # window 3: A rises (above its average), B falls (below), C flat (equal -> not above)
        closes = {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "B": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "C": [5.0] * 12}
        out = _run(closes)
        row = out.iloc[5]
        assert row["universe_pct_above_200dma"] == pytest.approx(100 / 3)
        assert row["universe_n_evaluable"] == 3

    def test_the_average_includes_the_current_day_and_is_simple_not_exponential(self):
        closes = {"A": [10, 10, 10, 13, 10, 10, 10, 10, 10, 10, 10, 10]}
        out = _run(closes, window=3)
        assert out.iloc[3]["universe_pct_above_200dma"] == 100.0        # 13 > mean(10, 10, 13) = 11
        assert out.iloc[4]["universe_pct_above_200dma"] == 0.0          # 10 < mean(10, 13, 10) = 11

    def test_strictly_above_equality_does_not_count(self):
        out = _run({"A": [4.0] * 12}, window=3)
        assert out.iloc[5]["universe_pct_above_200dma"] == 0.0

    def test_floating_point_noise_around_the_average_is_a_tie_not_above(self):
        """
        Two correct ways to average a flat window (np.mean vs a rolling mean) can differ in the last bit, e.g. np.mean of
        three 0.7s is 0.6999999999999998, below the close. 'Above' must not depend on that, so a close within 1e-9
        (relative) of its average is a tie. One 0.05 tick on a Rs 1,000 stock is 5e-5: nothing real is that close.
        """
        noise = _run({"A": [1.0] * 10 + [1.0 + 1e-13, 1.0 + 1e-13]}, window=2)
        assert noise.iloc[10]["universe_pct_above_200dma"] == 0.0        # 1.0000000000001 vs mean 1.00000000000005: noise
        assert _run({"A": [1.0] * 10 + [1.001, 1.001]}, window=2).iloc[10]["universe_pct_above_200dma"] == 100.0

    def test_the_tolerance_is_a_recorded_constant(self):
        assert EQUALITY_TOLERANCE == 1e-9

    def test_only_universe_members_of_that_year_count(self):
        closes = {"IN": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "OUT": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        out = _run(closes, members={2025: {"IN"}})
        assert out.iloc[6]["universe_pct_above_200dma"] == 100.0 and out.iloc[6]["universe_n_evaluable"] == 1

    def test_membership_is_by_the_year_of_the_session_not_a_single_snapshot(self):
        sessions = [date(2024, 12, 27) + timedelta(days=i) for i in range(10)]           # spans the new year
        closes = {"OLD": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "NEW": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        out = _run(closes, members={2024: {"OLD"}, 2025: {"NEW"}}, sessions=sessions)
        before = out[out["date"] == pd.Timestamp("2024-12-31")].iloc[0]
        after = out[out["date"] == pd.Timestamp("2025-01-05")].iloc[0]
        assert before["universe_pct_above_200dma"] == 100.0                             # OLD is rising
        assert after["universe_pct_above_200dma"] == 0.0                                # NEW is falling

    def test_a_stock_without_a_full_window_is_not_in_the_denominator(self):
        closes = {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "YOUNG": [np.nan] * 8 + [1, 2, 3, 4]}
        out = _run(closes, window=3)
        assert out.iloc[5]["universe_n_evaluable"] == 1                                 # YOUNG has no prices yet
        assert out.iloc[11]["universe_n_evaluable"] == 2


class TestMissingness:
    def test_a_session_with_no_price_rows_is_nan_and_flagged_not_zero_percent(self):
        sessions = SESSIONS
        panel = _panel({"A": list(range(1, 13))})
        panel = panel[panel["DATE"] != pd.Timestamp(sessions[7])]                        # e.g. 2020-07-13: the bhavcopy is missing
        out = compute_breadth_features(panel, {2025: {"A"}}, sessions, window=3, min_evaluable=1)
        gap = out.iloc[7]
        assert np.isnan(gap["universe_pct_above_200dma"]) and gap["is_missing_breadth"] and np.isnan(gap["universe_n_evaluable"])

    def test_a_real_zero_percent_is_not_missing(self):
        out = _run({"A": list(range(12, 0, -1)), "B": list(range(12, 0, -1))}, window=3)
        assert out.iloc[6]["universe_pct_above_200dma"] == 0.0 and not out.iloc[6]["is_missing_breadth"]

    def test_warm_up_sessions_are_nan_and_flagged(self):
        out = _run({"A": list(range(1, 13))}, window=5)
        assert out["universe_pct_above_200dma"].iloc[:4].isna().all() and out["is_missing_breadth"].iloc[:4].all()
        assert not out["is_missing_breadth"].iloc[4:].any()

    def test_a_thin_sample_is_not_the_market(self):
        closes = {f"S{i}": list(range(1, 13)) for i in range(5)}
        thin = _run(closes, window=3, min_evaluable=6)
        assert thin["universe_pct_above_200dma"].isna().all() and thin["is_missing_breadth"].all()
        assert thin["universe_n_evaluable"].iloc[5] == 5                                # the real count is still reported
        assert not _run(closes, window=3, min_evaluable=5).iloc[5]["is_missing_breadth"]

    def test_a_non_positive_or_missing_close_is_missing_not_a_price(self):
        closes = {"A": [1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12]}                         # a 0 close cannot be a real price
        out = _run(closes, window=3)
        for i in (3, 4, 5):                                                              # the bad close is in the window of rows 3, 4 and 5
            assert out.iloc[i]["universe_n_evaluable"] == 0                              # rows exist for the date; none is evaluable
            assert np.isnan(out.iloc[i]["universe_pct_above_200dma"]) and out.iloc[i]["is_missing_breadth"]
        assert out.iloc[6]["universe_n_evaluable"] == 1 and not out.iloc[6]["is_missing_breadth"]   # the window has slid past it

    def test_the_flag_is_a_plain_bool_and_the_value_is_nan_exactly_where_it_is_true(self):
        out = _run({"A": list(range(1, 13))}, window=5)
        assert out["is_missing_breadth"].dtype == bool
        assert (out["is_missing_breadth"] == out["universe_pct_above_200dma"].isna()).all()


class TestShape:
    def test_one_row_per_session_in_order_with_the_declared_columns(self):
        out = _run({"A": list(range(1, 13))})
        assert list(out.columns) == ["date", *BREADTH_COLUMNS, "is_missing_breadth"]
        assert out["date"].tolist() == [pd.Timestamp(s) for s in SESSIONS]

    def test_duplicate_symbol_dates_are_rejected(self):
        panel = pd.concat([_panel({"A": list(range(1, 13))})] * 2, ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            compute_breadth_features(panel, {2025: {"A"}}, SESSIONS, window=3, min_evaluable=1)

    def test_the_default_minimum_is_a_recorded_constant(self):
        assert MIN_EVALUABLE == 100
