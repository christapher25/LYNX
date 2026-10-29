"""
tests/test_fno_features.py

Unit tests for features/blocks/fno.py: the day summariser, the pure feature computation, and the loaders (on tiny synthetic
files). The silent-failure invariants live in tests/invariants/test_fno_feature_invariants.py.
"""

import numpy as np
import pandas as pd
import pytest

from features.blocks import fno
from features.blocks.fno import build_symbol_features, compute_fno_features, load_ban_lists, load_fno_summaries, summarise_fno_day

D = pd.Timestamp("2024-03-20")


def _contracts(rows):
    """rows: (symbol, instrument, expiry, strike, option_type, open_int)"""
    return pd.DataFrame(rows, columns=["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE", "OPEN_INT"]).assign(
        EXPIRY_DATE=lambda f: pd.to_datetime(f["EXPIRY_DATE"]), DATE=D)


def _obs(n=30, **overrides):
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = {
        "date": dates, "volume_adj_factor": 1.0, "fno_day_available": True, "ban_available": True, "in_ban": False,
        "fut_oi_raw": 1000.0, "near_oi": 900.0, "next_oi": 100.0, "days_to_expiry": 5.0, "put_oi": 50.0, "call_oi": 100.0,
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestSummariseDay:
    def test_near_next_and_totals(self):
        day = _contracts([
            ("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 700.0),
            ("ABC", "FUTSTK", "2024-04-25", np.nan, np.nan, 200.0),
            ("ABC", "FUTSTK", "2024-05-30", np.nan, np.nan, 100.0),
        ])
        row = summarise_fno_day(day, D).loc["ABC"]
        assert (row["fut_oi_raw"], row["near_oi"], row["next_oi"], row["days_to_expiry"]) == (1000.0, 700.0, 200.0, 8.0)

    def test_expired_contract_is_dropped_and_expiry_day_is_still_near(self):
        day = _contracts([
            ("ABC", "FUTSTK", "2024-03-19", np.nan, np.nan, 999.0),      # expired the day before: still listed in old files
            ("ABC", "FUTSTK", "2024-03-20", np.nan, np.nan, 300.0),      # expires today: near
            ("ABC", "FUTSTK", "2024-04-25", np.nan, np.nan, 100.0),
        ])
        row = summarise_fno_day(day, D).loc["ABC"]
        assert (row["fut_oi_raw"], row["near_oi"], row["next_oi"], row["days_to_expiry"]) == (400.0, 300.0, 100.0, 0.0)

    def test_no_next_contract_is_nan_not_zero(self):
        row = summarise_fno_day(_contracts([("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)]), D).loc["ABC"]
        assert row["near_oi"] == 5.0 and np.isnan(row["next_oi"])

    def test_index_and_other_instruments_are_ignored(self):
        day = _contracts([("NIFTY", "FUTIDX", "2024-03-28", np.nan, np.nan, 9e6), ("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)])
        assert list(summarise_fno_day(day, D).index) == ["ABC"]

    def test_option_sides_including_american_codes(self):
        day = _contracts([
            ("ABC", "OPTSTK", "2024-03-28", 100.0, "CE", 10.0), ("ABC", "OPTSTK", "2024-03-28", 110.0, "CA", 5.0),
            ("ABC", "OPTSTK", "2024-03-28", 100.0, "PE", 6.0), ("ABC", "OPTSTK", "2024-03-28", 110.0, "PA", 3.0),
        ])
        row = summarise_fno_day(day, D).loc["ABC"]
        assert (row["call_oi"], row["put_oi"]) == (15.0, 9.0) and np.isnan(row["fut_oi_raw"])

    def test_one_sided_options_leave_the_other_side_nan(self):
        row = summarise_fno_day(_contracts([("ABC", "OPTSTK", "2024-03-28", 100.0, "PE", 6.0)]), D).loc["ABC"]
        assert row["put_oi"] == 6.0 and np.isnan(row["call_oi"])

    def test_duplicate_live_contract_raises(self):
        day = _contracts([("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0), ("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 0.0)])
        with pytest.raises(ValueError, match="duplicated live futures"):
            summarise_fno_day(day, D)

    def test_duplicate_live_option_contract_raises(self):
        day = _contracts([("ABC", "OPTSTK", "2024-03-28", 100.0, "CE", 5.0), ("ABC", "OPTSTK", "2024-03-28", 100.0, "CE", 7.0)])
        with pytest.raises(ValueError, match="duplicated live options"):
            summarise_fno_day(day, D)

    def test_same_strike_different_side_is_not_a_duplicate(self):
        day = _contracts([("ABC", "OPTSTK", "2024-03-28", 100.0, "CE", 5.0), ("ABC", "OPTSTK", "2024-03-28", 100.0, "PE", 7.0)])
        assert summarise_fno_day(day, D).loc["ABC", "put_oi"] == 7.0

    def test_all_nan_open_interest_stays_nan(self):
        row = summarise_fno_day(_contracts([("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, np.nan)]), D).loc["ABC"]
        assert np.isnan(row["fut_oi_raw"])


class TestComputeFeatures:
    def test_levels_and_changes(self):
        oi = [100.0, 110.0, 121.0, 121.0, 121.0, 242.0]
        out = compute_fno_features(_obs(6, fut_oi_raw=oi))
        assert out["fut_oi"].tolist() == oi
        assert out["fut_oi_chg_1d"].iloc[1] == pytest.approx(0.10) and np.isnan(out["fut_oi_chg_1d"].iloc[0])
        assert out["fut_oi_chg_5d"].iloc[5] == pytest.approx(1.42) and out["fut_oi_chg_5d"].iloc[:5].isna().all()

    def test_adjustment_factor_removes_the_ex_date_jump(self):
        # 2:1 split: the pre-split raw OI is in old shares, the factor (ADJ/RAW) doubles it to match post-split shares
        raw = [100.0, 100.0, 100.0, 200.0, 200.0]
        factor = [2.0, 2.0, 2.0, 1.0, 1.0]
        adjusted = compute_fno_features(_obs(5, fut_oi_raw=raw, volume_adj_factor=factor))
        assert adjusted["fut_oi_chg_1d"].iloc[3] == pytest.approx(0.0)
        unadjusted = compute_fno_features(_obs(5, fut_oi_raw=raw))
        assert unadjusted["fut_oi_chg_1d"].iloc[3] == pytest.approx(1.0)       # what the factor exists to prevent

    def test_change_does_not_bridge_a_missing_day(self):
        raw = [100.0, np.nan, 120.0, 130.0]
        out = compute_fno_features(_obs(4, fut_oi_raw=raw))
        assert np.isnan(out["fut_oi_chg_1d"].iloc[2]) and out["fut_oi_chg_1d"].iloc[3] == pytest.approx(130 / 120 - 1)

    def test_ratio_needs_15_observed_days(self):
        out = compute_fno_features(_obs(30, fut_oi_raw=[10.0] * 30))
        assert out["fut_oi_ratio_20d"].iloc[:14].isna().all() and out["fut_oi_ratio_20d"].iloc[14] == pytest.approx(1.0)

    def test_zero_oi_window_ratio_is_undefined_not_zero(self):
        out = compute_fno_features(_obs(20, fut_oi_raw=[0.0] * 20))
        assert out["fut_oi_ratio_20d"].isna().all() and not out["is_missing_fno"].any()
        assert (out["fut_oi"] == 0).all()                                    # a real zero stays a zero

    def test_rollover_share(self):
        out = compute_fno_features(_obs(2, near_oi=[300.0, 0.0], next_oi=[100.0, 0.0]))
        assert out["fut_roll_next_share"].iloc[0] == pytest.approx(0.25) and np.isnan(out["fut_roll_next_share"].iloc[1])

    def test_rollover_with_no_next_is_nan_and_not_missing(self):
        out = compute_fno_features(_obs(2, next_oi=np.nan))
        assert out["fut_roll_next_share"].isna().all() and not out["is_missing_fno"].any()

    def test_real_zero_next_share_is_zero(self):
        out = compute_fno_features(_obs(2, near_oi=500.0, next_oi=0.0))
        assert (out["fut_roll_next_share"] == 0.0).all()

    def test_days_to_expiry_passes_through(self):
        assert compute_fno_features(_obs(2, days_to_expiry=[7.0, 0.0]))["fut_days_to_expiry"].tolist() == [7.0, 0.0]

    def test_pcr_and_its_edges(self):
        out = compute_fno_features(_obs(3, put_oi=[50.0, 50.0, np.nan], call_oi=[100.0, 0.0, 100.0]))
        assert out["opt_pcr_oi"].iloc[0] == 0.5 and np.isnan(out["opt_pcr_oi"].iloc[1]) and np.isnan(out["opt_pcr_oi"].iloc[2])
        assert out["is_missing_opt"].tolist() == [False, False, True]

    def test_symbol_absent_from_an_available_day_is_missing_fno_only(self):
        nan = np.nan
        out = compute_fno_features(_obs(2, fut_oi_raw=[nan, 5.0], near_oi=[nan, 5.0], next_oi=nan, days_to_expiry=[nan, 3.0],
                                        put_oi=nan, call_oi=nan))
        assert out["is_missing_fno"].tolist() == [True, False] and out["fno_day_unavailable"].tolist() == [False, False]

    def test_unavailable_day_sets_both_flags_and_nans_everything(self):
        out = compute_fno_features(_obs(3, fno_day_available=[True, False, True]))
        row = out.iloc[1]
        assert row["is_missing_fno"] and row["fno_day_unavailable"] and row["is_missing_opt"]
        assert row[["fut_oi", "fut_roll_next_share", "fut_days_to_expiry", "opt_pcr_oi"]].isna().all()

    def test_nonpositive_factor_is_unusable(self):
        out = compute_fno_features(_obs(2, volume_adj_factor=[np.nan, 0.0]))
        assert out["is_missing_fno"].all() and out["fut_oi"].isna().all()

    def test_unsorted_or_incomplete_input_raises(self):
        with pytest.raises(ValueError):
            compute_fno_features(_obs(3).iloc[::-1])
        with pytest.raises(ValueError):
            compute_fno_features(_obs(3).drop(columns=["in_ban"]))

    def test_output_columns_and_dtypes(self):
        out = build_symbol_features("ABC", _obs(3)[["date", "volume_adj_factor"]], None, set(), set(), set())
        assert list(out.columns) == fno.OUTPUT_COLUMNS
        assert all(out[c].dtype == bool for c in fno.FLAG_COLUMNS)


class TestBan:
    def test_unavailable_is_nan_and_none_is_zero(self):
        out = compute_fno_features(_obs(3, ban_available=[True, False, True], in_ban=[False, False, True]))
        assert out["fno_in_ban"].iloc[0] == 0.0 and np.isnan(out["fno_in_ban"].iloc[1]) and out["fno_in_ban"].iloc[2] == 1.0
        assert out["is_missing_ban"].tolist() == [False, True, False]

    def test_listed_but_unavailable_day_is_not_trusted(self):
        out = compute_fno_features(_obs(1, ban_available=False, in_ban=True))
        assert np.isnan(out["fno_in_ban"].iloc[0])

    def test_new_entry_needs_an_observed_previous_row(self):
        out = compute_fno_features(_obs(5, ban_available=[True, True, True, False, True], in_ban=[False, True, True, False, True]))
        assert out["fno_ban_new_entry"].iloc[1] == 1.0          # not banned -> banned
        assert out["fno_ban_new_entry"].iloc[2] == 0.0          # still banned: not new
        assert np.isnan(out["fno_ban_new_entry"].iloc[3]) and np.isnan(out["fno_ban_new_entry"].iloc[4])

    def test_rolling_count_needs_15_observed_lists(self):
        out = compute_fno_features(_obs(20, in_ban=[True] * 20))
        assert out["fno_ban_days_20d"].iloc[:14].isna().all() and out["fno_ban_days_20d"].iloc[14] == 15.0
        gaps = compute_fno_features(_obs(20, in_ban=True, ban_available=[True] * 5 + [False] * 10 + [True] * 5))
        assert np.isnan(gaps["fno_ban_days_20d"].iloc[19])      # only 10 observed lists in the window


class TestLoaders:
    def _write_bhav(self, directory, stem, date, rows):
        frame = _contracts(rows).assign(DATE=pd.Timestamp(date))
        frame.to_parquet(directory / f"{stem}.parquet", index=False)

    def test_missing_file_is_not_an_available_day(self, tmp_path):
        self._write_bhav(tmp_path, "2024-03-19", "2024-03-19", [("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)])
        self._write_bhav(tmp_path, "2024-03-21", "2024-03-21", [("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 6.0)])
        summaries, days = load_fno_summaries(directory=tmp_path)
        assert days == {pd.Timestamp("2024-03-19"), pd.Timestamp("2024-03-21")} and len(summaries) == 2

    def test_rows_are_dated_by_their_embedded_date_not_the_filename(self, tmp_path):
        self._write_bhav(tmp_path, "2024-03-21", "2024-03-20", [("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)])
        summaries, days = load_fno_summaries(directory=tmp_path)
        assert days == {pd.Timestamp("2024-03-20")} and summaries["date"].iloc[0] == pd.Timestamp("2024-03-20")

    def test_two_files_for_one_session_raise(self, tmp_path):
        for stem in ("2024-03-20", "2024-03-21"):
            self._write_bhav(tmp_path, stem, "2024-03-20", [("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)])
        with pytest.raises(ValueError, match="two files"):
            load_fno_summaries(directory=tmp_path)

    def test_a_file_that_mixes_dates_raises(self, tmp_path):
        frame = pd.concat([_contracts([("ABC", "FUTSTK", "2024-03-28", np.nan, np.nan, 5.0)]).assign(DATE=pd.Timestamp(d))
                           for d in ("2024-03-20", "2024-03-21")])
        frame.to_parquet(tmp_path / "2024-03-20.parquet", index=False)
        with pytest.raises(ValueError, match="distinct dates"):
            load_fno_summaries(directory=tmp_path)

    def _write_ban(self, directory, stem, date, symbols):
        pd.DataFrame({"DATE": pd.Timestamp(date), "SYMBOL": pd.array(symbols or [None], dtype="string")}).to_parquet(
            directory / f"{stem}.parquet", index=False)

    def test_nil_list_is_available_and_empty(self, tmp_path):
        self._write_ban(tmp_path, "2024-03-20", "2024-03-20", [])
        bans, available = load_ban_lists(directory=tmp_path)
        assert available == {pd.Timestamp("2024-03-20")} and bans.empty

    def test_ban_list_is_keyed_by_embedded_date(self, tmp_path):
        self._write_ban(tmp_path, "2014-02-19", "2014-03-19", ["ABC"])      # the stale re-serve shape
        bans, available = load_ban_lists(directory=tmp_path)
        assert available == {pd.Timestamp("2014-03-19")} and bans["date"].iloc[0] == pd.Timestamp("2014-03-19")

    def test_session_without_a_list_is_unavailable_not_none(self, tmp_path):
        self._write_ban(tmp_path, "2024-03-19", "2024-03-19", ["ABC"])
        _, available = load_ban_lists(directory=tmp_path)
        panel = pd.DataFrame({"date": pd.to_datetime(["2024-03-19", "2024-03-20"]), "volume_adj_factor": 1.0})
        out = build_symbol_features("XYZ", panel, None, set(), set(), available)
        assert out["fno_in_ban"].iloc[0] == 0.0 and np.isnan(out["fno_in_ban"].iloc[1])

    def test_range_selection_by_filename(self, tmp_path):
        for stem in ("2024-03-19", "2024-03-20", "2024-03-21"):
            self._write_ban(tmp_path, stem, stem, ["ABC"])
        _, available = load_ban_lists("2024-03-20", "2024-03-20", directory=tmp_path)
        assert available == {pd.Timestamp("2024-03-20")}
