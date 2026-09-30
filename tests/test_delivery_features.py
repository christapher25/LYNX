"""
tests/test_delivery_features.py

Invariant tests for features/blocks/delivery.py — Stage 10, delivery block. Covers the silent-failure
modes for delivery features: a missing observation read as a real number, a corporate action
masquerading as a delivery surge, an inconsistent NSE row trusted, and look-ahead.

compute_delivery_features() input, one row per price-panel date for ONE symbol, ascending:
    date, deliv_qty, deliv_per, volume, volume_adj_factor
where volume_adj_factor = Stage 4's ADJ_VOLUME / VOLUME for that row.
"""

import numpy as np
import pandas as pd
import pytest

from features.blocks.delivery import (
    DELIVERY_PCT_TOLERANCE_PP,
    FEATURE_COLUMNS,
    MIN_OBS_20D,
    compute_delivery_features,
)

OUTPUT_COLUMNS = ["date", *FEATURE_COLUMNS, "is_missing_delivery"]


def _obs(n: int = 60, seed: int = 3) -> pd.DataFrame:
    """Internally consistent observations: deliv_per == deliv_qty / volume * 100 (2 dp), factor 1."""
    rng = np.random.default_rng(seed)
    volume = rng.integers(100_000, 500_000, n).astype(float)
    deliv_qty = np.floor(volume * rng.uniform(0.2, 0.8, n))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "deliv_qty": deliv_qty,
        "deliv_per": np.round(deliv_qty / volume * 100, 2),
        "volume": volume,
        "volume_adj_factor": np.ones(n),
    })


def _from_qty(deliv_qty, volume) -> pd.DataFrame:
    """Observations from explicit quantities, deliv_per derived so the row is self-consistent."""
    deliv_qty = np.asarray(deliv_qty, dtype=float)
    volume = np.broadcast_to(np.asarray(volume, dtype=float), deliv_qty.shape).copy()
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(deliv_qty), freq="B"),
        "deliv_qty": deliv_qty,
        "deliv_per": np.round(deliv_qty / volume * 100, 2),
        "volume": volume,
        "volume_adj_factor": np.ones(len(deliv_qty)),
    })


def _blank(df: pd.DataFrame, rows) -> pd.DataFrame:
    """Simulate 'NSE has no delivery observation for these rows'."""
    df = df.copy()
    df.loc[rows, ["deliv_qty", "deliv_per"]] = np.nan
    return df


class TestDeliveryPct:
    def test_reported_percentage_passes_through_unchanged(self):
        obs = _obs()
        out = compute_delivery_features(obs)
        pd.testing.assert_series_equal(out["delivery_pct"], obs["deliv_per"], check_names=False)

    def test_a_real_zero_delivery_stays_zero_and_is_not_flagged_missing(self):
        """Zero delivered shares on a day that traded is a fact; it is not the same as no data."""
        obs = _obs()
        obs.loc[10, ["deliv_qty", "deliv_per"]] = [0.0, 0.0]
        out = compute_delivery_features(obs)
        assert out.loc[10, "delivery_pct"] == 0.0
        assert not out.loc[10, "is_missing_delivery"]

    def test_missing_stays_nan_never_zero(self):
        out = compute_delivery_features(_blank(_obs(), [10, 11]))
        assert out.loc[[10, 11], "delivery_pct"].isna().all()
        assert not (out["delivery_pct"] == 0).any()


class TestPercentagePointChanges:
    @staticmethod
    def _ramp(n=60) -> pd.DataFrame:
        volume = np.full(n, 1000.0)
        per = 10.0 + 0.5 * np.arange(n)          # exact in binary floating point
        return _from_qty(per * 10, volume)

    def test_chg_is_difference_in_percentage_points_against_n_rows_back(self):
        out = compute_delivery_features(self._ramp())
        assert out["delivery_pct_chg_5d"].iloc[5:].eq(2.5).all()     # 5 rows * 0.5pp
        assert out["delivery_pct_chg_20d"].iloc[20:].eq(10.0).all()  # 20 rows * 0.5pp

    def test_warm_up_rows_are_nan_not_zero(self):
        out = compute_delivery_features(self._ramp())
        assert out["delivery_pct_chg_5d"].iloc[:5].isna().all()
        assert out["delivery_pct_chg_20d"].iloc[:20].isna().all()

    def test_change_is_nan_when_the_reference_day_is_missing(self):
        out = compute_delivery_features(_blank(self._ramp(), [10]))
        assert np.isnan(out.loc[15, "delivery_pct_chg_5d"])          # its t-5 row is the missing one
        assert not np.isnan(out.loc[14, "delivery_pct_chg_5d"])
        assert np.isnan(out.loc[30, "delivery_pct_chg_20d"])


class TestQtyRatioUsesDeliveryQtyNotVolume:
    def test_ratio_is_delivered_quantity_over_its_own_20d_average(self):
        """Constant delivered qty, wildly varying total volume: a volume-based ratio would swing."""
        rng = np.random.default_rng(0)
        out = compute_delivery_features(_from_qty(np.full(60, 1000.0), rng.integers(5_000, 50_000, 60)))
        assert np.allclose(out["delivery_qty_ratio_20d"].iloc[MIN_OBS_20D - 1:], 1.0)

    def test_window_is_the_trailing_20_rows_including_today(self):
        qty = np.arange(1.0, 61.0) * 100
        out = compute_delivery_features(_from_qty(qty, 1_000_000.0))
        assert out.loc[25, "delivery_qty_ratio_20d"] == pytest.approx(qty[25] / qty[6:26].mean())

    def test_sparse_window_needs_min_obs_then_uses_only_observed_days(self):
        qty = np.arange(1.0, 61.0) * 100
        one_gap = compute_delivery_features(_blank(_from_qty(qty, 1_000_000.0), [30]))
        # Row 31's window is rows 12..31 with row 30 unobserved: 19 observations >= MIN_OBS_20D.
        assert one_gap.loc[31, "delivery_qty_ratio_20d"] == pytest.approx(
            qty[31] / np.delete(qty[12:32], 30 - 12).mean())
        many_gaps = compute_delivery_features(_blank(_from_qty(qty, 1_000_000.0), list(range(30, 37))))
        assert np.isnan(many_gaps.loc[38, "delivery_qty_ratio_20d"])  # only 13 of 20 observed < MIN_OBS_20D

    def test_all_zero_average_is_nan_not_inf(self):
        out = compute_delivery_features(_from_qty(np.zeros(40), 1_000.0))
        assert not np.isinf(out["delivery_qty_ratio_20d"]).any()
        assert out["delivery_qty_ratio_20d"].isna().all()


class TestCorporateActionsDoNotLookLikeDeliverySurges:
    """
    Raw DELIV_QTY is in shares as of each day, so a 2:1 split doubles it overnight with no change
    in real activity. Stage 4 rescales VOLUME to ADJ_VOLUME for exactly this reason; the delivery
    quantity must get the same rescaling or the ratio spikes for every split/bonus in history.
    """

    @staticmethod
    def _split_obs(adjust: bool) -> pd.DataFrame:
        pre, post = 1000.0, 2000.0                                   # 2:1 split at row 30, same real activity
        qty = np.r_[np.full(30, pre), np.full(30, post)]
        volume = np.r_[np.full(30, 5_000.0), np.full(30, 10_000.0)]
        obs = _from_qty(qty, volume)
        if adjust:
            obs["volume_adj_factor"] = np.r_[np.full(30, 2.0), np.full(30, 1.0)]  # Stage 4: ADJ_VOLUME / VOLUME
        return obs

    def test_split_adjusted_ratio_is_flat_across_the_ex_date(self):
        out = compute_delivery_features(self._split_obs(adjust=True))
        assert np.allclose(out["delivery_qty_ratio_20d"].iloc[MIN_OBS_20D - 1:], 1.0)

    def test_without_the_adjustment_the_same_data_shows_a_fake_surge(self):
        """Proves the test above is sensitive: unadjusted, the ex-date reads as a ~1.9x surge."""
        out = compute_delivery_features(self._split_obs(adjust=False))
        assert out.loc[30, "delivery_qty_ratio_20d"] > 1.5

    def test_percentage_is_unaffected_by_the_adjustment(self):
        adjusted = compute_delivery_features(self._split_obs(adjust=True))
        raw = compute_delivery_features(self._split_obs(adjust=False))
        pd.testing.assert_series_equal(adjusted["delivery_pct"], raw["delivery_pct"])


class TestSharedMissingFlag:
    def test_flag_is_a_plain_bool_with_no_nulls(self):
        out = compute_delivery_features(_blank(_obs(), [3, 4]))
        assert out["is_missing_delivery"].dtype == bool

    def test_flag_is_true_exactly_where_the_observation_is_missing(self):
        out = compute_delivery_features(_blank(_obs(), [3, 4, 20]))
        assert out.index[out["is_missing_delivery"]].tolist() == [3, 4, 20]

    def test_every_feature_is_nan_when_flagged(self):
        out = compute_delivery_features(_blank(_obs(), [3, 4, 20]))
        flagged = out[out["is_missing_delivery"]]
        assert flagged[FEATURE_COLUMNS].isna().all().all()

    def test_warm_up_nan_is_not_a_missing_observation(self):
        """chg_5d is NaN on row 2 because history is short, not because delivery was missing."""
        out = compute_delivery_features(_obs())
        assert np.isnan(out.loc[2, "delivery_pct_chg_5d"])
        assert not out.loc[2, "is_missing_delivery"]

    @pytest.mark.parametrize("column", ["deliv_qty", "deliv_per", "volume_adj_factor"])
    def test_a_single_absent_input_flags_the_whole_row(self, column):
        """One shared flag: a row is usable for all four features or for none of them."""
        obs = _obs()
        obs.loc[7, column] = np.nan
        out = compute_delivery_features(obs)
        assert out.loc[7, "is_missing_delivery"]
        assert out.loc[7, FEATURE_COLUMNS].isna().all()

    def test_all_missing_input_is_all_nan_and_all_flagged_never_zero(self):
        out = compute_delivery_features(_blank(_obs(), list(range(60))))
        assert out["is_missing_delivery"].all()
        assert out[FEATURE_COLUMNS].isna().all().all()


class TestInconsistentNseRowsAreQuarantined:
    """
    On five dates in the raw archive (2020-04-13, 2025-10-21, 2026-09-11, ...) NSE's DELIV_PER
    disagrees with its own DELIV_QTY / VOLUME, always upward (e.g. 41 traded, 1 delivered, "100%").
    Which field is wrong is unknowable, so the observation is treated as missing rather than trusted.
    """

    def test_reported_pct_contradicting_qty_over_volume_is_masked(self):
        obs = _obs()
        obs.loc[12, ["deliv_qty", "deliv_per", "volume"]] = [1.0, 100.0, 41.0]   # implied 2.44%, reported 100%
        out = compute_delivery_features(obs)
        assert out.loc[12, "is_missing_delivery"]
        assert out.loc[12, FEATURE_COLUMNS].isna().all()
        assert not out.loc[[11, 13], "is_missing_delivery"].any()               # neighbours untouched

    def test_two_decimal_rounding_is_not_mistaken_for_inconsistency(self):
        obs = _obs()
        obs.loc[5, ["deliv_qty", "deliv_per", "volume"]] = [17610.0, 59.33, 29683.0]  # implied 59.3278
        out = compute_delivery_features(obs)
        assert abs(17610 / 29683 * 100 - 59.33) < DELIVERY_PCT_TOLERANCE_PP
        assert not out.loc[5, "is_missing_delivery"]

    def test_delivered_more_than_traded_is_masked(self):
        obs = _obs()
        obs.loc[9, ["deliv_qty", "deliv_per", "volume"]] = [500.0, 50.0, 400.0]
        assert compute_delivery_features(obs).loc[9, "is_missing_delivery"]

    def test_row_that_cannot_be_validated_is_masked(self):
        obs = _obs()
        obs.loc[9, "volume"] = 0.0
        assert compute_delivery_features(obs).loc[9, "is_missing_delivery"]


class TestShapeAndValidation:
    def test_output_is_row_aligned_with_the_input(self):
        obs = _obs()
        out = compute_delivery_features(obs)
        assert list(out.columns) == OUTPUT_COLUMNS
        assert len(out) == len(obs) and (out["date"] == obs["date"]).all()

    def test_unsorted_dates_raise(self):
        with pytest.raises(ValueError, match="sorted ascending"):
            compute_delivery_features(_obs().iloc[::-1].reset_index(drop=True))

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            compute_delivery_features(_obs().drop(columns=["volume"]))

    def test_empty_input_returns_empty_frame_with_the_full_schema(self):
        out = compute_delivery_features(_obs().iloc[:0])
        assert list(out.columns) == OUTPUT_COLUMNS and out.empty


class TestCausality:
    """Adding future rows must not change any past value (a backtest that peeks looks better than live)."""

    @pytest.mark.parametrize("cut", [30, 45, 59])
    def test_past_values_unchanged_when_future_rows_are_added(self, cut):
        obs = _blank(_obs(n=60), [8, 9, 33])
        full = compute_delivery_features(obs)
        truncated = compute_delivery_features(obs.iloc[:cut])
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True), full.iloc[:cut].reset_index(drop=True),
            check_exact=False, rtol=1e-9, atol=1e-9,
        )
