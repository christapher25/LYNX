"""
tests/test_circuit_inference.py

Band inference for the years before sec_list exists (it starts 2019-12-30). Everything here is PIT-safe by construction: the estimate
for session t may use only prices and hit lists dated BEFORE t (the band for t is published in the file of the previous session, so
that session's own prices are the newest evidence that could have existed). Backfilling from a later file would fail the truncation
invariant (future data changing past values), so it is not an option.

Two kinds of evidence:
  hit       a stock that hit its band reveals it exactly: limit = floor(prev*(1+b)) to the 0.05 tick for the upper band, ceil for the lower
  envelope  a stock never moves beyond its band, so the tightest band containing the trailing maximum excursion is a LOWER BOUND
"""

import numpy as np
import pandas as pd
import pytest

from features.blocks.circuit_inference import (
    BANDS,
    excursion_pct,
    implied_band,
    infer_bands,
    lower_limit,
    upper_limit,
)


class TestLimits:
    @pytest.mark.parametrize("prev,band,upper", [(100.0, 5, 105.0), (100.0, 10, 110.0), (33.65, 20, 40.35), (3.05, 20, 3.65), (1.00, 5, 1.05)])
    def test_the_upper_limit_floors_to_the_tick(self, prev, band, upper):
        assert upper_limit(prev, band) == pytest.approx(upper)

    @pytest.mark.parametrize("prev,band,lower", [(100.0, 5, 95.0), (33.65, 20, 26.95), (3.05, 20, 2.45), (1.00, 5, 0.95)])
    def test_the_lower_limit_ceils_to_the_tick(self, prev, band, lower):
        assert lower_limit(prev, band) == pytest.approx(lower)


class TestExcursion:
    def test_the_larger_of_the_high_and_low_excursions_in_percent(self):
        assert excursion_pct(100.0, 108.0, 97.0) == pytest.approx(8.0)
        assert excursion_pct(100.0, 101.0, 91.0) == pytest.approx(9.0)

    def test_a_non_positive_or_missing_price_is_nan(self):
        assert np.isnan(excursion_pct(0.0, 5.0, 1.0)) and np.isnan(excursion_pct(100.0, np.nan, 90.0))


class TestImpliedBand:
    def test_an_upper_hit_names_its_band(self):
        assert implied_band(100.0, 110.0, "H") == 10.0 and implied_band(100.0, 105.0, "H") == 5.0

    def test_a_lower_hit_names_its_band(self):
        assert implied_band(100.0, 80.0, "L") == 20.0 and implied_band(100.0, 98.0, "L") == 2.0

    def test_a_price_that_matches_no_band_is_not_a_hit_we_can_use(self):
        assert np.isnan(implied_band(100.0, 107.3, "H"))

    def test_a_tick_collapsed_price_that_matches_two_bands_is_ambiguous_not_guessed(self):
        # at prev 0.10 every band rounds to the same 0.10-0.15 range: floor(0.105)=0.10, floor(0.11)=0.10 -> several bands match
        assert np.isnan(implied_band(0.10, 0.10, "H"))


def _rows(n, prev=100.0, spread=1.0):
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"date": dates, "prev_close": prev, "high": prev + spread, "low": prev - spread})


class TestInference:
    def test_no_history_gives_no_estimate(self):
        out = infer_bands(_rows(10), pd.DataFrame(columns=["date", "HIT"]), min_history=60)
        assert out["BAND_PCT"].isna().all()

    def test_a_calm_stock_gets_the_tightest_band_containing_its_excursions_labelled_envelope(self):
        out = infer_bands(_rows(120, spread=1.0), pd.DataFrame(columns=["date", "HIT"]), min_history=60)
        late = out.iloc[-1]
        assert late["BAND_PCT"] == 2.0 and late["METHOD"] == "envelope"          # a 1% excursion fits inside 2%

    def test_the_estimate_for_a_session_uses_only_earlier_sessions(self):
        rows = _rows(120, spread=1.0)
        rows.loc[100, ["high", "low"]] = [108.0, 92.0]                            # a violent day at index 100
        out = infer_bands(rows, pd.DataFrame(columns=["date", "HIT"]), min_history=60)
        assert out.loc[100, "BAND_PCT"] == 2.0                                    # the day itself is not yet known
        assert out.loc[101, "BAND_PCT"] == 10.0                                   # from the next session on it is

    def test_a_hit_reveals_the_exact_band_and_outranks_the_envelope(self):
        rows = _rows(120, spread=1.0)
        rows.loc[80, ["high"]] = 105.0                                            # upper circuit on a 5% band
        hits = pd.DataFrame({"date": [rows.loc[80, "date"]], "HIT": ["H"]})
        out = infer_bands(rows, hits, min_history=60)
        assert out.loc[81, "BAND_PCT"] == 5.0 and out.loc[81, "METHOD"] == "hit"
        assert out.loc[80, "METHOD"] != "hit"                                     # a hit is evidence only from the next session

    def test_a_hit_is_dropped_once_later_prices_exceed_its_band(self):
        rows = _rows(140, spread=1.0)
        rows.loc[80, "high"] = 105.0
        rows.loc[100, ["high", "low"]] = [112.0, 96.0]                            # a 12% excursion: the 5% band must have been widened
        hits = pd.DataFrame({"date": [rows.loc[80, "date"]], "HIT": ["H"]})
        out = infer_bands(rows, hits, min_history=60)
        assert out.loc[101, "BAND_PCT"] == 20.0 and out.loc[101, "METHOD"] == "envelope"

    def test_a_stale_hit_expires(self):
        rows = _rows(400, spread=1.0)
        rows.loc[80, "high"] = 105.0
        hits = pd.DataFrame({"date": [rows.loc[80, "date"]], "HIT": ["H"]})
        out = infer_bands(rows, hits, min_history=60, hit_ttl=100)
        assert out.loc[150, "METHOD"] == "hit" and out.loc[300, "METHOD"] == "envelope"

    def test_an_excursion_beyond_the_widest_static_band_gives_no_estimate(self):
        rows = _rows(120, spread=1.0)
        rows.loc[70, ["high", "low"]] = [130.0, 70.0]                             # 30%: a No Band stock, a corporate action or a 40% band
        out = infer_bands(rows, pd.DataFrame(columns=["date", "HIT"]), min_history=60)
        assert out.loc[71:, "BAND_PCT"].isna().all() and out.loc[70, "BAND_PCT"] == 2.0     # the day itself is not yet known

    def test_the_bands_offered_are_the_static_ones_up_to_20(self):
        assert BANDS == (2.0, 5.0, 10.0, 20.0)

    def test_truncation_adding_later_rows_never_changes_earlier_estimates(self):
        rows = _rows(200, spread=1.0)
        rows.loc[120, "high"] = 105.0
        hits = pd.DataFrame({"date": [rows.loc[120, "date"]], "HIT": ["H"]})
        full = infer_bands(rows, hits, min_history=60)
        part = infer_bands(rows.iloc[:150], hits, min_history=60)
        pd.testing.assert_frame_equal(full.iloc[:150].reset_index(drop=True), part.reset_index(drop=True))
