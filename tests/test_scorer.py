"""
tests/test_scorer.py

Invariant tests for signals/scorer.py — Stage 9. This extends the same
pattern as the four existing invariant tests (corporate actions, graph
leakage, normalization leakage, no-Viterbi): each test proves a specific
silent-failure mode CANNOT happen, not just that the happy path runs.

The test that matters most here is test_market_wide_selloff_does_not_
contaminate_scores — it's a direct, literal test of the exact bug
architecture v4 §6.3 describes: "RSI < 35 fires for every stock in a
market-wide selloff." If this test ever fails, the scorer has regressed
back into being a disguised market-timing signal.
"""

import numpy as np
import pandas as pd
import pytest

from signals.scorer import (
    FULL_WEIGHTS,
    build_composite_score,
    compute_market_context_score,
    compute_technical_score,
    cross_sectional_normalize,
)


# --------------------------------------------------------------------------- #
# cross_sectional_normalize — the core primitive
# --------------------------------------------------------------------------- #

class TestCrossSectionalNormalize:
    def test_produces_zero_mean_unit_std_within_each_date(self):
        """Basic correctness: each date group should z-score to mean~0, std~1."""
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
            "raw_score": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
        })
        df["z"] = cross_sectional_normalize(df, "raw_score")

        for d in df["date"].unique():
            group = df.loc[df["date"] == d, "z"]
            assert abs(group.mean()) < 1e-9
            assert abs(group.std(ddof=0) - 1.0) < 1e-9

    def test_zero_variance_group_guards_to_zero_not_nan(self):
        """
        A date where every symbol has the identical raw score (edge case, but
        reachable with a bounded threshold-based score) must not produce NaN
        — a NaN here would silently drop that date from any model that
        doesn't explicitly handle missing features, which is a worse failure
        than a slightly-wrong-but-present zero.
        """
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 4,
            "raw_score": [50.0, 50.0, 50.0, 50.0],
        })
        z = cross_sectional_normalize(df, "raw_score")
        assert not z.isna().any()
        assert (z == 0.0).all()

    def test_groups_are_independent(self):
        """A value's z-score must depend only on its own date's group, never
        leak information from another date — this is the graph/normalization
        leakage failure mode applied to the scorer specifically."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "raw_score": [10, 20, 10, 20],
        })
        z = cross_sectional_normalize(df, "raw_score")
        # identical raw values + identical within-group spread on both dates
        # must produce identical z-scores, regardless of what the other
        # date's absolute levels were
        assert z.iloc[0] == pytest.approx(z.iloc[2])
        assert z.iloc[1] == pytest.approx(z.iloc[3])


# --------------------------------------------------------------------------- #
# THE test — direct reproduction of the bug this whole design fixes
# --------------------------------------------------------------------------- #

class TestMarketWideSelloffInvariant:
    def test_market_wide_selloff_does_not_contaminate_scores(self):
        """
        Construct a synthetic universe of 50 stocks on a single date. 45 of
        them are in a uniform market-wide selloff (RSI 30, identical
        technical picture — the "everyone is oversold together" scenario).
        5 are genuine cross-sectional standouts: still oversold, but MORE
        oversold than their selloff-day peers, with confirming volume.

        BEFORE cross-sectional normalization, SignalForge's absolute-
        threshold logic gives all 45 selloff stocks the same "+20 oversold"
        bump — the composite can't distinguish the mundane 45 from the
        genuine 5 standouts, because everyone's absolute score moved
        together.

        AFTER cross-sectional normalization (this module), the 45 uniform
        stocks should z-score close to the group mean (near 0), while the 5
        genuine standouts should separate clearly into the tail. This is the
        literal invariant architecture v4 §6.3 exists to guarantee.
        """
        rng = np.random.default_rng(42)
        n_selloff, n_standout = 45, 5
        date = "2024-03-15"

        # 45 "everyone is oversold together" stocks — near-identical inputs
        # with small noise so they aren't a literal zero-variance edge case
        selloff_rows = []
        for _ in range(n_selloff):
            selloff_rows.append({
                "date": date,
                "close": 100.0,
                "rsi_14": 33.0 + rng.normal(0, 0.5),
                "macd_delta": -0.1 + rng.normal(0, 0.02),
                "ema_200": 105.0,
                "bb_position": 0.15 + rng.normal(0, 0.02),
                "atr_normalized": 0.03,
                "volume_ratio": 1.1 + rng.normal(0, 0.05),
                "obv_trend": -500 + rng.normal(0, 50),
            })

        # 5 genuine standouts — more oversold, WITH volume confirmation and
        # positive OBV trend (accumulation into weakness), which the interaction
        # term in compute_technical_score specifically rewards
        standout_rows = []
        for _ in range(n_standout):
            standout_rows.append({
                "date": date,
                "close": 100.0,
                "rsi_14": 22.0 + rng.normal(0, 0.5),
                "macd_delta": 0.15 + rng.normal(0, 0.02),  # already turning up
                "ema_200": 105.0,
                "bb_position": 0.05,
                "atr_normalized": 0.015,
                "volume_ratio": 2.2 + rng.normal(0, 0.1),
                "obv_trend": 800 + rng.normal(0, 50),
            })

        df = pd.DataFrame(selloff_rows + standout_rows)
        df["technical_score"] = df.apply(compute_technical_score, axis=1)

        scored = build_composite_score(
            df, weights={"technical": 1.0, "fundamental": 0, "sentiment": 0, "market_context": 0}
        )

        selloff_z = scored.iloc[:n_selloff]["technical_zscore"]
        standout_z = scored.iloc[n_selloff:]["technical_zscore"]

        # The 45 uniform selloff stocks should cluster near the group mean —
        # this is the actual assertion that matters: being "typically
        # oversold for a selloff day" must NOT read as an extreme signal.
        assert selloff_z.abs().mean() < 0.5, (
            "market-wide selloff stocks are not clustering near zero after "
            "cross-sectional normalization — the beta-contamination bug is back"
        )

        # The 5 genuine standouts must separate clearly into the positive tail
        assert standout_z.mean() > 1.5, (
            "genuine cross-sectional standouts are not separating from the "
            "selloff cohort — normalization is too aggressive or mis-applied"
        )

        # And the standouts must be unambiguously above the selloff cohort
        assert standout_z.min() > selloff_z.max(), (
            "standout and selloff cohorts overlap — the composite can't "
            "actually distinguish genuine outliers from a uniform market move"
        )


# --------------------------------------------------------------------------- #
# Composite weight renormalization
# --------------------------------------------------------------------------- #

class TestCompositeWeightRenormalization:
    def test_missing_layers_renormalize_rather_than_bias_low(self):
        """
        With only technical + market_context populated (current build phase
        — fundamental/sentiment are Phase 4), the composite for a stock with
        z-scores of exactly 0 on both present layers must be exactly 0, not
        pulled toward a negative value as if the two absent layers were
        silently scored as zero.
        """
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 3,
            "technical_score": [50.0, 60.0, 40.0],       # will z-score to [0, 1, -1]-ish
            "market_context_score": [50.0, 50.0, 50.0],  # zero variance -> guards to 0
        })
        scored = build_composite_score(df)

        # row 0: technical z=0 (it's the mean), market_context z=0 (zero-var
        # guard) -> composite must be exactly 0, not biased by absent layers
        assert scored.iloc[0]["deterministic_score"] == pytest.approx(0.0, abs=1e-9)

    def test_weights_sum_to_one_over_present_layers_only(self):
        present = {"technical", "market_context"}
        renorm = {k: FULL_WEIGHTS[k] / sum(FULL_WEIGHTS[l] for l in present) for k in present}
        assert sum(renorm.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Layer functions degrade gracefully on missing inputs
# --------------------------------------------------------------------------- #

class TestGracefulDegradation:
    def test_technical_score_handles_all_nan_row(self):
        row = pd.Series({
            "close": np.nan, "rsi_14": np.nan, "macd_delta": np.nan,
            "ema_200": np.nan, "bb_position": np.nan, "atr_normalized": np.nan,
            "volume_ratio": np.nan, "obv_trend": np.nan,
        })
        # must not raise — every condition should simply not fire
        score = compute_technical_score(row)
        assert score == 50.0  # neutral, no conditions fired

    def test_market_context_score_handles_all_nan_row(self):
        row = pd.Series({
            "india_vix": np.nan, "fii_net_flow_cr": np.nan,
            "nifty500_pct_above_200dma": np.nan, "usdinr_chg_pct": np.nan,
        })
        score = compute_market_context_score(row)
        assert score == 50.0