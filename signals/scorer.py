"""
signals/scorer.py

Stage 9 — Deterministic scorer for LYNX (architecture v4 §6.3).

This is NOT a port of SignalForge's scorer. Three structural changes, each
fixing something SignalForge's design couldn't support once LYNX demoted the
scorer from decision-maker to feature-provider-and-fallback:

1. CROSS-SECTIONAL NORMALIZATION (the actual fix, per §6.3).
   SignalForge's technical_score used absolute thresholds: `if rsi_14 < 35:
   score += 20`. On a market-wide selloff day, RSI < 35 fires for most of the
   universe simultaneously — the score stops measuring "is this stock
   oversold relative to its peers" and starts measuring "is the market down
   today," which is the market factor PRISM's ranking objective already
   removes at the label level (architecture v4 §3.1). LYNX keeps the same
   style of threshold-based sub-scores (they're auditable and that's the
   point of a deterministic fallback) but z-scores each layer's output
   CROSS-SECTIONALLY, within each trading date, before it enters the feature
   matrix. A stock that's oversold the same amount as everyone else nets to
   zero; only genuine cross-sectional outliers survive.

2. ONE COMPOSITE, NOT THREE MODE-WEIGHTED ONES.
   SignalForge computed separate INTRADAY/LONG-TERM composites because the
   composite WAS the product. In LYNX, PRISM's four horizon heads (§3.4)
   handle horizon differentiation by learning their own weights on whatever
   features move the needle — the deterministic scorer just needs to supply
   one honest, auditable composite feature, not pre-guess three different
   horizon-specific blends by hand.

3. FUNDAMENTAL AND SENTIMENT LAYERS ARE WIRED, NOT FAKED.
   SignalForge had SEC-filing RAG and FinBERT sentiment available from day
   one. LYNX doesn't yet — those come from Phase 4 (Stages 16-20, the agent
   extraction and knowledge graph layers). compute_fundamental_score and
   compute_sentiment_score return None until those stages exist, and the
   composite renormalizes its weights over whichever layers are actually
   available rather than silently treating a missing layer as zero (which
   would bias the composite low on every stock, corrupting the cross-
   sectional comparison this whole module exists to get right).

Market-context inputs also change: SignalForge used SPY/sector-ETF/Fed-
meeting proximity. LYNX uses the same four series the regime HMM consumes
(architecture v4 §3.6) — India VIX, FII net flow, Nifty 500 breadth, USDINR
change — so the scorer and the regime router agree on what "context" means.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# --------------------------------------------------------------------------- #
# Layer weights, used ONLY once a layer is non-null. Renormalized per row
# over whichever layers are actually present — see build_composite_score.
# Calibrate these once all four layers are live (Phase 4) and an ablation
# (architecture v4 §8.3, A8 specifically) shows what each layer is worth;
# these starting values are a reasonable prior, not a measured result.
# --------------------------------------------------------------------------- #
FULL_WEIGHTS = {
    "technical": 0.35,
    "fundamental": 0.30,
    "sentiment": 0.20,
    "market_context": 0.15,
}

# Indian VIX bands — India VIX trades at a structurally higher baseline than
# US VIX (long-run median closer to 15-18, not 12-15). These are a starting
# point; recalibrate against the actual India VIX distribution once Stage 8's
# full history is loaded, rather than trusting a borrowed US intuition.
VIX_CALM_THRESHOLD = 15.0
VIX_STRESSED_THRESHOLD = 25.0


# --------------------------------------------------------------------------- #
# Layer 1 — Technical score. Pure function of one row of Stage 9's indicator
# output (indicators.py) plus that day's close and volume_ratio.
# --------------------------------------------------------------------------- #

def compute_technical_score(row: pd.Series) -> float:
    """
    Raw technical score, 0-100, centered at 50. Same threshold-based style as
    SignalForge's original — the fix for cross-sectional contamination is
    the normalization step downstream (see module docstring), not the
    elimination of thresholds here.

    Expects `row` to contain: close, rsi_14, macd_delta, bb_position,
    atr_normalized, obv_trend, ema_200, volume_ratio. Missing/NaN inputs
    degrade gracefully — each condition simply doesn't fire rather than
    raising, since early-history rows legitimately have NaN indicators
    (see indicators.py's MIN_ROWS_REQUIRED gate for the alternative of
    excluding them entirely upstream).
    """
    score = 50.0

    rsi = row.get("rsi_14")
    if pd.notna(rsi):
        if rsi < 35:
            score += 20  # oversold
        elif rsi > 70:
            score -= 20  # overbought

    close, ema_200 = row.get("close"), row.get("ema_200")
    if pd.notna(close) and pd.notna(ema_200):
        if close > ema_200:
            score += 15  # above long-term trend

    macd_delta = row.get("macd_delta")
    obv_trend = row.get("obv_trend")
    if pd.notna(macd_delta) and macd_delta > 0:
        score += 15  # momentum positive
        if pd.notna(obv_trend) and obv_trend > 0:
            score += 10  # volume confirms the momentum — new vs. SignalForge:
            # a MACD-positive day with distribution (falling OBV) is a much
            # weaker signal than one with accumulation behind it; SignalForge
            # never had this interaction because its OBV feature fed XGBoost
            # directly rather than the hand-built score.

    volume_ratio = row.get("volume_ratio")
    if pd.notna(volume_ratio) and volume_ratio > 1.5:
        score += 10  # unusual volume

    bb_position = row.get("bb_position")
    if pd.notna(bb_position) and bb_position < 0.2:
        score += 15  # price near the lower band — substitutes SignalForge's
        # undefined "price_near_support" with an actually-computed condition

    atr_normalized = row.get("atr_normalized")
    if pd.notna(atr_normalized):
        if atr_normalized < 0.02:
            score += 10  # low volatility, cleaner signal
        elif atr_normalized > 0.05:
            score -= 10  # high volatility, noisier signal

    return float(np.clip(score, 0, 100))


# --------------------------------------------------------------------------- #
# Layer 2 — Market context. Uses the SAME four series as the regime HMM
# (architecture v4 §3.6), so this layer and the regime router never disagree
# about what "risk-on" means.
# --------------------------------------------------------------------------- #

def compute_market_context_score(row: pd.Series) -> float:
    """
    Raw market-context score, 0-100, centered at 50.

    Expects `row` to contain: india_vix, fii_net_flow_cr (FII net flow in
    crores, positive = net buying), nifty500_pct_above_200dma (breadth, 0-1),
    usdinr_chg_pct (day-over-day % change). These come from Stage 8's macro
    pipeline — the FII/DII figure specifically from the forward-capture
    ingester, so history before its start date will be sparse; that's a
    known, documented gap (see india-signal-app notes on the FII/DII source),
    not a bug in this function.
    """
    score = 50.0

    vix = row.get("india_vix")
    if pd.notna(vix):
        if vix < VIX_CALM_THRESHOLD:
            score += 15
        elif vix > VIX_STRESSED_THRESHOLD:
            score -= 15

    fii_flow = row.get("fii_net_flow_cr")
    if pd.notna(fii_flow):
        score += 15 if fii_flow > 0 else -10

    breadth = row.get("nifty500_pct_above_200dma")
    if pd.notna(breadth):
        if breadth > 0.5:
            score += 15
        elif breadth < 0.3:
            score -= 15

    usdinr_chg = row.get("usdinr_chg_pct")
    if pd.notna(usdinr_chg):
        if abs(usdinr_chg) < 0.005:
            score += 5  # currency stable
        elif usdinr_chg > 0.01:
            score -= 10  # rupee weakening fast — often coincides with FII outflow stress

    return float(np.clip(score, 0, 100))


# --------------------------------------------------------------------------- #
# Layers 3 & 4 — not active yet. Return None explicitly rather than a
# fabricated neutral value, so build_composite_score can tell "not available"
# apart from "genuinely neutral" and renormalize weights accordingly.
# --------------------------------------------------------------------------- #

def compute_fundamental_score(row: pd.Series) -> Optional[float]:
    """
    Activates in Phase 4 (Stage 18: LLM extraction pipeline) once structured
    financial fields are being pulled from results announcements per
    architecture v4 §4.3. Returns None until then — do not fake a neutral 50
    here, that would silently bias every stock's composite downward relative
    to a world where this layer is genuinely absent from the weighting.
    """
    return None


def compute_sentiment_score(row: pd.Series) -> Optional[float]:
    """
    Activates in Phase 4 (Stage 18) once the Qwen3-4B extraction layer is
    emitting structured event records with polarity/magnitude (architecture
    v4 §5.2). Returns None until then, same reasoning as
    compute_fundamental_score.
    """
    return None


# --------------------------------------------------------------------------- #
# Cross-sectional normalization — the actual fix from §6.3.
# --------------------------------------------------------------------------- #

def cross_sectional_normalize(
    df: pd.DataFrame, value_col: str, date_col: str = "date"
) -> pd.Series:
    """
    Z-score `value_col` within each `date_col` group independently.

    This is the invariant that makes the scorer safe to use as a PRISM
    feature: a level shift that hits every stock on the same day (a
    market-wide selloff, a policy announcement, peak-margin-style structural
    breaks) moves every value in the group together, so the group mean moves
    with it and the z-score sees nothing — exactly the "two-part defence"
    architecture v4 §6.3 describes. A date group with zero variance (a
    single-symbol day, or a day where a raw score happens to be identical
    across the whole universe — rare but not impossible for a bounded score
    with only a handful of achievable values) is guarded to 0.0 rather than
    left as NaN or inf, since a NaN here would silently drop that date's rows
    out of any downstream model that doesn't explicitly handle it.
    """
    def _zscore(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return df.groupby(date_col)[value_col].transform(_zscore)


# --------------------------------------------------------------------------- #
# Composite — combines whichever layers are non-null, per row, with weights
# renormalized over the available subset.
# --------------------------------------------------------------------------- #

def build_composite_score(
    df: pd.DataFrame,
    date_col: str = "date",
    weights: dict[str, float] = FULL_WEIGHTS,
) -> pd.DataFrame:
    """
    Given a DataFrame with raw layer-score columns
    (technical_score, market_context_score, and optionally fundamental_score,
    sentiment_score), cross-sectionally z-score each present layer, then
    combine them into `deterministic_score` using weights renormalized over
    whichever layers are non-null for that row.

    Returns the input df with the per-layer z-scored columns
    (`{layer}_zscore`) and the final `deterministic_score` column appended.
    """
    layer_cols = {
        "technical": "technical_score",
        "fundamental": "fundamental_score",
        "sentiment": "sentiment_score",
        "market_context": "market_context_score",
    }

    out = df.copy()
    zscore_cols = {}
    for layer, col in layer_cols.items():
        if col not in out.columns:
            continue
        zcol = f"{layer}_zscore"
        out[zcol] = cross_sectional_normalize(out, col, date_col=date_col)
        zscore_cols[layer] = zcol

    def _combine(row: pd.Series) -> float:
        present = {layer: row[zcol] for layer, zcol in zscore_cols.items() if pd.notna(row[zcol])}
        if not present:
            return np.nan
        raw_weight_sum = sum(weights[layer] for layer in present)
        return sum(weights[layer] * z for layer, z in present.items()) / raw_weight_sum

    out["deterministic_score"] = out.apply(_combine, axis=1)
    return out


# --------------------------------------------------------------------------- #
# Convenience entry point for ad-hoc inspection / the Stage 9 checkpoint
# ("scorer returns full dict for a sample ticker"). Scores a SINGLE row —
# cross-sectional normalization needs the full day's universe, so the
# z-scored/composite fields here are only meaningful when called as part of
# a full-date batch through build_composite_score, not in isolation. This
# function is for auditing raw layer scores on one symbol/date, not for
# producing a trustworthy final composite by itself.
# --------------------------------------------------------------------------- #

def score_signal(indicator_row: pd.Series, market_context_row: pd.Series) -> dict:
    """Return every raw layer score for one (symbol, date) as an audit dict."""
    technical = compute_technical_score(indicator_row)
    market_context = compute_market_context_score(market_context_row)
    fundamental = compute_fundamental_score(indicator_row)
    sentiment = compute_sentiment_score(indicator_row)

    active_layers = {
        k: v for k, v in {
            "technical": technical,
            "fundamental": fundamental,
            "sentiment": sentiment,
            "market_context": market_context,
        }.items() if v is not None
    }
    renormalized_weights = {
        k: FULL_WEIGHTS[k] / sum(FULL_WEIGHTS[l] for l in active_layers)
        for k in active_layers
    }
    naive_composite = sum(renormalized_weights[k] * v for k, v in active_layers.items())

    return {
        "technical_score": technical,
        "market_context_score": market_context,
        "fundamental_score": fundamental,  # None until Phase 4
        "sentiment_score": sentiment,      # None until Phase 4
        "active_layers": list(active_layers.keys()),
        "renormalized_weights": renormalized_weights,
        "single_row_composite_NOT_cross_sectional": naive_composite,
        # ^ diagnostic only — the real `deterministic_score` that enters the
        # feature matrix comes from build_composite_score() run over the
        # full day's universe, never from this single-row shortcut.
    }