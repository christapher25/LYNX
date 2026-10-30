"""
gates/override_gates.py

Stage 11 -- India hard gates (v6 section 6.2: circuit limits, F&O ban periods). Pure, deterministic, DB-free, no learned model.

A gate looks at ONE row of Stage 10 features -- (symbol, session t) -- and returns one of three states:

  PASS         the input is known and clear
  BLOCK        the input is known and the name must be dropped
  UNAVAILABLE  the input is not known (NaN, or its missing flag is set). It is never read as PASS.

Gates (inputs are the Stage 10 columns, unchanged):

  upper_circuit   hit_upper_circuit   / is_missing_circuit_hits   BLOCK if the stock touched its upper band on session t
  lower_circuit   hit_lower_circuit   / is_missing_circuit_hits   BLOCK if it touched its lower band on session t
  fno_ban         fno_in_ban          / is_missing_ban            BLOCK if on session t's F&O ban list
  liquidity_cliff liquidity_rank      / is_missing_liquidity_rank BLOCK if the name ranks outside the top `liquidity_top_n` (200) by
                                                                  trailing turnover (gates/liquidity.py, point-in-time)

Default enabled set is the two filters the v6 nightly timeline names for the pre-open re-filter ("drop names at an upper circuit or
in the F&O ban list"): upper_circuit and fno_ban. lower_circuit and liquidity_cliff exist but are off unless asked for. BUY_SIGNAL_GATES is the set the M1 run applies to buy signals
(user decisions 15 and 16, 2026-10-30): upper and lower circuit, F&O ban, liquidity cliff. NOT_APPLIED_GATES lists, with reasons, the
gates v6 section 6.2 names that are NOT applied, so a report can state them.

Combined verdict per row: any BLOCK -> BLOCK (a known block is never hidden by another gate's missing input); else any UNAVAILABLE ->
UNAVAILABLE; else PASS. `gate_allowed` is True for PASS only, unless the config sets unavailable_policy="allow", in which case an
UNAVAILABLE row is allowed but its status still says UNAVAILABLE (the policy is explicit and visible, never a silent pass).

Point-in-time. The verdict at (symbol, t) is a function of that row alone: the hit flag comes from the price-band hit list dated t
and the ban flag from the ban list dated t (Stage 10 same-session convention). Nothing is shifted, rolled or looked up from another row,
and no price band is read -- the band columns are ignored on purpose.

Corrupt input raises: a flag value outside {0, 1}, or a missing required column, is an error, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PASS = "PASS"
BLOCK = "BLOCK"
UNAVAILABLE = "UNAVAILABLE"
DISABLED = "DISABLED"

# gate name -> (value column, missing-flag column)
GATE_INPUTS: dict[str, tuple[str, str]] = {
    "upper_circuit": ("hit_upper_circuit", "is_missing_circuit_hits"),
    "lower_circuit": ("hit_lower_circuit", "is_missing_circuit_hits"),
    "fno_ban": ("fno_in_ban", "is_missing_ban"),
    "liquidity_cliff": ("liquidity_rank", "is_missing_liquidity_rank"),
}
DEFAULT_ENABLED = ("upper_circuit", "fno_ban")
BUY_SIGNAL_GATES = ("upper_circuit", "lower_circuit", "fno_ban", "liquidity_cliff")
LIQUIDITY_TOP_N = 200
NOT_APPLIED_GATES: dict[str, str] = {
    "corporate_action_frequency": "deferred by user decision 16 (2026-10-30): no threshold in v6, no input feature built",
    "pledging": "deferred by user decision 16 (2026-10-30): no threshold in v6; the pledging feature block exists but is not gated on",
    "news": "needs the Stage 15+ news/event inputs, not built",
    "volatility": "needs a volatility-gate threshold v6 does not give",
    "drawdown": "needs a drawdown-gate threshold v6 does not give",
    "event_proximity": "needs the Stage 15+ event calendar, not built",
}
GATE_COLUMNS = [f"gate_{name}" for name in GATE_INPUTS] + ["gate_status", "gate_allowed", "gate_reasons"]


@dataclass(frozen=True)
class GateConfig:
    enabled: tuple[str, ...] = DEFAULT_ENABLED
    unavailable_policy: str = "block"          # "block": an UNAVAILABLE row is not allowed. "allow": allowed, status stays UNAVAILABLE.
    liquidity_top_n: int = LIQUIDITY_TOP_N     # liquidity_cliff blocks a rank strictly greater than this

    def __post_init__(self):
        unknown = set(self.enabled) - set(GATE_INPUTS)
        if unknown:
            raise ValueError(f"unknown gate(s) {sorted(unknown)}; known: {sorted(GATE_INPUTS)}")
        if not self.enabled:
            raise ValueError("at least one gate must be enabled")
        if self.unavailable_policy not in ("block", "allow"):
            raise ValueError("unavailable_policy must be 'block' or 'allow'")
        if int(self.liquidity_top_n) < 1:
            raise ValueError("liquidity_top_n must be at least 1")


def classify(value: pd.Series, missing: pd.Series) -> pd.Series:
    """Three-state verdict for one gate. The missing flag wins over the value; NaN in either is UNAVAILABLE."""
    numeric = pd.to_numeric(value, errors="coerce")
    flagged = missing.astype("boolean")
    bad = numeric.notna() & ~numeric.isin([0.0, 1.0])
    if bad.any():
        raise ValueError(f"{value.name}: values outside {{0, 1}} found: {sorted(numeric[bad].unique())[:5]}")
    known = numeric.notna() & flagged.notna() & ~flagged.fillna(True).astype(bool)
    out = np.where(~known, UNAVAILABLE, np.where(numeric == 1.0, BLOCK, PASS))
    return pd.Series(out, index=value.index, dtype=object)


def classify_liquidity(value: pd.Series, missing: pd.Series, top_n: int) -> pd.Series:
    """Three-state verdict for the liquidity cliff. Same unknown handling as classify(); BLOCK iff rank > top_n. A rank must be a
    whole number >= 1, anything else is corrupt input and raises."""
    numeric = pd.to_numeric(value, errors="coerce")
    flagged = missing.astype("boolean")
    bad = numeric.notna() & ((numeric < 1) | (numeric != np.floor(numeric)))
    if bad.any():
        raise ValueError(f"{value.name}: ranks must be whole numbers >= 1, found {sorted(numeric[bad].unique())[:5]}")
    known = numeric.notna() & flagged.notna() & ~flagged.fillna(True).astype(bool)
    out = np.where(~known, UNAVAILABLE, np.where(numeric > top_n, BLOCK, PASS))
    return pd.Series(out, index=value.index, dtype=object)


def evaluate_gates(features: pd.DataFrame, config: GateConfig = GateConfig()) -> pd.DataFrame:
    """
    features : rows of Stage 10 features with at least the value and missing-flag columns of every enabled gate (extra columns,
               including price_band_pct, are ignored). Index is preserved.
    returns  : gate_<name> per gate (DISABLED for a gate not enabled), gate_status, gate_allowed (bool), gate_reasons
               (';'-joined '<gate>=BLOCK' / '<gate>=UNAVAILABLE', empty on PASS).
    """
    needed = {col for name in config.enabled for col in GATE_INPUTS[name]}
    absent = needed - set(features.columns)
    if absent:
        raise ValueError(f"missing required column(s): {sorted(absent)}")

    out = pd.DataFrame(index=features.index)
    for name, (value_col, flag_col) in GATE_INPUTS.items():
        if name not in config.enabled:
            out[f"gate_{name}"] = DISABLED
        elif name == "liquidity_cliff":
            out[f"gate_{name}"] = classify_liquidity(features[value_col], features[flag_col], int(config.liquidity_top_n))
        else:
            out[f"gate_{name}"] = classify(features[value_col], features[flag_col])

    states = out[[f"gate_{n}" for n in config.enabled]]
    any_block = (states == BLOCK).any(axis=1)
    any_unavailable = (states == UNAVAILABLE).any(axis=1)
    out["gate_status"] = np.where(any_block, BLOCK, np.where(any_unavailable, UNAVAILABLE, PASS))
    out["gate_allowed"] = (out["gate_status"] == PASS) | ((out["gate_status"] == UNAVAILABLE) & (config.unavailable_policy == "allow"))
    reasons = pd.Series("", index=features.index, dtype=object)
    for name in config.enabled:
        state = states[f"gate_{name}"]
        part = np.where(state.isin([BLOCK, UNAVAILABLE]), name + "=" + state, "")
        reasons = reasons.where(part == "", np.where(reasons == "", part, reasons + ";" + part))
    out["gate_reasons"] = reasons
    return out
