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

Default enabled set is the two filters the v6 nightly timeline names for the pre-open re-filter ("drop names at an upper circuit or
in the F&O ban list"): upper_circuit and fno_ban. lower_circuit exists but is off unless asked for (spec question, see verify_stage11).

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
}
DEFAULT_ENABLED = ("upper_circuit", "fno_ban")
GATE_COLUMNS = [f"gate_{name}" for name in GATE_INPUTS] + ["gate_status", "gate_allowed", "gate_reasons"]


@dataclass(frozen=True)
class GateConfig:
    enabled: tuple[str, ...] = DEFAULT_ENABLED
    unavailable_policy: str = "block"          # "block": an UNAVAILABLE row is not allowed. "allow": allowed, status stays UNAVAILABLE.

    def __post_init__(self):
        unknown = set(self.enabled) - set(GATE_INPUTS)
        if unknown:
            raise ValueError(f"unknown gate(s) {sorted(unknown)}; known: {sorted(GATE_INPUTS)}")
        if not self.enabled:
            raise ValueError("at least one gate must be enabled")
        if self.unavailable_policy not in ("block", "allow"):
            raise ValueError("unavailable_policy must be 'block' or 'allow'")


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
        out[f"gate_{name}"] = classify(features[value_col], features[flag_col]) if name in config.enabled else DISABLED

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
