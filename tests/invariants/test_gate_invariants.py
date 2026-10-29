"""
tests/invariants/test_gate_invariants.py

Stage 11 invariants for the India hard gates (gates/override_gates.py). Each is a violation counter run on the real code (must be 0)
and on a deliberately broken gate for the failure it exists to prevent (must be > 0):

  1. UNAVAILABLE IS NEVER PASS      a row whose input is NaN or flagged missing is UNAVAILABLE and, under the default policy, not allowed.
                                    Broken: NaN is zero-filled, so a missing ban list reads "nobody banned" and passes.
  2. FLAG BEATS VALUE               a value that looks clean (0.0) but carries a missing flag never passes.
  3. POINT-IN-TIME / ROW-LOCAL      the verdict at (symbol, date) depends only on that row: truncating the history never changes it.
                                    Broken: "banned on the next row" leaks in through shift(-1).
  4. BLOCK WINS, THEN UNAVAILABLE   a known block is never hidden by another gate's missing input.
  5. BAND IGNORED                   changing the price band column never changes a verdict (the gate reads hits and bans, never a band).
  6. ROW ORDER                      shuffling the input rows never changes a verdict.
"""

import numpy as np
import pandas as pd
import pytest

from gates.override_gates import BLOCK, PASS, UNAVAILABLE, GateConfig, evaluate_gates

ALL = GateConfig(enabled=("upper_circuit", "lower_circuit", "fno_ban"))


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return evaluate_gates(df, ALL)


def _world(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "symbol": rng.choice([f"S{i:02d}" for i in range(40)], n),
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n), "D"),
        "hit_upper_circuit": rng.choice([0.0, 1.0], n, p=[0.9, 0.1]),
        "hit_lower_circuit": rng.choice([0.0, 1.0], n, p=[0.9, 0.1]),
        "fno_in_ban": rng.choice([0.0, 1.0], n, p=[0.85, 0.15]),
        "price_band_pct": rng.choice([5.0, 10.0, 20.0], n),
    })
    hits_missing = rng.random(n) < 0.15
    ban_missing = rng.random(n) < 0.15
    df.loc[hits_missing, ["hit_upper_circuit", "hit_lower_circuit"]] = np.nan
    df.loc[ban_missing, "fno_in_ban"] = np.nan
    df["is_missing_circuit_hits"] = hits_missing
    df["is_missing_ban"] = ban_missing
    return df


# --- broken gates: each commits exactly the failure its invariant exists to catch ------------------------------------

def _broken_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    """NaN read as 0.0, flags ignored: a missing list passes."""
    filled = df.assign(hit_upper_circuit=df["hit_upper_circuit"].fillna(0.0), hit_lower_circuit=df["hit_lower_circuit"].fillna(0.0),
                       fno_in_ban=df["fno_in_ban"].fillna(0.0), is_missing_circuit_hits=False, is_missing_ban=False)
    return evaluate_gates(filled, ALL)


def _broken_flag_bypass(df: pd.DataFrame) -> pd.DataFrame:
    """Passes any row whose ban value is 0.0 even when the missing flag is set (value looked at, flag ignored)."""
    out = evaluate_gates(df, ALL)
    clean_looking = (df["fno_in_ban"] == 0.0) & df["is_missing_ban"] & (out["gate_status"] == UNAVAILABLE)
    out.loc[clean_looking, ["gate_status", "gate_allowed"]] = [PASS, True]
    return out


def _broken_lookahead(df: pd.DataFrame) -> pd.DataFrame:
    """'Banned on the next row' leaks into today's verdict."""
    out = df.sort_values(["symbol", "date"]).copy()
    nxt = out.groupby("symbol")["fno_in_ban"].shift(-1)
    out["fno_in_ban"] = np.fmax(out["fno_in_ban"], nxt.fillna(0.0)).where(out["fno_in_ban"].notna())
    return evaluate_gates(out, ALL)


def _broken_unavailable_hides_block(df: pd.DataFrame) -> pd.DataFrame:
    """Any unavailable gate overrides a known block."""
    out = evaluate_gates(df, ALL)
    any_unavailable = (out["gate_upper_circuit"] == UNAVAILABLE) | (out["gate_lower_circuit"] == UNAVAILABLE) | (out["gate_fno_ban"] == UNAVAILABLE)
    hidden = (out["gate_status"] == BLOCK) & any_unavailable
    out.loc[hidden, "gate_status"] = UNAVAILABLE
    return out


def _broken_reads_band(df: pd.DataFrame) -> pd.DataFrame:
    """Treats a 20% band as 'cannot be locked' and clears an upper hit: the verdict now depends on the band column."""
    out = evaluate_gates(df, ALL)
    cleared = (df["price_band_pct"] == 20.0) & (out["gate_upper_circuit"] == BLOCK) & (out["gate_status"] == BLOCK)
    out.loc[cleared, ["gate_status", "gate_allowed"]] = [PASS, True]
    return out


# --- violation counters ------------------------------------------------------------------------------------------------

def _unavailable_passes(gate, df) -> int:
    out = gate(df).reindex(df.index)
    unknown = df["is_missing_ban"] | df["is_missing_circuit_hits"] | df[["hit_upper_circuit", "hit_lower_circuit", "fno_in_ban"]].isna().any(axis=1)
    known_block = ((df["fno_in_ban"] == 1.0) & ~df["is_missing_ban"]) | \
                  (((df["hit_upper_circuit"] == 1.0) | (df["hit_lower_circuit"] == 1.0)) & ~df["is_missing_circuit_hits"])
    return int((unknown & ~known_block & ((out["gate_status"] == PASS) | out["gate_allowed"])).sum())


def _flag_bypass(gate, df) -> int:
    out = gate(df).reindex(df.index)
    return int(((out["gate_status"] == PASS) & (df["is_missing_ban"] | df["is_missing_circuit_hits"])).sum())


def _row_local_violations(gate, df, cut) -> int:
    """Verdicts for rows dated <= cut computed from history truncated at cut must equal those from the full history."""
    full = gate(df).sort_index()
    short = gate(df[df["date"] <= cut]).sort_index()
    cols = ["gate_status", "gate_allowed"]
    return int((full.loc[short.index, cols].to_numpy() != short[cols].to_numpy()).any(axis=1).sum())


def _block_hidden(gate, df) -> int:
    out = gate(df).reindex(df.index)
    blocked = ((df["fno_in_ban"] == 1.0) & ~df["is_missing_ban"]) | \
              (((df["hit_upper_circuit"] == 1.0) | (df["hit_lower_circuit"] == 1.0)) & ~df["is_missing_circuit_hits"])
    return int((blocked & (out["gate_status"] != BLOCK)).sum())


def _band_dependence(gate, df) -> int:
    base = gate(df).reindex(df.index)[["gate_status", "gate_allowed"]].to_numpy()
    moved = gate(df.assign(price_band_pct=df["price_band_pct"] + 1000.0)).reindex(df.index)[["gate_status", "gate_allowed"]].to_numpy()
    gone = gate(df.assign(price_band_pct=np.nan)).reindex(df.index)[["gate_status", "gate_allowed"]].to_numpy()
    return int(((base != moved).any(axis=1) | (base != gone).any(axis=1)).sum())


SEEDS = [1, 2, 3, 4]


@pytest.mark.parametrize("seed", SEEDS)
def test_unavailable_is_never_pass(seed):
    df = _world(seed)
    assert _unavailable_passes(_real, df) == 0
    assert _unavailable_passes(_broken_zero_fill, df) > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_missing_flag_beats_value(seed):
    df = _world(seed)
    df.loc[df["is_missing_ban"], "fno_in_ban"] = 0.0
    df.loc[df["is_missing_circuit_hits"], ["hit_upper_circuit", "hit_lower_circuit"]] = 0.0
    assert _flag_bypass(_real, df) == 0
    assert _flag_bypass(_broken_flag_bypass, df) > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_point_in_time_row_local(seed):
    df = _world(seed)
    cut = df["date"].iloc[len(df) // 2]
    assert _row_local_violations(_real, df, cut) == 0
    # plant the exact trap: the last visible row of a symbol is clear, the very next row is banned
    trap = df.copy()
    last = trap.index[trap["date"] <= cut][-1]
    trap.loc[last, ["fno_in_ban", "is_missing_ban", "hit_upper_circuit", "hit_lower_circuit", "is_missing_circuit_hits"]] = [0.0, False, 0.0, 0.0, False]
    nxt = trap.index[trap["date"] > cut][0]
    trap.loc[nxt, ["symbol", "fno_in_ban", "is_missing_ban"]] = [trap.loc[last, "symbol"], 1.0, False]
    assert _row_local_violations(_real, trap, cut) == 0
    assert _row_local_violations(_broken_lookahead, trap, cut) > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_known_block_never_hidden(seed):
    df = _world(seed)
    assert _block_hidden(_real, df) == 0
    assert _block_hidden(_broken_unavailable_hides_block, df) > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_band_columns_are_ignored(seed):
    df = _world(seed)
    assert _band_dependence(_real, df) == 0
    assert _band_dependence(_broken_reads_band, df) > 0


def test_row_order_does_not_matter():
    df = _world(7)
    a = _real(df).sort_index()
    b = _real(df.sample(frac=1.0, random_state=0)).sort_index()
    assert a.equals(b)
