"""
tests/invariants/test_circuits_invariants.py

The Stage 10 invariants for the circuits block (features/blocks/circuits.py):

  1. POINT-IN-TIME   the band on session t comes from a file dated before t, and the hit flag from the list dated t: rewriting
                     every band file dated on or after the cut and every hit list dated after it never changes a row up to the cut.
                     Reading the SAME-DAY band file fails this.
  2. TRUNCATION      adding future sessions and files never changes past rows.
  3. NO ZERO-FILL    a session with no published hit list, and a symbol absent from the previous band file, are NaN and flagged,
                     never 0 and never the last known band carried forward; a real 0 (published list, no hit) stays 0.0.
  4. AGREES WITH A LOOP-BASED REFERENCE on random worlds with weekend sessions, holidays, missing files and band changes.

Each is a violation counter run on the real code (must be 0) and on a deliberately buggy implementation of the failure it exists to
prevent (must be > 0).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.blocks.circuits import compute_circuit_features

COLS = ["price_band_pct", "is_no_band", "band_is_inferred", "hit_upper_circuit", "hit_lower_circuit"]


def _world(seed: int, n_sessions: int = 160):
    rng = np.random.default_rng(seed)
    sessions, d = [], date(2025, 1, 1)
    while len(sessions) < n_sessions:
        if d.weekday() < 5 and rng.random() > 0.04:
            sessions.append(d)                                     # weekday, with random holidays
        elif d.weekday() >= 5 and rng.random() < 0.04:
            sessions.append(d)                                     # a special weekend session
        d += timedelta(days=1)
    band, rows = 5.0, []
    for s in sessions:
        if rng.random() < 0.06:
            band = float(rng.choice([2, 5, 10, 20]))
        if rng.random() > 0.08:                                    # some files lack this symbol (or do not exist)
            no_band = rng.random() < 0.05
            rows.append((pd.Timestamp(s), np.nan if no_band else band, no_band))
    bands = pd.DataFrame(rows, columns=["DATE", "BAND_PCT", "IS_NO_BAND"])
    published = {pd.Timestamp(s) for s in sessions if rng.random() > 0.07}
    hit_rows = [(pd.Timestamp(s), rng.choice(["H", "L"])) for s in sessions if pd.Timestamp(s) in published and rng.random() < 0.15]
    hit_rows += [(pd.Timestamp(s), "H") for s in sessions[::23] if pd.Timestamp(s) in published]
    hit_rows += [(pd.Timestamp(s), "L") for s in sessions[::23] if pd.Timestamp(s) in published]          # both bands, same day
    hits = pd.DataFrame(hit_rows, columns=["DATE", "HIT"]).drop_duplicates()
    spine = [s for s in sessions if rng.random() > 0.05]           # the symbol misses some sessions (suspension)
    return sessions, spine, bands, hits, published


def _real(world, **override):
    sessions, spine, bands, hits, published = world
    return compute_circuit_features(override.get("spine", spine), override.get("bands", bands), override.get("hits", hits),
                                    override.get("published", published), override.get("sessions", sessions))


def _brute_force(world):
    sessions, spine, bands, hits, published = world
    order = [pd.Timestamp(s) for s in sessions]
    band_by = {r.DATE: (r.BAND_PCT, r.IS_NO_BAND) for r in bands.itertuples()}
    up = {r.DATE for r in hits.itertuples() if r.HIT == "H"}
    lo = {r.DATE for r in hits.itertuples() if r.HIT == "L"}
    out = []
    for s in spine:
        t = pd.Timestamp(s)
        i = order.index(t) if t in order else None
        prev = order[i - 1] if i is not None and i >= 1 else None
        if prev is not None and prev in band_by:
            pct, nb = band_by[prev]
            out.append((t, pct, float(nb), 0.0) + ((float(t in up), float(t in lo)) if t in published else (np.nan, np.nan)))
        else:
            out.append((t, np.nan, np.nan, np.nan) + ((float(t in up), float(t in lo)) if t in published else (np.nan, np.nan)))
    return pd.DataFrame(out, columns=["date", *COLS])


def _same(a, b):
    return np.isclose(a[COLS].to_numpy(float), b[COLS].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)


# ── buggy implementations ────────────────────────────────────────────────────────────────────────────────────────────

def _buggy_same_day_band(world):
    """LEAK: the band for session t is read from the file dated t itself (published that evening)."""
    sessions, spine, bands, hits, published = world
    out = _real(world)
    by = bands.set_index("DATE")
    pct = by["BAND_PCT"].reindex(pd.DatetimeIndex(spine)).to_numpy(float)
    out["price_band_pct"] = pct
    out["is_no_band"] = by["IS_NO_BAND"].reindex(pd.DatetimeIndex(spine)).astype("float64").to_numpy()
    out["is_missing_circuit_band"] = np.isnan(out["is_no_band"])
    return out


def _buggy_carry_forward(world):
    """STALE CARRY: a missing previous-session file is replaced by the last band seen."""
    out = _real(world)
    for c in ("price_band_pct", "is_no_band", "band_is_inferred"):
        out[c] = out[c].ffill()
    out["is_missing_circuit_band"] = out["is_no_band"].isna()
    return out


def _buggy_zero_fill_hits(world):
    out = _real(world)
    out[["hit_upper_circuit", "hit_lower_circuit"]] = out[["hit_upper_circuit", "hit_lower_circuit"]].fillna(0.0)
    return out


def _buggy_tomorrows_hits(world):
    """LOOK-AHEAD: the hit flag on session t is the NEXT session's list."""
    out = _real(world)
    out[["hit_upper_circuit", "hit_lower_circuit"]] = out[["hit_upper_circuit", "hit_lower_circuit"]].shift(-1)
    return out


def _buggy_full_sample_scale(world):
    out = _real(world)
    v = out["price_band_pct"]
    out["price_band_pct"] = (v - v.mean()) / v.std()
    return out


# ── the invariants as violation counters ─────────────────────────────────────────────────────────────────────────────

def reference_violations(fn, world) -> int:
    return int((~_same(fn(world).reset_index(drop=True), _brute_force(world))).sum())


def future_rewrite_violations(fn, world, cut_index: int, seed: int = 0) -> int:
    """Rows on or before the cut that change when every band file dated on/after the cut and every hit list after it is rewritten."""
    sessions, spine, bands, hits, published = world
    cut = pd.Timestamp(sessions[cut_index])
    rng = np.random.default_rng(seed)
    later_bands = bands["DATE"] >= cut
    rewritten_bands = bands.copy()
    rewritten_bands.loc[later_bands, "BAND_PCT"] = rng.choice([2.0, 5.0, 10.0, 20.0], later_bands.sum())
    rewritten_bands.loc[later_bands, "IS_NO_BAND"] = False
    rewritten_hits = pd.concat([hits[hits["DATE"] <= cut], pd.DataFrame({"DATE": [pd.Timestamp(s) for s in sessions if pd.Timestamp(s) > cut][::2],
                                                                        "HIT": "H"})], ignore_index=True)
    rewritten_published = {d for d in published if d <= cut} | {pd.Timestamp(s) for s in sessions if pd.Timestamp(s) > cut}
    before = fn(world)
    after = fn((sessions, spine, rewritten_bands, rewritten_hits, rewritten_published))
    keep = before["date"] <= cut
    return int((~_same(before[keep].reset_index(drop=True), after[keep].reset_index(drop=True))).sum())


def truncation_violations(fn, world, cut_index: int) -> int:
    sessions, spine, bands, hits, published = world
    cut = pd.Timestamp(sessions[cut_index])
    full = fn(world)
    part = fn(([s for s in sessions if pd.Timestamp(s) <= cut], [s for s in spine if pd.Timestamp(s) <= cut], bands[bands["DATE"] <= cut],
               hits[hits["DATE"] <= cut], {d for d in published if d <= cut}))
    a = full[full["date"] <= cut].reset_index(drop=True)
    return int((~_same(a, part.reset_index(drop=True))).sum())


def zero_fill_violations(fn, world) -> int:
    ref = _brute_force(world)
    got = fn(world).reset_index(drop=True)
    invented = int(((got["hit_upper_circuit"] == 0) & ref["hit_upper_circuit"].isna()).sum()
                   + ((got["hit_lower_circuit"] == 0) & ref["hit_lower_circuit"].isna()).sum())
    stale_band = int((got["is_no_band"].notna() & ref["is_no_band"].isna()).sum())
    flag_hits = int((got["is_missing_circuit_hits"] != got["hit_upper_circuit"].isna()).sum())
    flag_band = int((got["is_missing_circuit_band"] != got["is_no_band"].isna()).sum())
    return invented + stale_band + flag_hits + flag_band


SEEDS = [1, 2, 3, 4, 5, 6]


class TestAgainstReference:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_matches_loop_based_truth(self, seed):
        assert reference_violations(_real, _world(seed)) == 0

    def test_the_worlds_exercise_the_hard_cases(self):
        """Weekend sessions, band changes, missing files, both-bands days, unpublished lists must all occur, or agreement means little."""
        worlds = [_world(seed) for seed in SEEDS]
        assert any(any(pd.Timestamp(s).weekday() >= 5 for s in w[0]) for w in worlds)
        assert any(w[2]["BAND_PCT"].nunique() > 1 for w in worlds)
        assert any(len(w[4]) < len(w[0]) for w in worlds)
        assert any((w[3].groupby("DATE")["HIT"].nunique() == 2).any() for w in worlds)
        assert any(_brute_force(w)["is_no_band"].isna().any() for w in worlds)


class TestPointInTime:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [40, 90, 130])
    def test_rewriting_the_future_never_changes_an_earlier_row(self, seed, cut):
        assert future_rewrite_violations(_real, _world(seed), cut) == 0

    def test_reading_the_same_days_band_file_is_caught(self):
        assert any(future_rewrite_violations(_buggy_same_day_band, _world(seed), 90) > 0 for seed in SEEDS)

    def test_taking_tomorrows_hit_list_is_caught(self):
        assert any(future_rewrite_violations(_buggy_tomorrows_hits, _world(seed), 90) > 0 for seed in SEEDS)


class TestTruncation:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [30, 80, 140])
    def test_adding_future_data_never_changes_the_past(self, seed, cut):
        assert truncation_violations(_real, _world(seed), cut) == 0

    def test_a_full_sample_statistic_is_caught(self):
        assert truncation_violations(_buggy_full_sample_scale, _world(2), 80) > 0


class TestNoZeroFill:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_never_invents_a_zero_or_carries_a_stale_band(self, seed):
        assert zero_fill_violations(_real, _world(seed)) == 0

    def test_zero_filling_unpublished_hit_days_is_caught(self):
        assert any(zero_fill_violations(_buggy_zero_fill_hits, _world(seed)) > 0 for seed in SEEDS)

    def test_carrying_the_last_band_forward_is_caught(self):
        assert any(zero_fill_violations(_buggy_carry_forward, _world(seed)) > 0 for seed in SEEDS)

    def test_a_published_list_without_the_stock_is_a_real_zero(self):
        out = _real(_world(3))
        assert (out["hit_upper_circuit"] == 0).any() and (out["hit_lower_circuit"] == 0).any()
