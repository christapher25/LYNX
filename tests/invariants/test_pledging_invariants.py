"""
tests/invariants/test_pledging_invariants.py

The Stage 10 invariants for the pledging block (features/blocks/pledging.py):

  1. POINT-IN-TIME   no row depends on a filing that had not been broadcast before that session's 19:00 IST run: dropping every
                     filing broadcast after a cut leaves every row up to the cut unchanged. Keying on the quarter a filing
                     DESCRIBES (or ignoring the cut-off) fails this.
  2. TRUNCATION      adding future filings and sessions never changes past rows.
  3. NO ZERO-FILL    an unknown filing, a filing with no promoter group and a session before the first filing are NaN and
                     flagged, never 0 and never the previous quarter's value; a real 0% stays 0.0.
  4. AGREES WITH A LOOP-BASED REFERENCE on random filing sets with revisions, weekends, late-evening broadcasts, parse failures.

Each is a violation counter run on the real code (must be 0) and on a deliberately buggy implementation of the failure it exists
to prevent (must be > 0).
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from data.publication import FEATURE_CUTOFF_IST, IST
from features.blocks.pledging import compute_pledging_features

COLS = ["pledged_pct_of_promoter", "pledged_pct_chg", "pledge_rising", "pledging_age_days"]


def _world(seed: int, n_sessions: int = 420):
    rng = np.random.default_rng(seed)
    sessions, d = [], date(2024, 1, 1)
    while len(sessions) < n_sessions:
        if d.weekday() < 5 and rng.random() > 0.03:
            sessions.append(d)
        d += timedelta(days=1)
    quarter_ends = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30)]
    rows = []
    for qe in quarter_ends:
        for revision in (0, 1) if rng.random() < 0.3 else (0,):
            delay = int(rng.integers(8, 60)) + (int(rng.integers(30, 70)) if revision else 0)
            broadcast = datetime.combine(qe + timedelta(days=delay), time(int(rng.integers(6, 23)), int(rng.integers(0, 60))), tzinfo=IST)
            kind = rng.random()
            promoter = float(rng.integers(1_000, 1_000_000))
            if kind < 0.1:
                rows.append({"PERIOD_END": pd.Timestamp(qe), "BROADCAST_IST": broadcast.isoformat(), "PROMOTER_SHARES": None,
                             "ENCUMBERED_SHARES": None, "HAS_PROMOTER_GROUP": None, "PARSE_ERROR": "404"})
            elif kind < 0.2:
                rows.append({"PERIOD_END": pd.Timestamp(qe), "BROADCAST_IST": broadcast.isoformat(), "PROMOTER_SHARES": None,
                             "ENCUMBERED_SHARES": None, "HAS_PROMOTER_GROUP": False, "PARSE_ERROR": None})
            else:
                encumbered = 0.0 if rng.random() < 0.25 else float(int(promoter * rng.random()))
                rows.append({"PERIOD_END": pd.Timestamp(qe), "BROADCAST_IST": broadcast.isoformat(), "PROMOTER_SHARES": promoter,
                             "ENCUMBERED_SHARES": encumbered, "HAS_PROMOTER_GROUP": True, "PARSE_ERROR": None})
    return sessions, pd.DataFrame(rows)


def _ratio(rec):
    """A filing's ratio, or NaN unless it parsed and has a promoter group with shares. (pandas stores a missing string as NaN.)"""
    has_group = not pd.isna(rec["HAS_PROMOTER_GROUP"]) and bool(rec["HAS_PROMOTER_GROUP"])
    ok = (has_group and pd.isna(rec["PARSE_ERROR"]) and not pd.isna(rec["PROMOTER_SHARES"]) and rec["PROMOTER_SHARES"] > 0
          and not pd.isna(rec["ENCUMBERED_SHARES"]))
    return 100.0 * rec["ENCUMBERED_SHARES"] / rec["PROMOTER_SHARES"] if ok else np.nan


def _usable_day(broadcast_iso, sessions, ignore_cutoff=False):
    when = datetime.fromisoformat(broadcast_iso)
    for s in sessions:
        run = datetime.combine(s, time(23, 59, 59) if ignore_cutoff else FEATURE_CUTOFF_IST, tzinfo=IST)
        if run >= when:
            return s
    return None


def _brute_force(sessions, filings, key=None):
    """Independent loop-based truth. key(rec) -> the day a filing becomes visible (default: broadcast + 19:00 cut-off)."""
    key = key or (lambda rec: _usable_day(rec["BROADCAST_IST"], sessions))
    recs = [dict(r) for r in filings.to_dict("records") if r["BROADCAST_IST"] is not None]
    for r in recs:
        r["_usable"] = key(r)
        r["_published"] = datetime.fromisoformat(r["BROADCAST_IST"])
    out = []
    for s in sessions:
        visible = sorted((r for r in recs if r["_usable"] is not None and r["_usable"] <= s), key=lambda r: (r["_usable"], r["_published"]))
        by_period = {}
        for r in visible:
            by_period[r["PERIOD_END"]] = _ratio(r)
        if not by_period:
            out.append((pd.Timestamp(s), np.nan, np.nan, np.nan, np.nan))
            continue
        periods = sorted(by_period)
        cur = by_period[periods[-1]]
        prev = by_period[periods[-2]] if len(periods) > 1 else np.nan
        chg = cur - prev
        rising = np.nan if np.isnan(chg) else float(chg > 1e-9)
        age = (pd.Timestamp(s) - periods[-1]).days if not np.isnan(cur) else np.nan
        out.append((pd.Timestamp(s), cur, chg, rising, age))
    return pd.DataFrame(out, columns=["date", *COLS])


def _same(a, b):
    return np.isclose(a[COLS].to_numpy(float), b[COLS].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)


def _real(sessions, filings):
    return compute_pledging_features(sessions, filings, sessions)


# ── buggy implementations ────────────────────────────────────────────────────────────────────────────────────────────

def _buggy_period_keyed(sessions, filings):
    """THE CLASSIC LEAK: a filing is visible from the quarter it describes, not from when it was broadcast."""
    truth = _brute_force(sessions, filings, key=lambda r: next((s for s in sessions if pd.Timestamp(s) >= r["PERIOD_END"]), None))
    out = _real(sessions, filings)
    out[COLS] = truth[COLS].to_numpy()
    return out


def _buggy_no_cutoff(sessions, filings):
    """A filing broadcast at 22:30 is treated as public in that evening's 19:00 run."""
    truth = _brute_force(sessions, filings, key=lambda r: _usable_day(r["BROADCAST_IST"], sessions, ignore_cutoff=True))
    out = _real(sessions, filings)
    out[COLS] = truth[COLS].to_numpy()
    return out


def _buggy_fallback(sessions, filings):
    """STALE CARRY: an unknown current filing is replaced by the last known ratio."""
    out = _real(sessions, filings)
    out["pledged_pct_of_promoter"] = out["pledged_pct_of_promoter"].ffill()
    out["is_missing_pledging"] = out["pledged_pct_of_promoter"].isna()
    return out


def _buggy_zero_fill(sessions, filings):
    out = _real(sessions, filings)
    out[COLS] = out[COLS].fillna(0.0)
    return out


def _buggy_latest_broadcast_wins(sessions, filings):
    """A restated OLD quarter displaces the newer quarter as the current filing."""
    recs = [dict(r) for r in filings.to_dict("records") if r["BROADCAST_IST"] is not None]
    out = _real(sessions, filings)
    rows = []
    for s in sessions:
        visible = sorted((r for r in recs if _usable_day(r["BROADCAST_IST"], sessions) is not None and _usable_day(r["BROADCAST_IST"], sessions) <= s),
                         key=lambda r: datetime.fromisoformat(r["BROADCAST_IST"]))
        rows.append(_ratio(visible[-1]) if visible else np.nan)
    out["pledged_pct_of_promoter"] = rows
    return out


def _buggy_full_sample_zscore(sessions, filings):
    out = _real(sessions, filings)
    v = out["pledged_pct_of_promoter"]
    out["pledged_pct_of_promoter"] = (v - v.mean()) / v.std()
    return out


# ── the invariants as violation counters ─────────────────────────────────────────────────────────────────────────────

def against_reference_violations(fn, sessions, filings) -> int:
    return int((~_same(fn(sessions, filings).reset_index(drop=True), _brute_force(sessions, filings))).sum())


def unpublished_filing_violations(fn, sessions, filings, cut_index: int) -> int:
    """Rows up to the cut that change when every filing not yet broadcast by the cut's 19:00 run is removed."""
    cut = sessions[cut_index]
    run = datetime.combine(cut, FEATURE_CUTOFF_IST, tzinfo=IST)
    public = filings[[b is not None and datetime.fromisoformat(b) <= run for b in filings["BROADCAST_IST"]]]
    full = fn(sessions, filings)
    partial = fn(sessions, public)
    keep = full["date"] <= pd.Timestamp(cut)
    return int((~_same(full[keep].reset_index(drop=True), partial[keep].reset_index(drop=True))).sum())


def truncation_violations(fn, sessions, filings, cut_index: int) -> int:
    cut = sessions[cut_index]
    full = fn(sessions, filings)
    part = fn([s for s in sessions if s <= cut], filings)
    a = full[full["date"] <= pd.Timestamp(cut)].reset_index(drop=True)
    return int((~_same(a, part.reset_index(drop=True))).sum())


def zero_fill_violations(fn, sessions, filings) -> int:
    ref = _brute_force(sessions, filings)
    got = fn(sessions, filings).reset_index(drop=True)
    invented = int(((got["pledged_pct_of_promoter"] == 0) & ~(ref["pledged_pct_of_promoter"] == 0)).sum())
    flagged_with_value = int((got["is_missing_pledging"] & got["pledged_pct_of_promoter"].notna()).sum())
    unflagged_nan = int((~got["is_missing_pledging"] & got["pledged_pct_of_promoter"].isna()).sum())
    stale = int((got["pledged_pct_of_promoter"].notna() & ref["pledged_pct_of_promoter"].isna()).sum())
    return invented + flagged_with_value + unflagged_nan + stale


SEEDS = [1, 2, 3, 4, 5, 6]


class TestAgainstReference:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_matches_loop_based_truth(self, seed):
        sessions, filings = _world(seed)
        assert against_reference_violations(_real, sessions, filings) == 0

    def test_the_reference_can_tell_period_keying_from_broadcast_keying(self):
        sessions, filings = _world(1)
        assert (~_same(_buggy_period_keyed(sessions, filings), _real(sessions, filings))).any()

    def test_the_reference_sees_the_real_zero_percent_and_the_unknown_filings(self):
        """Both must occur in the random worlds, or 'agrees with the reference' would say nothing about them."""
        worlds = [_brute_force(*_world(seed)) for seed in SEEDS]
        assert any((w["pledged_pct_of_promoter"] == 0).any() for w in worlds)
        assert any(w["pledged_pct_of_promoter"].isna().any() and w["pledged_pct_of_promoter"].notna().any() for w in worlds)


class TestPointInTime:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [100, 220, 330])
    def test_filings_not_yet_broadcast_never_reach_an_earlier_row(self, seed, cut):
        sessions, filings = _world(seed)
        assert unpublished_filing_violations(_real, sessions, filings, cut) == 0

    def test_keying_on_the_period_a_filing_describes_is_caught(self):
        assert any(unpublished_filing_violations(_buggy_period_keyed, *_world(seed), 150) > 0 for seed in SEEDS)

    def test_ignoring_the_nightly_cut_off_is_caught(self):
        """A filing broadcast at 22:30 on the cut session itself: public for the NEXT run, not the one at 19:00 that evening."""
        sessions, filings = _world(1)
        evening = {"PERIOD_END": pd.Timestamp("2023-12-31"), "PROMOTER_SHARES": 1000.0, "ENCUMBERED_SHARES": 500.0,
                   "HAS_PROMOTER_GROUP": True, "PARSE_ERROR": None,
                   "BROADCAST_IST": datetime.combine(sessions[100], time(22, 30), tzinfo=IST).isoformat()}
        filings = pd.concat([filings, pd.DataFrame([evening])], ignore_index=True)
        assert unpublished_filing_violations(_real, sessions, filings, 100) == 0
        assert unpublished_filing_violations(_buggy_no_cutoff, sessions, filings, 100) > 0


class TestTruncation:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [90, 200, 320])
    def test_adding_future_data_never_changes_the_past(self, seed, cut):
        sessions, filings = _world(seed)
        assert truncation_violations(_real, sessions, filings, cut) == 0

    def test_a_full_sample_statistic_is_caught(self):
        sessions, filings = _world(2)
        assert truncation_violations(_buggy_full_sample_zscore, sessions, filings, 200) > 0


class TestNoZeroFill:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_never_invents_a_zero_or_carries_a_stale_value(self, seed):
        sessions, filings = _world(seed)
        assert zero_fill_violations(_real, sessions, filings) == 0

    def test_zero_filling_is_caught(self):
        assert any(zero_fill_violations(_buggy_zero_fill, *_world(seed)) > 0 for seed in SEEDS)

    def test_falling_back_to_the_previous_quarter_is_caught(self):
        assert any(zero_fill_violations(_buggy_fallback, *_world(seed)) > 0 for seed in SEEDS)

    def test_a_restated_old_quarter_displacing_the_newer_one_is_caught(self):
        assert any(against_reference_violations(_buggy_latest_broadcast_wins, *_world(seed)) > 0 for seed in SEEDS)

    def test_a_real_zero_percent_survives(self):
        sessions, filings = _world(3)
        out = _real(sessions, filings)
        assert (out["pledged_pct_of_promoter"] == 0).any()
