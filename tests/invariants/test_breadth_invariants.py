"""
tests/invariants/test_breadth_invariants.py

The Stage 10 invariants for the breadth block (features/blocks/breadth.py):

  1. POINT-IN-TIME   a row depends only on prices dated on or before it and on its own year's universe: rewriting the
                     future (prices, or later years' membership) never changes an earlier row.
  2. TRUNCATION      adding future data never changes past rows.
  3. ADJUSTMENT      a future corporate action rescales every earlier adjusted close by one constant; "close > average"
                     for any session before it must not change. (This is what makes an adjusted panel safe to use.)
  4. NO ZERO-FILL    a session without evaluable stocks is NaN and flagged, never 0%; a real 0% stays 0%.
  5. AGREES WITH A LOOP-BASED REFERENCE on random panels with gaps, late listings and year-changing membership.

Each is a violation counter run on the real code (must be 0) and on a deliberately buggy implementation of the exact
failure it exists to prevent (must be > 0), so a green result can be trusted to mean something.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.blocks.breadth import compute_breadth_features

WINDOW = 5
COLS = ["universe_pct_above_200dma", "universe_n_evaluable"]


def _world(seed: int, n_symbols: int = 14, n_sessions: int = 70):
    """Random-walk panel spanning a year change, with row gaps, late listings and different members each year."""
    rng = np.random.default_rng(seed)
    sessions, d = [], date(2024, 11, 4)
    while len(sessions) < n_sessions:
        if d.weekday() < 5:
            sessions.append(d)
        d += timedelta(days=1)
    rows = []
    for k in range(n_symbols):
        start = int(rng.integers(0, 15))
        price = 100.0 * rng.uniform(0.3, 3)
        for i, s in enumerate(sessions):
            price *= float(np.exp(rng.normal(0, 0.03)))
            if i < start or rng.random() < 0.04:                                 # not yet listed / a suspension: no row
                continue
            rows.append((f"S{k:02d}", pd.Timestamp(s), price))
    panel = pd.DataFrame(rows, columns=["SYMBOL", "DATE", "ADJ_CLOSE"])
    symbols = sorted(panel["SYMBOL"].unique())
    members = {2024: set(rng.choice(symbols, size=10, replace=False)), 2025: set(rng.choice(symbols, size=10, replace=False)),
               2026: set(symbols)}
    return panel, members, sessions


def _real(panel, members, sessions, **kw):
    return compute_breadth_features(panel, members, sessions, window=WINDOW, min_evaluable=kw.get("min_evaluable", 1))


def _brute_force(panel, members, sessions, min_evaluable=1):
    """Independent truth by loops: for each session and member, the mean of its last WINDOW rows (current included)."""
    by_symbol = {sym: g.sort_values("DATE") for sym, g in panel.groupby("SYMBOL")}
    rows = []
    for s in sessions:
        s = pd.Timestamp(s)
        evaluable = above = 0
        seen_any = False
        for sym, g in by_symbol.items():
            upto = g[g["DATE"] <= s]
            if upto.empty or upto["DATE"].iloc[-1] != s:
                continue
            seen_any = True
            if sym not in members.get(s.year, ()) or len(upto) < WINDOW:
                continue
            window = upto["ADJ_CLOSE"].to_numpy()[-WINDOW:]
            if not (window > 0).all():
                continue
            evaluable += 1
            above += int(window[-1] > window.mean() * (1.0 + 1e-9))
        pct = 100.0 * above / evaluable if evaluable >= min_evaluable and evaluable > 0 else np.nan
        rows.append((s, pct, float(evaluable) if seen_any else np.nan))
    return pd.DataFrame(rows, columns=["date", *COLS])


def _same(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    return np.isclose(a[COLS].to_numpy(float), b[COLS].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)


# ── buggy implementations ────────────────────────────────────────────────────────────────────────────────────────────

def _buggy_centred(panel, members, sessions):
    """LOOK-AHEAD: the average is centred, so each session's average includes future rows."""
    out = compute_breadth_features(panel, members, sessions, window=WINDOW, min_evaluable=1)
    work = panel.sort_values(["SYMBOL", "DATE"]).copy()
    work["sma"] = work.groupby("SYMBOL")["ADJ_CLOSE"].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW, center=True).mean())
    work["ev"] = work["sma"].notna() & work.apply(lambda r: r["SYMBOL"] in members.get(r["DATE"].year, ()), axis=1)
    work["ab"] = work["ev"] & (work["ADJ_CLOSE"] > work["sma"])
    daily = work.groupby("DATE").agg(n=("ev", "sum"), a=("ab", "sum")).reindex(pd.DatetimeIndex(sessions))
    out["universe_pct_above_200dma"] = (100 * daily["a"] / daily["n"]).where(daily["n"] >= 1).to_numpy()
    out["universe_n_evaluable"] = daily["n"].to_numpy(float)
    out["is_missing_breadth"] = out["universe_pct_above_200dma"].isna()
    return out


def _buggy_latest_universe_for_all_years(panel, members, sessions):
    """SURVIVORSHIP / PIT-MEMBERSHIP LEAK: today's universe applied to every year."""
    newest = members[max(members)]
    return compute_breadth_features(panel, {y: newest for y in members}, sessions, window=WINDOW, min_evaluable=1)


def _buggy_zero_fill(panel, members, sessions):
    out = compute_breadth_features(panel, members, sessions, window=WINDOW, min_evaluable=1)
    out[COLS] = out[COLS].fillna(0.0)
    return out


def _buggy_full_sample_rank(panel, members, sessions):
    """FULL-SAMPLE STATISTIC: express breadth as a percentile of the whole history; past rows move when data is added."""
    out = compute_breadth_features(panel, members, sessions, window=WINDOW, min_evaluable=1)
    out["universe_pct_above_200dma"] = out["universe_pct_above_200dma"].rank(pct=True) * 100
    return out


# ── the invariants as violation counters ─────────────────────────────────────────────────────────────────────────────

def future_rewrite_violations(fn, panel, members, sessions, cut_index: int, seed: int = 0) -> int:
    """Rows on or before the cut that change when prices after the cut are rewritten and later years' membership is redrawn."""
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(sessions[cut_index])
    rewritten = panel.copy()
    later = rewritten["DATE"] > cut
    rewritten.loc[later, "ADJ_CLOSE"] = rewritten.loc[later, "ADJ_CLOSE"] * rng.uniform(0.05, 20, later.sum())
    symbols = sorted(panel["SYMBOL"].unique())
    redrawn = {y: (m if y <= cut.year else set(rng.choice(symbols, size=4, replace=False))) for y, m in members.items()}
    before, after = fn(panel, members, sessions), fn(rewritten, redrawn, sessions)
    keep = before["date"] <= cut
    return int((~_same(before[keep].reset_index(drop=True), after[keep].reset_index(drop=True))).sum())


def truncation_violations(fn, panel, members, sessions, cut_index: int) -> int:
    cut = pd.Timestamp(sessions[cut_index])
    full = fn(panel, members, sessions)
    part = fn(panel[panel["DATE"] <= cut], members, [s for s in sessions if pd.Timestamp(s) <= cut])
    a = full[full["date"] <= cut].reset_index(drop=True)
    return int((~_same(a, part.reset_index(drop=True))).sum())


def adjustment_violations(fn, panel, members, sessions, split_index: int, factor: float) -> int:
    """
    A later corporate action rescales all of one stock's closes BEFORE it by `factor`. Sessions before the action must be
    unchanged: the window, the close and the average all move together, so "close > average" cannot change.
    """
    split_day = pd.Timestamp(sessions[split_index])
    symbol = panel["SYMBOL"].value_counts().index[0]
    rescaled = panel.copy()
    rows = (rescaled["SYMBOL"] == symbol) & (rescaled["DATE"] < split_day)
    rescaled.loc[rows, "ADJ_CLOSE"] *= factor
    before = fn(panel, members, sessions)
    after = fn(rescaled, members, sessions)
    keep = before["date"] < split_day
    return int((~_same(before[keep].reset_index(drop=True), after[keep].reset_index(drop=True))).sum())


def _make_buggy_raw_close(raw_panel):
    """
    ADJUSTMENT BUG, as a factory: pairs the current close taken from the UNADJUSTED series (`raw_panel`, never rescaled)
    with an average computed from whichever panel it is handed. Handed the retroactively rescaled panel, a stock's
    pre-action sessions then compare a raw close with a rescaled average.
    """
    raw = raw_panel.set_index(["SYMBOL", "DATE"])["ADJ_CLOSE"]

    def buggy(panel, members, sessions):
        work = panel.sort_values(["SYMBOL", "DATE"]).copy()
        work["sma"] = work.groupby("SYMBOL")["ADJ_CLOSE"].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).mean())
        work["cmp"] = raw.reindex(pd.MultiIndex.from_frame(work[["SYMBOL", "DATE"]])).to_numpy()
        work["ev"] = work["sma"].notna() & work.apply(lambda r: r["SYMBOL"] in members.get(r["DATE"].year, ()), axis=1)
        work["ab"] = work["ev"] & (work["cmp"] > work["sma"])
        daily = work.groupby("DATE").agg(n=("ev", "sum"), a=("ab", "sum")).reindex(pd.DatetimeIndex(sessions))
        out = pd.DataFrame({"date": pd.DatetimeIndex(sessions),
                            "universe_pct_above_200dma": (100 * daily["a"] / daily["n"]).where(daily["n"] >= 1).to_numpy(),
                            "universe_n_evaluable": daily["n"].to_numpy(float)})
        out["is_missing_breadth"] = out["universe_pct_above_200dma"].isna()
        return out
    return buggy


def zero_fill_violations(fn, panel, members, sessions) -> int:
    ref = _brute_force(panel, members, sessions)
    got = fn(panel, members, sessions).reset_index(drop=True)
    invented = int(((got["universe_pct_above_200dma"] == 0) & ~(ref["universe_pct_above_200dma"] == 0)).sum())
    flagged_with_value = int((got["is_missing_breadth"] & got["universe_pct_above_200dma"].notna()).sum())
    unflagged_nan = int((~got["is_missing_breadth"] & got["universe_pct_above_200dma"].isna()).sum())
    return invented + flagged_with_value + unflagged_nan


SEEDS = [1, 2, 3, 4]


class TestAgainstReference:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_matches_loop_based_truth_on_every_session(self, seed):
        panel, members, sessions = _world(seed)
        want = _brute_force(panel, members, sessions)
        got = _real(panel, members, sessions)
        assert _same(got, want).all()
        assert got["is_missing_breadth"].equals(want["universe_pct_above_200dma"].isna())

    def test_the_reference_can_tell_a_look_ahead_average_from_the_real_one(self):
        panel, members, sessions = _world(1)
        want = _brute_force(panel, members, sessions)
        assert (~_same(_buggy_centred(panel, members, sessions), want)).any()

    def test_the_reference_can_tell_a_single_snapshot_universe_from_yearly_membership(self):
        panel, members, sessions = _world(2)
        want = _brute_force(panel, members, sessions)
        assert (~_same(_buggy_latest_universe_for_all_years(panel, members, sessions), want)).any()


class TestPointInTime:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [25, 45, 60])
    def test_rewriting_the_future_never_changes_an_earlier_row(self, seed, cut):
        panel, members, sessions = _world(seed)
        assert future_rewrite_violations(_real, panel, members, sessions, cut) == 0

    def test_a_look_ahead_average_is_caught(self):
        panel, members, sessions = _world(1)
        assert future_rewrite_violations(_buggy_centred, panel, members, sessions, 45) > 0

    def test_keying_membership_on_the_newest_universe_is_caught(self):
        panel, members, sessions = _world(1)
        assert future_rewrite_violations(_buggy_latest_universe_for_all_years, panel, members, sessions, 25) > 0


class TestTruncation:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [20, 40, 60])
    def test_adding_future_data_never_changes_the_past(self, seed, cut):
        panel, members, sessions = _world(seed)
        assert truncation_violations(_real, panel, members, sessions, cut) == 0

    def test_a_full_sample_statistic_is_caught(self):
        panel, members, sessions = _world(2)
        assert truncation_violations(_buggy_full_sample_rank, panel, members, sessions, 40) > 0


class TestAdjustmentInvariance:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("factor", [0.5, 2.0, 0.1])
    def test_a_later_corporate_action_does_not_change_earlier_rows(self, seed, factor):
        panel, members, sessions = _world(seed)
        assert adjustment_violations(_real, panel, members, sessions, 50, factor) == 0

    def test_mixing_a_raw_close_with_an_adjusted_average_is_caught(self):
        panel, members, sessions = _world(1)
        assert adjustment_violations(_make_buggy_raw_close(panel), panel, members, sessions, 50, 0.1) > 0
        # and the mutant agrees with the real code when nothing was rescaled, so the difference above is the bug alone
        unrescaled = _make_buggy_raw_close(panel)(panel, members, sessions)
        assert _same(unrescaled, _real(panel, members, sessions)).all()


class TestNoZeroFill:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_never_invents_a_zero(self, seed):
        panel, members, sessions = _world(seed)
        assert zero_fill_violations(_real, panel, members, sessions) == 0

    def test_filling_missing_sessions_with_zero_is_caught(self):
        panel, members, sessions = _world(3)
        assert zero_fill_violations(_buggy_zero_fill, panel, members, sessions) > 0

    def test_a_real_zero_percent_survives(self):
        """Every evaluable stock falling: 0% is a true reading and must not be confused with missing."""
        sessions = [date(2025, 1, 1) + timedelta(days=i) for i in range(12)]
        panel = pd.DataFrame([(f"S{k}", pd.Timestamp(s), 100.0 - i) for k in range(4) for i, s in enumerate(sessions)],
                             columns=["SYMBOL", "DATE", "ADJ_CLOSE"])
        out = compute_breadth_features(panel, {2025: {f"S{k}" for k in range(4)}}, sessions, window=WINDOW, min_evaluable=1)
        assert (out["universe_pct_above_200dma"].iloc[WINDOW - 1:] == 0.0).all() and not out["is_missing_breadth"].iloc[WINDOW - 1:].any()
