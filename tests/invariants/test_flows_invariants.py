"""
tests/invariants/test_flows_invariants.py

The three invariants every Stage 10 block must satisfy, for the flows block:

  1. POINT-IN-TIME  no value is visible in a row before it was public.
  2. TRUNCATION     adding future data never changes past rows (as in Stage 9's indicators).
  3. NO ZERO-FILL   missing is NaN and flagged; a value is 0 only where the source was 0.

Each invariant is written as a check function that COUNTS violations for any implementation, and is run twice: against the
real code (must be 0) and against a deliberately buggy implementation of the exact failure it exists to prevent (must be
> 0). A check that cannot fail proves nothing; the second run is what shows this one can.
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from data.publication import FEATURE_CUTOFF_IST, IST
from features.blocks.flows import (
    FPI_PUBLICATION_TIME_IST,
    compute_dii_flow_features,
    compute_fii_flow_features,
)


def _world(seed: int, n: int = 90, gap_rate: float = 0.08):
    """Random weekday sessions with a few holidays, and a daily FII series with random gaps and some exact zeros."""
    rng = np.random.default_rng(seed)
    days, d = [], date(2025, 1, 1)
    while len(days) < n:
        if d.weekday() < 5 and rng.random() > 0.05:
            days.append(d)
        d += timedelta(days=1)
    values = rng.normal(0, 1500, n).round(2)
    values[rng.random(n) < 0.05] = 0.0
    values[rng.random(n) < gap_rate] = np.nan
    return days, pd.DataFrame({"date": pd.to_datetime(days), "net_cr": values})


# ── reference / buggy implementations ────────────────────────────────────────────────────────────────────────────────

def _brute_force_fii(daily, sessions):
    """
    Independent, loop-based truth. A figure becomes public on the first session strictly after its reporting day, so
    session s shows the newest figure whose reporting day t satisfies previous_session <= t < s. Nothing older is carried
    forward: if that figure is absent the row is NaN (missing means missing), never a stale value presented as current.
    The 5- and 20-day sums cover the last 5 / 20 REPORTED days (a day with no figure is skipped, not a zero slot), and are
    NaN only while fewer than k reported days exist.
    """
    net = {pd.Timestamp(r.date): r.net_cr for r in daily.itertuples() if not np.isnan(r.net_cr)}
    order = sorted(net)
    rows = []
    previous = None
    for s in sessions:
        s = pd.Timestamp(s)
        earlier = [t for t in net if t < s and (previous is None or t >= previous)]
        previous = s
        if not earlier:
            rows.append((s, np.nan, np.nan, np.nan, pd.NaT))
            continue
        t = max(earlier)
        pos = order.index(t)
        total = lambda k: sum(net[order[pos - j]] for j in range(k)) if pos - k + 1 >= 0 else np.nan
        rows.append((s, net[t], total(5), total(20), t))
    return pd.DataFrame(rows, columns=["date", "fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr", "source_date"])


def _buggy_same_day(daily, sessions):
    """THE LEAK: key each figure on the day it describes, as if it were public that same session."""
    out = pd.DataFrame({"date": pd.to_datetime(sessions)}).merge(
        daily.rename(columns={"net_cr": "fpi_equity_net_cr"}).assign(source_date=lambda d: d["date"]), on="date", how="left")
    out["fpi_equity_net_5d_cr"] = out["fpi_equity_net_cr"].rolling(5, min_periods=5).sum()
    out["fpi_equity_net_20d_cr"] = out["fpi_equity_net_cr"].rolling(20, min_periods=20).sum()
    out["is_missing_fii_flows"] = out["fpi_equity_net_cr"].isna()
    return out


def _buggy_centered(daily, sessions):
    """LOOK-AHEAD: the 5-session sum is centred, so today's row contains two future days."""
    out = compute_fii_flow_features(daily, sessions)
    out["fpi_equity_net_5d_cr"] = out["fpi_equity_net_cr"].rolling(5, min_periods=5, center=True).sum()
    return out


def _buggy_session_slot_windows(daily, sessions):
    """DATA LOSS: windows count SESSIONS, so a day with no report is a NaN slot that blanks the next 5 / 20 sums."""
    out = compute_fii_flow_features(daily, sessions)
    series = daily.set_index("date")["net_cr"].reindex(pd.DatetimeIndex(sessions))
    for col, k in (("fpi_equity_net_5d_cr", 5), ("fpi_equity_net_20d_cr", 20)):
        rolled = series.rolling(k, min_periods=k).sum()
        source = out["source_date"].map(rolled)
        out[col] = source.to_numpy()
    return out


def _buggy_zero_fill(daily, sessions):
    """ZERO-FILL: a missing figure becomes 0 (reads as 'foreign investors did nothing')."""
    out = compute_fii_flow_features(daily, sessions)
    out[["fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"]] = out[["fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"]].fillna(0.0)
    return out


# ── the three invariants, as violation counters ──────────────────────────────────────────────────────────────────────

def pit_violations(fn, daily, sessions) -> int:
    """Rows carrying a figure whose reporting day is not strictly before the row's session, PLUS rows that disagree with
    the independent brute-force reference (a value from the wrong day is also a leak)."""
    out = fn(daily, sessions)
    shown = out.dropna(subset=["source_date"])
    leaked = int((shown["source_date"] >= shown["date"]).sum())
    ref = _brute_force_fii(daily, sessions).set_index("date")
    got = out.set_index("date")
    cols = ["fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"]
    wrong = int((~np.isclose(got[cols].to_numpy(float), ref[cols].to_numpy(float), equal_nan=True)).any(axis=1).sum())
    return leaked + wrong


def truncation_violations(fn, daily, sessions, cut_index: int) -> int:
    """Rows up to the cut that change when the future (sessions and reporting days after the cut) is removed."""
    cut = pd.Timestamp(sessions[cut_index])
    full = fn(daily, sessions)
    part = fn(daily[daily["date"] <= cut], [s for s in sessions if pd.Timestamp(s) <= cut])
    a = full[full["date"] <= cut].reset_index(drop=True)
    b = part.reset_index(drop=True)
    cols = ["fpi_equity_net_cr", "fpi_equity_net_5d_cr", "fpi_equity_net_20d_cr"]
    return int((~np.isclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)).any(axis=1).sum())


def zero_fill_violations(fn, daily, sessions) -> int:
    """Rows where a feature is 0 although the figure it should carry is missing or non-zero, and flagged rows with a value."""
    out = fn(daily, sessions)
    ref = _brute_force_fii(daily, sessions).set_index("date")
    got = out.set_index("date")
    invented = int(((got["fpi_equity_net_cr"] == 0) & (ref["fpi_equity_net_cr"] != 0)).sum())
    flagged_with_value = int((got["is_missing_fii_flows"] & got["fpi_equity_net_cr"].notna()).sum())
    unflagged_nan = int((~got["is_missing_fii_flows"] & got["fpi_equity_net_cr"].isna()).sum())
    return invented + flagged_with_value + unflagged_nan


SEEDS = [1, 2, 3, 4, 5]


class TestFiiPointInTime:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_has_no_leak_and_matches_brute_force(self, seed):
        days, daily = _world(seed)
        assert pit_violations(compute_fii_flow_features, daily, days) == 0

    def test_the_same_day_keyed_bug_is_caught(self):
        days, daily = _world(1)
        assert pit_violations(_buggy_same_day, daily, days) > 0

    def test_windowing_by_session_slots_is_caught(self):
        """Not a leak but the reference must still tell the two window rules apart, or agreeing with it means nothing."""
        days, daily = _world(1)
        assert pit_violations(_buggy_session_slot_windows, daily, days) > 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_figure_is_never_in_a_row_whose_run_started_before_it_was_public(self, seed):
        """Assumed publication instant = 10:00 IST on the first session after the reporting day; that row's 19:00 run must not precede it."""
        days, daily = _world(seed)
        out = compute_fii_flow_features(daily, days).dropna(subset=["source_date"])
        assert len(out) > 30
        sessions = pd.DatetimeIndex(days)
        for row in out.itertuples():
            first_session_after = sessions[sessions > row.source_date][0]
            published = datetime.combine(first_session_after.date(), FPI_PUBLICATION_TIME_IST, tzinfo=IST)
            run = datetime.combine(row.date.date(), FEATURE_CUTOFF_IST, tzinfo=IST)
            assert run >= published, f"{row.date.date()} used a figure public only at {published}"
            assert row.date == first_session_after, "a figure must land on the first session after its reporting day, no later"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_missing_figure_is_not_papered_over_with_an_older_one(self, seed):
        days, daily = _world(seed)
        out = compute_fii_flow_features(daily, days).set_index("date")
        previous_missing = daily["net_cr"].isna().shift(1, fill_value=False).to_numpy()
        stale_candidates = out[previous_missing]
        assert len(stale_candidates) > 0
        assert stale_candidates["fpi_equity_net_cr"].isna().all() and stale_candidates["is_missing_fii_flows"].all()


class TestFiiTruncation:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [30, 55, 80])
    def test_the_real_implementation_never_changes_the_past(self, seed, cut):
        days, daily = _world(seed)
        assert truncation_violations(compute_fii_flow_features, daily, days, cut) == 0

    def test_a_centred_window_is_caught(self):
        days, daily = _world(2)
        assert truncation_violations(_buggy_centered, daily, days, 55) > 0


class TestFiiNoZeroFill:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_real_implementation_never_invents_a_zero(self, seed):
        days, daily = _world(seed)
        assert zero_fill_violations(compute_fii_flow_features, daily, days) == 0

    def test_zero_filling_missing_figures_is_caught(self):
        days, daily = _world(3)
        assert zero_fill_violations(_buggy_zero_fill, daily, days) > 0

    def test_real_zero_inputs_survive_as_zero(self):
        days, daily = _world(4)
        assert (daily["net_cr"] == 0).any()
        out = compute_fii_flow_features(daily, days)
        assert (out["fpi_equity_net_cr"] == 0).any()


# ── DII: the same three invariants, keyed on the capture timestamp ──────────────────────────────────────────────────

def _dii_world(seed: int, n: int = 40):
    rng = np.random.default_rng(seed)
    days, d = [], date(2026, 6, 1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for i, day in enumerate(days):
        if rng.random() < 0.7:                                              # the crawler misses some days entirely
            hour = int(rng.integers(6, 23))
            captured = datetime.combine(day, time(hour, int(rng.integers(0, 60))), tzinfo=IST)
            if rng.random() < 0.3:                                          # ... and sometimes captures the next morning
                captured = datetime.combine(days[min(i + 1, n - 1)], time(7, 30), tzinfo=IST)
            rows.append((day, float(rng.normal(0, 3000)), captured.isoformat()))
    return days, pd.DataFrame(rows, columns=["DATE", "NET_VALUE_CR", "CAPTURED_AT_IST"])


def _dii_pit_violations(fn, captures, days) -> int:
    out = fn(captures, days)
    caps = captures.assign(pub=pd.to_datetime(captures["CAPTURED_AT_IST"], utc=True), DATE=pd.to_datetime(captures["DATE"]))
    bad = 0
    for row in out.dropna(subset=["dii_net_cr"]).itertuples():
        cutoff = pd.Timestamp(datetime.combine(row.date.date(), FEATURE_CUTOFF_IST, tzinfo=IST)).tz_convert("UTC")
        source = caps[caps["DATE"] == row.source_date]
        bad += int(source.empty or (source["pub"].min() > cutoff))            # value used by a run that started before the capture
    return bad


def _buggy_dii_by_reporting_date(captures, sessions):
    """LEAK: key the DII figure on the day it describes, ignoring when it was captured."""
    out = pd.DataFrame({"date": pd.to_datetime(sessions)}).merge(
        captures.assign(date=pd.to_datetime(captures["DATE"]))[["date", "NET_VALUE_CR"]].rename(columns={"NET_VALUE_CR": "dii_net_cr"}),
        on="date", how="left")
    out["source_date"] = out["date"].where(out["dii_net_cr"].notna())
    out["is_missing_dii_flows"] = out["dii_net_cr"].isna()
    return out


class TestDii:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_value_is_used_by_a_run_that_started_before_it_was_captured(self, seed):
        days, caps = _dii_world(seed)
        assert _dii_pit_violations(compute_dii_flow_features, caps, days) == 0

    def test_keying_on_the_reporting_day_is_caught(self):
        days, caps = _dii_world(1)
        assert _dii_pit_violations(_buggy_dii_by_reporting_date, caps, days) > 0

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("cut", [10, 25])
    def test_truncation_adding_later_captures_never_changes_earlier_rows(self, seed, cut):
        days, caps = _dii_world(seed)
        cutoff = pd.Timestamp(days[cut])
        full = compute_dii_flow_features(caps, days)
        part = compute_dii_flow_features(caps[pd.to_datetime(caps["DATE"]) <= cutoff], [d for d in days if pd.Timestamp(d) <= cutoff])
        a = full[full["date"] <= cutoff].reset_index(drop=True)
        assert np.allclose(a["dii_net_cr"].to_numpy(float), part["dii_net_cr"].to_numpy(float), equal_nan=True)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_missing_days_are_nan_and_flagged_never_zero(self, seed):
        days, caps = _dii_world(seed)
        out = compute_dii_flow_features(caps, days)
        assert out.loc[out["is_missing_dii_flows"], "dii_net_cr"].isna().all()
        assert not ((out["dii_net_cr"] == 0) & ~out["dii_net_cr"].isin(caps["NET_VALUE_CR"])).any()
        assert out["is_missing_dii_flows"].any()
