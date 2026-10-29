"""
tests/invariants/vanta/test_vanta_label_fills.py

VANTA Stage 13A invariants N1-N4 (docs/vanta/VANTA_SPEC.md), tested on tiny hand-built OHLC inputs with known answers.

  N1  No fill on a locked open     upper hit-list day with raw open == raw high -> NO_FILL_UPPER_LOCK, no exit price, no payoff.
  N2  Stops respect lower locks    stop triggers on d; d and d+1 lower-locked at close, d+2 not -> exit at d+2's close,
                                   STOP_DELAYED_LOWER_LOCK, lock_delay_sessions == 2. Lock detection uses RAW prices.
  N3  Same-day ambiguity -> stop   stop and target both inside one day's range -> exit at the stop.
  N4  Gap fills                    gap below stop fills at the OPEN (STOP_GAP); gap above target fills at the TARGET.

Every check is a function of a simulator. It is run against (a) the real vanta.labels.simulate_trade (skipped until the module
exists), (b) a correct toy (proves the check is not over-strict), and (c) deliberately broken toys (each must make the check
FAIL, proving the test can fail). (b) and (c) run with no vanta package present.

Common setup: ATR=2, k_s=1, k_t=2, h=5, entry at 100 -> stop 98, target 104, costs forced to zero so y_net == gross_return.
"""

import math

import pandas as pd
import pytest

from vanta_toys_labels import ZERO_COST, isnan, make_bars, make_hits, toy

SYM, ATR, KS, KT, H = "TESTCO", 2.0, 1.0, 2.0, 5
T = 0  # the decision session is position 0; entry is position 1


def _run(sim, ohlc, upper=(), lower=(), adj_ohlc=None, atr=ATR):
    bars = make_bars(ohlc, adj_ohlc=adj_ohlc)
    hits, cov = make_hits(bars, SYM, upper=upper, lower=lower)
    return bars, sim(SYM, bars, bars.index[T], H, KS, KT, atr, hits, cov, cost_fn=ZERO_COST)


T_DAY = (99, 101, 98, 100)
ENTRY_DAY = (100, 101, 99, 100)     # quiet entry day: low 99 > stop 98, high 101 < target 104
FLAT = (100, 101, 99, 100)


# ---------------------------------------------------------------- N1
def check_n1(sim):
    """N1: a day on the hit list as UPPER with raw open == raw high gives NO_FILL_UPPER_LOCK, no exit price, no payoff."""
    locked = [T_DAY, (100, 100, 97, 98), FLAT, FLAT, FLAT, FLAT]
    _, r = _run(sim, locked, upper=[1])
    assert r["fill_status"] == "NO_FILL_UPPER_LOCK"
    assert isnan(r["exit_price"]) and isnan(r["gross_return"]) and isnan(r["y_net"])
    assert isnan(r["entry_price_raw"])

    # both conjuncts are required: on the list but open != high -> filled
    not_flat_open = [T_DAY, (100, 103, 97, 98), FLAT, FLAT, FLAT, FLAT]
    _, r = _run(sim, not_flat_open, upper=[1])
    assert r["fill_status"] == "FILLED" and r["entry_price_raw"] == 100

    # open == high but NOT on the hit list -> filled (a stock merely opening at its high is not locked)
    _, r = _run(sim, locked, upper=[])
    assert r["fill_status"] == "FILLED"

    # a LOWER hit with open == high is not an upper lock
    _, r = _run(sim, locked, lower=[1])
    assert r["fill_status"] == "FILLED"

    # lock detection uses RAW prices: adjusted rounding noise must not defeat a genuine lock
    adj = [tuple(x * 0.5 for x in b) for b in locked]
    adj[1] = (50.0, 50.01, 48.5, 49.0)
    _, r = _run(sim, locked, upper=[1], adj_ohlc=adj, atr=1.0)
    assert r["fill_status"] == "NO_FILL_UPPER_LOCK"


def test_n1_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n1(sim)


def test_n1_reference_toy_passes():
    check_n1(toy())


@pytest.mark.parametrize("bug", ["ignore_upper_lock", "no_hit_list_check", "hit_list_only", "lock_on_adjusted", "payoff_on_no_fill"])
def test_n1_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n1(toy(bug))


# ---------------------------------------------------------------- N2
N2_BARS = [
    T_DAY,
    ENTRY_DAY,                    # 1  entry at 100, stop 98
    (99, 99, 95, 95),             # 2  d: low 95 <= 98 -> stop triggers; close == low (locked)
    (94, 94, 90, 90),             # 3  d+1: locked at close
    (91, 93, 90.5, 92),           # 4  d+2: close 92 != low 90.5 -> first session not lower-locked
    (92, 93, 91, 92),             # 5
]


def check_n2(sim):
    """N2: stop on d; d, d+1 lower-locked at close, d+2 not -> exit d+2 close, STOP_DELAYED_LOWER_LOCK, delay 2."""
    bars, r = _run(sim, N2_BARS, lower=[2, 3])
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK"
    assert r["lock_delay_sessions"] == 2
    assert r["exit_date"] == bars.index[4]
    assert r["exit_price"] == 92
    assert r["gross_return"] == pytest.approx(92 / 100 - 1)
    assert r["label_end_date"] == bars.index[4]

    # no lock on the hit list (close == low alone is not a lock) -> ordinary STOP at the stop, no delay
    bars, r = _run(sim, N2_BARS, lower=[])
    assert r["exit_reason"] == "STOP" and r["lock_delay_sessions"] == 0 and r["exit_price"] == 98

    # on the list but close != low -> not locked
    unlocked = list(N2_BARS)
    unlocked[2] = (99, 99, 95, 96)
    bars, r = _run(sim, unlocked, lower=[2, 3])
    assert r["exit_reason"] == "STOP" and r["lock_delay_sessions"] == 0

    # only the stop day is locked -> exit at the next session's close, delay 1
    bars, r = _run(sim, N2_BARS, lower=[2])
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK" and r["lock_delay_sessions"] == 1
    assert r["exit_date"] == bars.index[3] and r["exit_price"] == 90

    # lock detection uses RAW prices: adjusted rounding noise (adj close != adj low) must not hide the lock
    adj = [tuple(x * 0.5 for x in b) for b in N2_BARS]
    adj[2] = (49.5, 49.5, 47.5, 47.51)
    adj[3] = (47.0, 47.0, 45.0, 45.01)
    bars, r = _run(sim, N2_BARS, lower=[2, 3], adj_ohlc=adj, atr=1.0)
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK" and r["lock_delay_sessions"] == 2
    assert r["exit_date"] == bars.index[4]


def test_n2_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n2(sim)


def test_n2_reference_toy_passes():
    check_n2(toy())


@pytest.mark.parametrize("bug", ["ignore_lower_lock", "delay_off_by_one", "lock_on_adjusted"])
def test_n2_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n2(toy(bug))


# ---------------------------------------------------------------- N3
def check_n3(sim):
    """N3: stop and target both inside one day's range -> the stop is assumed hit first; exit at the stop."""
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (100, 105, 97, 101), FLAT, FLAT, FLAT])
    assert r["exit_reason"] == "STOP" and r["exit_price"] == 98
    assert r["exit_date"] == bars.index[2]
    assert r["y_net"] == pytest.approx(-0.02)

    # same ambiguity on the entry day itself
    bars, r = _run(sim, [T_DAY, (100, 105, 97, 101), FLAT, FLAT, FLAT, FLAT])
    assert r["exit_reason"] == "STOP" and r["exit_price"] == 98 and r["exit_date"] == bars.index[1]

    # control: only the target inside the range is a TARGET exit at the target
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (100, 105, 99, 104), FLAT, FLAT, FLAT])
    assert r["exit_reason"] == "TARGET" and r["exit_price"] == 104


def test_n3_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n3(sim)


def test_n3_reference_toy_passes():
    check_n3(toy())


@pytest.mark.parametrize("bug", ["target_first"])
def test_n3_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n3(toy(bug))


# ---------------------------------------------------------------- N4
def check_n4(sim):
    """N4: a gap below the stop fills at the OPEN (STOP_GAP); a gap above the target fills at the TARGET (no windfall)."""
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (95, 96, 94, 95), FLAT, FLAT, FLAT])
    assert r["exit_reason"] == "STOP_GAP"
    assert r["exit_price"] == 95 and r["exit_date"] == bars.index[2]
    assert r["gross_return"] == pytest.approx(-0.05)

    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (106, 108, 105, 107), FLAT, FLAT, FLAT])
    assert r["exit_reason"] == "TARGET"
    assert r["exit_price"] == 104 and r["exit_date"] == bars.index[2]
    assert r["gross_return"] == pytest.approx(0.04)

    # boundary: opening exactly at the stop is a gap-through at the open (== stop here, so both agree)
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (98, 99, 97, 98), FLAT, FLAT, FLAT])
    assert r["exit_price"] == 98


def check_n4_gap_down_lock(sim):
    """N4 (gap-down rule, user decision 2026-10-29): a gap-down session (open <= stop) that is itself lower-locked exits at the
    close of the first non-locked session (STOP_DELAYED_LOWER_LOCK, delay recorded); an unlocked gap-down exits at the open
    (STOP_GAP, delay 0)."""
    # gap day 2 opens 90 <= stop 98 and is locked (lower hit, close == low); day 3 also locked; day 4 not -> exit d+2 close
    locked_gap = [T_DAY, ENTRY_DAY, (90, 90, 88, 88), (85, 85, 83, 83), (84, 87, 83.5, 86), FLAT]
    bars, r = _run(sim, locked_gap, lower=[2, 3])
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK"
    assert r["exit_price"] == 86 and r["exit_date"] == bars.index[4]
    assert r["lock_delay_sessions"] == 2
    assert r["gross_return"] == pytest.approx(86 / 100 - 1)

    # lock detection uses RAW prices: adjusted rounding noise (adj close != adj low) must not hide the lock on the gap day
    adj = [tuple(x * 0.5 for x in b) for b in locked_gap]
    adj[2] = (45.0, 45.0, 44.0, 44.01)
    adj[3] = (42.5, 42.5, 41.5, 41.51)
    bars, r = _run(sim, locked_gap, lower=[2, 3], adj_ohlc=adj, atr=1.0)
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK" and r["lock_delay_sessions"] == 2
    assert r["exit_date"] == bars.index[4]
    assert r["gross_return"] == pytest.approx(43 / 50 - 1)

    # only the gap day locked -> next session's close, delay 1
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (90, 90, 88, 88), (89, 92, 88.5, 91), FLAT, FLAT], lower=[2])
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK" and r["lock_delay_sessions"] == 1
    assert r["exit_price"] == 91 and r["exit_date"] == bars.index[3]

    # same gap, NOT locked (no lower hit): fills at the open, STOP_GAP, no delay
    bars, r = _run(sim, locked_gap, lower=[])
    assert r["exit_reason"] == "STOP_GAP" and r["lock_delay_sessions"] == 0
    assert r["exit_price"] == 90 and r["exit_date"] == bars.index[2]

    # on the hit list but close != low -> not locked -> STOP_GAP at the open
    bars, r = _run(sim, [T_DAY, ENTRY_DAY, (90, 91, 88, 90), FLAT, FLAT, FLAT], lower=[2])
    assert r["exit_reason"] == "STOP_GAP" and r["exit_price"] == 90 and r["lock_delay_sessions"] == 0


def test_n4_gap_down_lock_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n4_gap_down_lock(sim)


def test_n4_gap_down_lock_reference_toy_passes():
    check_n4_gap_down_lock(toy())


@pytest.mark.parametrize("bug", ["gap_ignores_lock", "delay_unlocked_gap", "lock_on_adjusted"])
def test_n4_gap_down_lock_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n4_gap_down_lock(toy(bug))


def test_n4_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n4(sim)


def test_n4_reference_toy_passes():
    check_n4(toy())


@pytest.mark.parametrize("bug", ["gap_stop_at_stop", "gap_target_at_open"])
def test_n4_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n4(toy(bug))
