"""Extra unit tests for vanta.labels.simulate_trade beyond the N1-N4/N6 invariants: censoring, delisting, calendar handling,
lock_source, NaN-not-zero, no look-ahead, and the default cost legs."""

import math

import pandas as pd
import pytest

from vanta.labels import BROKERAGE_REFERENCE_NOTIONAL_INR, default_cost_fn, simulate_trade
from costs.model import get_rates

SYM = "TESTCO"
ZERO = lambda *a: (0.0, 0.0, 0.0)  # noqa: E731
T_DAY, ENTRY, FLAT = (99, 101, 98, 100), (100, 101, 99, 100), (100, 101, 99, 100)


def bars_of(ohlc, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(ohlc))
    cols = {}
    for pre, off in (("raw_", 0), ("", 0)):
        for k, name in enumerate(("open", "high", "low", "close")):
            cols[pre + name] = [r[k] for r in ohlc]
    return pd.DataFrame(cols, index=idx)


def hits_of(bars, upper=(), lower=()):
    rows = [{"date": bars.index[i], "symbol": SYM, "side": "UPPER"} for i in upper]
    rows += [{"date": bars.index[i], "symbol": SYM, "side": "LOWER"} for i in lower]
    return pd.DataFrame(rows, columns=["date", "symbol", "side"]), set(bars.index)


def run(ohlc, h=5, upper=(), lower=(), **kw):
    b = bars_of(ohlc)
    hits, cov = hits_of(b, upper, lower)
    cov = kw.pop("cov", cov)
    return b, simulate_trade(SYM, b, b.index[0], h, 1.0, 2.0, 2.0, hits, cov, cost_fn=kw.pop("cost_fn", ZERO), **kw)


def test_horizon_past_end_of_bars_is_censored_not_truncated():
    _, r = run([T_DAY, ENTRY, FLAT, FLAT], h=5)           # only 3 sessions after t, h=5, nothing triggers
    assert r["fill_status"] == "NO_DATA"
    assert math.isnan(r["y_net"]) and math.isnan(r["exit_price"]) and math.isnan(r["entry_price_raw"])


def test_stop_before_data_ends_is_a_complete_label_even_if_horizon_runs_past():
    _, r = run([T_DAY, ENTRY, (99, 99, 97, 97)], h=5)
    assert r["fill_status"] == "FILLED" and r["exit_reason"] == "STOP"


def test_delisted_exit_at_last_traded_price():
    b, r = run([T_DAY, ENTRY, FLAT, (100, 100, 96, 97)], h=5, delisted_after=pd.Timestamp("2024-01-04"))
    assert r["exit_reason"] == "STOP"                      # stop still wins when it fires
    b, r = run([T_DAY, ENTRY, FLAT, (100, 101, 99, 100.5)], h=5, delisted_after=pd.Timestamp("2024-01-04"))
    assert r["exit_reason"] == "DELISTED" and r["exit_price"] == 100.5 and r["exit_date"] == b.index[-1]


def test_locked_to_the_end_of_bars_is_censored_unless_delisted():
    ohlc = [T_DAY, ENTRY, (99, 99, 95, 95), (94, 94, 90, 90)]
    _, r = run(ohlc, lower=[2, 3])
    assert r["fill_status"] == "NO_DATA"
    b, r = run(ohlc, lower=[2, 3], delisted_after=pd.Timestamp("2024-01-04"))
    assert r["exit_reason"] == "DELISTED" and r["exit_price"] == 90 and r["lock_delay_sessions"] == 1


def test_lock_source_none_when_a_session_has_no_hit_list():
    b = bars_of([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT])
    hits, cov = hits_of(b)
    _, r = run([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT])
    assert r["lock_source"] == "HIT_LIST"
    _, r = run([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT], cov=cov - {b.index[3]})
    assert r["lock_source"] == "NONE"
    _, r = run([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT], cov=cov - {b.index[0]})   # t itself is not part of the trade
    assert r["lock_source"] == "HIT_LIST"


def test_missing_atr_or_entry_is_nan_not_zero():
    b = bars_of([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT])
    hits, cov = hits_of(b)
    r = simulate_trade(SYM, b, b.index[0], 5, 1.0, 2.0, float("nan"), hits, cov, cost_fn=ZERO)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])
    b2 = b.copy()
    b2.iloc[1, b2.columns.get_loc("open")] = float("nan")
    r = simulate_trade(SYM, b2, b.index[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])


def test_decision_date_not_in_bars_raises():
    b = bars_of([T_DAY, ENTRY, FLAT])
    hits, cov = hits_of(b)
    with pytest.raises(ValueError):
        simulate_trade(SYM, b, pd.Timestamp("2030-01-01"), 5, 1.0, 2.0, 2.0, hits, cov)


def test_no_look_ahead_label_is_unchanged_by_bars_after_its_exit():
    ohlc = [T_DAY, ENTRY, (100, 105, 99, 104), (60, 61, 50, 55), (10, 11, 9, 9), FLAT]
    b, full = run(ohlc, h=5)
    assert full["exit_reason"] == "TARGET" and full["exit_date"] == b.index[2]
    b2 = bars_of(ohlc[:3])
    hits, cov = hits_of(b2)
    cut = simulate_trade(SYM, b2, b2.index[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO)
    assert cut["exit_date"] == full["exit_date"] and cut["y_net"] == full["y_net"]


def test_sessions_calendar_no_trade_and_time_exit_on_first_bar_after_horizon():
    cal = pd.bdate_range("2024-01-01", periods=10)
    b = bars_of([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT]).drop(cal[1])           # symbol did not trade on t+1
    hits, cov = hits_of(b)
    r = simulate_trade(SYM, b, cal[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO, sessions=cal)
    assert r["fill_status"] == "NO_FILL_NO_TRADE" and math.isnan(r["y_net"])
    b = bars_of([T_DAY, ENTRY, FLAT, FLAT, FLAT, FLAT, FLAT]).drop(cal[5])      # suspended on the t+h session
    hits, cov = hits_of(b)
    r = simulate_trade(SYM, b, cal[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO, sessions=cal)
    assert r["exit_reason"] == "TIME" and r["exit_date"] == cal[6]              # first bar on/after the t+5 session
    assert r["label_end_date"] == cal[6] and r["exit_date"] > r["entry_date"]
    # mutant (old behaviour: last traded bar before t+h) would give cal[4]; never a same-day round trip after a long gap
    b = bars_of([T_DAY, ENTRY, FLAT]).rename(index=dict(zip(bars_of([T_DAY, ENTRY, FLAT]).index, [cal[0], cal[1], cal[9]])))
    hits, cov = hits_of(b)
    r = simulate_trade(SYM, b, cal[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO, sessions=cal)
    assert r["exit_date"] == cal[9] and r["exit_date"] != r["entry_date"]
    b = bars_of([T_DAY, ENTRY, FLAT])                                           # never trades again on/after t+5
    hits, cov = hits_of(b)
    r = simulate_trade(SYM, b, cal[0], 5, 1.0, 2.0, 2.0, hits, cov, cost_fn=ZERO, sessions=cal)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])


def test_lock_delay_counts_market_sessions_not_bar_rows_when_bars_are_missing():
    cal = pd.bdate_range("2024-01-01", periods=10)
    ohlc = [T_DAY, ENTRY, (99, 99, 95, 95), (92, 95, 91, 93), FLAT]
    b = bars_of(ohlc)
    b.index = pd.DatetimeIndex([cal[0], cal[1], cal[2], cal[6], cal[7]])      # no bars on cal[3..5] while locked
    hits = pd.DataFrame([{"date": cal[2], "symbol": SYM, "side": "LOWER"}])
    r = simulate_trade(SYM, b, cal[0], 7, 1.0, 2.0, 2.0, hits, set(b.index), cost_fn=ZERO, sessions=cal)
    assert r["exit_reason"] == "STOP_DELAYED_LOWER_LOCK" and r["exit_date"] == cal[6]
    assert r["lock_delay_sessions"] == 4                                        # mutant (bar-row count) gives 1
    r = simulate_trade(SYM, b, cal[0], 7, 1.0, 2.0, 2.0, hits, set(b.index), cost_fn=ZERO)
    assert r["lock_delay_sessions"] == 1                                        # without `sessions`: bar rows, as documented


def test_default_cost_legs_match_the_rate_table_by_hand():
    r = get_rates(pd.Timestamp("2024-03-04").date())
    brok = r.brokerage_flat / BROKERAGE_REFERENCE_NOTIONAL_INR
    ce, cx, ci = default_cost_fn(SYM, pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-11"), 100.0, 110.0)
    assert ce == pytest.approx(r.stt_delivery_buy + r.stamp_duty_buy + r.exchange_txn_charge + r.sebi_turnover_fee
                               + brok + (brok + r.exchange_txn_charge) * r.gst_rate)
    assert cx == pytest.approx(1.1 * (r.stt_delivery_sell + r.exchange_txn_charge + r.sebi_turnover_fee) + brok
                               + (brok + 1.1 * r.exchange_txn_charge) * r.gst_rate)
    assert ci == 0.0


def test_pre_2004_has_no_stt():
    ce, cx, _ = default_cost_fn(SYM, pd.Timestamp("2003-05-05"), pd.Timestamp("2003-05-12"), 100.0, 100.0)
    r = get_rates(pd.Timestamp("2003-05-05").date())
    assert r.stt_delivery_buy == 0.0 and r.stt_delivery_sell == 0.0
    assert ce + cx < 0.002
