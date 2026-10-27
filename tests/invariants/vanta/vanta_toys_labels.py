"""
Reference ("good") and deliberately broken toy implementations of the Stage 13A label simulator, plus tiny input builders.

This is test support, not implementation: it exists so each invariant test can be shown to (a) accept a correct simulator and
(b) FAIL against a simulator with the specific bug the invariant exists to catch. It runs with no `vanta` package present.

Interface assumed (see the V1 note to the lead):
    simulate_trade(symbol, bars, decision_date, horizon, k_s, k_t, atr, hits, hit_coverage, cost_fn=None, delisted_after=None)
"""

from __future__ import annotations

import functools
import math
from datetime import date

import pandas as pd

NAN = float("nan")
ZERO_COST = lambda *a, **k: (0.0, 0.0, 0.0)  # noqa: E731  (cost_entry, cost_exit, cost_impact) as return fractions


def make_bars(ohlc, start="2024-01-01", adj_ohlc=None) -> pd.DataFrame:
    """ohlc: list of (open, high, low, close) RAW tuples, one per consecutive weekday session from `start`.
    adj_ohlc: optional list of the same shape for the ADJUSTED columns (default: identical to raw)."""
    idx = pd.bdate_range(start=start, periods=len(ohlc))
    adj_ohlc = adj_ohlc or ohlc
    df = pd.DataFrame(
        {
            "raw_open": [r[0] for r in ohlc], "raw_high": [r[1] for r in ohlc],
            "raw_low": [r[2] for r in ohlc], "raw_close": [r[3] for r in ohlc],
            "open": [r[0] for r in adj_ohlc], "high": [r[1] for r in adj_ohlc],
            "low": [r[2] for r in adj_ohlc], "close": [r[3] for r in adj_ohlc],
        },
        index=idx,
    )
    return df


def make_hits(bars: pd.DataFrame, symbol: str, upper=(), lower=()) -> tuple[pd.DataFrame, set]:
    """upper/lower: session positions (0-based into bars) on which the symbol is on the hit list.
    Coverage is every session of bars (a published list exists each day)."""
    rows = [{"date": bars.index[i], "symbol": symbol, "side": "UPPER"} for i in upper]
    rows += [{"date": bars.index[i], "symbol": symbol, "side": "LOWER"} for i in lower]
    hits = pd.DataFrame(rows, columns=["date", "symbol", "side"])
    return hits, set(bars.index)


def _default_cost(symbol, entry_date, exit_date, entry_price, exit_price, *, frozen_date=None):
    """Toy trade-date cost: STT (delivery) both legs from the real rate table at the ENTRY date; frozen_date is the bug."""
    from costs.model import get_rates

    r = get_rates(frozen_date or pd.Timestamp(entry_date).date())
    return r.stt_delivery_buy + r.stamp_duty_buy, r.stt_delivery_sell, 0.0


def simulate_trade(symbol, bars, decision_date, horizon, k_s, k_t, atr, hits, hit_coverage,
                   cost_fn=None, delisted_after=None, *, _bugs=frozenset()):
    """Correct reference semantics per VANTA_SPEC Part A, minus delisting. `_bugs` injects one named defect."""
    idx = list(bars.index)
    ti = idx.index(pd.Timestamp(decision_date))
    e, end = ti + 1, min(ti + horizon, len(idx) - 1)
    hs = {(pd.Timestamp(r.date), r.symbol, r.side) for r in hits.itertuples()}
    o, h, l, c = ("open", "high", "low", "close") if "lock_on_adjusted" in _bugs else ("raw_open", "raw_high", "raw_low", "raw_close")

    def upper_locked_open(j):
        if "no_hit_list_check" in _bugs:
            return bars[o].iloc[j] == bars[h].iloc[j]
        if "hit_list_only" in _bugs:
            return (idx[j], symbol, "UPPER") in hs
        return (idx[j], symbol, "UPPER") in hs and bars[o].iloc[j] == bars[h].iloc[j]

    def lower_locked_close(j):
        return (idx[j], symbol, "LOWER") in hs and bars[c].iloc[j] == bars[l].iloc[j]

    all_dates_covered = all(idx[j] in hit_coverage for j in range(e, end + 1))
    row = dict(
        symbol=symbol, decision_date=idx[ti], horizon=horizon, entry_date=idx[e], k_s=k_s, k_t=k_t,
        lock_source="HIT_LIST" if all_dates_covered else "NONE", lock_delay_sessions=0,
        entry_price_raw=NAN, stop_price=NAN, target_price=NAN, exit_date=pd.NaT, exit_price=NAN,
        exit_reason=None, gross_return=NAN, cost_entry=NAN, cost_exit=NAN, cost_impact=NAN, y_net=NAN,
        label_end_date=pd.NaT, fill_status="FILLED",
    )
    if upper_locked_open(e) and "ignore_upper_lock" not in _bugs:
        row["fill_status"] = "NO_FILL_UPPER_LOCK"
        if "payoff_on_no_fill" in _bugs:
            row.update(exit_price=bars["open"].iloc[e], gross_return=0.0, y_net=0.0)
        return row

    entry = bars["open"].iloc[e]
    stop, target = entry - k_s * atr, entry + k_t * atr
    row.update(entry_price_raw=bars["raw_open"].iloc[e], stop_price=stop, target_price=target)

    exit_j, exit_px, reason, delay = end, bars["close"].iloc[end], "TIME", 0
    for j in range(e, end + 1):
        op, hi, lo = bars["open"].iloc[j], bars["high"].iloc[j], bars["low"].iloc[j]
        gap = j > e
        if gap and op <= stop:
            px = stop if "gap_stop_at_stop" in _bugs else op
            exit_j, exit_px, reason = j, px, "STOP_GAP"
            break
        if gap and op >= target:
            px = op if "gap_target_at_open" in _bugs else target
            exit_j, exit_px, reason = j, px, "TARGET"
            break
        hit_stop, hit_target = lo <= stop, hi >= target
        if hit_stop and hit_target and "target_first" in _bugs:
            hit_stop = False
        if hit_stop:
            if "ignore_lower_lock" not in _bugs and lower_locked_close(j):
                k = j
                while lower_locked_close(k) and k < len(idx) - 1:
                    k += 1
                if "delay_off_by_one" in _bugs:
                    k -= 1
                exit_j, exit_px, reason, delay = k, bars["close"].iloc[k], "STOP_DELAYED_LOWER_LOCK", k - j
            else:
                exit_j, exit_px, reason = j, stop, "STOP"
            break
        if hit_target:
            exit_j, exit_px, reason = j, target, "TARGET"
            break

    gross = exit_px / entry - 1.0
    cf = cost_fn
    if cf is None:
        frozen = date(2020, 1, 1) if "frozen_cost_date" in _bugs else None
        cf = functools.partial(_default_cost, frozen_date=frozen)
    ce, cx, ci = cf(symbol, idx[e], idx[exit_j], entry, exit_px)
    row.update(exit_date=idx[exit_j], exit_price=exit_px, exit_reason=reason, lock_delay_sessions=delay,
               gross_return=gross, cost_entry=ce, cost_exit=cx, cost_impact=ci, y_net=gross - ce - cx - ci,
               label_end_date=idx[exit_j])
    return row


def toy(*bugs):
    """A simulate_trade with the named bugs injected (no names = the correct reference)."""
    return functools.partial(simulate_trade, _bugs=frozenset(bugs))


def purge_training_rows(labels_df, test_start, embargo, sessions, *, _bugs=frozenset()):
    """Reference purge: drop rows with label_end_date >= (test_start minus `embargo` SESSIONS on the calendar `sessions`)."""
    sessions = pd.DatetimeIndex(sessions)
    pos = int(sessions.searchsorted(pd.Timestamp(test_start)))
    if "calendar_days" in _bugs:
        cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=embargo)
    else:
        cutoff = sessions[pos - embargo]
    if "strict_inequality" in _bugs:
        return labels_df[labels_df["label_end_date"] <= cutoff]
    if "no_embargo" in _bugs:
        return labels_df[labels_df["label_end_date"] < pd.Timestamp(test_start)]
    if "uses_decision_date" in _bugs:
        return labels_df[labels_df["decision_date"] < cutoff]
    return labels_df[labels_df["label_end_date"] < cutoff]


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
