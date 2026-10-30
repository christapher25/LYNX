"""
Stage 13A -- NEL, net executable labels (docs/vanta/VANTA_SPEC.md Part A; signature per docs/vanta/INTERFACES.md section 1).

simulate_trade() replays the exact trade that would have been published after the close of decision session t:

    entry   raw open of t+1 (the adjusted open is the price used for returns). No fill when the symbol is an UPPER hit that
            day AND raw open == raw high (N1).
    stop    entry - k_s * ATR(14) at t;  target = entry + k_t * ATR(14) at t  (ATR in adjusted-price units).
    order   sessions after the entry day: open <= stop is a gap through the stop, fill at the OPEN (STOP_GAP); open >= target
            fills at the TARGET (no windfall). Inside a session the stop is assumed hit before the target (N3). The entry
            day's own range is checked too.
    locks   a session is lower-locked iff the symbol is a LOWER hit that day AND raw close == raw low. A stop (or a gap-down
            through the stop) on a lower-locked session exits at the ADJUSTED CLOSE of the first later session that is not
            lower-locked, STOP_DELAYED_LOWER_LOCK, lock_delay_sessions = sessions from the stop session to that exit (N2, N4).
    time    close of row t+h (TIME). With `sessions`, if the symbol has no bar on the t+h session the exit is the close of the
            FIRST session on/after t+h on which it does have a bar (the first day it could be sold) -- a CHOSEN reading; exit_date
            and label_end_date are that date. If it never trades again in the data the label is NO_DATA (or DELISTED when
            delisted_after applies). Stops/targets/locks are still scanned on every bar through that exit.
    cost    y_net = gross_return - cost_entry - cost_exit - cost_impact, every term a fraction of entry notional.

Cost conventions (CHOSEN readings, the spec is silent): the entry leg uses the rates in force on the entry date and the exit
leg those on the exit date; cost_impact is 0.0 because no NSE impact-cost feed exists yet, which UNDERSTATES costs for illiquid
names; flat brokerage uses BROKERAGE_REFERENCE_NOTIONAL_INR.

Lock detection reads RAW prices and the price-band hit list only; returns use Stage 4 adjusted prices. Labels are
policy-free (no incumbency) and a label uses only bars through its own exit. Missing means missing: a trade that cannot be
completed returns NaN (never 0) fields and a fill_status other than FILLED.

fill_status: FILLED, NO_FILL_UPPER_LOCK, NO_FILL_NO_TRADE (symbol has no bar on the market's session t+1; needs `sessions`),
NO_DATA (inputs missing, or the horizon runs past the end of the bars and the symbol is not known to be delisted).
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from costs.model import get_rates

NAN = float("nan")

BROKERAGE_REFERENCE_NOTIONAL_INR = 100_000.0
"""ASSUMPTION pending a user ruling (not in the spec): the flat per-order brokerage in costs/model.py (Rs.20) is not a fraction
of notional, so it is expressed against a reference position of Rs.1,00,000 per name."""

TERMINAL_RTOL = 1e-6
"""A Stage 5 terminal value must equal the bars' own last adjusted close to this relative tolerance, else it is in other price units
(an older adjustment) and the trade is left NO_DATA."""

REQUIRED_COLUMNS = ("raw_open", "raw_high", "raw_low", "raw_close", "open", "high", "low", "close")


def entry_leg_cost(r, brok_frac: float) -> float:
    """Buy leg as a fraction of entry notional: STT + stamp duty + exchange + SEBI + brokerage + GST on (brokerage + exchange)."""
    return (r.stt_delivery_buy + r.stamp_duty_buy + r.exchange_txn_charge + r.sebi_turnover_fee + brok_frac
            + (brok_frac + r.exchange_txn_charge) * r.gst_rate)


def exit_leg_cost(r, brok_frac: float, ratio: float) -> float:
    """Sell leg as a fraction of ENTRY notional (`ratio` = exit value / entry value). Stamp duty is buy-side only."""
    return (ratio * (r.stt_delivery_sell + r.exchange_txn_charge + r.sebi_turnover_fee) + brok_frac
            + (brok_frac + ratio * r.exchange_txn_charge) * r.gst_rate)


def default_cost_fn(symbol, entry_date, exit_date, entry_price, exit_price):
    """Stage 6 delivery costs at TRADE-DATE rates: the entry leg uses the rates in force on entry_date, the exit leg those on
    exit_date. Returns (cost_entry, cost_exit, cost_impact), each a fraction of entry notional.

    cost_impact is 0.0: the v6 document says NSE publishes per-security impact cost, but no impact-cost data exists in this
    repo yet (explicit assumption, flagged to the lead) until that feed is built."""
    re_ = get_rates(pd.Timestamp(entry_date).date())
    rx = get_rates(pd.Timestamp(exit_date).date())
    ratio = exit_price / entry_price
    return (entry_leg_cost(re_, re_.brokerage_flat / BROKERAGE_REFERENCE_NOTIONAL_INR),
            exit_leg_cost(rx, rx.brokerage_flat / BROKERAGE_REFERENCE_NOTIONAL_INR, ratio),
            0.0)


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def simulate_trade(symbol, bars, decision_date, horizon, k_s, k_t, atr, hits, hit_coverage,
                   cost_fn=None, delisted_after=None, *, sessions=None, terminal_value=None):
    """Simulate one NEL trade. See the module docstring; units and column names follow docs/vanta/INTERFACES.md section 1.

    `sessions` (optional, keyword-only): the market's sorted session calendar. With it, t+1 and t+h are market sessions, a
    symbol with no bar on t+1 is NO_FILL_NO_TRADE, and a time exit on a day the symbol did not trade happens at the close
    of the first session on or after t+h on which it has a bar (exit_date and label_end_date are that date; NO_DATA if it
    never trades again in the data). Lock delay counts market-session positions. Without it the rows of `bars` are taken
    to be consecutive market sessions and the delay counts bar rows.

    `delisted_after` (the symbol's last traded date) and `terminal_value` (its Stage 5 terminal value, the last traded adjusted
    close; see load_terminal_values) are passed together by terminal_kwargs(). A trade still open when a delisted name's history
    ends exits at `terminal_value` (DELISTED). If `terminal_value` is given but unusable (NaN, non-positive, or not the bars' own
    last adjusted close within TERMINAL_RTOL) the trade stays NO_DATA: never a guessed price. With `terminal_value` omitted the
    last adjusted close is used (the pre-wiring behaviour, kept for direct callers)."""
    if cost_fn is None:
        cost_fn = default_cost_fn
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing columns {missing}")
    t = pd.Timestamp(decision_date)
    idx = bars.index
    if t not in idx:
        raise ValueError(f"decision_date {t.date()} is not a session in bars for {symbol}")
    ti = int(idx.get_loc(t))
    n = len(idx)

    # side-specific hit lookups for this symbol (dates only)
    if len(hits):
        hs = hits[hits["symbol"] == symbol]
        upper_days = set(pd.to_datetime(hs.loc[hs["side"] == "UPPER", "date"]))
        lower_days = set(pd.to_datetime(hs.loc[hs["side"] == "LOWER", "date"]))
    else:
        upper_days, lower_days = set(), set()
    cov = hit_coverage if isinstance(hit_coverage, (set, frozenset)) else {pd.Timestamp(d) for d in hit_coverage}

    ro, rh, rl, rc = (bars[c].to_numpy(dtype=float) for c in ("raw_open", "raw_high", "raw_low", "raw_close"))
    ao, ah, al, ac = (bars[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))

    def upper_locked_open(j):
        return idx[j] in upper_days and ro[j] == rh[j]

    def lower_locked_close(j):
        return idx[j] in lower_days and rc[j] == rl[j]

    row = dict(symbol=symbol, decision_date=t, horizon=horizon, entry_date=pd.NaT, k_s=k_s, k_t=k_t,
               fill_status="NO_DATA", entry_price_raw=NAN, entry_price=NAN, stop_price=NAN, target_price=NAN,
               exit_date=pd.NaT, exit_price=NAN, exit_reason=None, lock_delay_sessions=0, lock_source="NONE",
               gross_return=NAN, cost_entry=NAN, cost_exit=NAN, cost_impact=NAN, y_net=NAN, label_end_date=pd.NaT)

    # ---- locate the entry row and the time-exit row (positions in bars)
    sess = pd.DatetimeIndex(sessions) if sessions is not None else None
    if sess is not None:
        sp = int(sess.get_loc(t))
        if sp + 1 >= len(sess):
            return row                                                  # t is the last known session
        entry_session = sess[sp + 1]
        if entry_session not in idx:
            row.update(fill_status="NO_FILL_NO_TRADE", entry_date=entry_session)
            return row
        e = int(idx.get_loc(entry_session))
        end_date = sess[sp + horizon] if sp + horizon < len(sess) else None
        complete = end_date is not None and idx[-1] >= end_date
        end = int(idx.searchsorted(end_date, side="left")) if complete else n - 1   # first bar on/after t+h
    else:
        if ti + 1 >= n:
            return row
        e = ti + 1
        complete = ti + horizon <= n - 1
        end = ti + horizon if complete else n - 1
    row["entry_date"] = idx[e]

    entry = ao[e]
    if _isnan(atr) or not atr > 0 or np.isnan(entry) or not entry > 0:
        return row                                                      # missing inputs: NaN, never a guessed price

    if upper_locked_open(e):
        row["fill_status"] = "NO_FILL_UPPER_LOCK"
        row["lock_source"] = "HIT_LIST" if idx[e] in cov else "NONE"
        return row

    stop, target = entry - k_s * atr, entry + k_t * atr

    def first_free_session(d):
        """First position after d that is not lower-locked at close; None if the bars run out first."""
        k = d + 1
        while k < n:
            if not lower_locked_close(k):
                return k
            k += 1
        return None

    def delay_of(stop_j, exit_k):
        if sess is None:
            return exit_k - stop_j
        return int(sess.get_loc(idx[exit_k])) - int(sess.get_loc(idx[stop_j]))

    exit_j, exit_px, reason, delay, stop_day = None, NAN, None, 0, e
    for j in range(e, end + 1):
        gap = j > e
        if gap and ao[j] <= stop:
            exit_j, exit_px, reason = j, ao[j], "STOP_GAP"
            if lower_locked_close(j):
                reason = "STOP_DELAYED_LOWER_LOCK"
                exit_j = first_free_session(j)
                exit_px = ac[exit_j] if exit_j is not None else NAN
                delay = delay_of(j, exit_j) if exit_j is not None else 0
                stop_day = j
            break
        if gap and ao[j] >= target:
            exit_j, exit_px, reason = j, target, "TARGET"
            break
        if al[j] <= stop:                                               # stop wins any same-day ambiguity
            exit_j, exit_px, reason = j, stop, "STOP"
            if lower_locked_close(j):
                reason = "STOP_DELAYED_LOWER_LOCK"
                exit_j = first_free_session(j)
                exit_px = ac[exit_j] if exit_j is not None else NAN
                delay = delay_of(j, exit_j) if exit_j is not None else 0
                stop_day = j
            break
        if ah[j] >= target:
            exit_j, exit_px, reason = j, target, "TARGET"
            break
    else:
        if complete:
            exit_j, exit_px, reason = end, ac[end], "TIME"

    last_j = n - 1
    if terminal_value is not None:
        # a Stage 5 terminal value belongs to the symbol's true last bar: bars cut earlier are truncated, not delisted
        ended = delisted_after is not None and idx[last_j] == pd.Timestamp(delisted_after)
    else:
        ended = delisted_after is not None and idx[last_j] <= pd.Timestamp(delisted_after)
    delist_px = ac[last_j]
    if terminal_value is not None:
        tv = float(terminal_value)
        usable = math.isfinite(tv) and tv > 0 and math.isfinite(delist_px) and abs(tv / delist_px - 1.0) <= TERMINAL_RTOL
        ended = ended and usable
        delist_px = tv
    if reason == "STOP_DELAYED_LOWER_LOCK" and exit_j is None:
        # lower-locked through the last available bar: only a known delisting gives a (terminal-value) exit
        if ended:
            exit_j, exit_px, reason, delay = last_j, delist_px, "DELISTED", delay_of(stop_day, last_j)
        else:
            reason = None
    elif reason is None and ended:
        exit_j, exit_px, reason = last_j, delist_px, "DELISTED"         # Stage 5 terminal value: last traded price

    if reason is None or exit_j is None or _isnan(float(exit_px)):
        return row                                                      # censored / unpriced: not a completed label

    gross = exit_px / entry - 1.0
    ce, cx, ci = cost_fn(symbol, idx[e], idx[exit_j], entry, exit_px)
    covered = all(idx[k] in cov for k in range(e, exit_j + 1))
    row.update(entry_price_raw=float(ro[e]), entry_price=entry, stop_price=stop, target_price=target,
               fill_status="FILLED", exit_date=idx[exit_j], exit_price=float(exit_px), exit_reason=reason,
               lock_delay_sessions=int(delay), lock_source="HIT_LIST" if covered else "NONE", gross_return=float(gross),
               cost_entry=ce, cost_exit=cx, cost_impact=ci, y_net=gross - ce - cx - ci, label_end_date=idx[exit_j])
    return row


def load_terminal_values(universe_dir=None, adjusted_dir=None) -> dict:
    """{symbol: (last_date, terminal_value)} for every symbol that is delisted by Stage 5's own rule (data/build_universe.py).

    Stage 5 flags a symbol delisted in a year's snapshot when its LAST_DATE is before that year's 1 January and stores TERMINAL_VALUE
    (last traded ADJ_CLOSE) only on such rows, and only for names in that year's top list. A name that died mid-year, or fell out of
    the list afterwards, therefore has no stored value although the same rule applies to it. This applies the rule at the NEWEST
    snapshot's cutoff to every symbol in any snapshot: delisted iff its latest LAST_DATE is before that cutoff. The value is
    Stage 5's TERMINAL_VALUE where one was stored, else the same derivation (last ADJ_CLOSE) from the Stage 4 file. Either way the
    Stage 4 file must confirm the last date and the value; a symbol without a usable terminal value is left out (its trades stay
    NO_DATA). Truncated histories are not delistings: symbols whose LAST_DATE is on or after the cutoff are never included."""
    from config import ADJUSTED_PRICES_DIR, UNIVERSE_DIR

    universe_dir = Path(universe_dir or UNIVERSE_DIR)
    adjusted_dir = Path(adjusted_dir or ADJUSTED_PRICES_DIR)
    files = sorted(universe_dir.glob("*.parquet"))
    if not files:
        return {}
    frames = [pd.read_parquet(f, columns=["SYMBOL", "LAST_DATE", "TERMINAL_VALUE"]).assign(YEAR=int(f.stem)) for f in files]
    allrows = pd.concat(frames, ignore_index=True)
    cutoff = pd.Timestamp(f"{int(allrows['YEAR'].max())}-01-01")
    allrows["LAST_DATE"] = pd.to_datetime(allrows["LAST_DATE"])
    latest = allrows.sort_values(["SYMBOL", "LAST_DATE", "YEAR"]).groupby("SYMBOL").tail(1).set_index("SYMBOL")
    stored = allrows[allrows["TERMINAL_VALUE"].notna()].sort_values("YEAR").groupby("SYMBOL").tail(1).set_index("SYMBOL")["TERMINAL_VALUE"]
    out = {}
    for symbol, row in latest.iterrows():
        last = row["LAST_DATE"]
        if pd.isna(last) or last >= cutoff:
            continue
        path = adjusted_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        px = pd.read_parquet(path, columns=["DATE", "ADJ_CLOSE"]).sort_values("DATE")
        if px.empty or pd.Timestamp(px["DATE"].iloc[-1]) != last:
            continue
        value = float(px["ADJ_CLOSE"].iloc[-1])
        if symbol in stored.index and not (abs(float(stored[symbol]) / value - 1.0) <= TERMINAL_RTOL if value > 0 else False):
            continue                                                  # Stage 5's stored value disagrees with Stage 4: stale, not trusted
        if math.isfinite(value) and value > 0:
            out[symbol] = (last, value)
    return out


def terminal_kwargs(symbol, terminal_values) -> dict:
    """simulate_trade keyword arguments for `symbol`: delisted_after and terminal_value if it has a terminal value, else {}."""
    if symbol not in terminal_values:
        return {}
    last, value = terminal_values[symbol]
    return {"delisted_after": last, "terminal_value": value}
