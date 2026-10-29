"""
Stage 13A coverage report: build NEL labels (vanta.labels.simulate_trade) for a SAMPLE of real symbols and print fill / no-fill /
exit-reason / lock-source counts by decision year. Optionally write a hand-reconcilable file of 20 simulated trades.

Inputs (all from config.py paths): Stage 4 adjusted prices (raw OPEN/HIGH/LOW/CLOSE for lock detection, ADJ_* for returns),
Stage 10 circuit features (hit_upper_circuit / hit_lower_circuit flags; NaN = no published list), ATR(14) from
signals.indicators.compute_indicators on the adjusted prices. Decision dates are the last session of each ISO week.

This is a sampling tool, not the full label build: memory-light, a few dozen symbols, no database. The full build is a job
for scripts/run_job.py.

Usage:
    python -m scripts.report_nel_coverage --symbols 40 --horizon 20 --k-s 1.5 --k-t 2.0 --seed 0
    python -m scripts.report_nel_coverage --symbols 150 --sample-trades logs/nel_sample_20_trades.md
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import ADJUSTED_PRICES_DIR, CIRCUIT_FEATURES_DIR, PROJECT_ROOT
from costs.model import get_rates
from data.trading_calendar import trading_sessions
from signals.indicators import compute_indicators, prepare_ohlcv
from vanta.labels import BROKERAGE_REFERENCE_NOTIONAL_INR, simulate_trade

FIRST_YEAR_NOTE = "lock_source NONE = a session between entry and exit has no published hit list (hit lists start 2010-01-04)"
REASONS = ("TARGET", "STOP", "STOP_GAP", "STOP_DELAYED_LOWER_LOCK", "TIME", "DELISTED")
STATUSES = ("FILLED", "NO_FILL_UPPER_LOCK", "NO_FILL_NO_TRADE", "NO_DATA")


def load_symbol(symbol: str):
    """Return (bars, atr_by_date, hits, coverage) for one symbol, or None when its files are missing."""
    adj_path, circ_path = ADJUSTED_PRICES_DIR / f"{symbol}.parquet", CIRCUIT_FEATURES_DIR / f"{symbol}.parquet"
    if not (adj_path.exists() and circ_path.exists()):
        return None
    raw = pd.read_parquet(adj_path).drop_duplicates("DATE").sort_values("DATE").reset_index(drop=True)
    ohlcv = prepare_ohlcv(raw)
    atr = pd.Series(compute_indicators(ohlcv)["atr_14"].to_numpy(), index=pd.DatetimeIndex(raw["DATE"]))
    # raw prices: a zero in the archive means missing, so it becomes NaN (never a price)
    rawpx = raw[["OPEN", "HIGH", "LOW", "CLOSE"]].where(raw[["OPEN", "HIGH", "LOW", "CLOSE"]] > 0)
    bars = pd.DataFrame({"raw_open": rawpx["OPEN"], "raw_high": rawpx["HIGH"], "raw_low": rawpx["LOW"], "raw_close": rawpx["CLOSE"],
                         "open": ohlcv["open"], "high": ohlcv["high"], "low": ohlcv["low"], "close": ohlcv["close"]})
    bars.index = pd.DatetimeIndex(raw["DATE"])
    circ = pd.read_parquet(circ_path).drop_duplicates("date").set_index("date")
    published = circ.index[~circ["is_missing_circuit_hits"].astype(bool)]
    up = circ.index[circ["hit_upper_circuit"].fillna(0) == 1.0]
    lo = circ.index[circ["hit_lower_circuit"].fillna(0) == 1.0]
    hits = pd.concat([pd.DataFrame({"date": up, "symbol": symbol, "side": "UPPER"}),
                      pd.DataFrame({"date": lo, "symbol": symbol, "side": "LOWER"})], ignore_index=True)
    return bars, atr, hits, set(pd.DatetimeIndex(published))


def weekly_decision_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = sessions.isocalendar()
    key = iso["year"].astype(int).to_numpy() * 100 + iso["week"].astype(int).to_numpy()
    last = pd.Series(sessions, index=key).groupby(level=0).max()
    return pd.DatetimeIndex(last.to_numpy())


def build_labels(symbols, horizon, k_s, k_t, sessions, keep_inputs=False):
    rows, inputs = [], {}
    decision_dates = weekly_decision_dates(sessions)
    for symbol in symbols:
        loaded = load_symbol(symbol)
        if loaded is None:
            logger.warning(f"{symbol}: adjusted or circuit file missing, skipped")
            continue
        bars, atr, hits, coverage = loaded
        for t in decision_dates.intersection(bars.index):
            a = atr.get(t, np.nan)
            if np.isnan(a):
                continue                                            # ATR warm-up: no decision is made, not a NaN label
            r = simulate_trade(symbol, bars, t, horizon, k_s, k_t, float(a), hits, coverage, sessions=sessions)
            r["atr"] = float(a)
            rows.append(r)
        if keep_inputs:
            inputs[symbol] = (bars, hits, coverage)
    return pd.DataFrame(rows), inputs


def coverage_tables(labels: pd.DataFrame) -> str:
    labels = labels.assign(year=pd.to_datetime(labels["decision_date"]).dt.year)
    filled = labels[labels["fill_status"] == "FILLED"]
    t1 = pd.crosstab(labels["year"], labels["fill_status"]).reindex(columns=list(STATUSES), fill_value=0)
    t1["total"] = t1.sum(axis=1)
    t2 = pd.crosstab(filled["year"], filled["exit_reason"]).reindex(columns=list(REASONS), fill_value=0)
    t3 = pd.crosstab(filled["year"], filled["lock_source"]).reindex(columns=["HIT_LIST", "NONE"], fill_value=0)
    extra = filled.groupby("year").agg(mean_y_net=("y_net", "mean"), mean_gross=("gross_return", "mean"),
                                       max_lock_delay=("lock_delay_sessions", "max"))
    return "\n".join([
        "FILL STATUS by decision year", t1.to_string(), "",
        "EXIT REASON (filled trades)", t2.to_string(), "",
        f"LOCK SOURCE (filled trades). {FIRST_YEAR_NOTE}", t3.to_string(), "",
        "MEAN RETURNS (filled trades, fraction)", extra.round(5).to_string(),
    ])


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return "NaN" if np.isnan(x) else f"{x:.10g}"
    if isinstance(x, pd.Timestamp):
        return "NaT" if pd.isna(x) else x.date().isoformat()
    return str(x)


def write_sample_trades(labels: pd.DataFrame, inputs: dict, path: Path, horizon, k_s, k_t, n=20, seed=0, sessions=None, pins=()):
    """Pick n trades stratified over outcomes and write every input and intermediate value for hand reconciliation."""
    rng = random.Random(seed)
    quota = ["NO_FILL_UPPER_LOCK"] * 2 + ["STOP_DELAYED_LOWER_LOCK"] * 3 + ["STOP_GAP"] * 3 + ["TARGET"] * 4 + ["STOP"] * 4 + ["TIME"] * 4
    chosen = []
    for sym, d in pins:                                                    # pinned trades first, in the order given
        hit = labels.index[(labels["symbol"] == sym) & (labels["decision_date"] == pd.Timestamp(d))]
        if len(hit):
            chosen.append(hit[0])
        else:
            logger.warning(f"pinned trade {sym} {d} is not in the label set, skipped")
    for want in ([] if len(chosen) >= n else quota):
        col = "fill_status" if want.startswith("NO_FILL") else "exit_reason"
        pool = labels[(labels[col] == want) & (~labels.index.isin(chosen))]
        pool = pool[pool["decision_date"] >= pd.Timestamp("2010-02-01")]   # hit lists exist: locks are decidable
        if len(pool):
            chosen.append(rng.choice(list(pool.index)))
    for i in rng.sample(list(labels.index), len(labels)):                  # top up with any outcome
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    lines = [
        "# NEL sample: 20 simulated real trades", "",
        f"Generated by `scripts/report_nel_coverage.py` (horizon h={horizon}, k_s={k_s}, k_t={k_t}). Every price below is copied from the "
        "Stage 4 adjusted parquet (`RAW` = unadjusted OPEN/HIGH/LOW/CLOSE, used only for lock detection; `ADJ` = ADJ_*, used for entry, "
        "stop, target, exit and return). ATR is ATR(14) Wilder at the decision close in ADJ units. `UP`/`LO` columns are the Stage 10 "
        "price-band hit flags (1 = on that day's hit list, 0 = list published and absent, `?` = no list published).", "",
        "Rules to reconcile against: entry = ADJ open of t+1; stop = entry - k_s*ATR; target = entry + k_t*ATR; after the entry day "
        "open <= stop exits at the open (STOP_GAP), open >= target exits at the target; within a day the stop is checked before the target; "
        "a stop session with LO=1 and RAW close == RAW low is lower-locked and the exit moves to the ADJ close of the first later session "
        "that is not lower-locked; no fill if UP=1 and RAW open == RAW high on t+1; TIME exits at the ADJ close of t+h. "
        f"Cost legs use the entry-date / exit-date STT, stamp, exchange, SEBI, GST rates from `costs/model.py` and Rs.20 flat brokerage on a "
        f"Rs.{BROKERAGE_REFERENCE_NOTIONAL_INR:,.0f} reference position (assumption). cost_impact = 0 (no impact-cost feed).", ""]
    for n_i, i in enumerate(chosen, 1):
        r = labels.loc[i]
        bars, hits, cov = inputs[r["symbol"]]
        t = pd.Timestamp(r["decision_date"])
        end = r["label_end_date"] if pd.notna(r["label_end_date"]) else r["entry_date"]
        lo_i = bars.index.get_loc(t)
        hi_i = bars.index.get_loc(end) if pd.notna(end) and end in bars.index else min(lo_i + 2, len(bars) - 1)
        window = bars.iloc[lo_i: hi_i + 1]                                   # t through the exit row inclusive
        up = set(pd.to_datetime(hits.loc[hits["side"] == "UPPER", "date"]))
        lw = set(pd.to_datetime(hits.loc[hits["side"] == "LOWER", "date"]))
        lines += [f"## Trade {n_i}: {r['symbol']}, decision {_fmt(t)}", "",
                  f"- inputs: h={int(r['horizon'])}, k_s={k_s}, k_t={k_t}, ATR(14)={_fmt(r['atr'])}",
                  f"- derived: entry (ADJ open of {_fmt(r['entry_date'])}) = {_fmt(r['entry_price'])}, "
                  f"stop = {_fmt(r['stop_price'])}, target = {_fmt(r['target_price'])}",
                  ]
        if sessions is not None and pd.notna(r["entry_date"]) and pd.notna(end):
            between = sessions[(sessions > r["entry_date"]) & (sessions < end)]
            lines.append(f"- calendar sessions strictly between entry and exit: {len(between)}"
                         + (f" ({', '.join(_fmt(pd.Timestamp(x)) for x in between)})" if len(between) <= 60 else " (first/last "
                            f"{_fmt(pd.Timestamp(between[0]))} / {_fmt(pd.Timestamp(between[-1]))})")
                         + f"; sessions with no bar for this symbol: {sum(1 for x in between if x not in bars.index)}")
        lines += ["", "| date | role | RAW O | RAW H | RAW L | RAW C | ADJ O | ADJ H | ADJ L | ADJ C | UP | LO |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d, b in window.iterrows():
            role = "t" if d == t else "entry" if d == r["entry_date"] else "exit" if d == r["exit_date"] else ""
            flag = lambda s: ("1" if d in s else "0") if d in cov else "?"          # noqa: E731
            lines.append(f"| {_fmt(d)} | {role} | " + " | ".join(_fmt(float(b[c])) for c in
                         ("raw_open", "raw_high", "raw_low", "raw_close", "open", "high", "low", "close")) + f" | {flag(up)} | {flag(lw)} |")
        lines += ["", f"- result: fill_status={r['fill_status']}, exit_reason={r['exit_reason']}, exit_date={_fmt(r['exit_date'])}, "
                  f"exit_price={_fmt(r['exit_price'])}, lock_delay_sessions={int(r['lock_delay_sessions'])}, lock_source={r['lock_source']}"]
        if r["fill_status"] == "FILLED":
            re_, rx = get_rates(pd.Timestamp(r["entry_date"]).date()), get_rates(pd.Timestamp(r["exit_date"]).date())
            lines += [f"- gross_return = exit/entry - 1 = {_fmt(r['exit_price'])}/{_fmt(r['entry_price'])} - 1 = {_fmt(r['gross_return'])}",
                      f"- entry-leg rates ({_fmt(r['entry_date'])}): stt_buy={re_.stt_delivery_buy}, stamp={re_.stamp_duty_buy}, "
                      f"exch={re_.exchange_txn_charge}, sebi={re_.sebi_turnover_fee}, gst={re_.gst_rate}, flat_brokerage={re_.brokerage_flat}",
                      f"- exit-leg rates ({_fmt(r['exit_date'])}): stt_sell={rx.stt_delivery_sell}, exch={rx.exchange_txn_charge}, "
                      f"sebi={rx.sebi_turnover_fee}, gst={rx.gst_rate}, flat_brokerage={rx.brokerage_flat}",
                      f"- cost_entry={_fmt(r['cost_entry'])}, cost_exit={_fmt(r['cost_exit'])} (exit value/entry value = "
                      f"{_fmt(r['exit_price'] / r['entry_price'])}), cost_impact={_fmt(r['cost_impact'])}",
                      f"- y_net = gross - cost_entry - cost_exit - cost_impact = {_fmt(r['y_net'])}"]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {len(chosen)} sample trades to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", type=int, default=40, help="number of sampled symbols (default 40)")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k-s", type=float, default=1.5)
    ap.add_argument("--k-t", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pin", nargs="*", default=[], help="SYMBOL:YYYY-MM-DD trades to force into the sample, in order")
    ap.add_argument("--sample-trades", type=Path, default=None, help="write the 20-trade reconciliation file here")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stdout, format="{message}")

    symbols = sorted(p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet"))
    symbols = random.Random(args.seed).sample(symbols, min(args.symbols, len(symbols)))
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    labels, inputs = build_labels(symbols, args.horizon, args.k_s, args.k_t, sessions, keep_inputs=args.sample_trades is not None)
    logger.info(f"{len(symbols)} symbols, {len(labels):,} weekly decisions, h={args.horizon}, k_s={args.k_s}, k_t={args.k_t}, seed={args.seed}")
    logger.info(coverage_tables(labels))
    if args.sample_trades is not None:
        write_sample_trades(labels, inputs, PROJECT_ROOT / args.sample_trades if not args.sample_trades.is_absolute() else args.sample_trades,
                            args.horizon, args.k_s, args.k_t, seed=args.seed, sessions=sessions,
                            pins=[tuple(x.split(":")) for x in args.pin])


if __name__ == "__main__":
    main()
