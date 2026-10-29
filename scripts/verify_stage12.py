"""
scripts/verify_stage12.py

Stage 12 checkpoint: a read as of date t returns only what was knowable at t, demonstrated on real Stage 10 feature files for a
small sample (20 symbols x ~30 as-of dates, centred on real events). Sample scale only: nothing is materialised.

Checks
  1 NO LEAK       every returned row is dated <= t and has availability <= t (and < t with before_nightly_run).
  2 EXPECTED      the read equals an independent plain-pandas lookup of the full block files (NaN-aware, flags included).
  3 TRUNCATION    the read equals the read over data and calendar deleted after t.
  4 RAW           values re-derived from the RAW archives: F&O ban flag from data/raw/fno_ban, circuit hits from
                  data/raw/price_band_hits, delivery % from data/raw/delivery.
  5 EVENTS        pledging change appears at t and not at t-1; FII row at t carries a source day strictly before t.
  6 LAG           the availability lags of delivery / F&O-OI / ban re-derived from data/raw/_publication/first_seen.jsonl (a source
                  last modified after 19:00 IST is usable next session), independent of pitstore's own availability code.
  7 MUTANT        a reader that admits t+1 is caught by check 1 on this same sample.

Exit 0 if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from config import FNO_BAN_DIR, PRICE_BAND_HITS_DIR, RAW_DATA_DIR  # noqa: E402
from data.trading_calendar import trading_sessions  # noqa: E402
from pitstore import store  # noqa: E402
from pitstore.store import BLOCK_SPECS, read_as_of, truncated_loader, default_loader  # noqa: E402

BLOCKS = ["delivery", "fno_oi", "fno_ban", "pledging", "circuits", "flows_fii", "flows_dii", "breadth"]
RECENT_START = pd.Timestamp("2026-07-01")
SEED = 12
FIRST_SEEN = RAW_DATA_DIR / "_publication" / "first_seen.jsonl"
CUTOFF_HOUR_IST = 19
# which first_seen item publishes each block (the ban list for target D was last modified on the evening of D-1: lag 0)
PUBLISHING_ITEM = {"delivery": "nse_sec_bhavdata_full", "fno_oi": "nse_fo_bhavcopy_udiff", "fno_ban": "nse_fo_ban"}


def expected_lags() -> dict[str, int]:
    """1 if any recorded last_modified of the block's source is after 19:00 IST on its TARGET date, else 0; read from the jsonl."""
    rows = [json.loads(line) for line in FIRST_SEEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    lags = {}
    for block, item in PUBLISHING_ITEM.items():
        records = [r for r in rows if r["item"] == item and r.get("detail")]
        if not records:
            raise RuntimeError(f"no first_seen record for {item}")
        lags[block] = int(any(pd.Timestamp(r["detail"]["last_modified"]) > pd.Timestamp(r["target_date"]).tz_localize("Asia/Kolkata")
                              + pd.Timedelta(hours=CUTOFF_HOUR_IST) for r in records))
    return lags


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        return False


def _raw_ban(sessions) -> dict:
    out = {}
    for day in sessions:
        path = FNO_BAN_DIR / f"{pd.Timestamp(day).date()}.parquet"
        if path.exists():
            out[pd.Timestamp(day)] = set(pd.read_parquet(path)["SYMBOL"].dropna())
    return out


def build_sample(sessions, universe: list[str]):
    rng = np.random.default_rng(SEED)
    recent = [pd.Timestamp(s) for s in sessions if pd.Timestamp(s) >= RECENT_START]
    ban_days = _raw_ban(recent)
    banned = sorted({s for v in ban_days.values() for s in v} & set(universe))
    ban_symbols = banned[:3]
    ban_event_days = [d for d, v in ban_days.items() if v & set(ban_symbols)]
    hits = {}
    for day in recent[-40:]:
        path = PRICE_BAND_HITS_DIR / f"{day.date()}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frame = frame[(frame["SERIES"] == "EQ") & (frame["HIT"] == "H") & frame["SYMBOL"].isin(universe)]
            if len(frame):
                hits[day] = sorted(frame["SYMBOL"])
    hit_symbols = sorted({s for v in hits.values() for s in v})[:3]
    # pledging change: first row where the pledged % moves, on a few symbols that have any pledging history
    pledge_symbols, pledge_days = [], []
    for symbol in universe[:400]:
        f = pd.read_parquet(BLOCK_SPECS["pledging"].path / f"{symbol}.parquet", columns=["date", "pledged_pct_chg", "source_period_end"])
        f = f[f["date"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True)
        new_filing = f[f["source_period_end"].notna() & (f["source_period_end"] != f["source_period_end"].shift()) & (f.index > 0)
                       & (f["pledged_pct_chg"].shift() != f["pledged_pct_chg"])]
        if len(new_filing):
            pledge_symbols.append(symbol)
            pledge_days.append(pd.Timestamp(new_filing["date"].iloc[0]))
        if len(pledge_symbols) >= 3:
            break
    fixed = ban_symbols + hit_symbols + pledge_symbols + ["RELIANCE", "TCS", "INFY"]
    others = [s for s in universe if s not in fixed]
    symbols = list(dict.fromkeys(fixed + list(rng.choice(others, size=20 - len(set(fixed)), replace=False))))[:20]
    events = []
    for d in ban_event_days[:4] + list(hits)[:4] + pledge_days:
        idx = [i for i, s in enumerate(sessions) if pd.Timestamp(s) == d]
        if idx and idx[0] > 0:
            events += [pd.Timestamp(sessions[idx[0] - 1]), d]
    spread = [recent[i] for i in np.linspace(0, len(recent) - 1, 12).astype(int)]
    dates = sorted(set(events + spread))[:30]
    return symbols, dates, ban_days, hits, pledge_symbols, pledge_days


def main() -> int:
    sessions = trading_sessions()
    universe = sorted(p.stem for p in BLOCK_SPECS["delivery"].path.glob("*.parquet"))
    symbols, dates, ban_days, hits, pledge_symbols, pledge_days = build_sample(sessions, universe)
    logger.info(f"sample: {len(symbols)} symbols, {len(dates)} as-of dates ({dates[0].date()} .. {dates[-1].date()}); "
                f"ban symbols/hit symbols/pledge symbols included: {symbols[:9]}")

    full = {b: (pd.concat([pd.read_parquet(BLOCK_SPECS[b].path / f"{s}.parquet") for s in symbols if (BLOCK_SPECS[b].path / f"{s}.parquet").exists()])
                if BLOCK_SPECS[b].scope == "symbol" else pd.read_parquet(BLOCK_SPECS[b].path)) for b in BLOCKS}
    indexes = {b: full[b].set_index(["symbol", "date"] if BLOCK_SPECS[b].scope == "symbol" else "date") for b in BLOCKS}
    lags = expected_lags()
    for b in BLOCKS:
        if BLOCK_SPECS[b].lag_sessions != lags.get(b, 0):
            raise SystemExit(f"{b}: BLOCK_SPECS lag {BLOCK_SPECS[b].lag_sessions} != lag {lags.get(b, 0)} derived from first_seen.jsonl")
    results: dict[str, list[str]] = {k: [] for k in ("NO LEAK", "EXPECTED", "TRUNCATION", "RAW", "LAG", "EVENTS", "MUTANT")}
    reads = {}

    for t in dates:
        for before in (False, True):
            out = read_as_of(symbols, t, BLOCKS, sessions, before_nightly_run=before, lookback_sessions=3)
            reads[(t, before)] = out
            limit = t - pd.Timedelta(days=1) if before else t
            if (out["date"] > limit).any():
                results["NO LEAK"].append(f"{t.date()} before={before}: row dated after {limit.date()}")
            for b in BLOCKS:
                a = out[f"{b}_available_on"]
                if ((a > limit) & ~out[f"{b}_row_absent"]).any():
                    results["NO LEAK"].append(f"{t.date()} {b}: availability after cutoff returned unmasked")
            # EXPECTED: independent lookup straight off the full frames (indexed once), every sampled symbol
            usable = [pd.Timestamp(x) for x in sessions if pd.Timestamp(x) <= limit]
            sess = usable[-3:]
            last_session = usable[-1]
            for b in BLOCKS:
                spec = BLOCK_SPECS[b]
                indexed = indexes[b]
                for symbol in symbols:
                    for day in sess:
                        got = out[(out["symbol"] == symbol) & (out["date"] == day)].iloc[0]
                        key = (symbol, day) if spec.scope == "symbol" else day
                        present = key in indexed.index
                        late = bool(spec.lag_sessions) and day >= last_session          # dated at/after the last usable session
                        for c in spec.value_columns:
                            if not present or late:
                                ok = np.isnan(got[c]) and bool(got[f"{b}_row_absent"])
                            else:
                                want = indexed.at[key, c]
                                ok = (np.isnan(want) and np.isnan(got[c])) or want == got[c] or bool(got[f"{b}_row_absent"] and spec.source_column)
                            if not ok:
                                results["EXPECTED"].append(f"{t.date()} {b} {symbol} {day.date()} {c}")
            # TRUNCATION
            cut = read_as_of(symbols, t, BLOCKS, [s for s in sessions if pd.Timestamp(s) <= t], lookback_sessions=3,
                             before_nightly_run=before, loader=truncated_loader(default_loader, t))
            if not _same(out, cut):
                results["TRUNCATION"].append(f"{t.date()} before={before}")

    # RAW: ban / hits / delivery at t, read with lookback 1 as of t
    raw_checked = {"ban": 0, "hit": 0, "delivery": 0}
    for t in dates:
        out = reads[(t, False)]
        out = out[out["date"] == t]
        if t in ban_days:
            for _, r in out.iterrows():
                if r["fno_ban_row_absent"] or r["is_missing_ban"]:
                    continue
                raw_checked["ban"] += 1
                if bool(r["fno_in_ban"]) != (r["symbol"] in ban_days[t]):
                    results["RAW"].append(f"ban {t.date()} {r['symbol']}: block={r['fno_in_ban']} raw={r['symbol'] in ban_days[t]}")
        path = PRICE_BAND_HITS_DIR / f"{t.date()}.parquet"
        if path.exists():
            raw = pd.read_parquet(path)
            up = set(raw.loc[(raw["SERIES"] == "EQ") & (raw["HIT"] == "H"), "SYMBOL"])
            for _, r in out.iterrows():
                if r["circuits_row_absent"] or r["is_missing_circuit_hits"]:
                    continue
                raw_checked["hit"] += 1
                if bool(r["hit_upper_circuit"]) != (r["symbol"] in up):
                    results["RAW"].append(f"hit {t.date()} {r['symbol']}")
        prev = [pd.Timestamp(x) for x in sessions if pd.Timestamp(x) < t][-1]
        out = reads[(t, False)]
        out = out[out["date"] == prev]                       # delivery for s is usable from the next session: check s = t-1
        dpath = RAW_DATA_DIR / "delivery" / f"{prev.date()}.parquet"
        if dpath.exists():
            raw = pd.read_parquet(dpath, columns=["SYMBOL", "SERIES", "DELIV_PER", "DELIV_QTY", "VOLUME"])
            raw = raw[raw["SERIES"] == "EQ"].set_index("SYMBOL")["DELIV_PER"]
            for _, r in out.iterrows():
                if r["delivery_row_absent"] or r["is_missing_delivery"] or r["symbol"] not in raw.index:
                    continue
                raw_checked["delivery"] += 1
                if abs(raw[r["symbol"]] - r["delivery_pct"]) > 1e-9:
                    results["RAW"].append(f"delivery {prev.date()} {r['symbol']}: {r['delivery_pct']} vs raw {raw[r['symbol']]}")

    # LAG: independent of compute_availability. For each sampled (symbol, t) whose raw block row dated t exists, a lag-1 block
    # must hide it at as_of=t, show the previous session's row, and show row t at the next session; a lag-0 block shows it.
    lag_checked = 0
    sess_all = [pd.Timestamp(x) for x in sessions]
    for t in dates:
        out = reads[(t, False)]
        prev = [x for x in sess_all if x < t][-1]
        for b, lag in lags.items():
            for symbol in symbols:
                if (symbol, t) not in indexes[b].index:
                    continue
                at_t = out[(out["symbol"] == symbol) & (out["date"] == t)].iloc[0][f"{b}_row_absent"]
                lag_checked += 1
                if bool(at_t) != bool(lag):
                    results["LAG"].append(f"{b} {symbol} {t.date()}: row dated t absent={at_t}, expected {bool(lag)}")
                if lag and (symbol, prev) in indexes[b].index:
                    at_prev = out[(out["symbol"] == symbol) & (out["date"] == prev)].iloc[0][f"{b}_row_absent"]
                    if at_prev:
                        results["LAG"].append(f"{b} {symbol}: previous-session row dated {prev.date()} hidden at as_of {t.date()}")
                if lag and symbol in symbols[:5]:
                    nxt = [x for x in sess_all if x > t]
                    if nxt:
                        later = read_as_of([symbol], nxt[0], [b], sessions, before_nightly_run=False, lookback_sessions=2)
                        if later[later["date"] == t][f"{b}_row_absent"].iloc[0]:
                            results["LAG"].append(f"{b} {symbol}: row dated {t.date()} still hidden at the next session")
    logger.info(f"lags from first_seen.jsonl: {lags}; lag cross-checks performed: {lag_checked}")

    # EVENTS
    sess_ts = [pd.Timestamp(s) for s in sessions]
    for symbol, day in zip(pledge_symbols, pledge_days):
        prev = sess_ts[sess_ts.index(day) - 1]
        a = read_as_of([symbol], prev, ["pledging"], sessions, before_nightly_run=False)
        b = read_as_of([symbol], day, ["pledging"], sessions, before_nightly_run=False)
        full_row = full["pledging"][(full["pledging"]["symbol"] == symbol) & (full["pledging"]["date"] == day)].iloc[0]
        if not (b["pledged_pct_chg"].iloc[0] == full_row["pledged_pct_chg"] and a["pledged_pct_chg"].iloc[0] != full_row["pledged_pct_chg"]
                and b["pledging_source_period_end"].iloc[0] == full_row["source_period_end"]
                and a["pledging_source_period_end"].iloc[0] != full_row["source_period_end"]):
            results["EVENTS"].append(f"pledging {symbol} {day.date()} change not confined to t")
    for t in dates:
        out = reads[(t, False)]
        fii = out[(out["symbol"] == symbols[0]) & ~out["flows_fii_row_absent"] & out["fpi_equity_net_cr"].notna()]
        if len(fii) and not (pd.to_datetime(fii["flows_fii_source_date"]) < fii["date"]).all():
            results["EVENTS"].append(f"FII source_date not strictly before row date at {t.date()}")

    # MUTANT: cutoff t+1 must trip check 1
    original = store.knowable_cutoff
    store.knowable_cutoff = lambda as_of, before: original(as_of, before) + pd.Timedelta(days=1)
    try:
        leaked = sum(int((read_as_of(symbols[:3], t, ["delivery"], sessions, before_nightly_run=False)["date"] > t).any()) for t in dates)
    finally:
        store.knowable_cutoff = original
    if leaked == 0:
        results["MUTANT"].append("t+1 reader NOT caught on this sample")

    failed = False
    logger.info(f"raw cross-checks performed: {raw_checked}")
    for name, problems in results.items():
        status = "PASS" if not problems else "FAIL"
        failed |= bool(problems)
        logger.info(f"{status} {name}" + (f": {len(problems)} problems, first {problems[:3]}" if problems else ""))
    logger.info(f"mutant (t+1 reader) leaked on {leaked}/{len(dates)} as-of dates")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
