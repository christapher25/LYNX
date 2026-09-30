"""
Stage 10 (delivery block) checkpoint verification.

Stage 10's passing check: "Delivery features validated against raw file; missingness masked with
indicator, never zero-filled". This script validates the Parquet written by features/blocks/delivery.py
against the RAW delivery files and the Stage 4 adjusted prices, using its own reader and its own
plain-loop arithmetic — it deliberately imports nothing from features.blocks.delivery, so a bug in
that module cannot hide from the code that checks it.

  1. One feature file per universe symbol, dates identical 1:1 to the symbol's Stage 4 price rows.
  2. Structural invariants, universe-wide: flag is a non-null bool; flag True => all four features NaN;
     flag False => delivery_pct present and within 0-100; no inf; nothing observed before the raw archive starts.
  3. is_missing_delivery matches an independently derived usability rule on EVERY row, every symbol.
  4. Zero is never a fill: rows with delivery_pct == 0 are exactly the raw rows that reported 0.
  5. Features recomputed from raw with loops match the stored values (sample symbols, all four features).
  6. Quarantine accounting: every raw observation that was withheld is one the raw file contradicts.
  7. Split/bonus days: delivery_qty_ratio_20d does not spike where the same days do spike unadjusted.

Usage: python -m scripts.verify_stage10_delivery
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from config import ADJUSTED_PRICES_DIR, DELIVERY_DIR, DELIVERY_FEATURES_DIR
from signals.indicators import list_universe_symbols

# Must match features/blocks/delivery.py. A drift between the two is precisely what checks 3 and 5 catch.
TOLERANCE_PP = 0.05
MIN_OBS_20D = 15
FEATURES = ["delivery_pct", "delivery_pct_chg_5d", "delivery_pct_chg_20d", "delivery_qty_ratio_20d"]
ALWAYS_SAMPLED = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "KIOCL"]

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


# --- independent raw access ----------------------------------------------------------------------

def load_raw_eq() -> pd.DataFrame:
    """Every EQ row from every raw file, dated by the DATE/DATE1 column inside the file (never its name)."""
    frames = []
    for path in sorted(DELIVERY_DIR.glob("*.parquet")):
        names = pq.read_schema(path).names
        date_col = "DATE1" if "DATE1" in names else "DATE"
        vol_col = "TTL_TRD_QNTY" if "TTL_TRD_QNTY" in names else "VOLUME"
        d = pd.read_parquet(path, columns=["SYMBOL", "SERIES", date_col, "DELIV_QTY", "DELIV_PER", vol_col])
        d = d[d["SERIES"].astype(str).str.strip() == "EQ"].drop(columns="SERIES")
        d.columns = ["symbol", "date", "qty", "per", "vol"]
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["symbol"] = raw["symbol"].astype(str).str.strip()
    raw["date"] = pd.to_datetime(raw["date"])
    return raw.drop_duplicates(["symbol", "date"])


def load_panel(symbol: str) -> pd.DataFrame:
    p = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE", "VOLUME", "ADJ_VOLUME"])
    p = p.sort_values("DATE").reset_index(drop=True)
    p["factor"] = p["ADJ_VOLUME"] / p["VOLUME"].where(p["VOLUME"] > 0)
    return p.rename(columns={"DATE": "date"})


def independent_usable(joined: pd.DataFrame) -> pd.Series:
    """Re-derivation of 'this raw observation may be trusted', written as plain boolean algebra."""
    have = joined["qty"].notna() & joined["per"].notna()
    vol_ok = joined["vol"].notna() & (joined["vol"] > 0)
    agrees = ((joined["qty"] / joined["vol"].where(vol_ok) * 100) - joined["per"]).abs() <= TOLERANCE_PP
    return have & vol_ok & agrees.fillna(False) & (joined["factor"] > 0)


def loop_features(joined: pd.DataFrame) -> pd.DataFrame:
    """The four features from first principles, one element at a time."""
    n = len(joined)
    usable = independent_usable(joined).to_numpy()
    per, qty, factor = joined["per"].to_numpy(float), joined["qty"].to_numpy(float), joined["factor"].to_numpy(float)
    pct = np.full(n, np.nan)
    adj = np.full(n, np.nan)
    for i in range(n):
        if usable[i]:
            pct[i] = per[i]
            adj[i] = qty[i] * factor[i]
    out = {k: np.full(n, np.nan) for k in FEATURES}
    out["delivery_pct"] = pct
    for i in range(n):
        if i >= 5 and not np.isnan(pct[i]) and not np.isnan(pct[i - 5]):
            out["delivery_pct_chg_5d"][i] = pct[i] - pct[i - 5]
        if i >= 20 and not np.isnan(pct[i]) and not np.isnan(pct[i - 20]):
            out["delivery_pct_chg_20d"][i] = pct[i] - pct[i - 20]
        window = adj[max(0, i - 19): i + 1]
        seen = window[~np.isnan(window)]
        if len(seen) >= MIN_OBS_20D and not np.isnan(adj[i]):
            mean = seen.mean()
            if mean > 0:
                out["delivery_qty_ratio_20d"][i] = adj[i] / mean
    return pd.DataFrame(out)


# --- checks ----------------------------------------------------------------------------------------

def main() -> int:
    symbols = list_universe_symbols()
    files = {p.stem for p in DELIVERY_FEATURES_DIR.glob("*.parquet")}
    check(files == set(symbols), "one delivery-feature file per universe symbol, no extras",
          f"({len(files):,} files, {len(symbols):,} symbols; missing={sorted(set(symbols) - files)[:5]}, "
          f"extra={sorted(files - set(symbols))[:5]})")

    logger.info("loading raw delivery archive with an independent reader ...")
    raw = load_raw_eq()
    raw_start = raw["date"].min()
    raw_by_symbol = {s: g.set_index("date") for s, g in raw.groupby("symbol", sort=False)}
    logger.info(f"raw EQ rows: {len(raw):,} across {len(raw_by_symbol):,} symbols, {raw_start.date()} .. {raw['date'].max().date()}")

    misaligned, row_mismatch, bad_dtype = [], [], []
    flag_leak = pct_missing_when_present = out_of_range = infs = observed_before_start = 0
    flag_disagree, zero_feature_rows, zero_raw_usable_rows = [], 0, 0
    total_rows = total_missing = 0
    quarantined_dates: Counter = Counter()
    quarantine_unexplained = 0
    events: list[tuple[str, pd.Timestamp, float, float]] = []
    sample_pool: list[str] = []
    split_symbols: list[str] = []

    for symbol in symbols:
        feats = pd.read_parquet(DELIVERY_FEATURES_DIR / f"{symbol}.parquet")
        panel = load_panel(symbol)
        if len(feats) != len(panel):
            row_mismatch.append(symbol)
        if feats["date"].tolist() != panel["date"].tolist():
            misaligned.append(symbol)
            continue
        if feats["is_missing_delivery"].dtype != bool or feats["is_missing_delivery"].isna().any():
            bad_dtype.append(symbol)

        flagged = feats["is_missing_delivery"].to_numpy(bool)
        block = feats[FEATURES]
        total_rows += len(feats)
        total_missing += int(flagged.sum())
        flag_leak += int(block[flagged].notna().any(axis=1).sum())
        pct_missing_when_present += int(feats.loc[~flagged, "delivery_pct"].isna().sum())
        out_of_range += int(((feats["delivery_pct"] < 0) | (feats["delivery_pct"] > 100)).sum())
        infs += int(np.isinf(block.to_numpy(float)).sum())
        observed_before_start += int(((feats["date"] < raw_start) & ~feats["is_missing_delivery"]).sum())

        joined = panel.join(raw_by_symbol[symbol][["qty", "per", "vol"]], on="date") if symbol in raw_by_symbol \
            else panel.assign(qty=np.nan, per=np.nan, vol=np.nan)
        usable = independent_usable(joined).to_numpy()
        if not np.array_equal(usable, ~flagged):
            flag_disagree.append(symbol)
        zero_feature_rows += int((feats["delivery_pct"] == 0).sum())
        zero_raw_usable_rows += int((usable & (joined["per"].to_numpy(float) == 0)).sum())

        present = (joined["qty"].notna() & joined["per"].notna()).to_numpy()
        withheld = present & flagged
        for d in joined.loc[withheld, "date"]:
            quarantined_dates[d.strftime("%Y-%m-%d")] += 1
        contradicted = present & ~usable
        quarantine_unexplained += int((withheld & ~contradicted).sum())

        if (~flagged).any():
            sample_pool.append(symbol)
            f = joined["factor"].to_numpy(float)
            for i in np.flatnonzero(np.abs(np.log(f[:-1] / f[1:])) > np.log(1.2)) + 1:   # factor jumps at an ex-date
                if not flagged[i] and joined["date"].iloc[i] >= raw_start:
                    split_symbols.append(symbol)
                    window = joined["qty"].to_numpy(float)[max(0, i - 19): i + 1]
                    seen = window[~np.isnan(window)]
                    if len(seen) >= MIN_OBS_20D and seen.mean() > 0:
                        events.append((symbol, joined["date"].iloc[i], float(feats["delivery_qty_ratio_20d"].iloc[i]),
                                       float(joined["qty"].iloc[i] / seen.mean())))

    check(not row_mismatch and not misaligned, "feature rows == Stage 4 price rows, dates identical 1:1, every symbol",
          f"({total_rows:,} rows; row-count mismatches={row_mismatch[:5]}, date mismatches={misaligned[:5]})")
    check(not bad_dtype, "is_missing_delivery is a non-null bool in every file", f"(violations: {bad_dtype[:5]})")
    check(flag_leak == 0, "flag True => all four features NaN (no value leaks through a missing row)", f"(violations: {flag_leak})")
    check(pct_missing_when_present == 0, "flag False => delivery_pct present", f"(violations: {pct_missing_when_present})")
    check(out_of_range == 0 and infs == 0, "delivery_pct within 0-100; no inf anywhere", f"(out of range: {out_of_range}, inf: {infs})")
    check(observed_before_start == 0, f"nothing observed before the raw archive starts ({raw_start.date()})", f"(violations: {observed_before_start})")
    check(not flag_disagree, "is_missing_delivery == independently derived usability, every row of every symbol",
          f"(symbols that disagree: {flag_disagree[:5]})")
    check(zero_feature_rows == zero_raw_usable_rows, "delivery_pct == 0 only where the raw file itself reported 0 (never a fill)",
          f"(feature rows at 0: {zero_feature_rows}, raw usable rows reporting 0: {zero_raw_usable_rows})")
    logger.info(f"flagged missing: {total_missing:,} of {total_rows:,} rows ({total_missing / total_rows:.1%}); "
                f"symbols with any delivery: {len(sample_pool):,}")

    # 5. loop recomputation vs stored, on a sample
    random.seed(11)
    sample = [s for s in ALWAYS_SAMPLED if s in sample_pool]
    sample += sorted(set(split_symbols))[:10]
    sample += random.sample([s for s in sample_pool if s not in sample], 12)
    bad_sample = []
    for symbol in sample:
        feats = pd.read_parquet(DELIVERY_FEATURES_DIR / f"{symbol}.parquet")
        panel = load_panel(symbol)
        joined = panel.join(raw_by_symbol[symbol][["qty", "per", "vol"]], on="date")
        expected = loop_features(joined)
        for col in FEATURES:
            a, b = feats[col].to_numpy(float), expected[col].to_numpy(float)
            if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)):
                bad_sample.append((symbol, col))
    check(not bad_sample, "all four features == loop recomputation from the raw files",
          f"({len(sample)} symbols sampled, {min(10, len(set(split_symbols)))} chosen for having corporate actions "
          f"[{len(set(split_symbols))} universe symbols have ex-dates in range]; mismatches={bad_sample[:5]})")

    # 6. quarantine accounting
    n_quarantined = sum(quarantined_dates.values())
    check(quarantine_unexplained == 0, "every withheld raw observation is one the raw file contradicts (none dropped without cause)",
          f"(unexplained: {quarantine_unexplained})")
    logger.info(f"withheld raw observations: {n_quarantined:,} on {len(quarantined_dates)} dates: {dict(quarantined_dates.most_common(8))}")

    # 7. split/bonus days
    if events:
        adj_dev = np.median([abs(a - 1) for _, _, a, _ in events])
        raw_dev = np.median([abs(r - 1) for _, _, _, r in events])
        check(len(events) >= 5 and adj_dev < raw_dev,
              "at real split/bonus ex-dates the adjusted ratio stays near 1 while the unadjusted one spikes",
              f"({len(events)} events in {len(set(s for s, *_ in events))} symbols; median |ratio-1|: adjusted={adj_dev:.2f}, unadjusted={raw_dev:.2f})")
        worst = sorted(events, key=lambda e: -abs(e[3] - 1))[:3]
        logger.info("largest unadjusted spikes: " + "; ".join(f"{s} {d:%Y-%m-%d} adj={a:.2f} raw={r:.2f}" for s, d, a, r in worst))
    else:
        check(False, "split/bonus ex-date events found to test against", "(none found)")

    if failures:
        logger.error(f"Stage 10 delivery checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 delivery checkpoint PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
