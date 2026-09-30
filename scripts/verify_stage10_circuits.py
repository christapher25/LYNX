"""
Stage 10 (circuits block) checkpoint verification, against the real files on disk.

Two sub-blocks, each with its own shared missing flag (their coverage differs by ten years, exactly as FII/DII flows do):
  BAND   price_band_pct, is_no_band, band_is_inferred      flag is_missing_circuit_band
  HITS   hit_upper_circuit, hit_lower_circuit               flag is_missing_circuit_hits

Every check below is a `*_violations(...) -> int` function. Each is run twice: on the real output (must return 0, summed
across the universe) and on an in-memory copy of real output with one bug deliberately injected (must return > 0) -- the
same discipline scripts/verify_stage10_flows.py uses. The independent-recompute check never imports
compute_circuit_features (own reader, own dict-lookup arithmetic on the raw sec_list / price_band_hits files), so a bug
shared between the block and its checker cannot hide from this script -- the same discipline
scripts/verify_stage10_delivery.py uses. The truncation check is the deliberate exception: it legitimately re-runs the
REAL function on a truncated real session spine.

  1. Raw coverage: sec_list (band) and price_band_hits (hit) file counts and date ranges, reported.
  2. One circuit-feature file per universe symbol, dates identical 1:1 to that symbol's Stage 4 price rows.
  3. Structure, both sub-blocks independently: flags are non-null bool; is_missing_circuit_band <=> is_no_band and
     band_is_inferred both NaN; is_missing_circuit_hits <=> both hit columns NaN; price_band_pct in {2,5,10,20,40} or
     NaN; hit columns in {0.0, 1.0} or NaN.
  4. Point-in-time BAND: the band on session t is read from the sec_list file dated the PREVIOUS session, never the
     same-day file.
  5. Point-in-time HITS: the hit flag on session t comes from the bh file dated t itself; a session with no published
     list is NaN + flagged, never a look-ahead to the next list.
  6. Independent recompute: both sub-blocks rebuilt from the raw files with this script's own dict-lookup code and
     compared to the stored output, every universe symbol, every row.
  7. No zero-fill / no stale carry: a flagged row never carries a value in any column; a real 0.0 (published list
     naming other symbols but not this one) exists and survives into the output unflagged.
  8. The core circuits.py docstring claim, checked on real data: the band that reproduces a real hit's exact limit
     price agrees with the PREVIOUS session's sec_list file far more often than with the same-day file.
  9. Truncation: cutting a symbol's session spine at several points never changes an earlier row (real
     compute_circuit_features, real full band/hit files).
  10. Injected bugs: same-day band leak, tomorrow's-hits look-ahead, zero-filling, stale carry-forward, injected into
      real output, must be caught by the violation-counters above.
  11. circuit_inference measurement, reported explicitly rather than silently trusted: reads
      data/raw/sec_list/_inference_measurement.csv (scripts/measure_band_inference.py) and prints coverage/accuracy by
      year and method. The combined method's measured accuracy (checked below against an illustrative bar) is well
      short of what would justify treating an inferred pre-2019 band as equivalent to a real one -- it is also
      degrading, not improving, toward the present -- so this checkpoint also confirms band_is_inferred is never 1.0
      anywhere in the real pipeline output: inference is measured and visible, and NOT applied.

Usage: python -m scripts.verify_stage10_circuits
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import ADJUSTED_PRICES_DIR, BHAVCOPY_DIR, CIRCUIT_FEATURES_DIR, PRICE_BAND_HITS_DIR, SEC_LIST_DIR
from data.trading_calendar import trading_sessions
from features.blocks.circuit_inference import implied_band
from features.blocks.circuits import BAND_COLUMNS, CIRCUIT_SERIES, FEATURE_COLUMNS, HIT_COLUMNS, compute_circuit_features
from signals.indicators import list_universe_symbols

INFERENCE_CSV = SEC_LIST_DIR / "_inference_measurement.csv"
# An illustrative bar an inferred band would need to clear before being treated as equivalent to a real one. Nothing in
# the pipeline reads this constant -- the point is to make a shortfall visible, never to gate production on it silently.
INFERENCE_TRUST_BAR = {"coverage": 0.95, "accuracy": 0.97}
failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


def expect_caught(bug: str, violations: int) -> None:
    caught = violations > 0
    (logger.success if caught else logger.error)(
        f"{'PASS' if caught else 'FAIL'}  injected bug caught: {bug}" + (f"  ({violations} violation(s))" if caught else "  <- THE CHECK DID NOT FIRE"))
    if not caught:
        failures.append(f"injected bug escaped: {bug}")


# --- independent raw access ----------------------------------------------------------------------------------------

def load_raw_bands() -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=["DATE", "SYMBOL", "SERIES", "BAND_PCT", "IS_NO_BAND"]) for p in sorted(Path(SEC_LIST_DIR).glob("*.parquet"))]
    both = pd.concat(frames, ignore_index=True)
    return both[both["SERIES"] == CIRCUIT_SERIES].drop(columns="SERIES")


def load_raw_hits() -> tuple[pd.DataFrame, set]:
    frames = [pd.read_parquet(p, columns=["DATE", "SYMBOL", "SERIES", "HIT"]) for p in sorted(Path(PRICE_BAND_HITS_DIR).glob("*.parquet"))]
    both = pd.concat(frames, ignore_index=True)
    published = set(pd.to_datetime(both["DATE"]).dt.normalize().unique())
    named = both[(both["SERIES"] == CIRCUIT_SERIES) & both["SYMBOL"].notna()].drop(columns="SERIES")
    return named, published


def load_panel(symbol: str) -> pd.DatetimeIndex:
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE"])
    return pd.DatetimeIndex(raw["DATE"].sort_values().reset_index(drop=True))


def independent_expected(symbol_bands: pd.DataFrame, symbol_hits: pd.DataFrame, hit_days: set, sessions: pd.DatetimeIndex,
                         rows: pd.DatetimeIndex) -> pd.DataFrame:
    """One symbol's circuit features rebuilt from raw files with plain dict lookups -- a separate code path from
    circuits.compute_circuit_features's numpy reindex/searchsorted implementation."""
    session_list = list(sessions)
    pos = {d: i for i, d in enumerate(session_list)}
    by_file = {pd.Timestamp(r.DATE): (r.BAND_PCT, bool(r.IS_NO_BAND)) for r in symbol_bands.itertuples()}
    up_days = {pd.Timestamp(r.DATE) for r in symbol_hits.itertuples() if r.HIT == "H"}
    lo_days = {pd.Timestamp(r.DATE) for r in symbol_hits.itertuples() if r.HIT == "L"}

    out = {c: [] for c in [*BAND_COLUMNS, "is_missing_circuit_band", *HIT_COLUMNS, "is_missing_circuit_hits"]}
    for row_date in rows:
        i = pos.get(row_date)
        prev_file = by_file.get(session_list[i - 1]) if i is not None and i >= 1 else None
        if prev_file is not None:
            pct, no_band = prev_file
            out["price_band_pct"].append(pct)
            out["is_no_band"].append(float(no_band))
            out["band_is_inferred"].append(0.0)
            out["is_missing_circuit_band"].append(False)
        else:
            out["price_band_pct"].append(np.nan)
            out["is_no_band"].append(np.nan)
            out["band_is_inferred"].append(np.nan)
            out["is_missing_circuit_band"].append(True)
        if row_date in hit_days:
            out["hit_upper_circuit"].append(float(row_date in up_days))
            out["hit_lower_circuit"].append(float(row_date in lo_days))
            out["is_missing_circuit_hits"].append(False)
        else:
            out["hit_upper_circuit"].append(np.nan)
            out["hit_lower_circuit"].append(np.nan)
            out["is_missing_circuit_hits"].append(True)
    df = pd.DataFrame(out)
    df.insert(0, "date", list(rows))
    return df[FEATURE_COLUMNS]


# --- violation counters: each run on real output (must be 0) and on an injected-bug copy (must be > 0) ---------------

def structure_violations(feats: pd.DataFrame) -> int:
    v = 0
    for flag in ("is_missing_circuit_band", "is_missing_circuit_hits"):
        if feats[flag].dtype != bool or feats[flag].isna().any():
            return len(feats)
    band_missing = feats["is_missing_circuit_band"].to_numpy(bool)
    v += int((band_missing & (feats["is_no_band"].notna() | feats["band_is_inferred"].notna())).sum())
    v += int((~band_missing & feats["is_no_band"].isna()).sum())
    hits_missing = feats["is_missing_circuit_hits"].to_numpy(bool)
    v += int((hits_missing & feats[HIT_COLUMNS].notna().any(axis=1)).sum())
    v += int((~hits_missing & feats[HIT_COLUMNS].isna().any(axis=1)).sum())
    numeric_band = feats["price_band_pct"].dropna()
    v += int((~numeric_band.isin([2.0, 5.0, 10.0, 20.0, 40.0])).sum())
    hit_vals = feats[HIT_COLUMNS].to_numpy(float)
    seen = hit_vals[~np.isnan(hit_vals)]
    v += int((~np.isin(seen, [0.0, 1.0])).sum())
    v += int(np.isinf(feats[["price_band_pct", *HIT_COLUMNS]].to_numpy(float)).sum())
    return v


def zero_fill_violations(feats: pd.DataFrame) -> int:
    """A flagged row must never carry a value in ANY of its own sub-block's columns."""
    band_missing = feats["is_missing_circuit_band"].to_numpy(bool)
    hits_missing = feats["is_missing_circuit_hits"].to_numpy(bool)
    v = int(feats.loc[band_missing, BAND_COLUMNS].notna().to_numpy().sum())
    v += int(feats.loc[hits_missing, HIT_COLUMNS].notna().to_numpy().sum())
    return v


def recompute_violations(feats: pd.DataFrame, expected: pd.DataFrame) -> int:
    a, b = feats.reset_index(drop=True), expected.reset_index(drop=True)
    if len(a) != len(b):
        return abs(len(a) - len(b))
    cols = [*BAND_COLUMNS, *HIT_COLUMNS]
    close = np.isclose(a[cols].to_numpy(float), b[cols].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)
    return int((~close.all(axis=1)).sum())


# --- driver checks -----------------------------------------------------------------------------------------------

def check_raw_coverage(bands: pd.DataFrame, hits: pd.DataFrame, hit_days: set) -> None:
    n_band_files, n_hit_files = bands["DATE"].nunique(), len(hit_days)
    logger.info(f"raw sec_list: {len(bands):,} EQ rows over {n_band_files:,} files "
                f"({bands['DATE'].min().date()} .. {bands['DATE'].max().date()})")
    logger.info(f"raw price_band_hits: {len(hits):,} EQ hit rows over {n_hit_files:,} published lists "
                f"(NIL days included; {hits['DATE'].min().date()} .. {hits['DATE'].max().date()})")
    check(n_band_files > 1500, "sec_list has substantial coverage (>1,500 files)", f"({n_band_files:,} files)")
    check(n_hit_files > 4000, "price_band_hits has substantial coverage (>4,000 published lists)", f"({n_hit_files:,} lists)")


def check_universe_files(symbols: list[str]) -> dict[str, pd.DataFrame]:
    files = {p.stem for p in CIRCUIT_FEATURES_DIR.glob("*.parquet")}
    check(files == set(symbols), "one circuit-feature file per universe symbol, no extras",
          f"({len(files):,} files, {len(symbols):,} symbols; missing={sorted(set(symbols) - files)[:5]}, "
          f"extra={sorted(files - set(symbols))[:5]})")
    loaded, row_mismatch, misaligned = {}, [], []
    for symbol in symbols:
        feats = pd.read_parquet(CIRCUIT_FEATURES_DIR / f"{symbol}.parquet")
        panel = load_panel(symbol)
        if len(feats) != len(panel):
            row_mismatch.append(symbol)
        elif feats["date"].tolist() != panel.tolist():
            misaligned.append(symbol)
        else:
            loaded[symbol] = feats
    check(not row_mismatch and not misaligned, "feature rows == Stage 4 price rows, dates identical 1:1, every symbol",
          f"(row-count mismatches={row_mismatch[:5]}, date mismatches={misaligned[:5]})")
    return loaded


_BAND_FILE_CACHE: dict = {}


def _band_lookup(symbol: str, file_date: pd.Timestamp) -> float:
    if file_date not in _BAND_FILE_CACHE:
        p = SEC_LIST_DIR / f"{file_date:%Y-%m-%d}.parquet"
        _BAND_FILE_CACHE[file_date] = (pd.read_parquet(p, columns=["SYMBOL", "SERIES", "BAND_PCT"]).query("SERIES == 'EQ'").set_index("SYMBOL")["BAND_PCT"]
                                       if p.exists() else pd.Series(dtype=float))
    s = _BAND_FILE_CACHE[file_date]
    return float(s[symbol]) if symbol in s.index and pd.notna(s[symbol]) else float("nan")


def load_raw_bhavcopy_day(trade_date: pd.Timestamp) -> pd.DataFrame:
    """The exchange's OWN unadjusted PREV_CLOSE/HIGH/LOW for one day, EQ series only -- circuit bands are computed
    against what NSE itself printed that day, never against Stage 4's split/bonus-adjusted prices (the bug that was
    here: using ADJUSTED_PRICES_DIR silently corrupts the implied band on every symbol with any adjustment factor,
    which is why a first pass of this check read 45% instead of the ~97% circuits.py's own docstring measured)."""
    p = BHAVCOPY_DIR / f"{trade_date:%Y-%m-%d}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["SYMBOL", "PREV_CLOSE", "HIGH", "LOW"]).set_index("SYMBOL")
    df = pd.read_parquet(p, columns=["SYMBOL", "SERIES", "PREV_CLOSE", "HIGH", "LOW"])
    return df[df["SERIES"] == "EQ"].drop(columns="SERIES").drop_duplicates("SYMBOL").set_index("SYMBOL")


def check_hit_reproduces_previous_session_band(hits: pd.DataFrame, sessions: pd.DatetimeIndex) -> None:
    """The core circuits.py docstring claim: the band that reproduces a real hit's exact limit price agrees with the
    PREVIOUS session's sec_list file far more often than with the same-day file. Own tick math via implied_band, on
    the exchange's own raw (unadjusted) bhavcopy prices -- the same source scripts/measure_band_inference.py uses."""
    random.seed(21)
    sec_list_start = pd.Timestamp("2019-12-30")          # sec_list has no file before this; a pre-2019 sample day would
    eligible_days = sorted(d for d in hits["DATE"].dt.normalize().unique() if pd.Timestamp(d) >= sec_list_start)
    sample_days = random.sample(eligible_days, min(40, len(eligible_days)))
    session_list = list(sessions)
    pos = {d: i for i, d in enumerate(session_list)}
    bhav_cache: dict = {}
    n_prev_match = n_same_match = n_total = 0
    for d in sample_days:
        d = pd.Timestamp(d)
        i = pos.get(d)
        if i is None or i < 1:
            continue
        if d not in bhav_cache:
            bhav_cache[d] = load_raw_bhavcopy_day(d)
        day_prices = bhav_cache[d]
        for rec in hits[hits["DATE"] == d].itertuples():
            symbol = rec.SYMBOL
            if symbol not in day_prices.index:
                continue
            prev_close = float(day_prices.loc[symbol, "PREV_CLOSE"])       # NSE's own reference price for THIS day's limits
            extreme = float(day_prices.loc[symbol, "HIGH"]) if rec.HIT == "H" else float(day_prices.loc[symbol, "LOW"])
            true_band = implied_band(prev_close, extreme, rec.HIT)
            if np.isnan(true_band):
                continue
            n_total += 1
            if _band_lookup(symbol, session_list[i - 1]) == true_band:
                n_prev_match += 1
            if _band_lookup(symbol, d) == true_band:
                n_same_match += 1
    if n_total == 0:
        check(False, "hit-reproduces-band sample produced comparable rows", "(0 comparable hits in the sample)")
        return
    prev_rate, same_rate = n_prev_match / n_total, n_same_match / n_total
    logger.info(f"hit-implied-band agreement (n={n_total}): previous-session file {prev_rate:.1%}, same-day file {same_rate:.1%}")
    check(prev_rate > 0.85 and prev_rate > same_rate, "the previous session's band file reproduces real hit limit prices "
          "far more often than the same-day file", f"(previous={prev_rate:.1%}, same-day={same_rate:.1%})")


def check_truncation(bands_by: dict, hits_by: dict, hit_days: set, sessions: pd.DatetimeIndex, universe: set[str]) -> None:
    """Full bands/hits/hit_days are passed UNCHANGED at every cut; only the session/date spine is truncated -- a future
    band or hit file cannot be looked up by a row that no longer exists, so this also doubles as a structural proof of
    point-in-time behaviour, the same logic the pledging verifier's truncation check relies on. Candidates are
    restricted to the adjusted-price universe: sec_list carries every NSE EQ symbol, most of which (e.g. PANORAMA)
    were never in a Stage 5 universe snapshot and have no data/processed/adjusted/ file to load a panel from."""
    random.seed(17)
    candidates = [s for s in bands_by if s in universe and len(bands_by[s]) >= 5]
    sample = random.sample(candidates, min(12, len(candidates)))
    empty_bands = pd.DataFrame(columns=["DATE", "BAND_PCT", "IS_NO_BAND"])
    empty_hits = pd.DataFrame(columns=["DATE", "HIT"])
    violations = []
    for symbol in sample:
        panel = load_panel(symbol)
        b, h = bands_by.get(symbol, empty_bands), hits_by.get(symbol, empty_hits)
        full = compute_circuit_features(panel, b, h, hit_days, sessions)
        for frac in (0.3, 0.6, 0.9):
            cut = sessions[int(len(sessions) * frac)]
            part_panel = panel[panel <= cut]
            part_sessions = sessions[sessions <= cut]
            if len(part_panel) < 2:
                continue
            part = compute_circuit_features(part_panel, b, h, hit_days, part_sessions)
            a = full[full["date"] <= cut].reset_index(drop=True)
            c = part[part["date"] <= cut].reset_index(drop=True)
            if recompute_violations(a, c) > 0:
                violations.append(symbol)
    check(not violations, "truncation: cutting a symbol's sessions never changes an earlier row (real function, real full files)",
          f"({len(sample)} symbols x 3 cuts sampled; violations: {sorted(set(violations))[:5]})")


def report_inference_measurement() -> None:
    if not INFERENCE_CSV.exists():
        check(False, "circuit_inference measurement evidence exists on disk", f"(missing: {INFERENCE_CSV}; run python -m scripts.measure_band_inference)")
        return
    table = pd.read_csv(INFERENCE_CSV)
    combined_all = table[(table["method"] == "combined") & (table["year"] == "all")]
    if combined_all.empty:
        check(False, "measurement table has an all-years 'combined' row", "(table malformed)")
        return
    row = combined_all.iloc[0]
    logger.info("circuit_inference measurement (scripts/measure_band_inference.py), combined method, all years: "
                f"coverage={row['coverage']:.1%}, accuracy={row['accuracy']:.1%}, "
                f"under-estimate={row['under']:.1%}, over-estimate={row['over']:.1%}")
    by_year = table[(table["method"] == "combined") & (table["year"] != "all")].sort_values("year")
    trend = ", ".join(f"{r.year}={r.accuracy:.1%}" for r in by_year.itertuples())
    logger.info(f"combined-method accuracy by year: {trend}")
    meets_bar = row["coverage"] >= INFERENCE_TRUST_BAR["coverage"] and row["accuracy"] >= INFERENCE_TRUST_BAR["accuracy"]
    logger.warning(f"combined method vs an illustrative trust bar (coverage>={INFERENCE_TRUST_BAR['coverage']:.0%}, "
                   f"accuracy>={INFERENCE_TRUST_BAR['accuracy']:.0%}): {'MEETS IT' if meets_bar else 'DOES NOT MEET IT'}"
                   f"{'' if meets_bar else ' -- this is why it is not applied to the real output.'}")


def check_inference_not_applied(loaded: dict[str, pd.DataFrame]) -> None:
    applied = sum(int((f["band_is_inferred"] == 1.0).sum()) for f in loaded.values())
    check(applied == 0, "band_is_inferred is never 1.0 anywhere in the real pipeline output "
          "(inference is measured and reported above, not applied)", f"(rows with inference applied: {applied})")


def main() -> int:
    logger.info("loading raw sec_list and price_band_hits archives independently ...")
    bands = load_raw_bands()
    hits, hit_days = load_raw_hits()
    check_raw_coverage(bands, hits, hit_days)
    bands_by = {s: g for s, g in bands.groupby("SYMBOL", sort=False)}
    hits_by = {s: g for s, g in hits.groupby("SYMBOL", sort=False)}
    empty_bands = pd.DataFrame(columns=["DATE", "BAND_PCT", "IS_NO_BAND"])
    empty_hits = pd.DataFrame(columns=["DATE", "HIT"])

    symbols = list_universe_symbols()
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions())).normalize()
    loaded = check_universe_files(symbols)

    # 3. structure, universe-wide
    struct_total = sum(structure_violations(f) for f in loaded.values())
    check(struct_total == 0, "structure: both sub-blocks' flags agree with their NaN patterns, values in range, no inf", f"(violations: {struct_total})")

    # 4-6. point-in-time + independent recompute, every universe symbol, every row (band/hits point-in-time IS the recompute here)
    recompute_total = 0
    n_real_zero_hits = 0
    for symbol, feats in loaded.items():
        expected = independent_expected(bands_by.get(symbol, empty_bands), hits_by.get(symbol, empty_hits), hit_days,
                                        sessions, pd.DatetimeIndex(feats["date"]))
        recompute_total += recompute_violations(feats, expected)
        n_real_zero_hits += int(((~feats["is_missing_circuit_hits"]) & (feats["hit_upper_circuit"] == 0.0)).sum())
    check(recompute_total == 0, "independent recompute (band from the previous session's file, hits from the same-day list) "
          "matches the stored output", f"({len(loaded):,} symbols checked; violations: {recompute_total})")

    # 7. no zero-fill, universe-wide, plus proof a real zero exists and survives
    zero_total = sum(zero_fill_violations(f) for f in loaded.values())
    check(zero_total == 0, "no flagged row carries a value in either sub-block, across every symbol", f"(violations: {zero_total})")
    logger.info(f"published-list-but-no-hit rows (real 0.0, not missing): {n_real_zero_hits:,}")
    check(n_real_zero_hits > 0, "a published hit list naming other symbols but not this one is a real 0.0 in the real output "
          "(otherwise the zero-fill check above proves nothing for HITS)")

    # 8. the docstring's own claim, on real data
    check_hit_reproduces_previous_session_band(hits, sessions)

    # 9. truncation
    check_truncation(bands_by, hits_by, hit_days, sessions, set(loaded))

    # 10. injected bugs
    logger.info("--- injected bugs (in-memory copies of real output; each must be caught) ---")
    symbol = next(s for s in sorted(loaded) if (~loaded[s]["is_missing_circuit_band"]).sum() > 30 and (~loaded[s]["is_missing_circuit_hits"]).any())
    feats = loaded[symbol]
    expected = independent_expected(bands_by.get(symbol, empty_bands), hits_by.get(symbol, empty_hits), hit_days, sessions, pd.DatetimeIndex(feats["date"]))

    same_day = feats.copy()
    same_day[BAND_COLUMNS] = same_day[BAND_COLUMNS].shift(-1)          # tomorrow's row's band pulled back = today reads its OWN file
    same_day["is_missing_circuit_band"] = same_day["is_no_band"].isna()
    expect_caught("same-day band file read instead of the previous session's", recompute_violations(same_day, expected))

    tomorrow_hits = feats.copy()
    tomorrow_hits[HIT_COLUMNS] = tomorrow_hits[HIT_COLUMNS].shift(-1)
    tomorrow_hits["is_missing_circuit_hits"] = tomorrow_hits["hit_upper_circuit"].isna()
    expect_caught("hit flag taken from the next session's list (look-ahead)", recompute_violations(tomorrow_hits, expected))

    zero_filled = feats.copy()
    zero_filled.loc[zero_filled["is_missing_circuit_hits"], HIT_COLUMNS] = 0.0
    expect_caught("missing hit rows filled with 0", zero_fill_violations(zero_filled))

    stale = feats.copy()
    for c in BAND_COLUMNS:
        stale[c] = stale[c].ffill()
    stale.loc[feats["is_missing_circuit_band"] & stale["is_no_band"].notna(), "is_missing_circuit_band"] = False
    expect_caught("stale carry-forward (ffill) of the last known band", recompute_violations(stale, expected))

    unflagged = feats.copy()
    missing_idx = feats.index[feats["is_missing_circuit_band"]]
    if len(missing_idx):
        unflagged.loc[missing_idx[-1], "is_missing_circuit_band"] = False
        expect_caught("missing band flag cleared on a NaN row", structure_violations(unflagged))

    # 11. inference measurement
    report_inference_measurement()
    check_inference_not_applied(loaded)

    if failures:
        logger.error(f"Stage 10 circuits checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 circuits checkpoint PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
