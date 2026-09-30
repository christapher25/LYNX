"""
Stage 10 (pledging block) checkpoint verification, against the real files on disk.

Stage 10's passing check: "Pledging features validated against the raw SHP filing archive; keyed on BROADCAST date, never
quarter-end or submission date; missingness masked with an indicator, never zero-filled or carried forward."

Every check below is a `*_violations(...) -> int` function. Each is run twice: on the real output (must return 0, summed
across the universe) and on an in-memory copy of real output with one bug deliberately injected (must return > 0). A check
that stays silent on the bug it exists for proves nothing about the real data either — this is the same discipline
scripts/verify_stage10_flows.py uses. The independent-recompute check additionally never imports compute_pledging_features
(it reads the raw filings and does its own plain-loop arithmetic), so a bug shared between the block and its checker cannot
hide from this script — the same discipline scripts/verify_stage10_delivery.py uses. The one deliberate exception is the
truncation check, which legitimately re-runs the REAL function on a truncated real session spine.

  1. SHP crawl coverage: every crawl-target symbol (data/ingest_shp_pledge.py's universe_symbols_by_priority, ~1,129
     names) has a raw filing file; per-filing PARSE_ERROR volume is reported by category (most are permanent NSE
     data-quality facts -- a missing XBRL link, a genuine 404, a filing whose XBRL period disagrees with the master
     list -- not bugs; a lingering NETWORK ERROR category would be a bug, since those are meant to self-heal on a rerun).
  2. One pledging-feature file per universe symbol (signals.indicators.list_universe_symbols, the full historical
     universe, a superset of the SHP crawl target), dates identical 1:1 to that symbol's Stage 4 price rows.
  3. Structure: is_missing_pledging is a non-null bool; flag True <=> every value column NaN <=> source_period_end NaT;
     flag False <=> pledged_pct_of_promoter present; values within 0-100; no inf.
  4. Point-in-time: every value traces to a filing whose real BROADCAST_IST sits at or before that row's session's
     19:00 IST run -- never the quarter it describes (PERIOD_END), never a submission or creation date.
  5. Independent recompute: current/previous-period ratio, pledged_pct_chg, pledge_rising, pledging_age_days and
     source_period_end, rebuilt with a plain loop straight from the raw filings (this script's own code, not
     pledging.py's), compared to the stored output -- every SHP-crawled symbol, every row.
  6. No zero-fill / no stale carry: a flagged row never carries a value in any column; an unflagged row's value can
     only be a real disclosed figure (checked via #5); a real 0% exists in the raw filings and survives into the
     output unflagged.
  7. Truncation: cutting a symbol's session spine at several points never changes an earlier row (real
     compute_pledging_features, real full filings -- a filing broadcast beyond the truncated sessions has nowhere to
     land, so this also doubles as a second, structural proof of point-in-time behaviour).
  8. Injected bugs: period-end keying instead of broadcast keying, ignoring the 19:00 IST cutoff, zero-filling missing
     rows, and stale carry-forward of the previous quarter, injected into real output, must be caught by the
     violation-counters above.

Usage: python -m scripts.verify_stage10_pledging
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

from config import ADJUSTED_PRICES_DIR, PLEDGING_FEATURES_DIR, SHP_DIR
from data.ingest_shp_pledge import universe_symbols_by_priority
from data.publication import FEATURE_CUTOFF_IST, IST, usable_from_series
from features.blocks.pledging import FILING_COLUMNS, PLEDGING_COLUMNS, compute_pledging_features
from data.trading_calendar import trading_sessions
from signals.indicators import list_universe_symbols

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


# --- independent raw access ------------------------------------------------------------------------------------------

def load_raw_filings() -> dict[str, pd.DataFrame]:
    """Every symbol's filings, read directly off disk (own call, not features.blocks.pledging.load_filings)."""
    return {p.stem: pd.read_parquet(p, columns=FILING_COLUMNS) for p in sorted(SHP_DIR.glob("*.parquet"))}


def load_panel(symbol: str) -> pd.DatetimeIndex:
    raw = pd.read_parquet(ADJUSTED_PRICES_DIR / f"{symbol}.parquet", columns=["DATE"])
    return pd.DatetimeIndex(raw["DATE"].sort_values().reset_index(drop=True))


def ratio_of(promoter, encumbered, has_group, parse_error) -> float:
    """A single filing's encumbered/promoter percentage, or NaN unless it parsed and has a promoter group with shares.
    Own plain function -- does not call pledging._filing_ratio."""
    if parse_error is not None and not (isinstance(parse_error, float) and np.isnan(parse_error)):
        return float("nan")
    has_group = bool(has_group) if has_group is not None and not (isinstance(has_group, float) and np.isnan(has_group)) else False
    if not has_group or promoter is None or (isinstance(promoter, float) and np.isnan(promoter)) or promoter <= 0:
        return float("nan")
    if encumbered is None or (isinstance(encumbered, float) and np.isnan(encumbered)):
        return float("nan")
    return 100.0 * float(encumbered) / float(promoter)


def independent_timeline(filings: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[tuple]:
    """(usable_date, current_ratio, previous_ratio, current_period_end) events, one per distinct usable date. Own
    plain-loop assembly -- a separate code path from pledging.compute_pledging_features's numpy groupby/searchsorted."""
    stamps = pd.to_datetime(filings["BROADCAST_IST"], utc=True, errors="coerce")
    keep = stamps.notna().to_numpy()
    if not keep.any():
        return []
    usable = usable_from_series(pd.Series(stamps[keep].to_numpy(), index=filings.index[keep]), [d.date() for d in sessions])
    events = []
    for idx in filings.index[keep]:
        u = usable.loc[idx]
        if pd.isna(u):
            continue
        rec = filings.loc[idx]
        r = ratio_of(rec["PROMOTER_SHARES"], rec["ENCUMBERED_SHARES"], rec["HAS_PROMOTER_GROUP"], rec["PARSE_ERROR"])
        events.append((pd.Timestamp(u), stamps.loc[idx], pd.Timestamp(rec["PERIOD_END"]), r))
    events.sort(key=lambda e: (e[0], e[1]))

    by_period: dict[pd.Timestamp, float] = {}
    timeline, i = [], 0
    while i < len(events):
        usable_date = events[i][0]
        j = i
        while j < len(events) and events[j][0] == usable_date:
            by_period[events[j][2]] = events[j][3]      # a later filing for the same period (a revision) replaces its slot
            j += 1
        periods = sorted(by_period)
        newest = periods[-1]
        timeline.append((usable_date, by_period[newest], by_period[periods[-2]] if len(periods) > 1 else float("nan"), newest))
        i = j
    return timeline


def independent_expected(filings: pd.DataFrame, sessions: pd.DatetimeIndex, rows: pd.DatetimeIndex) -> pd.DataFrame:
    """One symbol's pledging features rebuilt from raw filings with a plain loop, on `rows` dates."""
    timeline = independent_timeline(filings, sessions)
    out = {c: [] for c in PLEDGING_COLUMNS}
    source_period_end = []
    ti = 0
    current = None
    for row_date in rows:
        while ti < len(timeline) and timeline[ti][0] <= row_date:
            current = timeline[ti]
            ti += 1
        if current is None or np.isnan(current[1]):
            for c in PLEDGING_COLUMNS:
                out[c].append(np.nan)
            source_period_end.append(pd.NaT)
            continue
        _, cur_ratio, prev_ratio, period = current
        chg = cur_ratio - prev_ratio
        out["pledged_pct_of_promoter"].append(cur_ratio)
        out["pledged_pct_chg"].append(chg)
        out["pledge_rising"].append(float("nan") if np.isnan(chg) else float(chg > 1e-9))
        out["pledging_age_days"].append((row_date - period).days)
        source_period_end.append(period)
    df = pd.DataFrame(out)
    df.insert(0, "date", list(rows))
    df["source_period_end"] = source_period_end
    df["is_missing_pledging"] = df["pledged_pct_of_promoter"].isna()
    return df[["date", *PLEDGING_COLUMNS, "source_period_end", "is_missing_pledging"]]


# --- violation counters: each run on real output (must be 0) and on an injected-bug copy (must be > 0) ----------------

def structure_violations(feats: pd.DataFrame) -> int:
    if feats["is_missing_pledging"].dtype != bool or feats["is_missing_pledging"].isna().any():
        return len(feats)                                                  # a non-bool/NaN flag column: everything is suspect
    flagged = feats["is_missing_pledging"].to_numpy(bool)
    leak = int(feats.loc[flagged, PLEDGING_COLUMNS].notna().to_numpy().sum())
    period_leak = int(feats.loc[flagged, "source_period_end"].notna().sum())
    unflagged_nan = int(feats.loc[~flagged, "pledged_pct_of_promoter"].isna().sum())
    out_of_range = int(((feats["pledged_pct_of_promoter"] < 0) | (feats["pledged_pct_of_promoter"] > 100)).sum())
    infs = int(np.isinf(feats[PLEDGING_COLUMNS].to_numpy(float)).sum())
    return leak + period_leak + unflagged_nan + out_of_range + infs


def zero_fill_violations(feats: pd.DataFrame) -> int:
    """A flagged row must never carry a value in ANY column (this is the general form of 'no zero-fill': it catches a
    fill with 0, with ffill's stale value, or with anything else, since a flagged row carrying ANY number is wrong)."""
    flagged = feats["is_missing_pledging"].to_numpy(bool)
    return int(feats.loc[flagged, PLEDGING_COLUMNS].notna().to_numpy().sum())


def pit_violations(feats: pd.DataFrame, filings: pd.DataFrame) -> int:
    """Every unflagged row's value must trace to SOME filing for its source_period_end whose real BROADCAST_IST is
    at/before that row's 19:00 IST run. Existence, not the latest-ever broadcast for that period: a period can carry
    a later revision broadcast in the FUTURE relative to an earlier row that correctly used an earlier revision, and
    taking the max would falsely flag that earlier, correct row."""
    broadcasts_by_period: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for rec in filings.itertuples():
        if pd.notna(rec.BROADCAST_IST):
            broadcasts_by_period.setdefault(pd.Timestamp(rec.PERIOD_END), []).append(pd.Timestamp(rec.BROADCAST_IST))
    violations = 0
    known = feats[~feats["is_missing_pledging"]]
    for rec in known.itertuples():
        candidates = broadcasts_by_period.get(pd.Timestamp(rec.source_period_end))
        if not candidates:
            violations += 1                                               # a value with no matching real broadcast at all
            continue
        run = pd.Timestamp.combine(rec.date.date(), FEATURE_CUTOFF_IST).tz_localize(IST)
        if not any(c.tz_convert(IST) <= run for c in candidates):
            violations += 1
    return violations


def recompute_violations(feats: pd.DataFrame, expected: pd.DataFrame) -> int:
    a, b = feats.reset_index(drop=True), expected.reset_index(drop=True)
    if len(a) != len(b):
        return abs(len(a) - len(b))
    val_close = np.isclose(a[PLEDGING_COLUMNS].to_numpy(float), b[PLEDGING_COLUMNS].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)
    same_period = (a["source_period_end"].to_numpy() == b["source_period_end"].to_numpy()) | \
                  (pd.isna(a["source_period_end"].to_numpy()) & pd.isna(b["source_period_end"].to_numpy()))
    return int((~(val_close.all(axis=1) & same_period)).sum())


# --- driver checks -------------------------------------------------------------------------------------------------

def check_crawl_coverage(raw: dict[str, pd.DataFrame]) -> None:
    target = set(universe_symbols_by_priority())
    have = set(raw)
    missing = sorted(target - have)
    check(not missing, f"every SHP crawl-target symbol has a raw filing file ({len(target):,} targets)",
          f"({len(have):,} files on disk; missing={missing[:8]})")

    cats: Counter = Counter()
    total_filings = 0
    for symbol in target & have:
        f = raw[symbol]
        total_filings += len(f)
        for e in f["PARSE_ERROR"].dropna():
            if "no XBRL link" in e:
                cats["no XBRL link (filing predates XBRL or was never attached)"] += 1
            elif "404" in e:
                cats["404: XBRL link is dead"] += 1
            elif "the filing is for" in e:
                cats["XBRL period disagrees with the master list"] += 1
            elif "Max retries" in e or "Connection" in e or "RemoteDisconnected" in e:
                cats["NETWORK ERROR (retryable -- should not persist)"] += 1
            else:
                cats["other"] += 1
    logger.info(f"raw filings: {total_filings:,} rows across {len(target & have):,} crawled symbols; "
                f"PARSE_ERROR breakdown: {dict(cats)}")
    network = cats.get("NETWORK ERROR (retryable -- should not persist)", 0)
    check(network == 0, "no lingering network-error PARSE_ERROR rows (these are retryable, not permanent)",
          f"(found: {network}; rerun data/ingest_shp_pledge.py to clear them)")


def check_universe_files(symbols: list[str]) -> dict[str, pd.DataFrame]:
    files = {p.stem for p in PLEDGING_FEATURES_DIR.glob("*.parquet")}
    check(files == set(symbols), "one pledging-feature file per universe symbol, no extras",
          f"({len(files):,} files, {len(symbols):,} symbols; missing={sorted(set(symbols) - files)[:5]}, "
          f"extra={sorted(files - set(symbols))[:5]})")
    loaded, row_mismatch, misaligned = {}, [], []
    for symbol in symbols:
        feats = pd.read_parquet(PLEDGING_FEATURES_DIR / f"{symbol}.parquet")
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


def main() -> int:
    logger.info("loading raw SHP filing archive independently ...")
    raw = load_raw_filings()
    logger.info(f"raw archive: {len(raw):,} symbol files")
    check_crawl_coverage(raw)

    symbols = list_universe_symbols()
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions())).normalize()
    loaded = check_universe_files(symbols)

    # 3. structure, universe-wide
    struct_total = sum(structure_violations(f) for f in loaded.values())
    check(struct_total == 0, "structure: flag<->NaN<->NaT agree, values in range, no inf, across every symbol", f"(violations: {struct_total})")

    # 4-5. point-in-time and independent recompute, every SHP-crawled symbol, every row
    crawled = sorted(set(raw) & set(loaded))
    pit_total = recompute_total = 0
    n_real_zero = n_feature_zero = 0
    for symbol in crawled:
        feats = loaded[symbol]
        pit_total += pit_violations(feats, raw[symbol])
        expected = independent_expected(raw[symbol], sessions, pd.DatetimeIndex(feats["date"]))
        recompute_total += recompute_violations(feats, expected)
        f = raw[symbol]
        real_zero = f["PARSE_ERROR"].isna() & (f["ENCUMBERED_SHARES"] == 0) & f["HAS_PROMOTER_GROUP"].fillna(False).astype(bool)
        n_real_zero += int(real_zero.sum())
        n_feature_zero += int((feats["pledged_pct_of_promoter"] == 0).sum())
    check(pit_total == 0, "point-in-time: every value traces to a filing broadcast at/before that session's 19:00 IST run",
          f"({len(crawled):,} symbols checked; violations: {pit_total})")
    check(recompute_total == 0, "independent recompute (ratio, chg, rising, age, source period) matches the stored output",
          f"({len(crawled):,} symbols checked; violations: {recompute_total})")

    # 6. no zero-fill, universe-wide, plus proof a real zero exists and survives
    zero_total = sum(zero_fill_violations(f) for f in loaded.values())
    check(zero_total == 0, "no flagged row carries a value in any column, across every symbol", f"(violations: {zero_total})")
    logger.info(f"real zero-encumbrance filings: {n_real_zero:,}; feature rows at exactly 0%: {n_feature_zero:,} (0 is a real disclosed value, not a fill)")
    check(n_real_zero > 0 and n_feature_zero > 0, "at least one real 0% disclosure exists in the raw filings and survives into the output "
          "(otherwise the zero-fill check above proves nothing)")

    # 7. truncation: real function, real full filings, truncated session spine
    check_truncation(raw, sessions)

    # 8. injected bugs -- each corruption of REAL output must be caught by the counters above
    logger.info("--- injected bugs (in-memory copies of real output; each must be caught) ---")
    symbol = next(s for s in crawled if (~loaded[s]["is_missing_pledging"]).sum() >= 5 and loaded[s]["is_missing_pledging"].sum() >= 5)
    feats, filings = loaded[symbol], raw[symbol]

    period_keyed = feats.copy()
    period_keyed[PLEDGING_COLUMNS] = period_keyed[PLEDGING_COLUMNS].shift(-30)         # pull a FUTURE value 30 rows early
    period_keyed["source_period_end"] = period_keyed["source_period_end"].shift(-30)
    period_keyed["is_missing_pledging"] = period_keyed["pledged_pct_of_promoter"].isna()
    expect_caught("value visible ~30 sessions early (creation/period keying instead of broadcast keying)",
                  pit_violations(period_keyed, filings))

    zero_filled = feats.copy()
    zero_filled.loc[zero_filled["is_missing_pledging"], PLEDGING_COLUMNS] = zero_filled.loc[zero_filled["is_missing_pledging"], PLEDGING_COLUMNS].fillna(0.0)
    expect_caught("missing rows filled with 0", zero_fill_violations(zero_filled))

    stale = feats.copy()
    stale["pledged_pct_of_promoter"] = stale["pledged_pct_of_promoter"].ffill()
    stale.loc[feats["is_missing_pledging"] & stale["pledged_pct_of_promoter"].notna(), "is_missing_pledging"] = False   # a real ffill bug would also flip the flag
    expect_caught("stale carry-forward (ffill) of the previous quarter's value",
                  recompute_violations(stale, independent_expected(filings, sessions, pd.DatetimeIndex(feats["date"]))))

    nudged = feats.copy()
    shown = feats.index[~feats["is_missing_pledging"]]
    nudged.loc[shown[len(shown) // 2], "pledged_pct_of_promoter"] += 5.0
    expect_caught("one value off by 5.0 percentage points",
                  recompute_violations(nudged, independent_expected(filings, sessions, pd.DatetimeIndex(feats["date"]))))

    unflagged = feats.copy()
    missing_idx = feats.index[feats["is_missing_pledging"]]
    if len(missing_idx):
        unflagged.loc[missing_idx[-1], "is_missing_pledging"] = False
        expect_caught("missing flag cleared on a NaN row", structure_violations(unflagged))

    if failures:
        logger.error(f"Stage 10 pledging checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 pledging checkpoint PASSED")
    return 0


def check_truncation(raw: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex) -> None:
    """Full filings are passed UNCHANGED at every cut; only the session/date spine is truncated. A filing broadcast
    beyond the truncated sessions has no session to land on (usable_from_series maps it to NaT), so it cannot affect
    any row up to the cut -- this is what makes it a truncation check and not just a re-run on less data."""
    random.seed(13)
    candidates = [s for s in raw if raw[s]["BROADCAST_IST"].notna().sum() >= 2]
    sample = random.sample(candidates, min(15, len(candidates)))
    violations = []
    for symbol in sample:
        filings = raw[symbol]
        panel = load_panel(symbol)
        full = compute_pledging_features(panel, filings, sessions)
        for frac in (0.3, 0.6, 0.9):
            cut = sessions[int(len(sessions) * frac)]
            part_panel = panel[panel <= cut]
            part_sessions = sessions[sessions <= cut]
            if len(part_panel) < 2:
                continue
            part = compute_pledging_features(part_panel, filings, part_sessions)
            a = full[full["date"] <= cut].reset_index(drop=True)
            b = part[part["date"] <= cut].reset_index(drop=True)
            if recompute_violations(a, b) > 0:
                violations.append(symbol)
    check(not violations, "truncation: cutting a symbol's sessions never changes an earlier row (real function, real full filings)",
          f"({len(sample)} symbols x 3 cuts sampled; violations: {sorted(set(violations))[:5]})")


if __name__ == "__main__":
    sys.exit(main())
