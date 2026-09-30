"""
Stage 10 (F&O ingestion) checkpoint verification, against the real files on disk.

The ingesters validate every file as it arrives (one date, unique contracts, sane strikes and
expiries, ...). This script checks what a single-file validator cannot: that the whole archive is
complete, consistent across days, consistent across the 2024-07-08 format cutover, and consistent
with the other datasets.

  1. Manifest accounting: every day is ok / holiday; nothing failed; every 'ok' day has its file and
     the file's row count matches the manifest; no orphan files.
  2. Coverage: every equity trading day since F&O began has an F&O file (F&O trades whenever cash does).
  3. Schema: every file has exactly the declared columns; contract keys unique; no futures with strikes.
  4. NSE's own accounting identity, on real data: for a contract present on consecutive trading days,
     CHG_IN_OI(t) == OPEN_INT(t) - OPEN_INT(t-1). An independent check that OI was parsed into the right
     column with the right sign, across eras, INCLUDING the legacy -> UDiFF cutover pair.
  5. No step change at the cutover: total futures OI either side of 2024-07-08 moves like an ordinary day.
  6. Real-data format equivalence: 2024-07-05 was served in both formats; fetch both and compare all
     shared fields (traded value: UDiFF >= legacy, by < 0.01 lakh, since legacy truncates; everything else exactly).
  7. Ban list: every banned symbol actually had F&O contracts that day (the list is only meaningful
     for F&O stocks), every ban date is an F&O trading day, and coverage of F&O days is complete.

Usage: python -m scripts.verify_stage10_fno
"""

from __future__ import annotations

import io
import json
import random
import sys
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from loguru import logger

from config import BHAVCOPY_DIR, FNO_BAN_DIR, FNO_BHAVCOPY_DIR, MANIFEST_DIR
from data import ingest_fno_bhavcopy as fno
from data.ingest_common import build_session

CUTOVER_PAIR = (date(2024, 7, 5), date(2024, 7, 8))
VALUE_RESOLUTION_LAKHS = 0.01             # legacy traded value is TRUNCATED to 0.01 lakh (UDiFF >= legacy, by < 0.01)
# The OI identity is NSE's own accounting, but NSE breaks it on some rows for reasons that are source-data events, not
# parse faults: mergers and splits adjust OI (e.g. RELPETRO/RELIANCE/ICICI on 2002-02-18), and early files report
# CHG_IN_OI = 0 on some contracts that changed. Measured on the early archive: ~99.2% of continuing contracts reconcile,
# worst adjacent pair 98.7%. A parse fault (wrong column, flipped sign) would fail nearly EVERY row, so the checks are:
IDENTITY_MIN_MATCH = 0.99                 # share of continuing contracts that must reconcile, per era
IDENTITY_PAIR_FLOOR = 0.90                # no single adjacent pair may fall below this (a systematic fault would be ~0%)
# Days the archive itself cannot supply usably, each investigated and left out rather than guessed at. A day appears
# here only with its reason; a failed day that is NOT listed fails verification, and a listed day that now
# succeeds is reported so the list cannot go stale.
KNOWN_SOURCE_DEFECT_DAYS: dict[str, str] = {
    "2003-10-01": "57 far-month futures (48 expiring 2003-12-24, 9 expiring in 2004) are listed twice, once as an "
                  "all-zero placeholder and once with prices; the copies disagree, so the day is left out rather than "
                  "one copy chosen by guess",
}
# Equity trading days for which NSE's F&O archive returns 404, though futures traded whenever cash did. Probed
# 2026-09-24 against every reachable URL form (nsearchives, mixed-case month, UDiFF-style name; archives.nseindia.com
# answers 403/404, www1 fails TLS): nothing serves them, so they are archive gaps, not ingestion failures.
KNOWN_ARCHIVE_GAP_DAYS: tuple[str, ...] = (
    "2000-07-03", "2000-07-04", "2000-07-05", "2000-07-06",
    "2000-08-01", "2000-08-02", "2000-08-03", "2000-08-04", "2000-08-07", "2000-08-08", "2000-08-09",
    "2000-09-04", "2000-09-05", "2000-09-06", "2000-09-07", "2000-09-08", "2000-10-09",
    "2013-10-09", "2021-03-30",
)
# Ban-list equivalents, EMPTY until each entry is investigated and recorded with its evidence (same rule as above: an
# undocumented failed day fails verification).
#   KNOWN_BAN_SOURCE_DEFECT_DAYS  - days the ban archive cannot supply usably, e.g. a URL that serves another day's list
#                                   (seen: 2014-02-19 returns the 2014-03-19 list, 2014-08-25 the 2014-08-26 list).
#   KNOWN_BAN_DATES_WITHOUT_SESSION - ban files that exist for a date with no F&O session (seen: 2008-11-27).
# Evidence gathered 2026-09-24 by fetching NSE directly. Consumers (Stage 11 gates) must treat these three days as
# "ban list UNAVAILABLE", never as "nobody banned": all three were real sessions (equity and F&O bhavcopy exist).
KNOWN_BAN_SOURCE_DEFECT_DAYS: dict[str, str] = {
    "2008-11-28": "session day (equity and F&O bhavcopy exist) but the ban archive answers 404; the ban archive is off by "
                  "one around the 2008-11-27 closure (it holds a NIL list for the closed day instead)",
    "2014-02-19": "NSE's URL for this day serves a byte-identical copy of the 2014-03-19 list (header '19-Mar-2014', same "
                  "symbols); the real list for the session on 2014-02-19 is unavailable",
    "2014-08-25": "NSE's URL for this day serves the 2014-08-26 list (header '26-AUG-2014', same symbols); the real list "
                  "for the session on 2014-08-25 is unavailable",
}
KNOWN_BAN_DATES_WITHOUT_SESSION: dict[str, str] = {
    "2008-11-27": "NIL list published for a day both equity and F&O bhavcopy show as closed (404); NSE was shut the day "
                  "after the 26-Nov-2008 Mumbai attacks. Harmless (NIL bans on a closed day) but not a trading session.",
    "2014-02-22": "a real list dated Saturday 22-Feb-2014 naming AUROPHARMA, consistent with its neighbours (banned "
                  "2014-02-20, 02-21 and 02-24; ~12M stock-future OI on 02-21), but no equity or F&O session happened "
                  "that day. The only weekend ban file that is not a session. Key bans by session date, not file date.",
}
failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


def dates_in(directory) -> dict[date, "pd.Timestamp"]:
    return {date.fromisoformat(p.stem): p for p in sorted(directory.glob("*.parquet"))}


def latest_manifest_status(path) -> dict[str, dict]:
    """Last record per day: a later successful retry supersedes an earlier failure."""
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            latest[rec["trade_date"]] = rec
    return latest


def load_day(path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- 1-3: manifest, coverage, schema --------------------------------------------------------------------

def check_manifest_and_files(name: str, directory, manifest_path, allowed: dict[str, str] | None = None) -> dict[date, "pd.Timestamp"]:
    allowed = allowed or {}
    files = dates_in(directory)
    manifest = latest_manifest_status(manifest_path)
    by_status: dict[str, int] = {}
    for rec in manifest.values():
        by_status[rec["status"]] = by_status.get(rec["status"], 0) + 1
    failed = sorted(d for d, r in manifest.items() if r["status"] in ("failed", "empty"))
    unexpected = [d for d in failed if d not in allowed]
    check(not unexpected, f"{name}: every failed day is a documented source defect",
          f"(status counts: {by_status}; documented={[d for d in failed if d in allowed]}; UNEXPECTED={unexpected[:8]})")
    for d, reason in allowed.items():
        if d in manifest and manifest[d]["status"] == "ok":
            logger.info(f"{name}: documented defect day {d} now ingests fine; remove it from KNOWN_SOURCE_DEFECT_DAYS")
        elif d in failed:
            logger.info(f"{name}: known defect {d} left out: {reason}")

    ok_days = {date.fromisoformat(d) for d, r in manifest.items() if r["status"] == "ok"}
    check(ok_days == set(files), f"{name}: files on disk == days the manifest records as ok",
          f"({len(files):,} files; missing files={sorted(ok_days - set(files))[:5]}, orphans={sorted(set(files) - ok_days)[:5]})")

    wrong_rows = [d for d, p in files.items() if pq.ParquetFile(p).metadata.num_rows != manifest.get(d.isoformat(), {}).get("rows")]
    check(not wrong_rows, f"{name}: every file's row count matches its manifest record", f"(mismatches: {wrong_rows[:5]})")

    repaired = {d: r["format_era"] for d, r in manifest.items() if r["status"] == "ok" and "repaired" in r["format_era"]}
    logger.info(f"{name}: {len(repaired):,} days ingested after a documented repair (see ingest_fno_bhavcopy.py); e.g. "
                f"{dict(list(repaired.items())[:3])}")
    return files


def check_coverage(fno_files: dict[date, "pd.Timestamp"]) -> None:
    equity_days = {date.fromisoformat(p.stem) for p in BHAVCOPY_DIR.glob("*.parquet")}
    window = {d for d in equity_days if d >= fno.FIRST_AVAILABLE}
    documented = set(KNOWN_SOURCE_DEFECT_DAYS) | set(KNOWN_ARCHIVE_GAP_DAYS)
    missing = sorted(d for d in window - set(fno_files) if d.isoformat() not in documented)
    check(not missing, "every equity trading day since 2000-06-12 has an F&O file (documented archive gaps and defects excepted)",
          f"({len(window):,} equity days in window; {len(documented)} documented absences; UNEXPECTED missing: {missing[:8]})")
    healed = sorted(d for d in KNOWN_ARCHIVE_GAP_DAYS if date.fromisoformat(d) in fno_files)
    if healed:
        logger.info(f"documented archive-gap days that now exist on disk (remove from KNOWN_ARCHIVE_GAP_DAYS): {healed}")

    extra = sorted(set(fno_files) - equity_days)
    weekend = [d for d in extra if d.weekday() >= 5]
    logger.info(f"F&O sessions with no equity bhavcopy file: {len(extra)} ({len(weekend)} on weekends, e.g. "
                f"{[d.isoformat() for d in weekend[:6]]}; the rest are an equity-side gap or run past bhavcopy's last date: "
                f"{[d.isoformat() for d in extra if d.weekday() < 5][:6]})")
    # Weekend sessions are real (Budget-day Saturdays, Muhurat, special sessions). They must be present: without them
    # the next session's CHG_IN_OI spans two sessions and the OI identity cannot hold.
    must_exist = [date(2015, 2, 28), date(2020, 2, 1), date(2025, 2, 1), date(2024, 3, 2), date(2024, 5, 18),
                  date(2024, 1, 20), date(2019, 10, 27), date(2023, 11, 12), date(2016, 10, 30), date(2009, 10, 17)]
    absent = [d.isoformat() for d in must_exist if d not in fno_files]
    check(not absent, "known weekend trading sessions (Budget-day Saturdays, Muhurat, special sessions) are present",
          f"(absent: {absent})")


def check_schema_and_keys(files: dict[date, "pd.Timestamp"]) -> None:
    bad_schema, dup_days, strike_on_future, instruments_by_year = [], [], [], {}
    for d, path in files.items():
        cols = pq.read_schema(path).names
        if cols != fno.OUTPUT_COLS:
            bad_schema.append(d)
            continue
        df = pd.read_parquet(path, columns=fno.CONTRACT_KEY)
        if df.duplicated().any():
            dup_days.append(d)
        for inst in df["INSTRUMENT"].unique():
            instruments_by_year.setdefault(inst, set()).add(d.year)
        fut = df["INSTRUMENT"].str.startswith("FUT") | (df["INSTRUMENT"] == "UNDINT")
        if (df.loc[fut, "STRIKE"].notna() | df.loc[fut, "OPTION_TYPE"].notna()).any():
            strike_on_future.append(d)
    check(not bad_schema, "every file has exactly the declared columns, in order", f"(bad: {bad_schema[:5]})")
    check(not dup_days, "no contract appears twice on any day", f"(days: {dup_days[:5]})")
    check(not strike_on_future, "no future carries a strike or option type (never a false 0)", f"(days: {strike_on_future[:5]})")
    logger.info("instrument types by year range: " + str({k: (min(v), max(v)) for k, v in sorted(instruments_by_year.items())}))


# --- 4-5: NSE's OI identity, cutover continuity ---------------------------------------------------------

def oi_identity(prev_path, cur_path) -> tuple[int, int]:
    cols = fno.CONTRACT_KEY + ["OPEN_INT", "CHG_IN_OI"]
    prev, cur = load_day(prev_path)[cols], load_day(cur_path)[cols]
    prev = prev.rename(columns={"OPEN_INT": "OI_PREV"})[fno.CONTRACT_KEY + ["OI_PREV"]]
    both = cur.merge(prev, on=fno.CONTRACT_KEY, how="inner")
    exact = int(((both["OPEN_INT"] - both["OI_PREV"]) == both["CHG_IN_OI"]).sum())
    return exact, len(both)


def check_oi_identity(files: dict[date, "pd.Timestamp"], equity_days: set[date]) -> None:
    """Pairs are consecutive sessions on the union of the equity and F&O calendars, so a day missing from one
    archive (e.g. 2020-07-13, absent from equity bhavcopy) cannot masquerade as an adjacent pair."""
    calendar = sorted({d for d in equity_days if d >= fno.FIRST_AVAILABLE} | set(files))
    following = dict(zip(calendar[:-1], calendar[1:]))
    pairs = [(a, b) for a, b in following.items() if a in files and b in files]
    random.seed(5)
    by_era: dict[str, list[tuple[date, date]]] = {}
    for a, b in pairs:
        era = "2000-09" if b.year < 2010 else "2010-19" if b.year < 2020 else "2020-24" if b < CUTOVER_PAIR[1] else "udiff"
        by_era.setdefault(era, []).append((a, b))
    sample = {era: random.sample(v, min(40, len(v))) for era, v in by_era.items()}
    sample["cutover"] = [CUTOVER_PAIR] if CUTOVER_PAIR in pairs else []
    for era, ps in sorted(sample.items()):
        exact = total = 0
        worst = (1.0, None)
        for a, b in ps:
            e, n = oi_identity(files[a], files[b])
            exact += e
            total += n
            if n and e / n < worst[0]:
                worst = (e / n, b)
        share = exact / total if total else float("nan")
        check(total > 0 and share >= IDENTITY_MIN_MATCH and worst[0] >= IDENTITY_PAIR_FLOOR,
              f"OI identity CHG_IN_OI == OI(t) - OI(t-1) holds [{era}]",
              f"({len(ps)} adjacent pairs, {total:,} continuing contracts, {share:.4%} exact; "
              f"worst pair {worst[0]:.2%} on {worst[1]})")


def check_weekend_sessions_reconcile(files: dict[date, "pd.Timestamp"], equity_days: set[date]) -> None:
    """
    The sampled identity check cannot prove the weekend sessions are right, so test EVERY weekend file against both of its
    neighbours: (previous session -> weekend) and (weekend -> next session). Before these sessions were ingested the
    Monday after the 2025-02-01 Budget Saturday reconciled only 62.8% (its CHG_IN_OI was measured against the missing
    Saturday); with the Saturday present both pairs must reconcile like any ordinary pair.
    """
    calendar = sorted({d for d in equity_days if d >= fno.FIRST_AVAILABLE} | set(files))
    position = {d: i for i, d in enumerate(calendar)}
    weekend = sorted(d for d in files if d.weekday() >= 5)
    if not weekend:
        check(False, "weekend F&O sessions exist to test", "(none on disk; has --weekends run?)")
        return
    exact = total = 0
    rows = []
    for d in weekend:
        i = position[d]
        for a, b in ((calendar[i - 1], d), (d, calendar[i + 1] if i + 1 < len(calendar) else None)):
            if b is None or a not in files or b not in files:
                continue
            e, n = oi_identity(files[a], files[b])
            exact += e
            total += n
            rows.append((e / n if n else 1.0, a, b, n))
    worst = min(rows) if rows else (1.0, None, None, 0)
    share = exact / total if total else float("nan")
    substantial = [r for r in rows if r[3] >= 100]
    worst_substantial = min(substantial) if substantial else worst
    check(total > 0 and share >= IDENTITY_MIN_MATCH and worst_substantial[0] >= IDENTITY_PAIR_FLOOR,
          "every weekend session reconciles with both neighbouring sessions (CHG_IN_OI == OI(t) - OI(t-1))",
          f"({len(weekend)} weekend sessions, {len(rows)} pairs, {total:,} continuing contracts, {share:.4%} exact; "
          f"worst pair with >=100 contracts {worst_substantial[0]:.2%}: {worst_substantial[1]} -> {worst_substantial[2]})")


def check_cutover_continuity(files: dict[date, "pd.Timestamp"]) -> None:
    if not all(d in files for d in CUTOVER_PAIR):
        check(False, "cutover days present", str(CUTOVER_PAIR))
        return
    days = sorted(files)
    idx = days.index(CUTOVER_PAIR[1])
    window = days[idx - 8: idx + 8]
    totals = {d: load_day(files[d]).query("INSTRUMENT in ['FUTIDX','FUTSTK']")["OPEN_INT"].sum() for d in window}
    changes = pd.Series(totals).pct_change().dropna()
    cut = changes[pd.Timestamp(CUTOVER_PAIR[1]).date() if False else CUTOVER_PAIR[1]]
    typical = changes.drop(CUTOVER_PAIR[1]).abs().max()
    check(abs(cut) <= max(3 * typical, 0.05), "total futures OI moves across the format cutover like an ordinary day",
          f"(cutover {cut:+.2%}; largest ordinary move in the +-8 day window {typical:.2%})")


# --- 6: real-data format equivalence --------------------------------------------------------------------

def fetch_frame(session: requests.Session, url: str) -> pd.DataFrame:
    resp = session.get(url, timeout=90)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        return pd.read_csv(archive.open(archive.namelist()[0]), low_memory=False)


def check_real_overlap() -> None:
    session = build_session(total_retries=2)
    day = CUTOVER_PAIR[0]
    legacy_url = fno.LEGACY_URL.format(yyyy=f"{day:%Y}", MON=f"{day:%b}".upper(), DD=f"{day:%d}")
    udiff_url = fno.UDIFF_URL.format(yyyymmdd=f"{day:%Y%m%d}")
    try:
        legacy = fno.validate(fno.parse_legacy(fetch_frame(session, legacy_url)), day)
        udiff = fno.validate(fno.parse_udiff(fetch_frame(session, udiff_url)), day)
    except Exception as exc:                                                 # noqa: BLE001
        check(False, "fetch both formats for the overlap day", f"({type(exc).__name__}: {exc})")
        return
    key = fno.CONTRACT_KEY
    shared = [c for c in fno.OUTPUT_COLS if c not in ("UNDERLYING_PRICE", "LOT_SIZE", "NUM_TRADES")]
    a = legacy[shared].sort_values(key, na_position="first").reset_index(drop=True)
    b = udiff[shared].sort_values(key, na_position="first").reset_index(drop=True)
    same_keys = len(a) == len(b) and a[key].equals(b[key])
    bad = {}
    for col in shared:
        if col in key or col == "DATE":
            continue
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        if col == "VALUE_LAKHS":
            gap = y - x                                  # exact UDiFF minus truncated legacy: never negative, < 0.01
            differs = ~(both_nan | ((gap >= -1e-9) & (gap <= VALUE_RESOLUTION_LAKHS + 1e-9)))
        else:
            differs = ~(both_nan | (x == y))
        if differs.any():
            bad[col] = int(differs.sum())
    check(same_keys and not bad, f"{day}: legacy and UDiFF parse to the same contracts and values on real data",
          f"({len(a):,} vs {len(b):,} contracts; columns that differ: {bad or 'none'})")


# --- 7: ban list ------------------------------------------------------------------------------------------

def check_ban(ban_files: dict[date, "pd.Timestamp"], fno_files: dict[date, "pd.Timestamp"]) -> None:
    if not ban_files:
        check(False, "ban-list files exist on disk", "(none found; has the ban-list backfill run?)")
        return
    # A ban date is a session if we hold its F&O bhavcopy OR it is a documented archive-gap day (equity traded; only the
    # F&O file is missing), and a ban file dated after the newest F&O bhavcopy cannot be compared yet (the ban list for a
    # day is published before that day's bhavcopy).
    sessions = set(fno_files) | {date.fromisoformat(d) for d in KNOWN_ARCHIVE_GAP_DAYS}
    newest_fno = max(fno_files)
    too_new = sorted(d for d in ban_files if d > newest_fno)
    if too_new:
        logger.info(f"ban files dated after the newest F&O bhavcopy ({newest_fno}); not comparable yet: {[d.isoformat() for d in too_new]}")
    not_fno_days = sorted(d for d in ban_files if d not in sessions and d <= newest_fno
                          and d.isoformat() not in KNOWN_BAN_DATES_WITHOUT_SESSION)
    check(not not_fno_days, "every ban-list date is a session day (documented exceptions aside)",
          f"(UNEXPECTED: {not_fno_days[:8]}; documented: {sorted(KNOWN_BAN_DATES_WITHOUT_SESSION)})")
    documented_missing = set(KNOWN_BAN_SOURCE_DEFECT_DAYS)
    missing = sorted(d for d in fno_files if d >= date(2008, 1, 1) and d not in ban_files and d.isoformat() not in documented_missing)
    check(not missing, "every F&O trading day since 2008-01-01 has a ban list (documented defects aside)",
          f"(UNEXPECTED without a ban file: {missing[:8]}; documented: {sorted(documented_missing)})")

    frames = [pd.read_parquet(p) for p in ban_files.values()]
    ban = pd.concat(frames, ignore_index=True)
    nil_days = int(ban["SYMBOL"].isna().sum())
    banned = ban.dropna(subset=["SYMBOL"])
    logger.info(f"ban list: {len(ban_files):,} days, {nil_days:,} NIL days, {len(banned):,} (day, symbol) bans across {banned['SYMBOL'].nunique():,} symbols")

    random.seed(9)
    sample_days = random.sample(sorted(banned["DATE"].dt.date.unique()), min(120, banned["DATE"].nunique()))
    orphans = []
    for d in sample_days:
        if d not in fno_files:
            continue
        stock_rows = pd.read_parquet(fno_files[d], columns=["SYMBOL", "INSTRUMENT"])
        stock_symbols = set(stock_rows.loc[stock_rows["INSTRUMENT"].isin(["FUTSTK", "OPTSTK"]), "SYMBOL"])
        today = set(banned.loc[banned["DATE"] == pd.Timestamp(d), "SYMBOL"])
        orphans += [(d.isoformat(), s) for s in sorted(today - stock_symbols)]
    check(len(orphans) == 0, "every banned symbol has stock F&O contracts on that day (sampled 120 ban days)",
          f"(banned-but-no-contracts: {orphans[:6]})")


def main() -> int:
    fno_files = check_manifest_and_files("F&O bhavcopy", FNO_BHAVCOPY_DIR, MANIFEST_DIR / "fno_bhavcopy.jsonl", KNOWN_SOURCE_DEFECT_DAYS)
    ban_files = check_manifest_and_files("ban list", FNO_BAN_DIR, MANIFEST_DIR / "fno_ban.jsonl", KNOWN_BAN_SOURCE_DEFECT_DAYS)
    if not fno_files:
        logger.error("no F&O files on disk")
        return 1
    equity_days = {date.fromisoformat(p.stem) for p in BHAVCOPY_DIR.glob("*.parquet")}
    check_coverage(fno_files)
    check_schema_and_keys(fno_files)
    check_oi_identity(fno_files, equity_days)
    check_weekend_sessions_reconcile(fno_files, equity_days)
    check_cutover_continuity(fno_files)
    check_real_overlap()
    check_ban(ban_files, fno_files)

    if failures:
        logger.error(f"Stage 10 F&O checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 F&O checkpoint PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
