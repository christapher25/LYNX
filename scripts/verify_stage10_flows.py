"""
Stage 10 (flows) checkpoint verification, against the real files on disk.

Two blocks are checked, each with its own shared missing flag:
  FII  data/processed/features/flows/fii_flows.parquet  (NSDL daily FPI equity net, 1999 ->)
  DII  data/processed/features/flows/dii_flows.parquet  (NSE forward capture, timestamped)

Every check is a function that returns a list of problems. It is run twice: on the real output (must return none) and on
an in-memory copy with one bug injected (must return some). A check that stays silent on the bug it exists for proves
nothing, so an escaped bug fails the verifier exactly like a real defect does.

Real-data checks
  1. Structure   one row per trading session, exact columns, bool flag == value is NaN == source_date is NaT.
  2. Point-in-time   every figure sits on the first session strictly after its reporting day; none is visible earlier.
  3. Independent recompute   every row (value, 5- and 20-reported-day sums, source date, flag) rebuilt from the raw NSDL
                    files with plain loops and compared with the file: all sessions, not a sample.
  4. Zero policy   NaN + flag where absent; every 0.0 in the output traces to a 0.0 in the source.
  5. NSDL's own monthly totals   my monthly sums of the daily figures vs NSDL's published monthly equity totals.
  6. Truncation   rebuilding from data cut at 6 dates never changes any earlier row.
  7. DII         placement and point-in-time against the capture timestamps; same structure and zero policy.
  8. Publication timing   from data/publication_monitor.py evidence; PENDING (not PASS) until that evidence exists.

Usage: python -m scripts.verify_stage10_flows
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from config import FLOW_FEATURES_DIR, NSDL_FPI_DIR, PUBLICATION_LOG_DIR
from data.trading_calendar import trading_sessions
from data.publication import FEATURE_CUTOFF_IST, IST
from features.blocks import flows

# NSDL's published monthly equity total is rounded to whole Rs crore and mine sums ~21 daily figures each rounded to 0.1
# (0.01 after mid-2014). Measured over 267 months: mean gap 0.8, max 3.8. A lost or duplicated trading day is thousands.
MONTHLY_CROSSCHECK_TOLERANCE_CR = 4.0
MONTHLY_CROSSCHECK_MIN_MONTHS = 250
# NSDL's quarterly Equity vs my quarterly sums (2024 onward, where the monthly table is thin): measured gaps 0.3-14.9 crore on
# quarterly flows of 40,000-140,000 crore; the quarter still in progress differs by 0.6. Consistent with NSDL compiling the
# table from figures later revised on the daily page. A detector for lost or duplicated days, not a rounding identity.
QUARTERLY_TOLERANCE_CR = 20.0
QUARTERLY_MIN_QUARTERS = 10
KNOWN_QUARANTINED_DAYS = {
    date(2000, 2, 28): "NSDL lists 28-Feb-2000 twice with conflicting figures; ingest quarantines it rather than choose (data/raw/DATA_NOTES.md §12)",
}
FII_VALUE_COLUMNS = flows.FII_COLUMNS
failures: list[str] = []
pending: list[str] = []


def report(label: str, problems: list[str], detail: str = "") -> None:
    ok = not problems
    (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)
        for line in problems[:5]:
            logger.error(f"        {line}")


def expect_caught(bug: str, problems: list[str]) -> None:
    caught = bool(problems)
    (logger.success if caught else logger.error)(
        f"{'PASS' if caught else 'FAIL'}  injected bug caught: {bug}"
        + (f"  ({len(problems)} problem(s); e.g. {problems[0]})" if caught else "  <- THE CHECK DID NOT FIRE"))
    if not caught:
        failures.append(f"injected bug escaped: {bug}")


# ── raw inputs, read independently of the block's loaders ───────────────────────────────────────────────────────────

def read_reported_days() -> dict[pd.Timestamp, float]:
    """Equity net (Rs crore) per NSDL reporting day: the single no-route row (1999-2001 era) or the Sub-total row."""
    net: dict[pd.Timestamp, float] = {}
    duplicated: list[pd.Timestamp] = []
    for path in sorted(NSDL_FPI_DIR.glob("*.parquet")):
        frame = pq.read_table(path, columns=["DATE", "ASSET", "ROUTE", "NET_CR"]).to_pandas()
        equity = frame[(frame["ASSET"] == "Equity") & (frame["ROUTE"].isna() | (frame["ROUTE"] == "Sub-total"))]
        for day, value in zip(pd.to_datetime(equity["DATE"]), equity["NET_CR"]):
            if np.isnan(value):
                continue
            if day in net:
                duplicated.append(day)
            net[day] = float(value)
    if duplicated:
        raise RuntimeError(f"reporting days appear in more than one NSDL file: {duplicated[:5]}")
    return net


def read_captures() -> pd.DataFrame:
    return flows.load_dii_captures()


# ── independent expectations (plain loops; deliberately not the block's pandas code path) ───────────────────────────

def expected_fii(net: dict[pd.Timestamp, float], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """A figure lands on the first session strictly after its reporting day; sums cover the last 5 / 20 reported days."""
    reported = sorted(net)
    listed = list(sessions)
    landing: dict[pd.Timestamp, tuple] = {}
    for i, day in enumerate(reported):
        position = bisect_right(listed, day)
        if position == len(listed):
            continue
        w5 = sum(net[reported[i - k]] for k in range(5)) if i >= 4 else np.nan
        w20 = sum(net[reported[i - k]] for k in range(20)) if i >= 19 else np.nan
        landing[listed[position]] = (net[day], w5, w20, day)          # later reporting days overwrite earlier ones
    rows = []
    for s in listed:
        v = landing.get(s)
        rows.append((s, *(v if v else (np.nan, np.nan, np.nan, pd.NaT))))
    out = pd.DataFrame(rows, columns=["date", *FII_VALUE_COLUMNS, "source_date"])
    out["is_missing_fii_flows"] = out["source_date"].isna()
    return out


def expected_dii(captures: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    earliest: dict[pd.Timestamp, tuple[datetime, float]] = {}
    for rec in captures.itertuples():
        if rec.CAPTURED_AT_IST is None or pd.isna(rec.CAPTURED_AT_IST):
            continue
        when = datetime.fromisoformat(str(rec.CAPTURED_AT_IST))
        if when.tzinfo is None:
            raise RuntimeError(f"naive capture timestamp for {rec.DATE}")
        day = pd.Timestamp(rec.DATE).normalize()
        if day not in earliest or when < earliest[day][0]:
            earliest[day] = (when, float(rec.NET_VALUE_CR))
    landing: dict[pd.Timestamp, tuple] = {}
    for day in sorted(earliest):
        when, value = earliest[day]
        for s in sessions:
            if datetime.combine(s.date(), FEATURE_CUTOFF_IST, tzinfo=IST) >= when:
                landing[s] = (value, day)
                break
    rows = [(s, *(landing.get(s) or (np.nan, pd.NaT))) for s in sessions]
    out = pd.DataFrame(rows, columns=["date", "dii_net_cr", "source_date"])
    out["is_missing_dii_flows"] = out["source_date"].isna()
    return out


# ── the checks: each returns a list of problems ─────────────────────────────────────────────────────────────────────

def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    return ((a == b) | (a.isna() & b.isna())).to_numpy()


def structure_problems(out: pd.DataFrame, sessions: pd.DatetimeIndex, value_cols: list[str], flag: str) -> list[str]:
    problems = []
    if list(out.columns) != ["date", *value_cols, "source_date", flag]:
        return [f"columns are {list(out.columns)}"]
    if len(out) != len(sessions) or not (pd.DatetimeIndex(out["date"]) == sessions).all():
        problems.append(f"rows are not exactly the {len(sessions)} trading sessions")
    if out[flag].dtype != bool:
        problems.append(f"{flag} is {out[flag].dtype}, not a plain bool")
    first = value_cols[0]
    if (out[flag] != out[first].isna()).any():
        problems.append(f"{int((out[flag] != out[first].isna()).sum())} rows where {flag} disagrees with {first} being NaN")
    if (out[flag] != out["source_date"].isna()).any():
        problems.append(f"{int((out[flag] != out['source_date'].isna()).sum())} rows where {flag} disagrees with source_date being NaT")
    if out.loc[out[flag], value_cols].notna().any().any():
        problems.append("a flagged row carries a value")
    return problems


def fii_pit_problems(out: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[str]:
    problems = []
    if not flows.FPI_PUBLICATION_TIME_IST < FEATURE_CUTOFF_IST:
        problems.append("the assumed publication time is not before the nightly cut-off")
    shown = out.dropna(subset=["source_date"])
    early = shown[shown["source_date"] >= shown["date"]]
    if len(early):
        problems.append(f"{len(early)} rows show a figure reported on or after their own session, e.g. {early['date'].iloc[0].date()}")
    landing = sessions[np.minimum(sessions.searchsorted(shown["source_date"].to_numpy(), side="right"), len(sessions) - 1)]
    misplaced = shown[landing != pd.DatetimeIndex(shown["date"])]
    if len(misplaced):
        problems.append(f"{len(misplaced)} rows are not on the first session after their reporting day, e.g. {misplaced['date'].iloc[0].date()}")
    return problems


def independent_problems(out: pd.DataFrame, expected: pd.DataFrame, value_cols: list[str], flag: str) -> list[str]:
    a, b = out.reset_index(drop=True), expected.reset_index(drop=True)
    if len(a) != len(b):
        return [f"{len(a)} rows vs {len(b)} expected"]
    values = ~np.isclose(a[value_cols].to_numpy(float), b[value_cols].to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True)
    wrong = values.any(axis=1) | ~_same(a["source_date"], b["source_date"]) | (a[flag].to_numpy() != b[flag].to_numpy())
    if not wrong.any():
        return []
    first = a.loc[wrong, "date"].iloc[0].date()
    return [f"{int(wrong.sum())} of {len(a)} rows differ from the independent rebuild, first at {first}"]


def zero_policy_problems(out: pd.DataFrame, source_zero_days: set, value_col: str, flag: str, all_cols: list[str]) -> list[str]:
    problems = []
    zeros = out[out[value_col] == 0.0]
    invented = zeros[~zeros["source_date"].isin(source_zero_days)]
    if len(invented):
        problems.append(f"{len(invented)} zeros in {value_col} with no zero in the source, e.g. {invented['date'].iloc[0].date()}")
    if (out.loc[out[flag], all_cols] == 0.0).any().any():
        problems.append("a flagged (missing) row holds 0")
    return problems


def monthly_problems(net: dict[pd.Timestamp, float]) -> list[str]:
    table = pd.read_csv(NSDL_FPI_DIR / "_crosscheck_yearwise.csv")
    series = pd.Series(net)
    monthly = series.groupby(pd.DatetimeIndex(series.index).strftime("%Y-%m")).sum()
    problems = []
    if len(table) < MONTHLY_CROSSCHECK_MIN_MONTHS:
        problems.append(f"only {len(table)} months in the NSDL cross-check table (need >= {MONTHLY_CROSSCHECK_MIN_MONTHS})")
    for row in table.itertuples():
        if row.ym not in monthly.index:
            problems.append(f"{row.ym}: no daily data on disk")
            continue
        mine = float(monthly[row.ym])
        if abs(mine - row.nsdl_equity_cr) > MONTHLY_CROSSCHECK_TOLERANCE_CR:
            problems.append(f"{row.ym}: my daily sum {mine:.2f} vs NSDL monthly {row.nsdl_equity_cr:.1f}")
        elif abs(mine - row.mine_cr) > 0.011:
            problems.append(f"{row.ym}: on-disk data no longer matches the recorded cross-check ({mine:.2f} vs {row.mine_cr:.2f})")
    return problems


def quarterly_problems(net: dict[pd.Timestamp, float]) -> list[str]:
    """My quarterly sums vs NSDL's own quarterly Equity totals (data/crosscheck_nsdl_quarterly.py); the open quarter is cut at its 'up to' date."""
    table = pd.read_csv(NSDL_FPI_DIR / "_crosscheck_quarterly.csv", parse_dates=["up_to"])
    series = pd.Series(net)
    index = pd.DatetimeIndex(series.index)
    problems = []
    if len(table) < QUARTERLY_MIN_QUARTERS:
        problems.append(f"only {len(table)} quarters in the NSDL quarterly table (need >= {QUARTERLY_MIN_QUARTERS})")
    for row in table.itertuples():
        start = pd.Timestamp(year=int(row.year), month=3 * (int(row.quarter) - 1) + 1, day=1)
        end = row.up_to if pd.notna(row.up_to) else start + pd.offsets.QuarterEnd(0)
        mine = float(series[(index >= start) & (index <= end)].sum())
        if abs(mine - row.nsdl_equity_cr) > QUARTERLY_TOLERANCE_CR:
            problems.append(f"{int(row.year)}Q{int(row.quarter)}: my daily sum {mine:.1f} vs NSDL quarterly {row.nsdl_equity_cr:.0f}")
    return problems


def absent_problems(net: dict[pd.Timestamp, float], sessions: pd.DatetimeIndex) -> list[str]:
    """
    Every session from 1999 to the last reported day either has a figure, or NSDL's own page was audited and does not list
    it (data/audit_nsdl_absent_sessions.py). An unaudited gap, or one NSDL lists, is a lost day: this is the check that
    reaches 1999-2001, which the monthly table does not.
    """
    from data.audit_nsdl_absent_sessions import ABSENT, LOST

    path = NSDL_FPI_DIR / "_absent_sessions.csv"
    if not path.exists():
        return ["no audit file: run python -m data.audit_nsdl_absent_sessions"]
    audit = pd.read_csv(path, parse_dates=["session"]).set_index("session")["status"]
    problems = []
    for day in sessions[(sessions >= "1999-01-01") & (sessions <= max(net))]:
        if day in net:
            if audit.get(day) == ABSENT:
                problems.append(f"{day.date()}: has a figure now but was audited as absent on NSDL's page (stale audit)")
            continue
        status = audit.get(day)
        if status == ABSENT or (status == LOST and day.date() in KNOWN_QUARANTINED_DAYS):
            continue
        problems.append(f"{day.date()}: no figure and " + ("NSDL lists it: a lost day" if status == LOST else "not audited (run data.audit_nsdl_absent_sessions)"))
    return problems


def truncation_problems(net: dict[pd.Timestamp, float], sessions: pd.DatetimeIndex, fn=flows.compute_fii_flow_features) -> list[str]:
    daily = pd.DataFrame({"date": list(net), "net_cr": list(net.values())}).sort_values("date").reset_index(drop=True)
    full = fn(daily, list(sessions))
    modern = sessions[sessions >= "2000-01-01"]
    cuts = [modern[int(len(modern) * f)] for f in (0.15, 0.35, 0.55, 0.75, 0.9)] + [sessions[-3]]
    problems = []
    for cut in cuts:
        part = fn(daily[daily["date"] <= cut], list(sessions[sessions <= cut]))
        a = full[full["date"] <= cut].reset_index(drop=True)
        changed = ~np.isclose(a[FII_VALUE_COLUMNS].to_numpy(float), part[FII_VALUE_COLUMNS].to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True)
        changed = changed.any(axis=1) | ~_same(a["source_date"], part["source_date"].reset_index(drop=True))
        if changed.any():
            problems.append(f"cut at {cut.date()}: {int(changed.sum())} earlier rows changed when later data was removed")
    return problems


def dii_pit_problems(out: pd.DataFrame, captures: pd.DataFrame) -> list[str]:
    stamps = {}
    for rec in captures.itertuples():
        if rec.CAPTURED_AT_IST is None or pd.isna(rec.CAPTURED_AT_IST):
            continue
        when, day = datetime.fromisoformat(str(rec.CAPTURED_AT_IST)), pd.Timestamp(rec.DATE).normalize()
        stamps[day] = min(when, stamps.get(day, when))
    problems = []
    for row in out.dropna(subset=["source_date"]).itertuples():
        run = datetime.combine(row.date.date(), FEATURE_CUTOFF_IST, tzinfo=IST)
        captured = stamps.get(row.source_date)
        if captured is None:
            problems.append(f"{row.date.date()}: value with no capture timestamp")
        elif captured > run:
            problems.append(f"{row.date.date()}: used a figure captured {captured:%Y-%m-%d %H:%M} IST, after that session's 19:00 run")
    return problems


# ── publication timing ──────────────────────────────────────────────────────────────────────────────────────────────

def publication_problems() -> tuple[list[str], list[str]]:
    """(problems, notes). No evidence file yet -> ([], ['PENDING ...']): unmeasured is reported, never passed."""
    path = PUBLICATION_LOG_DIR / "first_seen.jsonl"
    if not path.exists():
        return [], ["PENDING: no evidence in data/raw/_publication/first_seen.jsonl yet (the monitor writes it when it stops)"]
    problems, notes, usable = [], [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["item"] == "nsdl_latest":
            notes.append(f"nsdl_latest (target {rec['target_date']}) IGNORED: that monitor read every date on the page, derivatives "
                         "table included, so it does not describe the cash table; superseded by nsdl_cash")
            continue
        if rec["item"] != "nsdl_cash":
            notes.append(f"{rec['item']}: target {rec['target_date']} first seen {rec['first_seen_ist']}"
                         + (f", last absent {rec['last_absent_ist']}" if rec.get("last_absent_ist") else ""))
            continue
        usable += 1
        target = date.fromisoformat(rec["target_date"])
        next_day = target + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)                       # next weekday: a proxy where the calendar has not reached it
        deadline = datetime.combine(next_day, FEATURE_CUTOFF_IST, tzinfo=IST)
        seen = rec["first_seen_ist"]
        if seen is None:
            problems.append(f"NSDL figure for {target} never appeared before the monitor stopped")
        elif datetime.fromisoformat(seen) > deadline:
            problems.append(f"NSDL figure for {target} first seen {seen}, after the {deadline:%Y-%m-%d} 19:00 run: placement would leak")
        else:
            assumed = datetime.combine(next_day, flows.FPI_PUBLICATION_TIME_IST, tzinfo=IST)
            late = datetime.fromisoformat(seen) > assumed
            notes.append(f"nsdl figure for {target} first seen {seen}" + (f" (later than the assumed {flows.FPI_PUBLICATION_TIME_IST}, still before 19:00: placement unchanged)" if late else ""))
    if usable == 0:
        notes.insert(0, "PENDING: no nsdl_cash evidence yet (run: python -m data.publication_monitor --date <next trading day>)")
    return problems, notes


# ── main ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions())).normalize()
    fii = pd.read_parquet(FLOW_FEATURES_DIR / "fii_flows.parquet")
    dii = pd.read_parquet(FLOW_FEATURES_DIR / "dii_flows.parquet")
    net = read_reported_days()
    captures = read_captures()
    source_zero_days = {t for t, v in net.items() if v == 0.0}
    want = expected_fii(net, sessions)

    logger.info(f"{len(sessions)} sessions, {len(net)} NSDL reporting days ({min(net).date()} -> {max(net).date()}), {len(captures)} DII captures")

    # 1-6: FII, real data
    report("FII structure: one row per session, exact columns, flag == NaN == NaT", structure_problems(fii, sessions, FII_VALUE_COLUMNS, "is_missing_fii_flows"))
    report("FII point-in-time: every figure on the first session after its reporting day, none earlier", fii_pit_problems(fii, sessions))
    report("FII independent rebuild from raw NSDL matches every row (values, 5/20-day sums, source date, flag)",
           independent_problems(fii, want, FII_VALUE_COLUMNS, "is_missing_fii_flows"), f"({len(fii)} rows)")
    report("FII zero policy: no zero without a zero in the source; missing rows hold no value",
           zero_policy_problems(fii, source_zero_days, "fpi_equity_net_cr", "is_missing_fii_flows", FII_VALUE_COLUMNS),
           f"({len(source_zero_days)} real source zero(s))")
    report(f"FII vs NSDL's own monthly totals within {MONTHLY_CROSSCHECK_TOLERANCE_CR} crore (2002-01 -> 2023-12, plus 3 later months)", monthly_problems(net))
    report(f"FII vs NSDL's own quarterly totals within {QUARTERLY_TOLERANCE_CR} crore (2024Q1 -> now)", quarterly_problems(net))
    report("FII completeness: every session with no figure is absent from NSDL's own page (1999 -> now)", absent_problems(net, sessions),
           f"({len(KNOWN_QUARANTINED_DAYS)} documented quarantine)")
    report("FII truncation: removing later data changes no earlier row (6 cuts)", truncation_problems(net, sessions))

    # 7: DII, real data
    dii_want = expected_dii(captures, sessions)
    report("DII structure: one row per session, exact columns, flag == NaN == NaT", structure_problems(dii, sessions, flows.DII_COLUMNS, "is_missing_dii_flows"))
    report("DII independent rebuild from the capture timestamps matches every row",
           independent_problems(dii, dii_want, flows.DII_COLUMNS, "is_missing_dii_flows"), f"({int((~dii['is_missing_dii_flows']).sum())} row(s) with data)")
    report("DII point-in-time: no figure is used by a run that started before it was captured", dii_pit_problems(dii, captures))
    report("DII zero policy", zero_policy_problems(dii, {pd.Timestamp(r.DATE) for r in captures.itertuples() if r.NET_VALUE_CR == 0.0},
                                                   "dii_net_cr", "is_missing_dii_flows", flows.DII_COLUMNS))

    # 8: publication timing
    problems, notes = publication_problems()
    if any(note.startswith("PENDING") for note in notes):
        pending.append("publication timing")                       # unmeasured is not passed: no PASS line is printed for it
        logger.warning("PENDING  FII publication timing: NSDL's figure is on its page before the next session's 19:00 run "
                       "(no evidence yet; the assumed 10:00 IST placement is unverified)")
    else:
        report("FII publication timing: NSDL's figure is on its page before the next session's 19:00 run", problems)
    for note in notes:
        logger.info(f"        {note}")

    # injected bugs: each must be caught by the check that exists for it
    logger.info("--- injected bugs (in-memory copies of the real output; each must be caught) ---")
    shown = fii.dropna(subset=["source_date"]).index
    leaked = fii.copy()
    leaked[FII_VALUE_COLUMNS + ["source_date"]] = fii[FII_VALUE_COLUMNS + ["source_date"]].shift(-1)
    leaked["is_missing_fii_flows"] = leaked["fpi_equity_net_cr"].isna()
    expect_caught("figure visible one session early (same-day keying)", fii_pit_problems(leaked, sessions))
    expect_caught("same bug, seen by the independent rebuild", independent_problems(leaked, want, FII_VALUE_COLUMNS, "is_missing_fii_flows"))

    zero_filled = fii.copy()
    zero_filled[FII_VALUE_COLUMNS] = zero_filled[FII_VALUE_COLUMNS].fillna(0.0)
    expect_caught("missing figures filled with 0", zero_policy_problems(zero_filled, source_zero_days, "fpi_equity_net_cr", "is_missing_fii_flows", FII_VALUE_COLUMNS))

    nudged = fii.copy()
    nudged.loc[shown[len(shown) // 2], "fpi_equity_net_cr"] += 1.0
    expect_caught("one figure off by 1.0 crore", independent_problems(nudged, want, FII_VALUE_COLUMNS, "is_missing_fii_flows"))

    centred = fii.copy()
    centred["fpi_equity_net_5d_cr"] = fii["fpi_equity_net_cr"].rolling(5, min_periods=5, center=True).sum()
    expect_caught("5-day sum centred (looks at future rows)", independent_problems(centred, want, FII_VALUE_COLUMNS, "is_missing_fii_flows"))

    unflagged = fii.copy()
    unflagged.loc[fii.index[fii["is_missing_fii_flows"]][-1], "is_missing_fii_flows"] = False
    expect_caught("missing flag cleared on a NaN row", structure_problems(unflagged, sessions, FII_VALUE_COLUMNS, "is_missing_fii_flows"))

    def without(day):
        return {t: v for t, v in net.items() if t != day}

    monthly_day = max((t for t in net if 2010 <= t.year <= 2023), key=lambda t: abs(net[t]))
    expect_caught(f"a reporting day lost in a month the monthly table covers ({monthly_day.date()}, {net[monthly_day]:.0f} crore)",
                  monthly_problems(without(monthly_day)))
    recent_day = max((t for t in net if t.year >= 2024), key=lambda t: abs(net[t]))
    expect_caught(f"a reporting day lost in 2024+ ({recent_day.date()}, {net[recent_day]:.0f} crore): the quarterly table",
                  quarterly_problems(without(recent_day)))
    early_day = max((t for t in net if t.year <= 2001), key=lambda t: abs(net[t]))
    expect_caught(f"a reporting day lost in 1999-2001 ({early_day.date()}, {net[early_day]:.1f} crore): only the absent-session audit reaches here",
                  absent_problems(without(early_day), sessions))
    quiet_day = min((t for t in net if 2015 <= t.year <= 2019 and t in set(sessions)), key=lambda t: abs(net[t]))
    expect_caught(f"a near-zero day lost ({quiet_day.date()}, {net[quiet_day]:.2f} crore): too small for any total, exact for the audit",
                  absent_problems(without(quiet_day), sessions))

    def demeaned(daily, sess):
        out = flows.compute_fii_flow_features(daily, sess)
        out["fpi_equity_net_cr"] = out["fpi_equity_net_cr"] - out["fpi_equity_net_cr"].mean()
        return out
    expect_caught("feature uses a full-sample statistic (past rows move when data is added)", truncation_problems(net, sessions, fn=demeaned))

    late = dii.copy()
    latest = captures.assign(_d=pd.to_datetime(captures["DATE"])).sort_values("CAPTURED_AT_IST").iloc[-1]
    row = late["date"] == pd.Timestamp(latest["_d"])
    late.loc[row, ["dii_net_cr", "source_date", "is_missing_dii_flows"]] = [float(latest["NET_VALUE_CR"]), pd.Timestamp(latest["_d"]), False]
    expect_caught("DII figure keyed on the day it describes, not when it was captured", dii_pit_problems(late, captures))

    if failures:
        logger.error(f"Stage 10 flows checkpoint FAILED: {failures}")
        return 1
    logger.success("Stage 10 flows checkpoint PASSED" + (f" (PENDING, not verified: {pending})" if pending else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
