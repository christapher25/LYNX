"""
R20-6: paper-log backfill wrapper. Runs the existing weekly paper log once per missed decision week, oldest first, flagged late, and stops
at the first refusal or failure. Resumable: a rerun skips weeks the decisions log already holds and resumes at the stopped week.

Nothing here writes to data/processed/paper_log/: every week is a child process `python -m scripts.run_paper_log --decision-date D
--allow-late` (the paper log's own, append-only path). The wrapper's own record is config.LOG_DIR / "paper_log_backfill.jsonl".

Decision weeks and the "already logged" test come from the paper log itself (scripts.run_paper_log calendar and vanta.paper_log
.read_decisions, the test run_decide uses for its own idempotence); there is no second definition of a decision date here.

Usage:
    python -m scripts.run_paper_log_backfill --dry-run              # list the weeks (reads only)
    python -m scripts.run_paper_log_backfill [--from 2026-09-30] [--to YYYY-MM-DD]

Exit 0 only if every week is already_logged or logged; 1 on a refusal, a failed check or a calendar error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

import config
from scripts import run_paper_log as rpl
from vanta import paper_log as P

LOG_DIR = config.LOG_DIR
DEFAULT_FROM = "2026-09-30"
STATEMENT = ("backfilled with --allow-late (late=True); model and code frozen at 068593b and later; "
             "no data after 2026-09-29 existed on disk before the model was frozen")


# ------------------------------------------------------------------------------------------ paper-log lookups (injectable in tests)
def _sessions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(rpl.trading_sessions()))


def _weekly(sessions) -> pd.DatetimeIndex:
    return rpl._weekly(sessions)


def _latest_complete(sessions) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    return P.latest_complete_decision_date(sessions, now)


def _latest_equity():
    return rpl.latest_equity_session()


def _decisions() -> pd.DataFrame:
    return P.read_decisions(rpl.LOG_DIR)


def _logged_dates() -> set:
    d = _decisions()
    return set() if d.empty else {pd.Timestamp(x) for x in d["decision_date"].unique()}


def _decision_rows(date) -> pd.DataFrame:
    d = _decisions()
    if d.empty:
        return pd.DataFrame({"decision_date": [], "late": []})
    return d[d["decision_date"] == pd.Timestamp(date)]


def _child_args(date) -> list[str]:
    return [sys.executable, "-m", "scripts.run_paper_log", "--decision-date", str(pd.Timestamp(date).date()), "--allow-late"]


def _run_child(args: list[str]) -> int:
    return subprocess.run(args, cwd=str(config.PROJECT_ROOT)).returncode


# ------------------------------------------------------------------------------------------ logic
def decision_weeks(start: pd.Timestamp, end: pd.Timestamp | None) -> list[pd.Timestamp]:
    """Weekly decision dates in [start, min(latest complete decision date, end)], ascending. An unevaluable calendar raises."""
    sessions = _sessions()
    weekly = pd.DatetimeIndex(_weekly(sessions))
    latest = _latest_complete(sessions)
    if latest is None or pd.isna(latest):
        raise RuntimeError("the paper-log calendar has no complete decision week; nothing can be evaluated")
    upper = pd.Timestamp(latest) if end is None else min(pd.Timestamp(latest), pd.Timestamp(end))
    return [pd.Timestamp(d) for d in sorted(weekly) if pd.Timestamp(start) <= d <= upper]


def _record(date, status, code, started, ended) -> None:
    rec = {"decision_date": str(pd.Timestamp(date).date()), "status": status, "child_exit_code": code,
           "started_utc": started, "ended_utc": ended, "statement": STATEMENT}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "paper_log_backfill.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_rows(date) -> str | None:
    """None if the log now holds rows for the date and every one is late whenever a later session exists; else the problem."""
    rows = _decision_rows(date)
    if len(rows) == 0:
        return "the child exited 0 but the decisions log holds no rows for this date"
    latest = _latest_equity()
    if latest is None or pd.isna(latest):
        return "the latest equity session is unknown: cannot verify the late flag"
    if pd.Timestamp(latest) > pd.Timestamp(date) and not rows["late"].astype(bool).all():
        return "a later session exists in the data but some rows are not flagged late"
    return None


def run(weeks: list[pd.Timestamp], dry_run: bool) -> dict:
    status: dict[str, str] = {}
    stopped = False
    for week in weeks:
        key = str(week.date())
        if stopped:
            status[key] = "not_attempted"
            continue
        if week in _logged_dates():
            status[key] = "already_logged"
            logger.info("{} already_logged", key)
            continue
        if dry_run:
            status[key] = "would_run"
            logger.info("{} would_run: {}", key, " ".join(_child_args(week)[1:]))
            continue
        logger.info("{} running the paper log (--allow-late)", key)
        started = _now()
        try:
            code = _run_child(_child_args(week))
        except Exception as e:                                                   # noqa: BLE001
            logger.error("{} child could not run: {}", key, e)
            _record(week, "failed", None, started, _now())
            status[key], stopped = "failed", True
            continue
        ended = _now()
        if code != 0:
            logger.error("{} REFUSED/FAILED: the paper log exited {}; stopping, a rerun resumes here", key, code)
            _record(week, "refused", code, started, ended)
            status[key], stopped = "refused", True
            continue
        problem = _check_rows(week)
        if problem:
            logger.error("{} FAILED: {}; stopping", key, problem)
            _record(week, "failed", code, started, ended)
            status[key], stopped = "failed", True
            continue
        _record(week, "logged", code, started, ended)
        status[key] = "logged"
        logger.info("{} logged", key)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--from", dest="start", default=DEFAULT_FROM, help=f"first decision date to backfill (default {DEFAULT_FROM})")
    ap.add_argument("--to", dest="end", default=None, help="last decision date to backfill (default: the latest complete one)")
    ap.add_argument("--dry-run", action="store_true", help="list the weeks; run nothing, write nothing")
    a = ap.parse_args(argv)
    try:
        weeks = decision_weeks(pd.Timestamp(a.start), pd.Timestamp(a.end) if a.end else None)
    except Exception as e:                                                       # noqa: BLE001
        logger.error("cannot evaluate the decision weeks: {}", e)
        return 1
    logger.info("{} decision week(s) from {}{}", len(weeks), a.start, f" to {a.end}" if a.end else "")
    status = run(weeks, a.dry_run)
    counts: dict[str, int] = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    logger.info("summary: {} | weeks: {}", counts or "nothing to do", status)
    return 0 if all(s in ("already_logged", "logged", "would_run") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
