"""Data catch-up wrapper: run the existing ingesters strictly one at a time.

    python -m scripts.run_data_refresh [--only STEP] [--start YYYY-MM-DD | --days N] [--step-timeout-hours H]
                                       [--downstream [--restart-downstream]]

Each step is `sys.executable -m <module>` with the same date range. Steps 1-7 run a weekday pass and then a --weekends
pass over that range. The ingesters skip dates they already hold, so the wrapper keeps no state and a rerun is safe. It
deletes nothing and stops at the first failing step, returning that step's exit code (or 124 on timeout, 130 on Ctrl-C).

Exit code 0 from an ingester is NOT trusted: data.ingest_common.run_ingestion catches per-date errors and records them as
`failed` in the manifest, then exits 0. After each pass the wrapper reads that ingester's manifest
(MANIFEST_DIR/<step>.jsonl, append-only, last record per date wins) and FAILS the step, naming the dates, if any date in
the requested range is `failed`. `empty` and `holiday` are not failures. Each step logs a currency line (newest date on
disk), and the run only says "complete" when every step passed.

fii_dii is forward capture of the latest day only: a failed fetch writes nothing and exits 0, which is not an error, but
the day is lost for good, so the wrapper logs a WARNING when there is no capture for today (IST) or the last session.

--days semantics differ for announcements: data.ingest_announcements takes `--days N` as a window that ENDS YESTERDAY
(N days, yesterday inclusive), whereas the other ingesters count back from today. The wrapper passes the same flag to all
of them; today's announcements are therefore picked up by the next day's run.

--downstream (R20-9): after ALL ingester steps have passed their checks, run the rebuild chain one child at a time and stop at the
first non-zero exit (the wrapper exits with that code): corporate-actions refresh (--refresh-from = the --start date), adjust_prices,
build_universe, the per-symbol feature blocks delivery / pledging / circuits / fno (always with --no-resume --list-orphans-only:
orphans are LISTED, never deleted), flows, breadth, and last the history guard (verify, owner-approved expected changes and fills).
A checkpoint (data/processed/data_refresh_checkpoint.json: run key = start date + today, completed step names) lets a rerun with
the same key skip completed downstream steps; the guard is never skipped. --restart-downstream ignores the checkpoint. The final
summary repeats every orphan line the children printed, per-step status and seconds, newest-on-disk per source and the guard's
last line. The wrapper deletes nothing. --downstream cannot be combined with --only.

No point-in-time store step is included (pending a user decision). Logs go to the console and to
LOG_DIR/data_refresh_<timestamp>.log so a detached run leaves a record.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (ANNOUNCEMENTS_DIR, BHAVCOPY_DIR, DELIVERY_DIR, FII_DII_DIR, FNO_BAN_DIR, FNO_BHAVCOPY_DIR,  # noqa: E402
                    LOG_DIR, MANIFEST_DIR, PRICE_BAND_HITS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, SEC_LIST_DIR)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAYS = 21
DEFAULT_STEP_TIMEOUT_HOURS = 2.0
IST = timezone(timedelta(hours=5, minutes=30))
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Where each step's data lands (for the currency line and the fii_dii check). Manifests: MANIFEST_DIR/<step>.jsonl.
DATA_DIRS = {
    "bhavcopy": BHAVCOPY_DIR,
    "bhavdata": DELIVERY_DIR,
    "fno_bhavcopy": FNO_BHAVCOPY_DIR,
    "fno_ban": FNO_BAN_DIR,
    "sec_list": SEC_LIST_DIR,
    "price_band_hits": PRICE_BAND_HITS_DIR,
    "indices": RAW_DATA_DIR / "indices",
    "fii_dii": FII_DII_DIR,
    "announcements": ANNOUNCEMENTS_DIR,
}


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    takes_range: bool = True
    weekend_pass: bool = True
    has_manifest: bool = True


STEPS = [
    Step("bhavcopy", "data.ingest_bhavcopy"),
    Step("bhavdata", "data.ingest_bhavdata"),
    Step("fno_bhavcopy", "data.ingest_fno_bhavcopy"),
    Step("fno_ban", "data.ingest_fno_ban"),
    Step("sec_list", "data.ingest_sec_list"),
    Step("price_band_hits", "data.ingest_price_band_hits"),
    Step("indices", "data.ingest_indices"),
    Step("fii_dii", "data.ingest_fii_dii", takes_range=False, weekend_pass=False, has_manifest=False),
    Step("announcements", "data.ingest_announcements", weekend_pass=False),
]


def _today() -> date:
    return datetime.now(IST).date()


def _commands(step: Step, range_args: list[str]) -> list[list[str]]:
    base = [sys.executable, "-m", step.module]
    if not step.takes_range:
        return [base]
    cmds = [base + range_args]
    if step.weekend_pass:
        cmds.append(base + range_args + ["--weekends"])
    return cmds


# ------------------------------------------------------------------------------------------ downstream chain (R20-9)
GUARD_STEP = "history_guard"
GUARD_CORRECTIONS = "docs/agent-team/corrections/2026-11-10_precutoff_corporate_actions.json"
GUARD_FILLS = "docs/agent-team/corrections/2026-11-12_guard_expected_fills.json"
CHECKPOINT_PATH = PROCESSED_DATA_DIR / "data_refresh_checkpoint.json"
ORPHAN_RE = re.compile(r"orphan symbol file\(s\).*NOT deleted")


def _per_symbol_block(module: str) -> list[str]:
    """The ONLY way this wrapper builds a per-symbol feature block command: orphans are listed, never deleted."""
    return [sys.executable, "-m", module, "--no-resume", "--list-orphans-only"]


def downstream_commands(start: date) -> list[tuple[str, list[str]]]:
    """The rebuild chain, in order, as (step name, argument list). Never run through a shell."""
    py = [sys.executable, "-m"]
    return [
        ("corporate_actions", py + ["data.ingest_corporate_actions", "--refresh-from", start.isoformat()]),
        ("adjust_prices", py + ["data.adjust_prices"]),
        ("universe", py + ["data.build_universe"]),
        ("delivery", _per_symbol_block("features.blocks.delivery")),
        ("pledging", _per_symbol_block("features.blocks.pledging")),
        ("circuits", _per_symbol_block("features.blocks.circuits")),
        ("fno", _per_symbol_block("features.blocks.fno")),
        ("flows", py + ["features.blocks.flows"]),
        ("breadth", py + ["features.blocks.breadth"]),
        (GUARD_STEP, py + ["scripts.history_guard", "verify", "--skip-db",
                           "--expected-changes", GUARD_CORRECTIONS, "--expected-fills", GUARD_FILLS]),
    ]


def _run_child(cmd: list[str], timeout: int) -> tuple[int, list[str]]:
    """Run one downstream child, relay its output line by line to the log, and return (exit code, kept lines). Kept lines are the
    orphan lines plus the last non-empty line (the guard's verdict); exit code 124 on timeout."""
    kept: list[str] = []
    last = ""
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            logger.info("  | {}", line)
            if ORPHAN_RE.search(line):
                kept.append(line)
            if line.strip():
                last = line
        proc.wait()
    finally:
        timer.cancel()
    if last:
        kept.append(last)
    return (124 if timed_out.is_set() else proc.returncode), kept


def _read_checkpoint(key: dict) -> list[str]:
    try:
        data = json.loads(Path(CHECKPOINT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    done = data.get("completed") if isinstance(data, dict) and data.get("key") == key else None
    return [s for s in done if isinstance(s, str)] if isinstance(done, list) else []


def _write_checkpoint(key: dict, completed: list[str]) -> None:
    """Atomic (tmp + os.replace): a crash leaves the previous valid file. A failed write is a warning, not a failed step. The fixed
    temp name is simply overwritten (and consumed by the replace) by the next write; this module deletes nothing."""
    path = Path(CHECKPOINT_PATH)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"key": key, "completed": completed}, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("could not write the downstream checkpoint {}: {} (a rerun will redo steps)", path.name, e)


def _log_wrapup(rows: list[tuple[str, str, float]], orphans: list[str], guard_line: str, total: float) -> None:
    logger.info("data refresh summary (step, status, seconds):")
    for name, status, secs in rows:
        logger.info("  {:<18} {:<22} {:>8.1f}", name, status, secs)
    logger.info("  total time {:.1f}s", total)
    newest = {n: (str(newest_on_disk(d)) if d is not None else "none") for n, d in DATA_DIRS.items()}
    logger.info("  newest on disk: {}", newest)
    if orphans:
        logger.warning("  orphan lines reported by the children (listed, NOT deleted; {}):", len(orphans))
        for line in orphans:
            logger.warning("    {}", line)
    else:
        logger.info("  no orphan lines reported")
    if guard_line:
        logger.info("  guard last line: {}", guard_line)


def _run_downstream(args, start: date, today: date, timeout: int, rows: list, t0: float) -> int:
    key = {"start": start.isoformat(), "today": today.isoformat()}
    completed = [] if args.restart_downstream else _read_checkpoint(key)
    if completed:
        logger.info("downstream checkpoint found for {}: skipping {}", key, completed)
    orphans: list[str] = []
    guard_line = ""
    rc_out = 0
    for name, cmd in downstream_commands(start):
        if name != GUARD_STEP and name in completed:
            rows.append((name, "skipped (checkpoint)", 0.0))
            continue
        logger.info("downstream step {}: {}", name, " ".join(cmd[1:]))
        began = time.monotonic()
        rc, lines = _run_child(cmd, timeout)
        secs = time.monotonic() - began
        orphans += [ln for ln in lines if ORPHAN_RE.search(ln)]
        last = next((ln for ln in reversed(lines) if ln.strip()), "")
        if name == GUARD_STEP:
            guard_line = last
        if rc != 0:
            rows.append((name, f"FAILED (exit {rc})", secs))
            if name == GUARD_STEP:
                logger.error("history guard FAILED: stop, do not use the rebuilt data until the owner decides (exit {})", rc)
            elif rc == 124:
                logger.error("downstream step {} FAILED: timed out after {}s: {}", name, timeout, " ".join(cmd[1:]))
            else:
                logger.error("downstream step {} FAILED with exit code {}: {}", name, rc, " ".join(cmd[1:]))
            rc_out = rc
            break
        rows.append((name, "ok", secs))
        if name != GUARD_STEP:
            completed.append(name)
            _write_checkpoint(key, completed)
    _log_wrapup(rows, orphans, guard_line, time.monotonic() - t0)
    return rc_out


# ------------------------------------------------------------------------------------------ checks
def newest_on_disk(directory: Path) -> date | None:
    """Newest date embedded in a *.parquet file name (the last date for range-named chunks)."""
    best = None
    for path in Path(directory).glob("*.parquet"):
        found = _DATE_RE.findall(path.stem)
        if found:
            d = date.fromisoformat(found[-1])
            best = d if best is None or d > best else best
    return best


def last_records(manifest_path: Path) -> dict[str, tuple[str, str]]:
    """key -> (status, error) of the LAST record per key. Undecodable lines are skipped with a WARNING."""
    path = Path(manifest_path)
    last: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return last
    for number, raw in enumerate(path.read_bytes().splitlines(), 1):
        line = raw.decode("utf-8", errors="replace")
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            last[rec["trade_date"]] = (rec["status"], str(rec.get("error") or ""))
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("manifest {}: skipping undecodable line {} ({}...)", path.name, number, line[:60])
    return last


def failed_records(manifest_path: Path, start: date, end: date) -> dict[str, str]:
    """key -> error for keys whose LAST manifest record is `failed` and that fall in [start, end]. A range key `a_b` counts if it
    overlaps."""
    out = {}
    for key, (status, error) in sorted(last_records(manifest_path).items()):
        if status != "failed":
            continue
        found = _DATE_RE.findall(key)
        if not found:
            continue
        try:
            a, b = date.fromisoformat(found[0]), date.fromisoformat(found[-1])
        except ValueError:
            # matches the date pattern but is not a calendar date: cannot be placed in the window, so never ignore it
            logger.warning("manifest {}: failed key {!r} is not a valid calendar date; counted as FAILED", Path(manifest_path).name, key)
            out[key] = error
            continue
        if a <= end and b >= start:
            out[key] = error
    return out


def failed_dates(manifest_path: Path, start: date, end: date) -> list[str]:
    return list(failed_records(manifest_path, start, end))


_STALE_RE = re.compile(r"file dated (\d{4}-\d{2}-\d{2}), requested (\d{4}-\d{2}-\d{2})")


def is_expected_stale_reserve(step_name: str, key: str, error: str,
                              equity_last: dict[str, tuple[str, str]] | None = None) -> bool:
    """True iff this `failed` record is NSE re-serving an EARLIER day's file on a non-session, confirmed by the equity bhavcopy.
    All of: (a) the error is `file dated A, requested B` with B == key and A < B; (b) no equity bhavcopy file exists for the
    date; (c) the equity bhavcopy manifest's last record for the date is `holiday` (weekdays and weekends alike: the equity
    weekend pass records `holiday` for non-sessions). A step never excuses itself: the bhavcopy step's own failures always
    fail. The ingester's header-date check is what rejected the file and is never loosened; this only recognises its known,
    benign outcome. Anything else stays a failure. `equity_last` is the equity manifest's last_records(), read once per step by
    the caller (read here when omitted)."""
    if step_name == "bhavcopy":
        return False
    m = _STALE_RE.search(error or "")
    if not m or m.group(2) != key:
        return False
    try:
        file_date, requested = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
    except ValueError:
        return False
    if not file_date < requested:
        return False
    if (Path(DATA_DIRS["bhavcopy"]) / f"{key}.parquet").exists():
        return False
    if equity_last is None:
        equity_last = last_records(MANIFEST_DIR / "bhavcopy.jsonl")
    equity = equity_last.get(key)
    return bool(equity) and equity[0] == "holiday"


def _first_date(key: str) -> date | None:
    """First date of a manifest key (the only date of a single-date key); None if absent or not a calendar date."""
    found = _DATE_RE.findall(key)
    try:
        return date.fromisoformat(found[0]) if found else None
    except ValueError:
        return None


def _currency(step: Step) -> None:
    newest = newest_on_disk(DATA_DIRS[step.name])
    if newest is None:
        logger.info("step {}: no files on disk in {}", step.name, DATA_DIRS[step.name])
    else:
        logger.info("step {}: newest on disk {}", step.name, newest)


def _check_fii_dii(today: date) -> None:
    """Forward capture only: no file for today or the last session means that day's DII figure is lost. Visible, not fatal."""
    newest = newest_on_disk(DATA_DIRS["fii_dii"])
    last_session = None
    for path in Path(DATA_DIRS["bhavcopy"]).glob("*.parquet"):
        found = _DATE_RE.findall(path.stem)
        if found and date.fromisoformat(found[-1]) <= today:
            d = date.fromisoformat(found[-1])
            last_session = d if last_session is None or d > last_session else last_session
    wanted = {today} | ({last_session} if last_session else set())
    present = {date.fromisoformat(_DATE_RE.findall(p.stem)[-1]) for p in Path(DATA_DIRS["fii_dii"]).glob("*.parquet")
               if _DATE_RE.findall(p.stem)}
    if not (present & wanted):
        logger.warning("step fii_dii: no capture for today ({}) or the last session ({}); newest on disk {}. DII is forward-capture "
                       "only, so the missing day is lost for good (NSE serves only the latest day).", today, last_session, newest)


def _parse_date(text: str) -> str:
    try:
        date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid YYYY-MM-DD date: {text!r}")
    return text


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parse(argv):
    p = argparse.ArgumentParser(description="Run the data ingesters one at a time.")
    p.add_argument("--only", choices=[s.name for s in STEPS], help="run just this step")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", type=_parse_date, help="start date YYYY-MM-DD")
    g.add_argument("--days", type=_positive_int, help=f"last N days (default {DEFAULT_DAYS})")
    p.add_argument("--step-timeout-hours", type=_positive_float, default=DEFAULT_STEP_TIMEOUT_HOURS,
                   help="kill a single ingester process after this long (default 2)")
    p.add_argument("--downstream", action="store_true",
                   help="after the ingesters: corporate-actions refresh, adjust_prices, universe, feature blocks (orphans listed, "
                        "never deleted), flows, breadth, then the history guard; resumable through a checkpoint")
    p.add_argument("--restart-downstream", action="store_true",
                   help="with --downstream: ignore the downstream checkpoint and start the chain from the first step")
    args = p.parse_args(argv)
    if args.downstream and args.only:
        p.error("--downstream cannot be combined with --only")
    if args.restart_downstream and not args.downstream:
        p.error("--restart-downstream is only valid together with --downstream")
    return args


# ------------------------------------------------------------------------------------------ run
def _run(args) -> int:
    today = _today()
    if args.start is not None:
        range_args, start = ["--start", args.start], date.fromisoformat(args.start)
    else:
        days = args.days if args.days is not None else DEFAULT_DAYS
        range_args, start = ["--days", str(days)], today - timedelta(days=days)
    timeout = int(args.step_timeout_hours * 3600)

    t0 = time.monotonic()
    rows: list[tuple[str, str, float]] = []
    steps = [s for s in STEPS if args.only in (None, s.name)]
    for step in steps:
        began = time.monotonic()
        for cmd in _commands(step, range_args):
            logger.info("step {}: {}", step.name, " ".join(cmd[1:]))
            try:
                rc = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                logger.error("step {} FAILED: timed out after {}s: {}", step.name, timeout, " ".join(cmd[1:]))
                return 124
            if rc != 0:
                logger.error("step {} FAILED with exit code {}: {}", step.name, rc, " ".join(cmd[1:]))
                return rc
        if step.has_manifest:
            failed = failed_records(MANIFEST_DIR / f"{step.name}.jsonl", start, today)
            # pending = not yet published: the key STARTS today or later. A range that starts earlier covers days that should exist.
            pending = [k for k in failed if _first_date(k) is not None and _first_date(k) >= today]
            if pending:
                logger.info("step {}: {} not yet published, not checked (today or later, next run retries): {}",
                            step.name, len(pending), pending)
                failed = {k: e for k, e in failed.items() if k not in pending}
            equity_last = last_records(MANIFEST_DIR / "bhavcopy.jsonl") if failed else {}
            excused = [k for k, e in failed.items() if is_expected_stale_reserve(step.name, k, e, equity_last)]
            if excused:
                logger.info("step {}: {} date(s) with an expected stale re-serve on a non-session (excused): {}",
                            step.name, len(excused), excused)
            bad = [k for k in failed if k not in excused]
            if bad:
                logger.error("step {} FAILED: the manifest records `failed` for {} date(s) in {}..{} despite exit 0: {}",
                             step.name, len(bad), start, today, bad[:20])
                _currency(step)
                return 1
        _currency(step)
        if step.name == "fii_dii":
            _check_fii_dii(today)
        rows.append((step.name, "ok", time.monotonic() - began))
    logger.info("data refresh complete: {} step(s), no failed dates", len(steps))
    if args.downstream:
        rc = _run_downstream(args, start, today, timeout, rows, t0)
        if rc != 0:
            return rc
        logger.info("data refresh with downstream chain complete: {} ingester step(s) and the history guard passed", len(steps))
    return 0


def main(argv=None) -> int:
    args = _parse(argv)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(LOG_DIR) / f"data_refresh_{datetime.now(IST):%Y%m%d_%H%M%S}.log"
    sink = logger.add(log_path, level="INFO", encoding="utf-8")
    try:
        return _run(args)
    except KeyboardInterrupt:
        logger.error("data refresh interrupted (Ctrl-C); exiting nonzero")
        return 130
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    sys.exit(main())
