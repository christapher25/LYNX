"""Read-only audit of a corporate-actions refresh (data/ingest_corporate_actions.py --refresh-from).

    python -m scripts.audit_corp_action_refresh [--since 2026-09-30] [--log logs/jobs/<refresh>.err.log]

1. Possible reworded duplicates: two rows of the same symbol with ex_date >= since that share (ex_date, action_type, rounded
   adjustment_factor) and a NON-null factor. A refresh run under the old code could append such a row when NSE reworded a
   subject, which would double-adjust prices. Rows without a factor (dividends) are never reported.
2. Factor-bearing actions (splits, bonuses...) per symbol with ex_date >= since.
3. If --log is given: the `NOT added` lines (new actions dated before the refresh date), by symbol. Those log lines carry no
   action type, so a by-type grouping is not available from the log and is not guessed.

Writes nothing; exit code is always 0 (it reports, the reader decides).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SINCE = date(2026, 9, 30)
_NOT_ADDED_RE = re.compile(r"(\S+): (\d+) new action\(s\) dated before \S+ .*?were NOT added: \[(.*)\]")


def _window(directory: Path, since: date):
    cutoff = pd.Timestamp(since)
    for path in sorted(Path(directory).glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["symbol", "ex_date", "raw_subject", "action_type", "adjustment_factor"])
        yield path.stem, frame[pd.to_datetime(frame["ex_date"]) >= cutoff]


def find_possible_duplicates(directory: Path, since: date) -> list[dict]:
    out = []
    for symbol, rows in _window(directory, since):
        rows = rows[rows["adjustment_factor"].notna()].copy()
        if rows.empty:
            continue
        rows["_factor"] = rows["adjustment_factor"].astype(float).round(12)
        rows["_day"] = pd.to_datetime(rows["ex_date"]).dt.strftime("%Y-%m-%d")
        for (day, action_type, factor), group in rows.groupby(["_day", "action_type", "_factor"], sort=True):
            if len(group) > 1:
                out.append({"symbol": symbol, "ex_date": day, "action_type": action_type, "factor": float(factor),
                            "subjects": list(group["raw_subject"])})
    return out


def factor_bearing_counts(directory: Path, since: date) -> dict[str, dict[str, int]]:
    out = {}
    for symbol, rows in _window(directory, since):
        rows = rows[rows["adjustment_factor"].notna()]
        if len(rows):
            out[symbol] = dict(Counter(rows["action_type"]))
    return out


def parse_not_added(log_path: Path) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _NOT_ADDED_RE.search(line)
        if m:
            dates = [d.strip().strip("'\"") for d in m.group(3).split(",") if d.strip()]
            rows.append({"symbol": m.group(1), "count": int(m.group(2)), "dates": dates})
    return rows


def group_by_type(rows: list[dict]) -> dict | None:
    """The refresh log lines carry no action type, so there is nothing to group by: None, never a guess."""
    return None if not any("action_type" in r for r in rows) else {}


def main(argv=None) -> int:
    from config import CORPORATE_ACTIONS_DIR
    parser = argparse.ArgumentParser(description="Read-only audit of a corporate-actions refresh.")
    parser.add_argument("--since", type=date.fromisoformat, default=DEFAULT_SINCE)
    parser.add_argument("--log", type=Path, help="refresh job err log to summarise its 'NOT added' lines")
    args = parser.parse_args(argv)

    dups = find_possible_duplicates(CORPORATE_ACTIONS_DIR, args.since)
    logger.info("possible reworded duplicates (same symbol/ex_date/type/factor, ex_date >= {}): {}", args.since, len(dups))
    for d in dups:
        logger.warning("{} {} {} factor {}: {}", d["symbol"], d["ex_date"], d["action_type"], d["factor"], d["subjects"])

    counts = factor_bearing_counts(CORPORATE_ACTIONS_DIR, args.since)
    total = Counter()
    for symbol, by_type in counts.items():
        total.update(by_type)
        logger.info("factor-bearing since {}: {} {}", args.since, symbol, by_type)
    logger.info("factor-bearing actions since {}: {} symbol(s), by type {}", args.since, len(counts), dict(total))

    if args.log:
        rows = parse_not_added(args.log)
        logger.info("'NOT added' lines in {}: {} symbol(s), {} action(s)", args.log, len(rows), sum(r["count"] for r in rows))
        for r in rows:
            logger.info("NOT added {}: {} {}", r["symbol"], r["count"], r["dates"])
        if group_by_type(rows) is None:
            logger.info("the log lines carry no action type, so no by-type grouping is possible from the log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
