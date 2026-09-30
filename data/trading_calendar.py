"""
data/trading_calendar.py

The trading-session calendar every India-block feature is aligned to: the union of the days for which NSE published an
equity bhavcopy or an F&O bhavcopy. The union matters: the equity archive is missing 2020-07-13 (Stage 3's parser
rejected that day's two-digit-year dates) and both archives were crawled weekdays-only until the F&O `--weekends`
pass found 30 weekend sessions (Budget Saturdays, Muhurat). A calendar that lacks a real session mis-counts every
"last N sessions" window that spans it.

Named trading_calendar, NOT calendar: `python data/ingest_x.py` puts data/ first on sys.path, so a data/calendar.py would
replace the standard-library `calendar` for everything imported afterwards and break every ingester
(tests/invariants/test_no_stdlib_shadowing.py guards this).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

from config import BHAVCOPY_DIR, FNO_BHAVCOPY_DIR

DEFAULT_SESSION_DIRS = (BHAVCOPY_DIR, FNO_BHAVCOPY_DIR)


def trading_sessions(directories: Iterable[Path] = DEFAULT_SESSION_DIRS) -> list[date]:
    """Sorted, unique session dates from `{YYYY-MM-DD}.parquet` file names. Non-date files are ignored."""
    days: set[date] = set()
    for directory in directories:
        if not Path(directory).exists():
            continue
        for path in Path(directory).glob("*.parquet"):
            try:
                days.add(date.fromisoformat(path.stem))
            except ValueError:
                continue
    if not days:
        raise ValueError(f"no sessions found in {[str(d) for d in directories]}")
    return sorted(days)
