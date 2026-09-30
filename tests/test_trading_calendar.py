"""tests/test_trading_calendar.py — data/trading_calendar.py: the trading-session calendar every India-block feature is aligned to."""

from datetime import date

import pandas as pd
import pytest

from data.trading_calendar import trading_sessions


def _touch(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.parquet").write_bytes(b"")


def test_union_of_all_given_directories_sorted_and_unique(tmp_path):
    a, b = tmp_path / "equity", tmp_path / "fno"
    _touch(a, ["2020-07-10", "2020-07-14"])
    _touch(b, ["2020-07-10", "2020-07-13", "2020-07-14"])          # F&O has 07-13, which equity bhavcopy lacks
    assert trading_sessions([a, b]) == [date(2020, 7, 10), date(2020, 7, 13), date(2020, 7, 14)]


def test_weekend_sessions_are_included(tmp_path):
    a = tmp_path / "fno"
    _touch(a, ["2025-01-31", "2025-02-01", "2025-02-03"])           # Saturday 2025-02-01 was the Budget session
    assert date(2025, 2, 1) in trading_sessions([a])


def test_non_date_files_are_ignored_not_fatal(tmp_path):
    a = tmp_path / "x"
    _touch(a, ["2020-01-02", "_manifest", "notes"])
    assert trading_sessions([a]) == [date(2020, 1, 2)]


def test_missing_directory_is_skipped_but_all_missing_raises(tmp_path):
    a = tmp_path / "exists"
    _touch(a, ["2020-01-02"])
    assert trading_sessions([a, tmp_path / "absent"]) == [date(2020, 1, 2)]
    with pytest.raises(ValueError, match="no sessions"):
        trading_sessions([tmp_path / "absent"])


def test_default_directories_are_the_equity_and_fno_bhavcopy_archives():
    from config import BHAVCOPY_DIR, FNO_BHAVCOPY_DIR
    from data.trading_calendar import DEFAULT_SESSION_DIRS
    assert list(DEFAULT_SESSION_DIRS) == [BHAVCOPY_DIR, FNO_BHAVCOPY_DIR]
