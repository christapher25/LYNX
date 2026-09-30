"""
tests/invariants/test_delivery_row_date.py

Invariant: the delivery loader keys every row on the date EMBEDDED in the file, never on the
filename.

Why this can silently go wrong: five files in data/raw/delivery/ are named for one day but hold
another day's rows, because NSE's archive re-served stale content for that URL:

    file name        rows inside     what the file really is
    2019-09-30   ->  2019-06-27      a three-month-old snapshot
    2019-10-02   ->  2019-10-01      an exact copy of the 2019-10-01 file
    2019-10-08   ->  2019-10-07      an exact copy of the 2019-10-07 file
    2019-10-21   ->  2019-10-18      an exact copy of the 2019-10-18 file
    2019-10-28   ->  2019-10-27      the Sunday Muhurat session, not Monday

A loader that trusts the filename would file Sunday's rows under Monday, and 3-month-old rows under
2019-09-30 — real-looking delivery numbers attached to the wrong day, with no error anywhere. All
five sit in the 24 files that predate the current ingester's own date check, and they use NSE's raw
column names (DATE1, TTL_TRD_QNTY, ...) rather than the normalized ones.

The real-data tests skip when data/raw/delivery is absent (CI has no data); the synthetic tests
reproduce the same shapes and always run.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import DELIVERY_DIR
from features.blocks.delivery import load_delivery, read_delivery_file

# filename date -> the date actually embedded in that file's rows (measured on the real archive)
KNOWN_MISMATCHED_FILES = {
    "2019-09-30": "2019-06-27",
    "2019-10-02": "2019-10-01",
    "2019-10-08": "2019-10-07",
    "2019-10-21": "2019-10-18",
    "2019-10-28": "2019-10-27",
}

NEW_SCHEMA = ["SYMBOL", "SERIES", "DATE", "PREV_CLOSE", "OPEN", "HIGH", "LOW", "LAST", "CLOSE", "VWAP",
              "VOLUME", "TURNOVER_LACS", "NUM_TRADES", "DELIV_QTY", "DELIV_PER"]
OLD_SCHEMA_RENAMES = {"DATE": "DATE1", "OPEN": "OPEN_PRICE", "HIGH": "HIGH_PRICE", "LOW": "LOW_PRICE",
                      "LAST": "LAST_PRICE", "CLOSE": "CLOSE_PRICE", "VWAP": "AVG_PRICE",
                      "VOLUME": "TTL_TRD_QNTY", "NUM_TRADES": "NO_OF_TRADES"}


def _rows(row_date: str, deliv_qty: float = None) -> pd.DataFrame:
    """Two EQ symbols and one BE symbol, values derived from the row date so a mix-up is visible."""
    day = pd.Timestamp(row_date)
    qty = 100.0 + day.day if deliv_qty is None else deliv_qty
    frame = pd.DataFrame({
        "SYMBOL": ["AAA", "BBB", "CCC"], "SERIES": ["EQ", "EQ", "BE"], "DATE": day,
        "PREV_CLOSE": 10.0, "OPEN": 10.0, "HIGH": 11.0, "LOW": 9.0, "LAST": 10.0, "CLOSE": 10.0,
        "VWAP": 10.0, "VOLUME": 1000, "TURNOVER_LACS": 1.0, "NUM_TRADES": 5,
        "DELIV_QTY": qty, "DELIV_PER": qty / 10.0,
    })
    return frame[NEW_SCHEMA]


def _write(directory, file_name: str, row_date: str, *, old_schema: bool = False, deliv_qty: float = None):
    frame = _rows(row_date, deliv_qty)
    if old_schema:
        frame = frame.rename(columns=OLD_SCHEMA_RENAMES)
    path = directory / f"{file_name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _filename_keyed_load(paths) -> pd.DataFrame:
    """The bug this file guards against: dating rows by the file's name."""
    frames = []
    for p in paths:
        frame = read_delivery_file(p)
        frame["date"] = pd.Timestamp(p.stem)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def stale_archive(tmp_path):
    """The five real failure shapes in miniature, mixed schemas included."""
    _write(tmp_path, "2019-09-30", "2019-06-27", old_schema=True)   # stale months-old snapshot
    _write(tmp_path, "2019-10-01", "2019-10-01", old_schema=True)
    _write(tmp_path, "2019-10-02", "2019-10-01", old_schema=True)   # exact copy of the day before
    _write(tmp_path, "2019-10-03", "2019-10-03")                    # new schema, honest
    return sorted(tmp_path.glob("*.parquet"))


class TestReaderKeysOnEmbeddedDate:
    def test_file_named_for_one_day_holding_another_returns_the_embedded_date(self, tmp_path):
        path = _write(tmp_path, "2019-09-30", "2019-06-27")
        out = read_delivery_file(path)
        assert set(out["date"]) == {pd.Timestamp("2019-06-27")}
        assert pd.Timestamp("2019-09-30") not in set(out["date"])

    def test_legacy_column_names_are_normalised(self, tmp_path):
        out = read_delivery_file(_write(tmp_path, "2019-10-01", "2019-10-01", old_schema=True))
        assert {"symbol", "series", "date", "deliv_qty", "deliv_per", "volume"} <= set(out.columns)
        assert set(out["date"]) == {pd.Timestamp("2019-10-01")}
        assert (out["volume"] == 1000).all()

    def test_file_with_more_than_one_embedded_date_is_rejected(self, tmp_path):
        mixed = pd.concat([_rows("2024-01-02"), _rows("2024-01-03")], ignore_index=True)
        path = tmp_path / "2024-01-03.parquet"
        mixed.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="more than one"):
            read_delivery_file(path)

    def test_file_with_an_unparseable_date_is_rejected_not_dated_by_its_name(self, tmp_path):
        frame = _rows("2024-01-02")
        frame["DATE"] = pd.NaT
        path = tmp_path / "2024-01-02.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="date"):
            read_delivery_file(path)


class TestLoaderOverADirectory:
    def test_every_row_carries_its_embedded_date(self, stale_archive):
        loaded = load_delivery(stale_archive)
        assert set(loaded["date"]) == {pd.Timestamp(d) for d in ("2019-06-27", "2019-10-01", "2019-10-03")}

    def test_filename_only_dates_never_appear(self, stale_archive):
        """2019-09-30 and 2019-10-02 exist only as file names; no real data was ever recorded for them."""
        loaded_dates = {d.date() for d in loaded_dates_of(stale_archive)}
        assert date(2019, 9, 30) not in loaded_dates
        assert date(2019, 10, 2) not in loaded_dates

    def test_stale_copy_collapses_to_one_observation_per_symbol_and_day(self, stale_archive):
        loaded = load_delivery(stale_archive)
        assert not loaded.duplicated(["symbol", "date"]).any()
        assert len(loaded[(loaded["symbol"] == "AAA") & (loaded["date"] == "2019-10-01")]) == 1

    def test_stale_snapshot_keeps_its_own_values_on_its_own_date(self, stale_archive):
        loaded = load_delivery(stale_archive).set_index(["symbol", "date"])
        assert loaded.loc[("AAA", pd.Timestamp("2019-06-27")), "deliv_qty"] == 100.0 + 27
        assert loaded.loc[("AAA", pd.Timestamp("2019-10-01")), "deliv_qty"] == 100.0 + 1

    def test_only_eq_series_is_loaded(self, stale_archive):
        """Stage 4's adjusted-price panel is EQ-only; other series would never join to it."""
        assert set(load_delivery(stale_archive)["series"]) == {"EQ"}

    def test_two_files_disagreeing_about_the_same_symbol_and_day_raise(self, tmp_path):
        _write(tmp_path, "2024-03-04", "2024-03-04", deliv_qty=500.0)
        _write(tmp_path, "2024-03-05", "2024-03-04", deliv_qty=999.0)   # claims the same day, different numbers
        with pytest.raises(ValueError, match="conflicting"):
            load_delivery(sorted(tmp_path.glob("*.parquet")))

    def test_a_loader_that_trusts_filenames_is_caught_by_these_assertions(self, stale_archive):
        """Sensitivity check: the reference bug produces exactly the wrong dates the tests above forbid."""
        buggy_dates = {d.date() for d in _filename_keyed_load(stale_archive)["date"]}
        assert date(2019, 9, 30) in buggy_dates and date(2019, 10, 2) in buggy_dates      # phantom dates
        assert date(2019, 6, 27) not in buggy_dates                                       # real date lost


def loaded_dates_of(paths):
    return set(load_delivery(paths)["date"])


# --- the five real files ------------------------------------------------------------------------

def _real_file(name: str):
    path = DELIVERY_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present (no raw delivery data in this environment)")
    return path


@pytest.mark.parametrize("file_name,embedded", sorted(KNOWN_MISMATCHED_FILES.items()))
def test_known_mismatched_real_file_is_dated_by_its_rows(file_name, embedded):
    out = read_delivery_file(_real_file(file_name))
    assert len(out) > 1000
    assert set(out["date"]) == {pd.Timestamp(embedded)}
    assert pd.Timestamp(file_name) not in set(out["date"])


def test_real_archive_slice_containing_all_five_never_invents_their_filename_dates():
    """
    Load the whole legacy-era slice (2019-09-30 .. 2019-10-31, 24 files). Every date in the result
    must be a date some file actually holds rows for; the five filename-only dates must be absent
    (2019-10-02/08/21/28 were NSE holidays; 2019-09-30 was a real trading day whose data NSE never served).
    """
    paths = [_real_file(n) for n in sorted(KNOWN_MISMATCHED_FILES)]
    slice_paths = sorted(p for p in DELIVERY_DIR.glob("2019-*.parquet") if "2019-09-30" <= p.stem <= "2019-10-31")
    assert set(paths) <= set(slice_paths)

    loaded_dates = {d.strftime("%Y-%m-%d") for d in load_delivery(slice_paths)["date"]}
    embedded_dates = {d.strftime("%Y-%m-%d") for p in slice_paths for d in read_delivery_file(p)["date"].unique()}
    assert loaded_dates == embedded_dates
    assert not loaded_dates & set(KNOWN_MISMATCHED_FILES)                 # none of the five filename dates
    assert set(KNOWN_MISMATCHED_FILES.values()) <= loaded_dates           # every embedded date is present
    assert not load_delivery(slice_paths).duplicated(["symbol", "date"]).any()
