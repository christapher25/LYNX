"""
tests/test_ingest_bhavdata.py

Tests for data/ingest_bhavdata.py — Stage 3 sec_bhavdata_full (OHLCV + delivery %) ingestion. Zero coverage
existed before this file (found during Stage 10's cleanup of the known gaps in data/raw/DATA_NOTES.md §11).

Two real, dated failures, both confirmed against the live archive and both now fixed here:

  2022-08-08  NSE served this one day's "sec_bhavdata_full_08082022.csv" as a genuine XLSX file (ZIP magic
              bytes `PK\\x03\\x04`, real OOXML spreadsheet parts inside) despite the .csv URL and Content-Type.
              read_content() blindly ran pd.read_csv on the raw bytes decoded as text, producing garbage that
              choked the C tokenizer ("Expected 1 fields in line 5, saw 2" — the manifest's actual error). The
              real fixture (tests/fixtures/bhavdata/sec_bhavdata_full_08082022.xlsx) is 2,255 valid rows.
  2024-02-19  Exactly one row (WTICAB, series ST — not even EQ) has DELIV_QTY > VOLUME, a genuine impossible
              value on NSE's side. validate() rejected the WHOLE ~2,646-row file for this one row. The fix is
              DATA_NOTES.md §11's own prescription: quarantine the offending row's delivery fields (NaN, never
              guessed), not the file — mirroring how features/blocks/delivery.py already quarantines individual
              bad rows downstream, just moved to where it belongs: ingestion, so the ~2,645 good rows are not
              lost with it. A file where an unreasonable SHARE of rows are impossible must still fail loudly
              (that is a parser/format problem, not a few bad data points) — the mutation-check below proves it.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.ingest_bhavdata import (
    FIRST_AVAILABLE,
    OUTPUT_COLS,
    parse_bhavdata,
    read_bhavdata_content,
    validate,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bhavdata"

# (symbol, series, prev, open, high, low, last, close, vwap, volume, turnover_lacs, num_trades, deliv_qty, deliv_per)
ROWS = [
    ("RELIANCE", " EQ", 2848.0, 2850.0, 2880.0, 2845.0, 2866.0, 2865.5, 2860.0, 5_500_000, 157500.0, 120_000, 2_000_000, "36.36"),
    ("TCS", " EQ", 3818.0, 3820.0, 3850.0, 3810.0, 3841.0, 3840.0, 3830.0, 1_200_000, 46000.0, 45_000, 600_000, "50.00"),
]
CSV_HEADER = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, "
              "AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER")


def _csv_text(trade_date_text="08-Aug-2022", rows=ROWS) -> str:
    lines = [CSV_HEADER]
    for sym, series, prev, o, h, l, last, c, vwap, vol, turn, n, dq, dp in rows:
        lines.append(f"{sym},{series},{trade_date_text},{prev},{o},{h},{l},{last},{c},{vwap},{vol},{turn},{n},{dq},{dp}")
    return "\n".join(lines) + "\n"


class TestParseBhavdata:
    def test_the_declared_columns_in_order(self):
        assert list(parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text()))).columns) == OUTPUT_COLS

    def test_values_round_trip(self):
        out = parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text())))
        assert out.loc[0, "SYMBOL"] == "RELIANCE"
        assert out.loc[0, "DATE"] == pd.Timestamp("2022-08-08")
        assert out.loc[0, "DELIV_PER"] == pytest.approx(36.36)

    def test_dash_sentinel_is_missing_not_zero(self):
        rows = list(ROWS)
        rows[0] = ("RELIANCE", " EQ", 2848.0, 2850.0, 2880.0, 2845.0, 2866.0, 2865.5, 2860.0, 5_500_000, 157500.0, 120_000, "-", "-")
        out = parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text(rows=rows))))
        assert out.loc[0, "DELIV_QTY"] is None or pd.isna(out.loc[0, "DELIV_QTY"])
        assert pd.isna(out.loc[0, "DELIV_PER"])


class TestValidate:
    def test_a_clean_day_passes(self):
        validate(parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text()))), date(2022, 8, 8))

    def test_a_date_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="requested"):
            validate(parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text()))), date(2022, 8, 9))

    def test_deliv_per_out_of_range_is_still_a_hard_failure(self):
        rows = list(ROWS)
        rows[0] = ("RELIANCE", " EQ", 2848.0, 2850.0, 2880.0, 2845.0, 2866.0, 2865.5, 2860.0, 5_500_000, 157500.0, 120_000, 2_000_000, "150")
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            validate(parse_bhavdata(pd.read_csv(pd.io.common.StringIO(_csv_text(rows=rows)))), date(2022, 8, 8))


class TestDeliveryQuarantineNotWholeFileRejection:
    """
    The 2024-02-19 fix: DATA_NOTES.md §11 — "one row with DELIV_QTY > VOLUME rejected the whole ~700-stock
    file; fix by rejecting the row, not the file". A single impossible row's delivery fields become NaN
    (missing, never guessed); its OHLCV stays intact; the day is no longer lost.
    """

    def _rows_with_one_impossible(self, n_good=100):
        """A realistically sized file (real ones run 700-2,800 rows): one impossible row among many good
        ones, so the SHARE (not just the count) of impossible rows is realistically small."""
        rows = list(ROWS)
        for i in range(n_good):
            rows.append((f"SYM{i:04d}", " EQ", 100.0, 101.0, 102.0, 99.0, 100.5, 100.5, 100.5, 10_000, 10.0, 100, 5_000, "50.00"))
        rows.append(("WTICAB", " ST", 147.0, 195.0, 204.75, 195.0, 204.75, 204.75, 199.92, 2_561_000, 51198.8, 1_875, 2_566_000, "99.92"))
        return rows

    def test_a_single_impossible_row_is_quarantined_not_the_whole_file(self):
        rows = self._rows_with_one_impossible()
        raw = pd.read_csv(pd.io.common.StringIO(_csv_text(rows=rows)))
        out = validate(parse_bhavdata(raw), date(2022, 8, 8))
        assert len(out) == len(rows), "the other rows must survive"
        wticab = out[out["SYMBOL"] == "WTICAB"].iloc[0]
        assert pd.isna(wticab["DELIV_QTY"]) and pd.isna(wticab["DELIV_PER"]), "quarantined: missing, not guessed"
        assert wticab["CLOSE"] == pytest.approx(204.75), "OHLCV is untouched -- only the delivery fields are suspect"

    def test_the_good_rows_are_completely_unaffected(self):
        raw = pd.read_csv(pd.io.common.StringIO(_csv_text(rows=self._rows_with_one_impossible())))
        out = validate(parse_bhavdata(raw), date(2022, 8, 8))
        reliance = out[out["SYMBOL"] == "RELIANCE"].iloc[0]
        assert reliance["DELIV_PER"] == pytest.approx(36.36)

    def test_a_file_where_every_row_is_impossible_still_fails_loudly(self):
        """Mutation check: the quarantine must not silently swallow a genuine parser/format regression. If
        DELIV_QTY and VOLUME were parsed from the wrong columns (a real historical-class bug), most or all
        rows would look impossible -- that must still raise, not quarantine everything into an all-missing,
        silently-accepted file."""
        rows = [(sym, series, prev, o, h, l, last, c, vwap, vol, turn, n, vol + 1000, dp)
                for sym, series, prev, o, h, l, last, c, vwap, vol, turn, n, dq, dp in ROWS]
        raw = pd.read_csv(pd.io.common.StringIO(_csv_text(rows=rows)))
        with pytest.raises(ValueError, match="DELIV_QTY > VOLUME"):
            validate(parse_bhavdata(raw), date(2022, 8, 8))

    def test_the_threshold_is_a_share_of_rows_not_a_fixed_count(self):
        """The same absolute count (2 bad rows) that would be fine buried in a 103-row file must still fail
        when it is a large SHARE of a small file (2 of 9, 22%) -- the check is proportional, not a flat count."""
        rows = self._rows_with_one_impossible(n_good=5)
        sym, series, prev, o, h, l, last, c, vwap, vol, turn, n, dq, dp = rows[0]
        rows[0] = (sym, series, prev, o, h, l, last, c, vwap, vol, turn, n, vol + 1000, dp)
        raw = pd.read_csv(pd.io.common.StringIO(_csv_text(rows=rows)))
        with pytest.raises(ValueError, match="DELIV_QTY > VOLUME"):
            validate(parse_bhavdata(raw), date(2022, 8, 8))


class TestXlsxDisguisedAsCsv:
    """The 2022-08-08 fix: this one day's response is a real XLSX file (ZIP container, OOXML parts) served at
    the plain .csv URL. read_bhavdata_content() must detect this and parse it, not hand raw zip bytes to the
    CSV tokenizer."""

    def test_the_real_2022_08_08_fixture_is_a_zip_not_text(self):
        content = (FIXTURE_DIR / "sec_bhavdata_full_08082022.xlsx").read_bytes()
        assert content[:4] == b"PK\x03\x04", "sanity: this fixture is the real captured response"

    def test_read_bhavdata_content_parses_the_real_xlsx_fixture(self):
        content = (FIXTURE_DIR / "sec_bhavdata_full_08082022.xlsx").read_bytes()
        raw = read_bhavdata_content(content)
        assert len(raw) == 2255
        parsed = parse_bhavdata(raw)
        out = validate(parsed, date(2022, 8, 8))
        assert len(out) == 2255
        row = out[out["SYMBOL"] == "20MICRONS"].iloc[0]
        assert row["CLOSE"] == pytest.approx(111.15)
        assert row["DELIV_PER"] == pytest.approx(33.89)

    def test_read_bhavdata_content_still_parses_a_plain_csv(self):
        """The fallback for one bad day must not break every ordinary day."""
        raw = read_bhavdata_content(_csv_text().encode("utf-8"))
        assert len(raw) == 2
        assert list(raw.columns.str.strip()) == ["SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE",
                                                   "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE",
                                                   "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS", "NO_OF_TRADES",
                                                   "DELIV_QTY", "DELIV_PER"]


class TestWeekendsCLI:
    """Weekend equity/delivery sessions exist in the F&O archive on the same dates and must be probed here too
    (DATA_NOTES.md §10: 'Whether NSE serves the weekend equity files was not probed')."""

    def test_weekends_flag_selects_the_weekend_day_selector(self, monkeypatch):
        import sys
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_bhavdata as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_bhavdata.py", "--weekends", "--days", "5"])
        mod.main()
        assert captured["days"] == "weekends"

    def test_default_is_weekdays(self, monkeypatch):
        import sys
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_bhavdata as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_bhavdata.py", "--days", "5"])
        mod.main()
        assert captured["days"] == "weekdays"


def test_first_available_is_the_measured_date():
    assert FIRST_AVAILABLE == date(2019, 9, 30)
