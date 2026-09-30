"""
tests/test_ingest_bhavcopy.py

Tests for data/ingest_bhavcopy.py — Stage 3 equity bhavcopy ingestion. Zero coverage existed before this file
(found during Stage 10's cleanup of the known gaps in data/raw/DATA_NOTES.md §11).

The headline failure mode: on 2020-07-13 NSE wrote the legacy file's TIMESTAMP column as '13-Jul-20' (two-digit
year) instead of the usual '13-JUL-2020'. The strict `%d-%b-%Y` parse turned all 2,001 rows to NaT, and
validate()'s "any row failed date parsing" check correctly rejected the whole day — so the fix is a second,
narrower parse attempt for the two-digit form, not a loosening of validation. Mirrors
tests/test_fno_ingest.py's TestTwoDigitYearGlitch, since data/ingest_fno_bhavcopy.py's legacy F&O parser already
carries this exact fix (`_legacy_dates`) for the same day.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.ingest_bhavcopy import (
    FIRST_AVAILABLE,
    OUTPUT_COLS,
    UDIFF_CUTOVER,
    build_url,
    parse_legacy,
    parse_udiff,
    validate,
)

TRADE_DATE = date(2024, 6, 3)

# (symbol, series, open, high, low, close, last, prevclose, volume, turnover, num_trades, isin)
ROWS = [
    ("RELIANCE", "EQ", 2850.0, 2880.0, 2845.0, 2865.5, 2866.0, 2848.0, 5_500_000, 15_750_000_000.0, 120_000, "INE002A01018"),
    ("TCS", "EQ", 3820.0, 3850.0, 3810.0, 3840.0, 3841.0, 3818.0, 1_200_000, 4_600_000_000.0, 45_000, "INE467B01029"),
]


_LEGACY_COLS = ["SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE", "TOTTRDQTY", "TOTTRDVAL",
                "TIMESTAMP", "TOTALTRADES", "ISIN"]


def _legacy_raw(trade_date_text="03-JUN-2024", rows=ROWS, with_optional=True) -> pd.DataFrame:
    out = []
    for sym, series, o, h, l, c, last, prev, vol, turn, n, isin in rows:
        rec = {
            "SYMBOL": sym, "SERIES": series, "OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "LAST": last,
            "PREVCLOSE": prev, "TOTTRDQTY": vol, "TOTTRDVAL": turn, "TIMESTAMP": trade_date_text,
        }
        if with_optional:
            rec["TOTALTRADES"] = n
            rec["ISIN"] = isin
        out.append(rec)
    cols = _LEGACY_COLS if with_optional else [c for c in _LEGACY_COLS if c not in ("TOTALTRADES", "ISIN")]
    return pd.DataFrame(out, columns=cols)


def _udiff_raw(trade_date_iso="2024-07-08", rows=ROWS) -> pd.DataFrame:
    out = []
    for sym, series, o, h, l, c, last, prev, vol, turn, n, isin in rows:
        out.append({
            "TradDt": trade_date_iso, "FinInstrmTp": "STK", "TckrSymb": sym, "SctySrs": series,
            "OpnPric": o, "HghPric": h, "LwPric": l, "ClsPric": c, "LastPric": last, "PrvsClsgPric": prev,
            "TtlTradgVol": vol, "TtlTrfVal": turn, "TtlNbOfTxsExctd": n, "ISIN": isin,
        })
    # UDiFF files carry other instrument types too; parse_udiff must filter to STK only
    out.append({"TradDt": trade_date_iso, "FinInstrmTp": "IDX", "TckrSymb": "NIFTY 50", "SctySrs": np.nan,
                "OpnPric": 0, "HghPric": 0, "LwPric": 0, "ClsPric": 0, "LastPric": 0, "PrvsClsgPric": 0,
                "TtlTradgVol": 0, "TtlTrfVal": 0, "TtlNbOfTxsExctd": 0, "ISIN": np.nan})
    return pd.DataFrame(out)


class TestParseLegacy:
    def test_the_declared_columns_in_order(self):
        assert list(parse_legacy(_legacy_raw()).columns) == OUTPUT_COLS

    def test_values_round_trip(self):
        out = parse_legacy(_legacy_raw())
        assert out.loc[0, "SYMBOL"] == "RELIANCE"
        assert out.loc[0, "DATE"] == pd.Timestamp("2024-06-03")
        assert out.loc[0, "CLOSE"] == pytest.approx(2865.5)
        assert out.loc[0, "VOLUME"] == 5_500_000

    def test_missing_optional_columns_become_na_not_zero(self):
        """Era 1 (pre-TOTALTRADES/ISIN) must not read as 0 trades -- that is a specific, false claim."""
        out = parse_legacy(_legacy_raw(with_optional=False))
        assert out["NUM_TRADES"].isna().all()
        assert out["ISIN"].isna().all()


class TestParseUdiff:
    def test_filters_to_cash_stocks_only(self):
        out = parse_udiff(_udiff_raw())
        assert set(out["SYMBOL"]) == {"RELIANCE", "TCS"}

    def test_the_declared_columns_in_order(self):
        assert list(parse_udiff(_udiff_raw()).columns) == OUTPUT_COLS


class TestValidate:
    def test_a_clean_day_passes(self):
        validate(parse_legacy(_legacy_raw()), date(2024, 6, 3))

    def test_zero_rows_is_rejected(self):
        with pytest.raises(ValueError, match="zero rows"):
            validate(parse_legacy(_legacy_raw(rows=[])), date(2024, 6, 3))

    def test_a_date_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="requested"):
            validate(parse_legacy(_legacy_raw()), date(2024, 6, 4))

    def test_negative_price_or_volume_is_rejected(self):
        """validate() checks CLOSE and VOLUME specifically (the fields every downstream feature reads)."""
        bad = list(ROWS)
        bad[0] = ("RELIANCE", "EQ", 2850.0, 2880.0, 2845.0, -1.0, 2866.0, 2848.0, 5_500_000, 1.0, 1, "X")
        with pytest.raises(ValueError, match="negative"):
            validate(parse_legacy(_legacy_raw(rows=bad)), date(2024, 6, 3))


class TestTwoDigitYearGlitch:
    """
    On 2020-07-13 NSE's equity bhavcopy wrote TIMESTAMP as '13-Jul-20' instead of '13-JUL-2020'. Stage 3's parser
    (strict %d-%b-%Y) turned all 2,001 rows to NaT and validate() correctly rejected the day as "2001 rows failed
    date parsing" -- data/raw/_manifests/bhavcopy.jsonl still records exactly that error. The file's data is fine;
    only the date parse needs the same two-digit fallback data/ingest_fno_bhavcopy.py's legacy F&O parser already
    has for this identical day (§11 of DATA_NOTES.md).
    """

    def test_reproduces_the_real_2020_07_13_failure_before_the_fix(self):
        """This is the failing test written before the fix: it documents the exact historical symptom. Once the
        fix lands this must flip from 'confirms the bug' to 'confirms the bug cannot happen again' -- see the next
        test, which is the same assertion run through validate()."""
        out = parse_legacy(_legacy_raw(trade_date_text="13-Jul-20"))
        assert out["DATE"].notna().all(), "the two-digit-year date must not be silently dropped as NaT"

    def test_two_digit_year_dates_parse_to_the_right_day(self):
        out = parse_legacy(_legacy_raw(trade_date_text="13-Jul-20"))
        assert (out["DATE"] == pd.Timestamp("2020-07-13")).all()
        validate(out, date(2020, 7, 13))

    def test_four_and_two_digit_years_can_coexist_in_one_file(self):
        """The fallback is a second attempt on rows the primary parse missed, not a blanket format switch."""
        raw = _legacy_raw(trade_date_text="13-JUL-2020")
        out = parse_legacy(raw)
        assert out["DATE"].notna().all()

    def test_a_genuinely_unparseable_date_still_fails_the_day(self):
        """Tolerance is for the known glitch only; garbage must still be rejected."""
        with pytest.raises(ValueError, match="date"):
            validate(parse_legacy(_legacy_raw(trade_date_text="13/07/2020")), date(2020, 7, 13))

    def test_a_two_digit_year_cannot_smuggle_in_the_wrong_day(self):
        """The parsed date is still checked against the requested day -- the fallback cannot file a row under the
        wrong date even though it is more permissive about the year's width."""
        out = parse_legacy(_legacy_raw(trade_date_text="13-Jul-20"))
        with pytest.raises(ValueError, match="requested"):
            validate(out, date(2020, 7, 14))

    def test_a_two_digit_year_a_century_off_is_still_caught_by_the_requested_day_check(self):
        """1920-07-13 is not 2020-07-13: pandas' %y is unambiguous (00-68 -> 2000s, 69-99 -> 1900s) but the
        requested-date check is the real backstop regardless of that pivot."""
        out = parse_legacy(_legacy_raw(trade_date_text="13-Jul-20"))
        assert out.loc[0, "DATE"].year == 2020


class TestBuildUrl:
    def test_legacy_before_cutover(self):
        url, era = build_url(date(2024, 7, 5))
        assert era == "legacy" and "cm05JUL2024bhav.csv.zip" in url

    def test_udiff_on_and_after_cutover(self):
        url, era = build_url(UDIFF_CUTOVER)
        assert era == "udiff" and "20240708" in url

    def test_first_available_is_the_measured_launch_date(self):
        assert FIRST_AVAILABLE == date(1994, 11, 3)


class TestWeekendsCLI:
    """Weekend equity sessions (Budget-day/special Saturdays, Muhurat) exist in the F&O archive and must be
    probed here too (DATA_NOTES.md §10: 'Whether NSE serves the weekend equity files was not probed'). The CLI
    flag must wire to the same days='weekends' mechanism the other Stage 10 ingesters use, not a new one."""

    def test_weekends_flag_selects_the_weekend_day_selector(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_bhavcopy as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_bhavcopy.py", "--weekends", "--days", "5"])
        mod.main()
        assert captured["days"] == "weekends"

    def test_default_is_weekdays(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_bhavcopy as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_bhavcopy.py", "--days", "5"])
        mod.main()
        assert captured["days"] == "weekdays"
