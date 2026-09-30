"""
tests/test_ingest_sec_list.py

NSE's daily security list (sec_list_DDMMYYYY.csv) carries each security's price band. It has no date inside the file, and
the Last-Modified header is the only evidence of when it was produced, so that is kept with every row (it is also the
publication clock the Stage 10 circuits block keys on).
"""

from datetime import date

import pandas as pd
import pytest

from data.ingest_sec_list import OUTPUT_COLS, parse_sec_list

DAY = date(2026, 9, 23)
LAST_MODIFIED = "Wed, 23 Sep 2026 12:20:39 GMT"
HEADER = "Symbol,Series,Security Name,Band,Remarks"


def _text(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


class TestParse:
    def test_real_rows_parse_with_band_percent_and_no_band_flag(self):
        text = _text('21STCENMGM,EQ,21ST CENTURY MANAGEMENT SERVICES LIMITED,2,"-"',
                     'HDFCBANK,EQ,HDFC BANK LIMITED,No Band,"-"',
                     'ABAN,BE,ABAN OFFSHORE LIMITED,20,"-"')
        out = parse_sec_list(text, DAY, LAST_MODIFIED)
        assert list(out.columns) == OUTPUT_COLS
        by = out.set_index("SYMBOL")
        assert by.loc["21STCENMGM", "BAND_PCT"] == 2.0 and not by.loc["21STCENMGM", "IS_NO_BAND"]
        assert pd.isna(by.loc["HDFCBANK", "BAND_PCT"]) and by.loc["HDFCBANK", "IS_NO_BAND"] and by.loc["HDFCBANK", "BAND_TEXT"] == "No Band"
        assert by.loc["ABAN", "BAND_PCT"] == 20.0

    def test_every_band_the_exchange_uses_is_accepted(self):
        text = _text(*[f'S{b},EQ,NAME {b},{b},"-"' for b in (2, 5, 10, 20, 40)], 'NB,EQ,NO BAND CO,No Band,"-"')
        out = parse_sec_list(text, DAY, LAST_MODIFIED)
        assert sorted(out["BAND_PCT"].dropna().tolist()) == [2.0, 5.0, 10.0, 20.0, 40.0]

    def test_a_name_containing_a_comma_inside_quotes_does_not_shift_the_columns(self):
        out = parse_sec_list(_text('XYZ,EQ,"XYZ INDUSTRIES, LIMITED",5,"-"'), DAY, LAST_MODIFIED)
        assert out.iloc[0]["SYMBOL"] == "XYZ" and out.iloc[0]["BAND_PCT"] == 5.0

    def test_the_file_date_is_the_requested_date_and_the_publication_stamp_is_kept_in_utc(self):
        out = parse_sec_list(_text('A,EQ,A LTD,5,"-"'), DAY, LAST_MODIFIED)
        assert out["DATE"].iloc[0] == pd.Timestamp(DAY)
        assert out["LAST_MODIFIED_UTC"].iloc[0] == "2026-09-23T12:20:39+00:00"

    def test_a_missing_last_modified_header_is_stored_as_null_never_invented(self):
        out = parse_sec_list(_text('A,EQ,A LTD,5,"-"'), DAY, None)
        assert out["LAST_MODIFIED_UTC"].isna().all()

    def test_the_same_symbol_in_two_series_is_two_rows(self):
        out = parse_sec_list(_text('A,EQ,A LTD,5,"-"', 'A,BE,A LTD,5,"-"'), DAY, LAST_MODIFIED)
        assert len(out) == 2

    def test_a_lowercase_no_band_is_the_same_token(self):
        """2021-12-30 wrote 'No band' for nine securities and 'No Band' for the rest."""
        out = parse_sec_list(_text('A,EQ,A LTD,No band,"-"', 'B,EQ,B LTD,No Band,"-"'), DAY, LAST_MODIFIED)
        assert out["IS_NO_BAND"].all() and out["BAND_PCT"].isna().all() and set(out["BAND_TEXT"]) == {"No Band"}

    def test_a_repeated_row_with_the_same_band_collapses_to_one(self):
        """2020-04-09 DHARSUGAR (remarks differ: '-' vs 'GSM Stage - III') and 2021-03-09 ROML (exact repeat)."""
        out = parse_sec_list(_text('D,BE,D LTD,5,"-"', 'D,BE,D LTD,5,GSM Stage - III', 'R,BE,R LTD,5,"-"', 'R,BE,R LTD,5,"-"'), DAY, LAST_MODIFIED)
        assert out[["SYMBOL", "SERIES", "BAND_PCT"]].values.tolist() == [["D", "BE", 5.0], ["R", "BE", 5.0]]

    def test_the_2020_02_05_extended_layout_with_a_copy_of_the_band_and_a_check_column(self):
        # measured over the file's 1,903 rows: the copy always equals the band; Check is 0, or the spreadsheet error #VALUE! for No Band
        text = "Symbol,Series,Security Name,Band,Remarks,,Check\n" + "\n".join(['A,EQ,A LTD,2,-,2,0', 'B,EQ,B LTD,No Band,-,No Band,#VALUE!'])
        out = parse_sec_list(text, DAY, LAST_MODIFIED)
        assert out[["SYMBOL", "BAND_TEXT"]].values.tolist() == [["A", "2"], ["B", "No Band"]]


class TestRejections:
    """The parser rejects rather than skips: a band read wrongly would be applied to every price move that day."""

    def test_an_unknown_band_value_is_rejected(self):
        with pytest.raises(ValueError, match="band"):
            parse_sec_list(_text('A,EQ,A LTD,15,"-"'), DAY, LAST_MODIFIED)

    def test_a_blank_band_is_rejected_not_treated_as_no_band(self):
        with pytest.raises(ValueError, match="band"):
            parse_sec_list(_text('A,EQ,A LTD,,"-"'), DAY, LAST_MODIFIED)

    def test_a_changed_header_is_rejected(self):
        with pytest.raises(ValueError, match="header"):
            parse_sec_list("Symbol,Series,Security Name,PriceBand,Remarks\nA,EQ,A LTD,5,-\n", DAY, LAST_MODIFIED)

    def test_duplicate_symbol_series_with_conflicting_bands_is_rejected(self):
        with pytest.raises(ValueError, match="conflicting"):
            parse_sec_list(_text('A,EQ,A LTD,5,"-"', 'A,EQ,A LTD,10,"-"'), DAY, LAST_MODIFIED)

    def test_the_extended_layout_is_checked_not_trusted(self):
        header = "Symbol,Series,Security Name,Band,Remarks,,Check\n"
        with pytest.raises(ValueError, match="extended"):
            parse_sec_list(header + 'A,EQ,A LTD,2,-,5,0', DAY, LAST_MODIFIED)          # the copied band disagrees
        with pytest.raises(ValueError, match="extended"):
            parse_sec_list(header + 'A,EQ,A LTD,2,-,2,1', DAY, LAST_MODIFIED)          # a check value other than 0
        with pytest.raises(ValueError, match="extended"):
            parse_sec_list(header + 'A,EQ,A LTD,No Band,-,No Band,0', DAY, LAST_MODIFIED)   # No Band rows carry #VALUE!, not 0
        with pytest.raises(ValueError, match="extended"):
            parse_sec_list(header + 'A,EQ,A LTD,2,-', DAY, LAST_MODIFIED)              # a short row under the long header

    def test_an_empty_or_header_only_file_is_rejected(self):
        for text in ("", HEADER + "\n"):
            with pytest.raises(ValueError):
                parse_sec_list(text, DAY, LAST_MODIFIED)

    def test_a_naive_or_unparseable_last_modified_is_rejected(self):
        with pytest.raises(ValueError, match="Last-Modified"):
            parse_sec_list(_text('A,EQ,A LTD,5,"-"'), DAY, "sometime yesterday")

    def test_a_row_with_the_wrong_number_of_fields_is_rejected(self):
        with pytest.raises(ValueError):
            parse_sec_list(_text('A,EQ,A LTD,5'), DAY, LAST_MODIFIED)
