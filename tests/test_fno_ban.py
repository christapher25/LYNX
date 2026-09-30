"""
tests/test_fno_ban.py

Tests for data/ingest_fno_ban.py — Stage 10, F&O ingestion (securities in the F&O ban period).

The ban list is what Stage 11's gates will read to drop names before a signal is published, so the
silent failures that matter are: an empty list ("NIL" — nobody banned) being indistinguishable from
a day with no file, a symbol dropped because its line did not parse, and a file dated for one day
being filed under another.

Header shapes observed in the archive (2008-2026):
    "Securities in Ban For Trade Date 11-JUN-2020: "  then  "1,BHEL" ...
    "Securities in Ban For Trade Date 03-JUN-2024: NIL"
    "Sr. No.,Securities in Ban For Trade Date 01-Jan-2008"  then  "1,ADLABSFILM" ...
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.ingest_fno_ban import (
    FIRST_AVAILABLE,
    OUTPUT_COLS,
    build_url,
    parse_ban_list,
    validate,
)

DAY = date(2020, 6, 11)
LIST_2020 = "Securities in Ban For Trade Date 11-JUN-2020: \n1,BHEL\n2,IDEA\n3,JINDALSTEL\n"


class TestParsing:
    def test_symbols_are_read_one_row_each(self):
        out = parse_ban_list(LIST_2020, DAY)
        assert list(out.columns) == OUTPUT_COLS
        assert out["SYMBOL"].tolist() == ["BHEL", "IDEA", "JINDALSTEL"]
        assert (out["DATE"] == pd.Timestamp("2020-06-11")).all()

    def test_nil_day_is_recorded_as_a_published_empty_list_not_dropped(self):
        """'NIL' is information (the list was published and nobody was banned); a 404 is a different fact."""
        out = parse_ban_list("Securities in Ban For Trade Date 03-JUN-2024: NIL", date(2024, 6, 3))
        assert len(out) == 1
        assert out.loc[0, "DATE"] == pd.Timestamp("2024-06-03")
        assert pd.isna(out.loc[0, "SYMBOL"])

    def test_2008_era_header_with_serial_number_column(self):
        text = "Sr. No.,Securities in Ban For Trade Date 01-Jan-2008\n1,ADLABSFILM\n2,ARVINDMILL\n"
        assert parse_ban_list(text, date(2008, 1, 1))["SYMBOL"].tolist() == ["ADLABSFILM", "ARVINDMILL"]

    def test_windows_line_endings_bom_and_padding_are_tolerated(self):
        text = "﻿Securities in Ban For Trade Date 11-JUN-2020:\r\n 1 , BHEL \r\n2,IDEA\r\n\r\n"
        assert parse_ban_list(text, DAY)["SYMBOL"].tolist() == ["BHEL", "IDEA"]

    def test_symbols_with_punctuation_survive(self):
        text = "Securities in Ban For Trade Date 11-JUN-2020:\n1,M&M\n2,BAJAJ-AUTO\n3,NIFTY_X\n"
        assert parse_ban_list(text, DAY)["SYMBOL"].tolist() == ["M&M", "BAJAJ-AUTO", "NIFTY_X"]

    def test_nil_marker_is_case_and_spacing_insensitive(self):
        assert pd.isna(parse_ban_list("Securities in Ban For Trade Date 11-JUN-2020:   nil  ", DAY).loc[0, "SYMBOL"])


class TestNothingIsSilentlyDropped:
    def test_a_line_that_does_not_parse_raises_instead_of_vanishing(self):
        text = "Securities in Ban For Trade Date 11-JUN-2020:\n1,BHEL\nthis is not a symbol row\n3,IDEA\n"
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list(text, DAY)

    def test_a_row_with_no_symbol_raises(self):
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list("Securities in Ban For Trade Date 11-JUN-2020:\n1,\n", DAY)

    def test_header_dated_for_another_day_raises(self):
        with pytest.raises(ValueError, match="requested"):
            parse_ban_list(LIST_2020, date(2020, 6, 12))

    def test_missing_header_raises(self):
        with pytest.raises(ValueError, match="header"):
            parse_ban_list("1,BHEL\n2,IDEA\n", DAY)

    def test_empty_file_raises(self):
        with pytest.raises(ValueError, match="header"):
            parse_ban_list("   \n", DAY)

    def test_the_same_symbol_twice_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_ban_list("Securities in Ban For Trade Date 11-JUN-2020:\n1,BHEL\n2,BHEL\n", DAY)

    def test_nil_together_with_symbols_raises(self):
        with pytest.raises(ValueError, match="NIL"):
            parse_ban_list("Securities in Ban For Trade Date 11-JUN-2020: NIL\n1,BHEL\n", DAY)


class TestValidation:
    def test_parsed_output_passes(self):
        assert len(validate(parse_ban_list(LIST_2020, DAY), DAY)) == 3

    def test_file_dated_for_another_day_is_rejected(self):
        with pytest.raises(ValueError, match="requested"):
            validate(parse_ban_list(LIST_2020, DAY), date(2020, 6, 12))

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError, match="zero rows"):
            validate(pd.DataFrame(columns=OUTPUT_COLS), DAY)


class TestUrl:
    def test_path_is_the_archives_sec_ban_location_not_content_fo(self):
        """Probed 2026-09-24: /content/fo/fo_secban_* is a 404 for every date; /archives/fo/sec_ban/ serves them."""
        assert build_url(date(2020, 6, 11)).endswith("/archives/fo/sec_ban/fo_secban_11062020.csv")

    def test_first_available_is_the_measured_boundary(self):
        """Probed 2026-09-24: 2007-12-31 -> 404, 2008-01-01 -> 200."""
        assert FIRST_AVAILABLE == date(2008, 1, 1)


# --- fetch_day, exercised with a fake session (no network) ------------------------------------------

import requests

from data.ingest_fno_ban import fetch_day


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class _FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, timeout=None):
        return self.response


class TestFetchDay:
    def test_a_50_byte_nil_file_is_ingested_as_a_published_empty_list(self):
        """These files are tiny by nature; a size cutoff would erase every 'nobody banned' day."""
        body = b"Securities in Ban For Trade Date 03-JUN-2024: NIL"
        df, era, nbytes = fetch_day(_FakeSession(_FakeResponse(200, body)), date(2024, 6, 3))
        assert nbytes == len(body) and len(df) == 1 and df["SYMBOL"].isna().all()

    def test_404_means_no_file_for_that_date(self):
        assert fetch_day(_FakeSession(_FakeResponse(404, b"<html/>")), date(2020, 1, 1)) == (None, "sec_ban", 0)

    def test_a_html_page_served_with_200_fails_the_day(self):
        with pytest.raises(ValueError, match="header"):
            fetch_day(_FakeSession(_FakeResponse(200, b"<html>Access Denied</html>")), date(2020, 6, 11))

    def test_utf8_bom_from_the_server_is_handled(self):
        body = "﻿Securities in Ban For Trade Date 11-JUN-2020:\n1,BHEL\n".encode("utf-8")
        df, _, _ = fetch_day(_FakeSession(_FakeResponse(200, body)), date(2020, 6, 11))
        assert df["SYMBOL"].tolist() == ["BHEL"]

    def test_a_server_error_propagates(self):
        with pytest.raises(requests.HTTPError):
            fetch_day(_FakeSession(_FakeResponse(500)), date(2020, 6, 11))


class TestHeaderVariantsFound2008To2009:
    """
    The 2008-2009 files (found by the full backfill, absent from the first samples) write the header date with
    a full month name and use several separators. Each shape below is a real file.
    """

    def test_full_month_name_with_space_before_the_colon_and_Nil(self):
        out = parse_ban_list("Securities in Ban For Trade Date 02-June-2008 : Nil", date(2008, 6, 2))
        assert len(out) == 1 and pd.isna(out.loc[0, "SYMBOL"]) and out.loc[0, "DATE"] == pd.Timestamp("2008-06-02")

    def test_comma_after_the_date_then_symbols_on_following_lines(self):
        text = "Securities in Ban For Trade Date 24-September-2008,\n1,AKRUTI\n"
        assert parse_ban_list(text, date(2008, 9, 24))["SYMBOL"].tolist() == ["AKRUTI"]

    def test_comma_then_Nil_on_the_same_line(self):
        out = parse_ban_list("Securities in Ban For Trade Date 12-Mar-2008,Nil", date(2008, 3, 12))
        assert len(out) == 1 and pd.isna(out.loc[0, "SYMBOL"])

    def test_full_month_name_with_colon_and_NIL(self):
        assert pd.isna(parse_ban_list("Securities in Ban For Trade Date 07-October-2008: NIL", date(2008, 10, 7)).loc[0, "SYMBOL"])

    @pytest.mark.parametrize("month,number", [("January", 1), ("February", 2), ("March", 3), ("April", 4), ("May", 5), ("June", 6),
                                              ("July", 7), ("August", 8), ("September", 9), ("October", 10), ("November", 11),
                                              ("December", 12), ("Sept", 9)])
    def test_every_full_month_name_maps_to_the_right_month(self, month, number):
        out = parse_ban_list(f"Securities in Ban For Trade Date 15-{month}-2009: NIL", date(2009, number, 15))
        assert out.loc[0, "DATE"] == pd.Timestamp(2009, number, 15)

    def test_a_full_month_name_does_not_let_a_wrong_date_through(self):
        with pytest.raises(ValueError, match="requested"):
            parse_ban_list("Securities in Ban For Trade Date 24-September-2008,\n1,AKRUTI\n", date(2008, 9, 25))

    def test_a_month_that_is_not_a_month_is_rejected(self):
        with pytest.raises(ValueError, match="month"):
            parse_ban_list("Securities in Ban For Trade Date 24-Foo-2008: NIL", date(2008, 9, 24))

    def test_other_text_after_the_header_is_still_rejected(self):
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list("Securities in Ban For Trade Date 24-September-2008: something else", date(2008, 9, 24))


class TestTrailingCommaOnSymbolLines:
    """From 2012-09 some files end each symbol line with a comma ('1,HDIL,'); found by the full backfill."""

    HEADER = "Securities in Ban For Trade Date 27-Sep-2012:\n"

    def test_symbol_lines_with_a_trailing_comma_parse(self):
        assert parse_ban_list(self.HEADER + "1,HDIL,\n2,IDEA,\n", date(2012, 9, 27))["SYMBOL"].tolist() == ["HDIL", "IDEA"]

    def test_mixed_shapes_within_one_file(self):
        assert parse_ban_list(self.HEADER + "1,HDIL,\n2,IDEA\n", date(2012, 9, 27))["SYMBOL"].tolist() == ["HDIL", "IDEA"]

    def test_an_empty_symbol_is_still_rejected(self):
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list(self.HEADER + "1,,\n", date(2012, 9, 27))

    def test_extra_fields_after_the_symbol_are_still_rejected(self):
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list(self.HEADER + "1,HDIL,extra\n", date(2012, 9, 27))

    def test_the_trailing_comma_does_not_leak_into_the_symbol(self):
        assert parse_ban_list(self.HEADER + "1,HDIL,\n", date(2012, 9, 27)).loc[0, "SYMBOL"] == "HDIL"


class TestCommaPaddingIsOnlyEverEmptyFields:
    """
    Raw bytes checked 2026-09-24 (2012-08-29, 09-21, 09-27, 2015-12-23): the extra commas are fixed-width CSV padding.
    Every field after the symbol is empty ('1,INDIACEM,,,') and the header carries the same padding
    ('...23-DEC-2015:,,,,'). Padding is accepted only when EVERY field after the symbol is empty.
    """

    def test_real_2015_file_shape(self):
        text = ("Securities in Ban For Trade Date 23-DEC-2015:,,,,\n1,INDIACEM,,,\n2,JINDALSTEL,,,\n"
                "3,JISLJALEQS,,,\n4,JPASSOCIAT,,,\n5,UNITECH,,,\n")
        assert parse_ban_list(text, date(2015, 12, 23))["SYMBOL"].tolist() == [
            "INDIACEM", "JINDALSTEL", "JISLJALEQS", "JPASSOCIAT", "UNITECH"]

    def test_real_2012_file_shape_with_double_comma_header(self):
        text = "Securities in Ban For Trade Date 29-August-2012,,\n1,COREEDUTEC,\n2,IFCI,\n3,MTNL,\n"
        assert parse_ban_list(text, date(2012, 8, 29))["SYMBOL"].tolist() == ["COREEDUTEC", "IFCI", "MTNL"]

    def test_any_number_of_empty_padding_fields_is_accepted(self):
        for pad in (",", ",,", ",,,", ", , ,"):
            text = f"Securities in Ban For Trade Date 23-DEC-2015:\n1,INDIACEM{pad}\n"
            assert parse_ban_list(text, date(2015, 12, 23)).loc[0, "SYMBOL"] == "INDIACEM", pad

    @pytest.mark.parametrize("line", [
        "1,HDIL,x",            # data after the symbol
        "1,HDIL,,x",           # data after padding
        "1,HDIL,,,x,,",        # data hidden inside the padding
        "1,,",                 # no symbol at all
        "1,,,",
        "1,HDIL x,,",          # a second token inside the symbol field
    ])
    def test_padding_never_excuses_data_or_a_missing_symbol(self, line):
        with pytest.raises(ValueError, match="unparseable"):
            parse_ban_list("Securities in Ban For Trade Date 23-DEC-2015:\n" + line + "\n", date(2015, 12, 23))

    def test_a_header_dated_for_another_day_is_still_rejected_despite_padding(self):
        """The archive serves other days' lists under some URLs (2014-02-19 returned the 2014-03-19 list); never loosen this."""
        with pytest.raises(ValueError, match="requested"):
            parse_ban_list("Securities in Ban For Trade Date 19-MAR-2014:,,,,\n1,HDIL,,,\n", date(2014, 2, 19))
