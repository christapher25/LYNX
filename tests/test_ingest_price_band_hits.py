"""
tests/test_ingest_price_band_hits.py

The PR zip's bhDDMMYY(YY).csv lists the securities that hit their price band during the day: H = upper circuit, L = lower.
Facts measured 2026-09-24 that these tests pin down:

  * starts 2010-01-04 (every earlier date -> 404);
  * the member is named bhDDMMYYYY.csv (2010-11, 2026) or bhDDMMYY.csv (2012-25): the date INSIDE the name is the only
    embedded date, so it is what proves a zip is the requested day's and not a stale re-serve;
  * three headers: SYMBOL,SR,... (2010), SYMBOL,SERIES,...,INDEX FLAG (2010-2025), SYMBOL,SERIES,SECURITY,HIGH/LOW (2026);
    INDEX FLAG was blank on all 3,345 rows sampled;
  * a stock that hit BOTH bands in a day appears twice (H and L): 1-61 such stocks per file, so the key is (symbol, series, hit).
"""

import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from data.ingest_price_band_hits import OUTPUT_COLS, parse_pr_zip

DAY = date(2020, 5, 12)
LAST_MODIFIED = "Tue, 12 May 2020 12:05:04 GMT"


def _zip(member: str | None, body: str | None, extra: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("readme.txt", "info")
        if member is not None:
            z.writestr(member, body if body is not None else "")
        for name, content in (extra or {}).items():
            z.writestr(name, content)
    return buffer.getvalue()


def _bh(*rows: str, header: str = "SYMBOL,SERIES,SECURITY,HIGH/LOW,INDEX FLAG") -> str:
    return "\n".join([header, *rows]) + "\n"


class TestParse:
    def test_upper_and_lower_hits_with_the_exchange_series(self):
        content = _zip("bh120520.csv", _bh("ALANKIT,EQ,ALANKIT LIMITED,H,", "BAGFILMS,BE,B.A.G FILMS AND MEDIA LTD,L,"))
        out = parse_pr_zip(content, DAY, LAST_MODIFIED)
        assert list(out.columns) == OUTPUT_COLS
        assert out[["SYMBOL", "SERIES", "HIT"]].values.tolist() == [["ALANKIT", "EQ", "H"], ["BAGFILMS", "BE", "L"]]

    def test_a_stock_that_hit_both_bands_appears_twice_and_is_not_a_duplicate(self):
        content = _zip("bh120520.csv", _bh("ALANKIT,EQ,ALANKIT LIMITED,H,", "ALANKIT,EQ,ALANKIT LIMITED,L,"))
        out = parse_pr_zip(content, DAY, LAST_MODIFIED)
        assert sorted(out["HIT"]) == ["H", "L"] and len(out) == 2

    @pytest.mark.parametrize("member", ["bh12052020.csv", "bh120520.csv", "BH120520.CSV"])
    def test_both_member_name_forms_and_case_are_accepted_when_the_date_matches(self, member):
        assert len(parse_pr_zip(_zip(member, _bh("A,EQ,A LTD,H,")), DAY, LAST_MODIFIED)) == 1

    def test_the_2010_header_with_SR_and_the_2026_header_without_INDEX_FLAG(self):
        old = parse_pr_zip(_zip("bh12052020.csv", _bh("A,BE,A LTD,H,", header="SYMBOL,SR,SECURITY,HIGH/LOW,INDEX FLAG")), DAY, LAST_MODIFIED)
        new = parse_pr_zip(_zip("bh12052020.csv", _bh("A,EQ,A LTD,L", header="SYMBOL,SERIES,SECURITY,HIGH/LOW")), DAY, LAST_MODIFIED)
        assert old.iloc[0]["SERIES"] == "BE" and new.iloc[0]["HIT"] == "L"

    def test_symbols_with_punctuation_survive(self):
        out = parse_pr_zip(_zip("bh120520.csv", _bh("M&M,EQ,MAHINDRA,H,", "BAJAJ-AUTO,EQ,BAJAJ AUTO,L,")), DAY, LAST_MODIFIED)
        assert out["SYMBOL"].tolist() == ["M&M", "BAJAJ-AUTO"]

    def test_date_and_publication_stamp(self):
        out = parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,A LTD,H,")), DAY, LAST_MODIFIED)
        assert out["DATE"].iloc[0] == pd.Timestamp(DAY) and out["LAST_MODIFIED_UTC"].iloc[0] == "2020-05-12T12:05:04+00:00"
        assert parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,A LTD,H,")), DAY, None)["LAST_MODIFIED_UTC"].isna().all()

    def test_a_published_file_with_no_hits_is_one_null_symbol_row_not_no_rows(self):
        """'Nobody hit a band' is information; it is not the same fact as 'no file for this date' (same rule as the ban list)."""
        out = parse_pr_zip(_zip("bh120520.csv", _bh()), DAY, LAST_MODIFIED)
        assert len(out) == 1 and out["SYMBOL"].isna().all() and out["DATE"].iloc[0] == pd.Timestamp(DAY)


class TestRejections:
    """Reject rather than skip: a hit dropped or misdated is an upper circuit the override gates would never see."""

    def test_a_zip_whose_member_is_another_days_file_is_a_stale_reserve(self):
        with pytest.raises(ValueError, match="dated"):
            parse_pr_zip(_zip("bh110520.csv", _bh("A,EQ,A LTD,H,")), DAY, LAST_MODIFIED)
        with pytest.raises(ValueError, match="dated"):
            parse_pr_zip(_zip("bh11052020.csv", _bh("A,EQ,A LTD,H,")), DAY, LAST_MODIFIED)

    def test_a_zip_without_a_bh_member_is_an_error_not_an_empty_day(self):
        with pytest.raises(ValueError, match="no bh"):
            parse_pr_zip(_zip(None, None, {"pr120520.csv": "x"}), DAY, LAST_MODIFIED)

    def test_two_bh_members_are_ambiguous(self):
        with pytest.raises(ValueError, match="more than one"):
            parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,A LTD,H,"), {"bh12052020.csv": _bh("A,EQ,A LTD,H,")}), DAY, LAST_MODIFIED)

    def test_an_unknown_hit_token_is_rejected(self):
        with pytest.raises(ValueError, match="HIGH/LOW"):
            parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,A LTD,X,")), DAY, LAST_MODIFIED)

    def test_the_index_flag_is_an_index_name_and_is_ignored_not_rejected(self):
        """2010-08-11/13/19: 'CNX Nifty Junior' for RAYMOND and WOCKPHARMA. It labels index membership; it is not a hit type."""
        out = parse_pr_zip(_zip("bh120520.csv", _bh("RAYMOND,EQ,RAYMOND LTD,H,CNX Nifty Junior")), DAY, LAST_MODIFIED)
        assert out[["SYMBOL", "HIT"]].values.tolist() == [["RAYMOND", "H"]]

    def test_trailing_empty_header_cells_from_the_fixed_width_export_are_ignored(self):
        """2010-09-13, 09-17, 11-16: 'SYMBOL,SR,SECURITY,HIGH/LOW,INDEX FLAG,,,,,'."""
        padded = _bh("A,BE,A LTD,H,,,,,,", header="SYMBOL,SR,SECURITY,HIGH/LOW,INDEX FLAG,,,,,")
        assert parse_pr_zip(_zip("bh120520.csv", padded), DAY, LAST_MODIFIED)["HIT"].tolist() == ["H"]
        no_flag = _bh("A,BE,A LTD,L,,,,,,", header="SYMBOL,SR,SECURITY,HIGH/LOW,,,,,,")
        assert parse_pr_zip(_zip("bh120520.csv", no_flag), DAY, LAST_MODIFIED)["HIT"].tolist() == ["L"]

    def test_the_INDEXFLAG_spelling_without_a_space_is_accepted(self):
        """2011-04-05."""
        text = _bh("A,BE,A LTD,H", header="SYMBOL,SR,SECURITY,HIGH/LOW,INDEXFLAG")
        assert len(parse_pr_zip(_zip("bh120520.csv", text), DAY, LAST_MODIFIED)) == 1

    def test_members_nested_in_a_folder_are_found_by_their_base_name(self):
        """2013-01-10: the zip holds 'nupr100113/bh100113.csv' next to 'nupr100113/Bc100113.csv'."""
        nested = _zip("nupr120520/bh120520.csv", _bh("A,EQ,A LTD,H,"))
        assert len(parse_pr_zip(nested, DAY, LAST_MODIFIED)) == 1

    def test_a_folder_entry_named_like_a_bh_file_is_not_a_member(self):
        with pytest.raises(ValueError, match="no bh"):
            parse_pr_zip(_zip("bh120520.csv/", None), DAY, LAST_MODIFIED)

    def test_the_same_symbol_series_and_direction_twice_is_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,A LTD,H,", "A,EQ,A LTD,H,")), DAY, LAST_MODIFIED)

    def test_an_unrecognised_header_is_rejected(self):
        with pytest.raises(ValueError, match="header"):
            parse_pr_zip(_zip("bh120520.csv", _bh("A,EQ,H", header="SYM,SER,HL")), DAY, LAST_MODIFIED)

    def test_content_that_is_not_a_zip_is_an_error(self):
        with pytest.raises(ValueError, match="zip"):
            parse_pr_zip(b"<html>Access Denied</html>", DAY, LAST_MODIFIED)
