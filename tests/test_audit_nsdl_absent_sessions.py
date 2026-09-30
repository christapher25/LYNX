"""
tests/test_audit_nsdl_absent_sessions.py

data/audit_nsdl_absent_sessions.py answers one question exactly: for every NSE session with no NSDL figure, does NSDL's
own page for that month list the date? If not, NSDL did not report it (custodians closed) and our data are complete. If it
does, our data lost a day. The date list is read straight from the page's date cells, independently of the table parser.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.audit_nsdl_absent_sessions import ABSENT, LOST, classify, page_dates
from data.ingest_nsdl_fpi import parse_cash_table

FIXTURES = Path(__file__).parent / "fixtures" / "nsdl_fpi"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestPageDates:
    @pytest.mark.parametrize("name", ["1999_01.html", "2003_06.html", "2008_06.html", "2011_06.html", "2015_06.html", "2026_09.html"])
    def test_agrees_with_the_table_parser_on_every_era(self, name):
        page = _fixture(name)
        assert page_dates(page) == {d.date() for d in parse_cash_table(page)["DATE"]}

    @pytest.mark.parametrize("name", ["full_2025_04.html", "full_2009_06.html"])
    def test_agrees_with_the_table_parser_on_full_real_pages(self, name):
        page = _fixture(name)
        assert page_dates(page) == {d.date() for d in parse_cash_table(page)["DATE"]}

    def test_a_date_that_only_the_derivatives_table_lists_is_not_a_cash_row(self):
        """
        NSDL's page carries a second table (FPI F&O activity) with its own date cells. 1-Apr-2025 is in it, because F&O
        traded that day, but the cash table has no row for it (custodians were closed). Reading dates from the whole page
        turned 127 legitimate 'NSDL did not report' days into false 'lost data' verdicts.
        """
        page = _fixture("full_2025_04.html")
        assert "01-Apr-2025" in page
        assert date(2025, 4, 1) not in page_dates(page)

    HEADER = "<tr><th>Reporting Date</th><th>Debt/Equity</th><th>Gross Purchases</th></tr>"

    def test_the_query_date_in_the_form_is_not_mistaken_for_a_row(self):
        page = f'<input name="txtDate" value="31-Jan-2026"><table>{self.HEADER}<tr><td>05-Jan-2026</td></tr></table>'
        assert page_dates(page) == {date(2026, 1, 5)}

    def test_a_date_cell_wrapped_in_markup_is_still_found(self):
        assert page_dates(f"<table>{self.HEADER}<tr><td rowspan=3><span> 06-Jan-2026 </span></td></tr></table>") == {date(2026, 1, 6)}

    def test_a_nested_derivatives_table_after_a_second_header_is_not_read(self):
        """Malformed older pages nest the derivatives table inside the cash table, starting at a second header."""
        page = (f"<table>{self.HEADER}<tr><td>05-Jan-2026</td></tr>"
                f"<table><tr><th>Reporting Date</th><th>Turnover</th></tr><tr><td>06-Jan-2026</td></tr></table></table>")
        assert page_dates(page) == {date(2026, 1, 5)}

    def test_a_page_with_no_cash_table_lists_nothing_so_the_positive_control_aborts_instead_of_calling_everything_absent(self):
        rejected = "<html><body>Request Rejected. The requested URL was rejected.</body></html>"
        assert page_dates(rejected) == set()
        with pytest.raises(ValueError, match="positive control"):
            classify([date(2025, 4, 1)], page_dates(rejected), {date(2025, 4, 2)})


class TestClassify:
    def test_a_session_the_page_does_not_list_is_confirmed_absent(self):
        rows = classify(sessions_without_figure=[date(2025, 4, 1)], page={date(2025, 4, 2)}, have={date(2025, 4, 2)})
        assert rows == [(date(2025, 4, 1), ABSENT)]

    def test_a_session_the_page_lists_but_our_data_lack_is_a_loss(self):
        rows = classify(sessions_without_figure=[date(2025, 4, 1)], page={date(2025, 4, 1), date(2025, 4, 2)}, have={date(2025, 4, 2)})
        assert rows == [(date(2025, 4, 1), LOST)]

    def test_positive_control_a_day_we_have_but_the_page_does_not_list_aborts_the_audit(self):
        """If our own days are not on the page the page was misread, and 'absent' would prove nothing."""
        with pytest.raises(ValueError, match="positive control"):
            classify(sessions_without_figure=[date(2025, 4, 1)], page={date(2025, 4, 3)}, have={date(2025, 4, 2), date(2025, 4, 3)})
