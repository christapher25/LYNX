"""
tests/test_publication_monitor.py

Tests for the pure parts of data/publication_monitor.py: URL builders for each watched source and the parsers that
decide whether a source already shows a given day's data. The polling loop itself is exercised with a fake fetcher.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from data.publication import IST
from data.publication_monitor import (
    ARCHIVE_SOURCES,
    archive_url,
    dates_in_nse_fii_dii,
    dates_in_nsdl_page,
    poll_once,
)

D = date(2026, 9, 24)


class TestArchiveUrls:
    def test_every_watched_archive_source_builds_a_distinct_url_for_the_day(self):
        urls = {name: archive_url(name, D) for name in ARCHIVE_SOURCES}
        assert len(set(urls.values())) == len(urls)
        assert all(u.startswith("https://nsearchives.nseindia.com/") for u in urls.values())

    @pytest.mark.parametrize("name,expected", [
        ("nse_cm_bhavcopy_udiff", "/content/cm/BhavCopy_NSE_CM_0_0_0_20260924_F_0000.csv.zip"),
        ("nse_fo_bhavcopy_udiff", "/content/fo/BhavCopy_NSE_FO_0_0_0_20260924_F_0000.csv.zip"),
        ("nse_sec_bhavdata_full", "/products/content/sec_bhavdata_full_24092026.csv"),
        ("nse_pr_zip", "/archives/equities/bhavcopy/pr/PR240926.zip"),
        ("nse_sec_list", "/content/equities/sec_list_24092026.csv"),
        ("nse_cm_security", "/content/cm/NSE_CM_security_24092026.csv.gz"),
        ("nse_fo_ban", "/archives/fo/sec_ban/fo_secban_24092026.csv"),
    ])
    def test_url_shapes_match_the_probed_real_ones(self, name, expected):
        assert archive_url(name, D).endswith(expected)


class TestParsers:
    def test_nse_fii_dii_dates_are_read_from_every_record(self):
        payload = [{"category": "DII", "date": "23-Sep-2026", "netValue": "1"}, {"category": "FII/FPI", "date": "23-Sep-2026"}]
        assert dates_in_nse_fii_dii(payload) == {date(2026, 9, 23)}

    def test_nse_fii_dii_mixed_dates(self):
        payload = [{"date": "23-Sep-2026"}, {"date": "24-Sep-2026"}]
        assert dates_in_nse_fii_dii(payload) == {date(2026, 9, 23), date(2026, 9, 24)}

    def test_nse_fii_dii_garbage_is_empty_not_an_error(self):
        assert dates_in_nse_fii_dii({"error": "x"}) == set()
        assert dates_in_nse_fii_dii([{"date": "not a date"}, {"nodate": 1}]) == set()

    HEADER = "<tr><th>Reporting Date</th><th>Debt/Equity</th><th>Gross Purchases</th></tr>"

    def test_nsdl_page_dates_are_read_from_the_cash_table(self):
        html = (f"<table>{self.HEADER}<tr><td>23-Sep-2026</td><td>Equity</td></tr><tr><td>22-Sep-2026</td></tr></table>"
                " <p>Rs. 95.4</p> 15-Sept-2026x")
        assert dates_in_nsdl_page(html) == {date(2026, 9, 23), date(2026, 9, 22)}

    def test_a_date_only_the_derivatives_table_lists_is_not_a_cash_publication(self):
        """
        The first version read every date on the page. NSDL's page also carries an FPI F&O table whose rows for a day appear
        on days the cash table has none, so 'first seen' would have recorded the wrong table's publication time.
        """
        html = (f"<table>{self.HEADER}<tr><td>23-Sep-2026</td></tr></table>"
                "<table><tr><th>Derivatives</th></tr><tr><td>24-Sep-2026</td></tr></table>")
        assert dates_in_nsdl_page(html) == {date(2026, 9, 23)}

    def test_nsdl_page_without_a_cash_table_lists_nothing(self):
        assert dates_in_nsdl_page("<html>maintenance 23-Sep-2026</html>") == set()


class TestPollOnce:
    """poll_once(state, fetch, now) records the first time each item is seen and never overwrites it."""

    @staticmethod
    def _now(h, m=0):
        return datetime(2026, 9, 24, h, m, tzinfo=IST)

    def test_first_sighting_is_recorded_with_the_previous_absence_as_a_lower_bound(self):
        state = {}
        seen = {"a": False}
        fetch = lambda item: (seen[item], {"last_modified": None})
        poll_once(state, ["a"], fetch, self._now(17, 0))
        assert state["a"]["first_seen_ist"] is None and state["a"]["last_absent_ist"] == self._now(17, 0)
        seen["a"] = True
        poll_once(state, ["a"], fetch, self._now(17, 3))
        assert state["a"]["first_seen_ist"] == self._now(17, 3)
        assert state["a"]["last_absent_ist"] == self._now(17, 0)           # published between 17:00 and 17:03

    def test_a_recorded_sighting_is_never_overwritten(self):
        state = {}
        fetch = lambda item: (True, {})
        poll_once(state, ["a"], fetch, self._now(17, 3))
        poll_once(state, ["a"], fetch, self._now(18, 30))
        assert state["a"]["first_seen_ist"] == self._now(17, 3)

    def test_items_already_seen_are_not_fetched_again(self):
        calls = []
        def fetch(item):
            calls.append(item)
            return True, {}
        state = {}
        poll_once(state, ["a", "b"], fetch, self._now(17))
        poll_once(state, ["a", "b"], fetch, self._now(17, 3))
        assert calls == ["a", "b"]

    def test_a_failing_fetch_counts_as_not_yet_seen_not_as_a_sighting(self):
        def fetch(item):
            raise ConnectionError("boom")
        state = {}
        poll_once(state, ["a"], fetch, self._now(17))
        assert state["a"]["first_seen_ist"] is None and "ConnectionError" in state["a"]["last_error"]

    def test_detail_such_as_last_modified_is_kept_with_the_sighting(self):
        fetch = lambda item: (True, {"last_modified": "2026-09-24T17:50:39+05:30"})
        state = {}
        poll_once(state, ["a"], fetch, self._now(18))
        assert state["a"]["detail"]["last_modified"] == "2026-09-24T17:50:39+05:30"

    def test_now_must_be_timezone_aware(self):
        with pytest.raises(ValueError, match="timezone"):
            poll_once({}, ["a"], lambda i: (True, {}), datetime(2026, 9, 24, 17))
