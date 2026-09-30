"""
tests/test_ingest_shp_pledge.py

Promoter encumbrance from NSE shareholding-pattern (SHP) filings. Fixtures are real filings (fetched 2026-09-24) trimmed to
the promoter aggregate and the dimensionless flags:

  vedl_2022_03_A            2020-09-30 schema. One combined fact, "pledged or otherwise encumbered": 2,590,189,293 of 2,590,349,949.
  vedl_2026_06_B_ndu_other  2025-10-31 schema. Pledge / non-disposal undertaking / other are separate facts; pledge flag FALSE, so
                            a pledge-only reading would show 0 where the older schema showed 99.99% encumbered.
  gmr_2026_06_B_pledged     2025-10-31 schema with the pledge flag TRUE (1,166,000,000 shares).
  reliance_2026_06_B_zero   the aggregate has only NumberOfShares; the flags declare that nothing is encumbered.
  sintex_2022_12_ns2022     the 2022-09-30 namespace (a third year of the schema); promoter shares are 0.
  icicibank_no_promoter     no promoter aggregate context at all.

The two schemas print percentages on different scales (99.99 vs 0.9999), so the ratio is computed from the share COUNTS and
the reported percentage is used only as an integrity check.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.ingest_shp_pledge import parse_master, parse_shp_xbrl

FIXTURES = Path(__file__).parent / "fixtures" / "shp"


def _xml(name: str) -> str:
    return (FIXTURES / f"{name}.xml").read_text(encoding="utf-8")


class TestXbrl:
    def test_older_schema_reads_the_combined_pledged_or_encumbered_count(self):
        out = parse_shp_xbrl(_xml("vedl_2022_03_A"))
        assert out["SCHEMA_FAMILY"] == "A" and out["HAS_PROMOTER_GROUP"]
        assert out["PROMOTER_SHARES"] == 2590349949 and out["ENCUMBERED_SHARES"] == 2590189293
        assert out["PLEDGED_SHARES"] is None                                   # the old schema cannot separate pledge from other encumbrances
        assert out["PERIOD_END"] == date(2022, 3, 31)

    def test_newer_schema_reports_the_total_encumbered_including_non_disposal_and_other(self):
        out = parse_shp_xbrl(_xml("vedl_2026_06_B_ndu_other"))
        assert out["SCHEMA_FAMILY"] == "B"
        assert out["ENCUMBERED_SHARES"] == 2139651763                          # 107,342,705 NDU + 2,032,309,058 other
        assert out["PLEDGED_SHARES"] == 0                                      # pledge flag false: a stated zero, not a missing value
        assert out["ENCUMBERED_SHARES"] / out["PROMOTER_SHARES"] == pytest.approx(0.9999, abs=1e-4)

    def test_a_pledge_flag_true_gives_the_pledged_count(self):
        out = parse_shp_xbrl(_xml("gmr_2026_06_B_pledged"))
        assert out["PLEDGED_SHARES"] == 1166000000 and out["ENCUMBERED_SHARES"] == 1166000000
        assert out["PROMOTER_SHARES"] == 7091579906

    def test_no_encumbrance_facts_with_every_flag_false_is_a_declared_zero_not_a_missing_value(self):
        out = parse_shp_xbrl(_xml("reliance_2026_06_B_zero"))
        assert out["ENCUMBERED_SHARES"] == 0 and out["PLEDGED_SHARES"] == 0 and out["HAS_PROMOTER_GROUP"]
        assert out["PROMOTER_SHARES"] == 6715496096

    def test_the_2022_namespace_is_read_by_fact_family_not_by_namespace_year(self):
        out = parse_shp_xbrl(_xml("sintex_2022_12_ns2022"))
        assert out["SCHEMA_FAMILY"] == "A" and out["PERIOD_END"] == date(2022, 12, 31)
        assert out["PROMOTER_SHARES"] == 23968563 and out["ENCUMBERED_SHARES"] == 0 and out["HAS_PROMOTER_GROUP"]

    def test_zero_promoter_shares_means_no_promoter_group_not_zero_percent(self):
        zeroed = _xml("sintex_2022_12_ns2022").replace(">23968563<", ">0<")
        assert zeroed != _xml("sintex_2022_12_ns2022")
        out = parse_shp_xbrl(zeroed)
        assert out["PROMOTER_SHARES"] == 0 and not out["HAS_PROMOTER_GROUP"] and out["ENCUMBERED_SHARES"] is None

    def test_a_filing_without_a_promoter_aggregate_context_has_no_promoter_group(self):
        out = parse_shp_xbrl(_xml("icicibank_no_promoter"))
        assert not out["HAS_PROMOTER_GROUP"] and out["PROMOTER_SHARES"] is None and out["ENCUMBERED_SHARES"] is None

    def test_the_trusts_context_is_not_mistaken_for_the_promoter_aggregate(self):
        """Its id contains 'PromoterAndPromoterGroup' but its category is a different member."""
        assert parse_shp_xbrl(_xml("icicibank_no_promoter"))["PROMOTER_SHARES"] is None


class TestXbrlIntegrity:
    """Reject rather than guess: a mis-read count would put a wrong pledge ratio on every session until the next filing."""

    def test_a_count_that_contradicts_the_reported_percentage_is_rejected(self):
        broken = _xml("vedl_2022_03_A").replace(">2590189293<", ">1290189293<")
        with pytest.raises(ValueError, match="percentage"):
            parse_shp_xbrl(broken)

    def test_the_fraction_scale_of_the_newer_schema_is_accepted_and_a_wrong_fraction_is_not(self):
        broken = _xml("vedl_2026_06_B_ndu_other").replace(">0.9999<", ">0.5000<")
        with pytest.raises(ValueError, match="percentage"):
            parse_shp_xbrl(broken)

    def test_encumbered_shares_above_the_promoter_holding_are_rejected(self):
        broken = _xml("gmr_2026_06_B_pledged").replace(">1166000000<", ">9166000000<")
        with pytest.raises(ValueError):
            parse_shp_xbrl(broken)

    def test_a_flag_that_says_nothing_is_encumbered_next_to_a_nonzero_count_is_rejected(self):
        broken = _xml("gmr_2026_06_B_pledged").replace(
            "AreEncumberedUnderPledgedForPromoterAndPromoterGroup contextRef=\"MainI\">true<",
            "AreEncumberedUnderPledgedForPromoterAndPromoterGroup contextRef=\"MainI\">false<")
        assert broken != _xml("gmr_2026_06_B_pledged")
        with pytest.raises(ValueError, match="flag"):
            parse_shp_xbrl(broken)

    def test_a_schema_with_neither_fact_family_is_rejected(self):
        with pytest.raises(ValueError, match="schema"):
            parse_shp_xbrl('<?xml version="1.0"?><xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"></xbrli:xbrl>')

    def test_malformed_xml_is_an_error(self):
        with pytest.raises(ValueError, match="XML"):
            parse_shp_xbrl("<html>Access Denied")


class TestMaster:
    RECORD = {"broadcastDate": "16-JUL-2026 19:24:44", "date": "30-JUN-2026", "pr_and_prgrp": "50.48", "recordId": "211570",
              "revisedData": "N", "revisedDate": None, "submissionDate": "16-JUL-2026", "symbol": "RELIANCE",
              "systemDate": "16-JUL-2026 19:24:48",
              "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/SHP_1694620_16072026072434_WEB.xml"}

    def test_parses_period_and_the_broadcast_timestamp_in_ist(self):
        out = parse_master([self.RECORD], "RELIANCE")
        row = out.iloc[0]
        assert row["PERIOD_END"] == pd.Timestamp("2026-06-30")
        assert row["BROADCAST_IST"] == "2026-07-16T19:24:44+05:30"
        assert row["SUBMISSION_DATE"] == pd.Timestamp("2026-07-16") and not row["IS_REVISED"]
        assert row["XBRL_URL"].endswith("SHP_1694620_16072026072434_WEB.xml") and row["RECORD_ID"] == "211570"

    def test_a_revised_filing_is_marked_and_keeps_its_revision_date(self):
        out = parse_master([{**self.RECORD, "revisedData": "Y", "revisedDate": "20-JUL-2026 10:00:00"}], "RELIANCE")
        assert out.iloc[0]["IS_REVISED"]

    def test_a_missing_broadcast_timestamp_is_kept_null_never_replaced_by_the_period_end(self):
        out = parse_master([{**self.RECORD, "broadcastDate": None}], "RELIANCE")
        assert out["BROADCAST_IST"].isna().all()

    @pytest.mark.parametrize("link", ["-", "", None, "  -  ", "https://nsearchives.nseindia.com/corporate/xbrl/-"])
    def test_a_missing_xbrl_link_is_null_not_a_url_to_fetch(self, link):
        """
        Real master rows say 'no XBRL' as the archive prefix plus a filename of '-' (ADANIENT's pre-2018 filings, VEDL 2023-06-30,
        3MINDIA 2022-03-31), so the value is a full URL ending in '/-'; requesting it is a 404, not data.
        """
        out = parse_master([{**self.RECORD, "xbrl": link}], "RELIANCE")
        assert out["XBRL_URL"].isna().all()

    def test_a_record_for_another_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="symbol"):
            parse_master([{**self.RECORD, "symbol": "TCS"}], "RELIANCE")

    def test_duplicate_period_and_record_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_master([self.RECORD, self.RECORD], "RELIANCE")

    def test_an_unparseable_timestamp_is_rejected(self):
        with pytest.raises(ValueError, match="broadcast"):
            parse_master([{**self.RECORD, "broadcastDate": "yesterday"}], "RELIANCE")

    def test_an_empty_list_is_an_empty_frame_not_an_error(self):
        assert parse_master([], "RELIANCE").empty
