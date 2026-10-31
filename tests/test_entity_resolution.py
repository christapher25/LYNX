"""
tests/test_entity_resolution.py

Entity resolution maps an announcement record to an NSE ticker. The one rule that matters: a record is either
resolved by an unambiguous, stated method or it is flagged unresolved with a reason. A ticker is NEVER guessed.
"""

from datetime import date

import pandas as pd
import pytest

from data.entity_resolution import (
    SymbolMaster,
    build_name_index,
    build_symbol_master,
    normalise_name,
    resolve,
)

MASTER_FRAME = pd.DataFrame({
    "SYMBOL": ["TCS", "INFY", "OLDCO", "NEWCO", "NOISIN"],
    "ISIN": ["INE467B01029", "INE009A01021", "INE111A01011", "INE111A01011", None],
    "FIRST_DATE": pd.to_datetime(["2004-08-25", "1995-01-01", "2005-01-01", "2015-06-01", "2005-01-01"]),
    "LAST_DATE": pd.to_datetime(["2026-09-29", "2026-09-29", "2015-05-29", "2026-09-29", "2026-09-29"]),
})


@pytest.fixture
def master():
    return SymbolMaster.from_frame(MASTER_FRAME)


def rec(symbol=None, isin=None, name=None, day="2020-01-02"):
    return {"SYMBOL_RAW": symbol, "ISIN": isin, "COMPANY_NAME": name, "PUBLISHED_DATE": pd.Timestamp(day)}


def one(master, name_index=None, **kw):
    out = resolve(pd.DataFrame([rec(**kw)]), master, name_index or {})
    return out.iloc[0]


class TestSymbolFirst:
    def test_known_symbol_resolves(self, master):
        r = one(master, symbol="TCS", isin="INE467B01029")
        assert r["TICKER"] == "TCS" and r["RESOLUTION_METHOD"] == "symbol" and pd.isna(r["UNRESOLVED_REASON"])

    def test_symbol_is_normalised(self, master):
        assert one(master, symbol=" tcs ")["TICKER"] == "TCS"

    def test_symbol_contradicting_isin_is_unresolved_not_trusted(self, master):
        r = one(master, symbol="TCS", isin="INE009A01021")           # INFY's ISIN on a TCS tag
        assert pd.isna(r["TICKER"]) and r["RESOLUTION_METHOD"] == "unresolved"
        assert r["UNRESOLVED_REASON"] == "symbol_isin_conflict"

    def test_symbol_with_unknown_isin_still_resolves(self, master):
        # an ISIN change (e.g. after a face-value split) must not turn into a false conflict
        assert one(master, symbol="TCS", isin="INE000000000")["TICKER"] == "TCS"

    def test_symbol_not_in_master_is_not_rescued_by_isin(self, master):
        r = one(master, symbol="GHOSTSYM", isin="INE467B01029")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_not_in_master"


class TestIsinFallback:
    @pytest.mark.parametrize("blank", [None, "", "-", "  "])
    def test_blank_symbol_uses_unique_isin(self, master, blank):
        r = one(master, symbol=blank, isin="INE467B01029")
        assert r["TICKER"] == "TCS" and r["RESOLUTION_METHOD"] == "isin"

    def test_shared_isin_is_decided_by_date_window(self, master):
        assert one(master, isin="INE111A01011", day="2010-03-01")["TICKER"] == "OLDCO"
        assert one(master, isin="INE111A01011", day="2020-03-01")["TICKER"] == "NEWCO"

    def test_shared_isin_outside_every_window_is_ambiguous(self, master):
        r = one(master, isin="INE111A01011", day="2001-01-01")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "isin_ambiguous"

    def test_malformed_isin_is_ignored_not_matched(self, master):
        r = one(master, isin="NOT-AN-ISIN")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "unmatched"


class TestNameFallback:
    def test_normalisation_collapses_legal_suffix_and_punctuation(self):
        assert normalise_name("Tata Consultancy Services Limited") == normalise_name("TATA CONSULTANCY SERVICES LTD.")
        assert normalise_name("A & B  Co. Ltd") == normalise_name("a and b co limited")

    def test_unique_name_resolves(self, master):
        idx = {normalise_name("Infosys Limited"): {"INFY"}}
        r = one(master, {normalise_name("Infosys Limited"): {"INFY"}}, name="Infosys Ltd.")
        assert r["TICKER"] == "INFY" and r["RESOLUTION_METHOD"] == "name"
        assert idx

    def test_name_with_two_symbols_is_ambiguous(self, master):
        idx = {normalise_name("Twin Limited"): {"TCS", "INFY"}}
        r = one(master, idx, name="Twin Limited")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "name_ambiguous"

    def test_name_pointing_at_symbol_outside_master_is_unresolved(self, master):
        idx = {normalise_name("Ghost Limited"): {"GHOST"}}
        assert pd.isna(one(master, idx, name="Ghost Limited")["TICKER"])

    def test_no_fuzzy_matching(self, master):
        idx = {normalise_name("Infosys Limited"): {"INFY"}}
        assert pd.isna(one(master, idx, name="Infosys Technologies Limited")["TICKER"])

    def test_nothing_usable_is_unmatched(self, master):
        r = one(master)
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "unmatched"


class TestNameIndex:
    def test_built_only_from_symbol_resolved_records(self, master):
        df = pd.DataFrame([
            rec(symbol="INFY", name="Infosys Limited"),
            rec(symbol="INFY", name="Infosys Limited"),
            rec(symbol=None, isin="INE467B01029", name="Tata Consultancy Services Limited"),   # isin-resolved: excluded
        ])
        idx = build_name_index(resolve(df, master, {}))
        assert idx == {normalise_name("Infosys Limited"): {"INFY"}}

    def test_name_seen_with_two_symbols_keeps_both_so_it_is_ambiguous(self, master):
        df = pd.DataFrame([rec(symbol="TCS", name="Same Name"), rec(symbol="INFY", name="Same Name")])
        assert build_name_index(resolve(df, master, {}))[normalise_name("Same Name")] == {"TCS", "INFY"}


class TestResolveFrame:
    def test_row_order_and_input_untouched(self, master):
        df = pd.DataFrame([rec(symbol="INFY"), rec(), rec(symbol="TCS")])
        before = df.copy()
        out = resolve(df, master, {})
        assert list(out["TICKER"].fillna("-")) == ["INFY", "-", "TCS"]
        pd.testing.assert_frame_equal(df, before)

    def test_empty_frame(self, master):
        out = resolve(pd.DataFrame(columns=["SYMBOL_RAW", "ISIN", "COMPANY_NAME", "PUBLISHED_DATE"]), master, {})
        assert out.empty and {"TICKER", "RESOLUTION_METHOD", "UNRESOLVED_REASON"} <= set(out.columns)


class TestBuildSymbolMaster:
    def test_from_bhavcopy_files(self, tmp_path):
        for d, rows in {"2020-01-01": [("AAA", "INE000A01011"), ("BBB", None)], "2020-01-02": [("AAA", "INE000A01011")]}.items():
            pd.DataFrame({"SYMBOL": [r[0] for r in rows], "SERIES": "EQ", "ISIN": [r[1] for r in rows]}).to_parquet(tmp_path / f"{d}.parquet")
        m = build_symbol_master(tmp_path)
        assert m.has_symbol("AAA") and m.has_symbol("BBB") and not m.has_symbol("CCC")
        (sym, first, last), = m.symbols_for_isin("INE000A01011")
        assert sym == "AAA" and first == date(2020, 1, 1) and last == date(2020, 1, 2)


class TestSymbolTradedWindow:
    """A symbol tag is only trusted inside the symbol's traded window (-30d before first, +365d after last)."""

    FRAME = pd.DataFrame({
        "SYMBOL": ["REUSE", "REUSE", "GONE"],
        "ISIN": ["INE222A01011", "INE333B01011", "INE444C01011"],
        "FIRST_DATE": pd.to_datetime(["2005-01-01", "2018-01-01", "2010-01-01"]),
        "LAST_DATE": pd.to_datetime(["2008-12-31", "2026-09-29", "2015-06-30"]),
    })

    @pytest.fixture
    def m(self):
        return SymbolMaster.from_frame(self.FRAME)

    def test_inside_window_resolves(self, m):
        assert one(m, symbol="GONE", day="2012-05-01")["TICKER"] == "GONE"

    def test_before_first_trade_beyond_slack_is_refused(self, m):
        r = one(m, symbol="GONE", day="2009-06-01")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_outside_traded_window"

    def test_pre_listing_notice_within_30_days_resolves(self, m):
        assert one(m, symbol="GONE", day="2009-12-10")["TICKER"] == "GONE"
        assert pd.isna(one(m, symbol="GONE", day="2009-11-20")["TICKER"])

    def test_post_delisting_notice_within_a_year_resolves_then_refused(self, m):
        assert one(m, symbol="GONE", day="2016-05-01")["TICKER"] == "GONE"
        r = one(m, symbol="GONE", day="2016-07-02")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_outside_traded_window"

    def test_reused_symbol_gap_between_two_companies_is_refused(self, m):
        assert one(m, symbol="REUSE", day="2007-01-01")["TICKER"] == "REUSE"
        assert one(m, symbol="REUSE", day="2020-01-01")["TICKER"] == "REUSE"
        r = one(m, symbol="REUSE", day="2014-01-01")                     # between the two companies' windows
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_outside_traded_window"

    def test_missing_publication_date_cannot_be_window_checked(self, m):
        r = resolve(pd.DataFrame([{"SYMBOL_RAW": "GONE", "ISIN": None, "COMPANY_NAME": None, "PUBLISHED_DATE": pd.NaT}]), m, {}).iloc[0]
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_outside_traded_window"


class TestNanIsinWindowsSplitAtGaps:
    """Early bhavcopy has no ISIN. Its NaN-ISIN days must not become one window spanning a reused-symbol gap."""

    def _write(self, tmp_path, days, isin=None):
        for d in days:
            pd.DataFrame({"SYMBOL": ["XYZ"], "SERIES": "EQ", "ISIN": [isin]}).to_parquet(tmp_path / f"{d}.parquet")

    def test_gap_over_395_days_splits_the_window(self, tmp_path):
        self._write(tmp_path, ["2005-01-03", "2005-01-04", "2012-01-02", "2012-01-03"])
        m = build_symbol_master(tmp_path)
        assert m.traded_near("XYZ", date(2005, 1, 4))
        assert m.traded_near("XYZ", date(2012, 1, 2))
        assert not m.traded_near("XYZ", date(2008, 6, 1))

    def test_gap_up_to_395_days_stays_one_window(self, tmp_path):
        self._write(tmp_path, ["2005-01-03", "2006-02-02"])                     # exactly 395 days
        m = build_symbol_master(tmp_path)
        assert m.traded_near("XYZ", date(2005, 8, 1))

    def test_split_helper(self):
        from data.entity_resolution import split_at_gaps
        days = pd.to_datetime(["2005-01-03", "2005-01-10", "2012-01-02"])
        assert split_at_gaps(days) == [(date(2005, 1, 3), date(2005, 1, 10)), (date(2012, 1, 2), date(2012, 1, 2))]
