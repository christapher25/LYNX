"""
tests/test_ingest_nsdl_fpi.py

Tests for data/ingest_nsdl_fpi.py — Stage 10, flows. NSDL's daily FPI trends page is an ASP.NET report whose cash table
uses rowspan and colspan (the date, asset class and FX rate are each written once per group), whose columns changed
across three eras (no route column until ~2010, growing debt categories since), and whose page also holds a derivatives
table. The fixtures in tests/fixtures/nsdl_fpi/ are the real cash-table HTML, captured 2026-09-24, for five eras.

Silent-failure modes covered: a merged cell shifting every number one column; a derivatives table read as cash flows;
"(5432.83)" read as positive; a blank read as zero; the FX rate lost on all but the first row of a day.
"""

import pathlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.ingest_nsdl_fpi import (
    FIRST_DAILY,
    OUTPUT_COLS,
    month_query_date,
    parse_cash_table,
    parse_fx,
    parse_number,
    identity_tolerances,
    validate_month,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "nsdl_fpi"


def _load(name: str) -> pd.DataFrame:
    return parse_cash_table((FIXTURES / f"{name}.html").read_text(encoding="utf-8"))


def _pick(df, day, asset, route=None):
    m = (df["DATE"] == pd.Timestamp(day)) & (df["ASSET"] == asset)
    m &= df["ROUTE"].isna() if route is None else (df["ROUTE"] == route)
    assert m.sum() == 1, f"{day} {asset} {route}: {m.sum()} rows"
    return df[m].iloc[0]


class TestNumbers:
    @pytest.mark.parametrize("text,expected", [
        ("24655.41", 24655.41), ("(6139.56)", -6139.56), ("1,234.50", 1234.5), ("0.00", 0.0), ("(0.30)", -0.30),
        (" 25.00 ", 25.0),
    ])
    def test_values(self, text, expected):
        assert parse_number(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "   ", "-", "NA", "n/a", None])
    def test_blank_or_missing_is_nan_never_zero(self, text):
        assert np.isnan(parse_number(text))

    @pytest.mark.parametrize("text,expected", [("Rs.63.7615", 63.7615), (" Rs. 42.48", 42.48), ("Rs. 47.03", 47.03)])
    def test_fx(self, text, expected):
        assert parse_fx(text) == pytest.approx(expected)

    def test_fx_garbage_is_nan(self):
        assert np.isnan(parse_fx("")) and np.isnan(parse_fx("Rs."))


class TestFirstEra1999:
    """No route column; two rows (Equity, Debt) per day; date and FX in rowspan cells."""

    def test_shape_and_columns(self):
        df = _load("1999_01")
        assert list(df.columns) == OUTPUT_COLS and len(df) == 6

    def test_a_positive_day(self):
        row = _pick(_load("1999_01"), "1999-01-01", "Equity")
        assert (row.GROSS_PURCHASES_CR, row.GROSS_SALES_CR, row.NET_CR, row.NET_USD_MN) == (32.5, 7.5, 25.0, 5.9)
        assert row.FX_INR_PER_USD == pytest.approx(42.48) and pd.isna(row.ROUTE)

    def test_a_negative_day_keeps_its_sign(self):
        row = _pick(_load("1999_01"), "1999-01-04", "Equity")
        assert row.NET_CR == -1.4 and row.NET_USD_MN == -0.3

    def test_the_debt_row_inherits_the_date_and_the_fx_from_the_merged_cells(self):
        row = _pick(_load("1999_01"), "1999-01-04", "Debt")
        assert row.FX_INR_PER_USD == pytest.approx(42.47) and row.NET_CR == 0.0


class TestMiddleEras:
    def test_2003_and_2008_have_no_route_and_only_equity_and_debt(self):
        for name in ("2003_06", "2008_06"):
            df = _load(name)
            assert df["ROUTE"].isna().all() and set(df["ASSET"]) == {"Equity", "Debt"}, name

    def test_2003_values(self):
        row = _pick(_load("2003_06"), "2003-06-02", "Equity")
        assert (row.GROSS_PURCHASES_CR, row.GROSS_SALES_CR, row.NET_CR) == (563.2, 439.6, 123.6)


class TestRouteEra2015:
    def test_seven_rows_per_day_with_routes(self):
        df = _load("2015_06")
        assert len(df) == 21 and set(df["ROUTE"].dropna()) == {"Stock Exchange", "Primary market & others", "Sub-total", "Total"}

    def test_equity_routes_and_subtotal(self):
        df = _load("2015_06")
        se = _pick(df, "2015-06-01", "Equity", "Stock Exchange")
        assert (se.GROSS_PURCHASES_CR, se.GROSS_SALES_CR, se.NET_CR, se.NET_USD_MN) == (24655.41, 22382.28, 2273.13, 356.51)
        assert se.FX_INR_PER_USD == pytest.approx(63.7615)
        assert _pick(df, "2015-06-01", "Equity", "Primary market & others").NET_CR == 35.38
        assert _pick(df, "2015-06-01", "Equity", "Sub-total").NET_CR == 2308.51

    def test_a_negative_primary_market_figure(self):
        assert _pick(_load("2015_06"), "2015-06-02", "Equity", "Primary market & others").NET_CR == -2.82

    def test_the_debt_group_keeps_its_own_rows_apart_from_equity(self):
        df = _load("2015_06")
        assert _pick(df, "2015-06-01", "Debt", "Stock Exchange").NET_CR == 160.98
        assert _pick(df, "2015-06-01", "Debt", "Sub-total").NET_CR == 530.94

    def test_the_total_row_is_labelled_total_not_absorbed_into_debt(self):
        """Its asset cell is carried down from the Debt rowspan, but a 'Total' row is not a Debt row."""
        row = _pick(_load("2015_06"), "2015-06-01", "Total", "Total")
        assert row.NET_CR == 2839.45 and row.GROSS_PURCHASES_CR == 27901.83


class TestLatestEra2026:
    def test_more_debt_categories_and_hybrid(self):
        df = _load("2026_09")
        assert {"Equity", "Debt-General Limit", "Debt-VRR", "Debt-FAR", "Hybrid"} <= set(df["ASSET"])

    def test_equity_stock_exchange_negative_day(self):
        df = _load("2026_09")
        assert _pick(df, "2026-09-01", "Equity", "Stock Exchange").NET_CR == -6139.56
        assert _pick(df, "2026-09-01", "Equity", "Sub-total").NET_CR == -5432.83

    def test_fx_is_present_on_every_row_of_every_day(self):
        for name in ("1999_01", "2003_06", "2008_06", "2015_06", "2026_09"):
            assert _load(name)["FX_INR_PER_USD"].notna().all(), name


class TestInternalIdentities:
    """Facts about the source that any correct parse must reproduce (a merged-cell shift would break them)."""

    @pytest.mark.parametrize("name", ["1999_01", "2003_06", "2008_06", "2011_06", "2015_06", "2026_09"])
    def test_net_equals_purchases_minus_sales(self, name):
        df = _load(name)
        gap = (df["GROSS_PURCHASES_CR"] - df["GROSS_SALES_CR"] - df["NET_CR"]).abs()
        assert (gap <= identity_tolerances(df)).all(), df.loc[(gap - identity_tolerances(df)).idxmax()].to_dict()

    def test_tolerance_follows_the_precision_of_each_row_not_the_route_column(self):
        """
        Measured on real months: 1999 to 2014-06-02 figures are rounded to ONE decimal (leaf/sub-total gap up to 0.10, Total
        rows up to 0.20); two decimals after (gap <= 0.01). The route column arrived years before the second decimal, so
        keying on it (or on the whole month) failed 55 then 29 months of a real backfill.
        """
        tol = lambda name: identity_tolerances(_load(name))
        eight = _load("2008_06")                                               # no route, one decimal
        nonzero = (eight[["GROSS_PURCHASES_CR", "GROSS_SALES_CR", "NET_CR"]] != 0).any(axis=1)
        assert set(tol("2008_06")[nonzero]) == {0.11} and set(tol("2008_06")[~nonzero]) == {0.02}   # all-zero rows: fine
        eleven = _load("2011_06")
        assert eleven["ROUTE"].notna().any()
        assert set(tol("2011_06")[eleven["ROUTE"] != "Total"]) == {0.11}       # ROUTE PRESENT, one decimal
        assert set(tol("2011_06")[eleven["ROUTE"] == "Total"]) == {0.22}       # Total rows: two rounded sub-totals
        fifteen = _load("2015_06")
        assert set(tol("2015_06")[fifteen["ROUTE"] != "Total"]) == {0.02} and set(tol("2015_06")[fifteen["ROUTE"] == "Total"]) == {0.04}
        gap = lambda n: ((lambda d: (d.GROSS_PURCHASES_CR - d.GROSS_SALES_CR - d.NET_CR).abs().max())(_load(n)))
        assert gap("2008_06") == pytest.approx(0.10, abs=1e-6) and gap("2015_06") <= 0.02

    def test_a_month_that_switches_precision_mid_month_is_judged_row_by_row(self):
        """2014-06: the first day is one-decimal, the rest two-decimal. Each row gets the tolerance of its own precision."""
        coarse, fine = _load("2011_06").iloc[:3].copy(), _load("2015_06").iloc[:3].copy()
        both = pd.concat([coarse, fine], ignore_index=True)
        assert list(identity_tolerances(both)) == [0.11] * 3 + [0.02] * 3

    def test_an_all_zero_row_carries_no_precision_information_and_uses_the_fine_tolerance(self):
        """Real case: 2015-06-03 Equity / Primary market shows 0.00 / 0.00 / 0.00 in a two-decimal era."""
        row = pd.DataFrame({"ROUTE": ["Primary market & others"], "GROSS_PURCHASES_CR": [0.0], "GROSS_SALES_CR": [0.0], "NET_CR": [0.0]})
        assert list(identity_tolerances(row)) == [0.02]

    def test_a_total_row_may_be_off_by_the_measured_two_subtotal_rounding_but_a_leaf_may_not(self):
        df = _load("2011_06")
        total = df.index[df["ROUTE"] == "Total"][0]
        df.loc[total, "NET_CR"] = df.loc[total, "GROSS_PURCHASES_CR"] - df.loc[total, "GROSS_SALES_CR"] + 0.20
        validate_month(df, date(2011, 6, 1))                                     # 0.20 is within a Total row's tolerance
        leaf = df.index[df["ROUTE"] == "Stock Exchange"][0]
        df.loc[leaf, "NET_CR"] = df.loc[leaf, "GROSS_PURCHASES_CR"] - df.loc[leaf, "GROSS_SALES_CR"] + 0.20
        with pytest.raises(ValueError, match="purchases"):
            validate_month(df, date(2011, 6, 1))                                 # but not on an ordinary row

    def test_a_column_shift_is_still_caught_by_a_wide_margin(self):
        df = _load("2011_06")
        df["NET_CR"] = df["NET_CR"].shift(1).fillna(0.0)
        with pytest.raises(ValueError, match="purchases"):
            validate_month(df, date(2011, 6, 1))

    def test_the_2011_fixture_validates_as_a_month(self):
        validate_month(_load("2011_06"), date(2011, 6, 1))

    def test_the_wider_tolerance_does_not_excuse_a_real_error_in_the_early_era(self):
        df = _load("2003_06")
        df.loc[0, "NET_CR"] += 1.0
        with pytest.raises(ValueError, match="purchases"):
            validate_month(df, date(2003, 6, 1))

    @pytest.mark.parametrize("name", ["2015_06", "2026_09"])
    def test_equity_subtotal_is_exchange_plus_primary(self, name):
        df = _load(name)
        for day in df["DATE"].unique():
            eq = df[(df["DATE"] == day) & (df["ASSET"] == "Equity")].set_index("ROUTE")["NET_CR"]
            assert eq["Sub-total"] == pytest.approx(eq["Stock Exchange"] + eq["Primary market & others"], abs=0.02)

    @pytest.mark.parametrize("name", ["1999_01", "2003_06", "2008_06", "2015_06", "2026_09"])
    def test_usd_net_is_inr_net_over_fx(self, name):
        df = _load(name)
        implied = df["NET_CR"] * 10 / df["FX_INR_PER_USD"]           # crore INR -> million USD: x10 / rate
        assert ((implied - df["NET_USD_MN"]).abs() <= 0.06 + 0.001 * df["NET_USD_MN"].abs()).all()


class TestTableSelection:
    def test_the_cash_table_is_chosen_over_a_derivatives_table_on_the_same_page(self):
        derivatives = ("<table><tr><th>Reporting Date</th><th>Derivative Products</th><th>Buy</th><th>Sell</th></tr>"
                       "<tr><td>01-Jun-2015</td><td>Index Futures</td><td>248948.00</td><td>5515.58</td></tr></table>")
        page = "<html>" + derivatives + (FIXTURES / "1999_01.html").read_text(encoding="utf-8") + "</html>"
        assert len(parse_cash_table(page)) == 6

    def test_a_page_with_no_cash_table_raises(self):
        with pytest.raises(ValueError, match="cash table"):
            parse_cash_table("<html><table><tr><td>maintenance</td></tr></table></html>")

    def test_dates_come_out_as_timestamps_in_ascending_order(self):
        df = _load("2015_06")
        assert pd.api.types.is_datetime64_any_dtype(df["DATE"]) and df["DATE"].is_monotonic_increasing


class TestMonthQueryDate:
    """The archive returns month-start THROUGH the queried date, so a full month needs the month's last day."""

    def test_a_past_month_queries_its_last_day(self):
        assert month_query_date(date(2015, 6, 1), today=date(2026, 9, 24)) == date(2015, 6, 30)
        assert month_query_date(date(2016, 2, 1), today=date(2026, 9, 24)) == date(2016, 2, 29)

    def test_the_current_month_queries_today(self):
        assert month_query_date(date(2026, 9, 1), today=date(2026, 9, 24)) == date(2026, 9, 24)

    def test_a_future_month_is_rejected(self):
        with pytest.raises(ValueError, match="future"):
            month_query_date(date(2026, 10, 1), today=date(2026, 9, 24))

    def test_first_daily_is_the_measured_start(self):
        """Probed 2026-09-24: 31-Dec-1998 is a cumulative 'since inception' seed row; daily data starts 1999-01-01."""
        assert FIRST_DAILY == date(1999, 1, 1)


class TestValidateMonth:
    def _good(self):
        return _load("2015_06")

    def test_a_clean_month_passes(self):
        df = self._good()
        assert len(validate_month(df, date(2015, 6, 1))) == len(df)

    def test_dates_outside_the_month_are_rejected(self):
        df = self._good()
        df.loc[0, "DATE"] = pd.Timestamp("2015-07-01")
        with pytest.raises(ValueError, match="month"):
            validate_month(df, date(2015, 6, 1))

    def test_a_duplicated_row_is_rejected(self):
        df = self._good()
        with pytest.raises(ValueError, match="duplicate"):
            validate_month(pd.concat([df, df.iloc[[0]]], ignore_index=True), date(2015, 6, 1))

    def test_net_that_disagrees_with_purchases_minus_sales_is_rejected(self):
        df = self._good()
        df.loc[0, "NET_CR"] += 5.0
        with pytest.raises(ValueError, match="purchases"):
            validate_month(df, date(2015, 6, 1))

    def test_an_empty_month_is_rejected(self):
        with pytest.raises(ValueError, match="no rows"):
            validate_month(self._good().iloc[:0], date(2015, 6, 1))

    def test_a_day_without_an_equity_row_is_rejected(self):
        df = self._good()
        df = df[df["ASSET"] != "Equity"]
        with pytest.raises(ValueError, match="equity"):
            validate_month(df, date(2015, 6, 1))

    def test_a_missing_fx_rate_is_rejected(self):
        df = self._good()
        df.loc[0, "FX_INR_PER_USD"] = np.nan
        with pytest.raises(ValueError, match="fx"):
            validate_month(df, date(2015, 6, 1))


# --- failure classes found by the full 1999-2026 backfill ---------------------------------------------------------

from data.ingest_nsdl_fpi import resolve_duplicate_dates


class TestMalformedPagesWithNestedDerivatives:
    """
    From ~2003 to ~2014 the page's HTML is malformed (3 <table> opens, 2 closes): the FPI derivatives table sits INSIDE
    the matched cash table, after a second 'Reporting Date' header. Read as cash, index/stock futures rows broke
    net = purchases - sales on ~80 rows a month and Saturday sessions (derivatives only) looked like days with no equity.
    """

    def _page(self):
        cash = (FIXTURES / "2003_06.html").read_text(encoding="utf-8").replace("</table>", "")      # unclosed, as served
        derivatives = (
            "<tr><th>Reporting Date</th><th>Derivative Products</th><th colspan='2'>Buy</th><th colspan='2'>Sell</th>"
            "<th colspan='2'>Open Interest at the end of the date</th></tr>"
            "<tr><th></th><th></th><th>No. of Contracts</th><th>Amount in Crore</th><th>No. of Contracts</th>"
            "<th>Amount in Crore</th><th>No. of Contracts</th><th>Amount in Crore</th></tr>"
            "<tr><td rowspan='2'>02-Jun-2003</td><td>Index Futures</td><td>2560.00</td><td>75.96</td><td>2020.00</td><td>59.91</td><td>15463.00</td><td>469.04</td></tr>"
            "<tr><td>Stock Futures</td><td>4832.00</td><td>176.03</td><td>5490.00</td><td>199.32</td><td>69534.00</td><td>2555.20</td></tr>"
            "<tr><td>25-Oct-2003</td><td>Index Futures</td><td>1.00</td><td>2.00</td><td>3.00</td><td>4.00</td><td>5.00</td><td>6.00</td></tr>"
            "</table>")
        return "<html>" + cash + derivatives + "</html>"

    def test_derivative_rows_are_not_read_as_cash_flows(self):
        df = parse_cash_table(self._page())
        assert set(df["ASSET"]) == {"Equity", "Debt"}
        assert len(df) == 6

    def test_a_derivatives_only_saturday_never_becomes_a_cash_date(self):
        assert pd.Timestamp("2003-10-25") not in set(parse_cash_table(self._page())["DATE"])

    def test_the_cash_rows_before_the_second_header_are_untouched(self):
        df = parse_cash_table(self._page())
        assert _pick(df, "2003-06-02", "Equity").NET_CR == 123.6
        validate_month(df, date(2003, 6, 1))


class TestDuplicateDates:
    """
    2000-02-28 appears twice with DIFFERENT figures (one entry is presumably really 2000-02-29, mislabelled, but which
    is unknowable). Exact copies are redundant and collapse; a date whose copies disagree is ambiguous and is quarantined
    whole, reported, and never guessed at.
    """

    def _frame(self):
        return _load("2015_06")

    def test_exact_duplicate_rows_collapse_silently(self):
        df = self._frame()
        out, quarantined, collapsed = resolve_duplicate_dates(pd.concat([df, df.iloc[:7]], ignore_index=True))
        assert len(out) == len(df) and quarantined == [] and collapsed == 7

    def test_conflicting_copies_quarantine_the_whole_date_and_keep_the_others(self):
        df = self._frame()
        clash = df[df["DATE"] == pd.Timestamp("2015-06-02")].copy()
        clash["NET_CR"] = clash["NET_CR"] + 1.0
        out, quarantined, _ = resolve_duplicate_dates(pd.concat([df, clash], ignore_index=True))
        assert quarantined == [pd.Timestamp("2015-06-02")]
        assert pd.Timestamp("2015-06-02") not in set(out["DATE"])
        pd.testing.assert_frame_equal(out.reset_index(drop=True), df[df["DATE"] != pd.Timestamp("2015-06-02")].reset_index(drop=True))

    def test_the_quarantine_never_picks_a_winner(self):
        df = self._frame()
        clash = df.iloc[:3].copy()
        clash["GROSS_SALES_CR"] = clash["GROSS_SALES_CR"] + 5
        out, quarantined, _ = resolve_duplicate_dates(pd.concat([df, clash], ignore_index=True))
        assert not ((out["DATE"] == df.iloc[0]["DATE"])).any()

    def test_a_clean_frame_is_returned_untouched(self):
        df = self._frame()
        out, quarantined, collapsed = resolve_duplicate_dates(df)
        pd.testing.assert_frame_equal(out, df)
        assert quarantined == [] and collapsed == 0

    def test_after_resolution_the_frame_validates(self):
        df = self._frame()
        clash = df[df["DATE"] == pd.Timestamp("2015-06-02")].copy()
        clash["NET_CR"] += 1.0
        out, _, _ = resolve_duplicate_dates(pd.concat([df, clash], ignore_index=True))
        validate_month(out, date(2015, 6, 1))
