"""
tests/test_crosscheck_nsdl_quarterly.py

NSDL publishes its own quarterly equity totals (QuarterlyWise.aspx). data/crosscheck_nsdl_quarterly.py turns them into an
independent check on the daily series for the years the monthly cross-check table does not cover. Fixtures are the real
pages fetched 2026-09-24.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.crosscheck_nsdl_quarterly import parse_quarterly

FIXTURES = Path(__file__).parent / "fixtures" / "nsdl_fpi"


def _page(year: str) -> str:
    return (FIXTURES / f"quarterly_{year}.html").read_text(encoding="utf-8")


class TestParse:
    def test_2024_takes_the_equity_column_not_the_neighbouring_debt_column(self):
        out = parse_quarterly(_page("2024"))
        assert out["quarter"].tolist() == [1, 2, 3, 4] and (out["year"] == 2024).all()
        assert out["nsdl_equity_cr"].tolist() == [10894.0, -7694.0, 97395.0, -100168.0]   # Q1 debt is 55841

    def test_2025(self):
        out = parse_quarterly(_page("2025"))
        assert out["nsdl_equity_cr"].tolist() == [-116559.0, 38668.0, -76609.0, -11760.0]

    def test_negative_numbers_keep_their_sign(self):
        assert (parse_quarterly(_page("2026"))["nsdl_equity_cr"].iloc[:2] < 0).all()

    def test_only_the_footnoted_quarter_is_partial_and_only_with_an_up_to_date(self):
        current = parse_quarterly(_page("2026"))
        assert current["up_to"].notna().sum() == 1
        assert current.loc[current["quarter"] == 3, "up_to"].iloc[0] == pd.Timestamp("2026-09-24")
        assert current["quarter"].tolist() == [1, 2, 3]
        # 2024 and 2025 also print '**' beside Q3 (a Debt-FAR/MF/AIF inclusion note), which is NOT a partial-quarter marker
        assert parse_quarterly(_page("2024"))["up_to"].isna().all() and parse_quarterly(_page("2025"))["up_to"].isna().all()

    def test_the_pages_own_total_row_equals_the_sum_of_the_quarters(self):
        for year in ("2024", "2025", "2026"):
            out = parse_quarterly(_page(year))
            assert out["nsdl_equity_cr"].sum() == {"2024": 427.0, "2025": -166260.0, "2026": -237030.0}[year]

    def test_a_misread_quarter_fails_the_total_check_instead_of_passing_quietly(self):
        page = _page("2024").replace("97395", "97359", 1)
        with pytest.raises(ValueError, match="total"):
            parse_quarterly(page)

    def test_a_page_with_no_quarter_rows_raises(self):
        with pytest.raises(ValueError):
            parse_quarterly("<html><body>Request Rejected</body></html>")

    def test_scripts_and_markup_do_not_leak_into_the_numbers(self):
        out = parse_quarterly(_page("2026"))
        assert out["nsdl_equity_cr"].notna().all() and np.isfinite(out["nsdl_equity_cr"]).all()
