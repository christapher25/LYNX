"""
tests/invariants/test_fno_format_equivalence.py

Invariant: the same trading day parses to the same rows whichever NSE format it arrives in.

Why: NSE replaced the legacy F&O bhavcopy with UDiFF on 2024-07-08, with a different column set,
different units (rupees vs lakhs), a different encoding of "this is a future" (strike 0 / type XX vs
blank), and different date formats. A parser that mishandled any of these would put a step change
into open interest, traded value or strike at the cutover, which downstream features would read as
a market event. Both formats were served for 2024-07-05 with identical contract counts (33,930), so
the equivalence is checkable on real data too: scripts/verify_stage10_fno.py fetches both and
compares them. This file pins it on synthetic twins so it runs anywhere, including CI.
"""

import numpy as np
import pandas as pd
import pytest

from data.ingest_fno_bhavcopy import parse_legacy, parse_udiff
from tests.test_fno_ingest import CONTRACTS, _legacy_raw, _udiff_raw

# Fields the legacy files never carried; everything else must agree exactly.
UDIFF_ONLY = ["UNDERLYING_PRICE", "LOT_SIZE", "NUM_TRADES"]
SHARED_KEY = ["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE"]


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    shared = frame.drop(columns=UDIFF_ONLY)
    return shared.sort_values(SHARED_KEY, na_position="first").reset_index(drop=True)


# Legacy files TRUNCATE traded value to 0.01 lakh (Rs 1,000); UDiFF is exact rupees. Measured on the real overlap
# day (2024-07-05, both formats, 12,531 traded contracts): UDiFF >= legacy on every contract, by less than 0.01
# lakh, and every other shared field agrees bit for bit (e.g. 1216.599997 -> legacy 1216.59, not 1216.60).
LEGACY_VALUE_RESOLUTION_LAKHS = 0.01


def _legacy_shortfall(legacy: pd.DataFrame, udiff: pd.DataFrame) -> pd.Series:
    """How far the truncated legacy value sits below the exact UDiFF value, per contract."""
    return udiff["VALUE_LAKHS"] - legacy["VALUE_LAKHS"]


def test_twin_days_parse_to_identical_rows_across_the_cutover():
    legacy = _canonical(parse_legacy(_legacy_raw()))
    udiff = _canonical(parse_udiff(_udiff_raw()))
    pd.testing.assert_frame_equal(legacy.drop(columns="VALUE_LAKHS"), udiff.drop(columns="VALUE_LAKHS"),
                                  check_exact=False, rtol=1e-12, atol=1e-9)
    shortfall = _legacy_shortfall(legacy, udiff)
    assert (shortfall >= -1e-9).all() and (shortfall <= LEGACY_VALUE_RESOLUTION_LAKHS + 1e-9).all()


def test_every_shared_numeric_field_agrees_contract_by_contract():
    legacy, udiff = _canonical(parse_legacy(_legacy_raw())), _canonical(parse_udiff(_udiff_raw()))
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PRICE", "CONTRACTS", "OPEN_INT", "CHG_IN_OI"]:
        assert np.array_equal(legacy[col].to_numpy(float), udiff[col].to_numpy(float), equal_nan=True), col


def test_a_units_mistake_at_the_cutover_would_be_caught():
    """Sensitivity check: skip UDiFF's rupees -> lakhs conversion and the twins visibly diverge."""
    wrong = parse_udiff(_udiff_raw())
    wrong["VALUE_LAKHS"] = wrong["VALUE_LAKHS"] * 100_000            # what forgetting the conversion looks like
    shortfall = _legacy_shortfall(_canonical(parse_legacy(_legacy_raw())), _canonical(wrong))
    assert shortfall.abs().max() > 1_000 * LEGACY_VALUE_RESOLUTION_LAKHS      # unmistakably not truncation


def test_rounding_instead_of_truncating_would_be_visible_in_the_direction_of_the_gap():
    """
    Legacy never exceeds the exact value. A parser that assumed rounding would expect gaps of either sign; the
    real archive has none below zero, and the synthetic twins must reproduce that.
    """
    legacy, udiff = _canonical(parse_legacy(_legacy_raw())), _canonical(parse_udiff(_udiff_raw()))
    assert (_legacy_shortfall(legacy, udiff) >= -1e-9).all()


def test_futures_encoding_differs_in_the_raw_files_but_not_after_parsing():
    raw_legacy, raw_udiff = _legacy_raw(), _udiff_raw()
    fut_legacy = raw_legacy[raw_legacy["INSTRUMENT"] == "FUTIDX"].iloc[0]
    fut_udiff = raw_udiff[raw_udiff["FinInstrmTp"] == "IDF"].iloc[0]
    assert fut_legacy["STRIKE_PR"] == 0.0 and fut_legacy["OPTION_TYP"] == "XX"     # legacy: sentinels
    assert np.isnan(fut_udiff["StrkPric"]) and np.isnan(fut_udiff["OptnTp"])       # UDiFF: blank
    for parsed in (parse_legacy(raw_legacy), parse_udiff(raw_udiff)):
        future = parsed[parsed["INSTRUMENT"] == "FUTIDX"].iloc[0]
        assert np.isnan(future["STRIKE"]) and pd.isna(future["OPTION_TYPE"])


def test_contract_count_is_preserved_by_both_parsers():
    assert len(parse_legacy(_legacy_raw())) == len(parse_udiff(_udiff_raw())) == len(CONTRACTS)
