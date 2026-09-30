"""
tests/test_ingest_fii_dii.py

Tests for data/ingest_fii_dii.py — the NSE FII/DII forward capture. NSE serves only the latest day, so a day nobody
captures is lost for good (22-Sep and 23-Sep-2026 were). What each capture must record, beyond the figures, is WHEN it
was captured: that timestamp is the only evidence of when the value was public, and the point-in-time rule for the DII
flow feature keys on it.
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.ingest_fii_dii import OUTPUT_COLS, parse_payload

IST = timezone(timedelta(hours=5, minutes=30))
CAPTURED = datetime(2026, 9, 24, 19, 41, 12, tzinfo=IST)
PAYLOAD = [
    {"buyValue": "14404.9", "category": "DII", "date": "23-Sep-2026", "netValue": "2341.46", "sellValue": "12063.44"},
    {"buyValue": "13580.4", "category": "FII/FPI", "date": "23-Sep-2026", "netValue": "1617.45", "sellValue": "11962.95"},
]


def test_columns_and_values():
    df = parse_payload(PAYLOAD, CAPTURED, "fiidiiTradeReact")
    assert list(df.columns) == OUTPUT_COLS
    fii = df[df["CATEGORY"] == "FII/FPI"].iloc[0]
    assert (fii["BUY_VALUE_CR"], fii["SELL_VALUE_CR"], fii["NET_VALUE_CR"]) == (13580.4, 11962.95, 1617.45)
    assert fii["DATE"] == pd.Timestamp("2026-09-23")


def test_every_row_carries_the_capture_time_and_source():
    df = parse_payload(PAYLOAD, CAPTURED, "fiidiiTradeReact")
    assert (df["CAPTURED_AT_IST"] == CAPTURED.isoformat()).all() and (df["SOURCE"] == "fiidiiTradeReact").all()


def test_negative_net_keeps_its_sign():
    payload = [PAYLOAD[0], dict(PAYLOAD[1], netValue="-576.2")]
    df = parse_payload(payload, CAPTURED, "x")
    assert df[df["CATEGORY"] == "FII/FPI"].iloc[0]["NET_VALUE_CR"] == -576.2


def test_a_naive_capture_time_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        parse_payload(PAYLOAD, datetime(2026, 9, 24, 19, 41), "fiidiiTradeReact")


def test_a_blank_value_is_nan_not_zero():
    payload = [dict(PAYLOAD[0], netValue=""), PAYLOAD[1]]
    assert np.isnan(parse_payload(payload, CAPTURED, "x").iloc[0]["NET_VALUE_CR"])


@pytest.mark.parametrize("payload", [None, {}, [], {"error": "x"}, [PAYLOAD[0]]])
def test_unexpected_shapes_return_nothing_rather_than_a_partial_frame(payload):
    """The endpoint must yield both FII and DII rows; anything else is a failed capture, not data."""
    assert parse_payload(payload, CAPTURED, "x") is None


def test_rows_for_two_different_dates_are_rejected():
    payload = [PAYLOAD[0], dict(PAYLOAD[1], date="24-Sep-2026")]
    assert parse_payload(payload, CAPTURED, "x") is None
