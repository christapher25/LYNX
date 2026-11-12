"""TESTER edge cases for R20-7 (guard late fills + one-time exemption). Synthetic data only."""
import numpy as np
import pandas as pd

from scripts import history_guard as hg
from tests.test_history_guard import (CUTOFF, MISSING_VALUES, _circ_world, _fill_all, _fill_in, _fills_file,
                                      _verify_fills, failing_text)


def test_fill_dates_after_the_cutoff_cannot_excuse_history_dates(tmp_path):
    w = _circ_world(tmp_path); _fill_all(w)
    r = _verify_fills(w, dates=["2026-10-05"])
    assert "features_circuits:AAA" in failing_text(r)


def test_a_listed_date_that_went_from_real_to_missing_fails(tmp_path):
    def pre(w):
        _fill_in(w.features / "circuits" / "AAA.parquet", dates=["2026-09-28"])      # real band at snapshot
    w = _circ_world(tmp_path, pre)
    p = w.features / "circuits" / "AAA.parquet"
    df = pd.read_parquet(p); m = df["date"] == pd.Timestamp("2026-09-28")
    df.loc[m, ["price_band_pct", "is_no_band", "band_is_inferred"]] = np.nan
    df.loc[m, "is_missing_circuit_band"] = True
    df.to_parquet(p)
    assert "features_circuits:AAA" in failing_text(_verify_fills(w))


def test_only_one_of_the_two_listed_dates_filled_is_allowed(tmp_path):
    w = _circ_world(tmp_path)
    for sym in ("AAA", "BBB"):
        _fill_in(w.features / "circuits" / f"{sym}.parquet", dates=["2026-09-28"])
    r = _verify_fills(w)
    assert r.failures == [] and r.kinds["owner-approved late fill"] == 2


def test_an_unchanged_real_value_on_one_listed_date_beside_a_fill_on_the_other_fails_closed(tmp_path):
    def pre(w):
        _fill_in(w.features / "circuits" / "AAA.parquet", dates=["2026-09-28"])      # 09-28 already real at snapshot
    w = _circ_world(tmp_path, pre)
    _fill_in(w.features / "circuits" / "AAA.parquet", dates=["2026-09-29"])
    r = _verify_fills(w)
    assert r.failures, "expected fail-closed (masking resets the unchanged real 09-28 row)"


def test_exempting_a_name_not_in_the_index_fails_and_does_not_hide_circuits_failures(tmp_path):
    w = _circ_world(tmp_path); _fill_all(w)
    r = hg.verify(w.sources(), exempt=["features_breadth"])          # circuits differ, no fills; breadth not in this fixture's index
    assert "features_circuits" in failing_text(r) and "not in the snapshot index" in failing_text(r)


def test_dataset_both_exempt_and_listed_in_fills_does_not_crash(tmp_path):
    w = _circ_world(tmp_path); _fill_all(w)
    r = hg.verify(w.sources(), exempt=["features_circuits"], fills=hg.load_fills(_fills_file(w)))
    assert r.exempt == ["features_circuits"] and not _only_circuits(r)


def _only_circuits(r):
    return [f for f in r.failures if f.startswith("features_circuits")]


def test_fill_column_absent_from_the_file_is_ignored_but_the_unit_is_still_checked(tmp_path):
    w = _circ_world(tmp_path); _fill_all(w)
    r = _verify_fills(w, missing={**MISSING_VALUES, "no_such_column": None})
    assert r.failures == []


def test_empty_fills_list_is_a_no_op(tmp_path):
    w = _circ_world(tmp_path)
    p = w.root / "f.json"; p.write_text('{"history_cutoff": "%s", "fills": []}' % CUTOFF)
    assert hg.verify(w.sources(), fills=hg.load_fills(p)).failures == []
    _fill_all(w)
    assert hg.verify(w.sources(), fills=hg.load_fills(p)).failures


def test_nan_in_missing_values_is_accepted_like_null(tmp_path):
    w = _circ_world(tmp_path); _fill_all(w)
    p = w.root / "f.json"
    p.write_text('{"history_cutoff": "%s", "fills": [{"dataset": "features_circuits", "dates": ["2026-09-28", "2026-09-29"], '
                 '"missing_values": {"price_band_pct": NaN, "is_no_band": null, "band_is_inferred": null, '
                 '"is_missing_circuit_band": true}}]}' % CUTOFF)
    assert hg.verify(w.sources(), fills=hg.load_fills(p)).failures == []
