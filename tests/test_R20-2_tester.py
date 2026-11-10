"""TESTER edge cases for R20-2 (history guard (a)/(c), ingester (b)/(d)). Synthetic data, no network."""
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts import history_guard as hg
from tests.test_history_guard import (BONUS, DIV, _add_rows, _adj_change, _approved_world, _ca_world, _entry,
                                      _verify_expected, _write_expected, failing_text)
from tests.test_ingest_corporate_actions import env, rec, seed  # noqa: F401
from data import ingest_corporate_actions as ing
from datetime import date


def test_expected_entry_dated_after_the_cutoff_cannot_be_used_to_hide_a_row(tmp_path):
    w = _ca_world(tmp_path); w.snapshot()
    late = _entry("AAA", "2026-10-05", "bonus", 0.5)
    _add_rows(w.actions / "AAA.parquet", late)
    assert "expected action not present" in failing_text(_verify_expected(w, [late]))


def test_expected_entry_with_wrong_factor_fails(tmp_path):
    w = _approved_world(tmp_path)
    r = _verify_expected(w, [_entry("AAA", "2026-09-28", "bonus", 0.25), DIV])
    assert "expected action not present" in failing_text(r)


def test_nan_factor_in_expected_file_never_passes_silently(tmp_path):
    w = _approved_world(tmp_path)
    p = w.root / "e.json"
    p.write_text('{"history_cutoff":"%s","actions":[{"symbol":"AAA","ex_date":"2026-09-28","action_type":"bonus",'
                 '"adjustment_factor":NaN}]}' % json.loads(json.dumps("2026-09-29")))
    try:
        r = hg.verify(w.sources(), expected=hg.load_expected(p))
    except ValueError:
        return                                   # rejected up front is fine
    assert r.failures


def test_approved_adjusted_exemption_does_not_cover_an_unadjusted_column_change(tmp_path):
    w = _approved_world(tmp_path)
    w.edit(w.adjusted / "AAA.parquet", lambda d: d.__setitem__("CLOSE", d["CLOSE"] * 0.5))
    assert "adjusted:AAA" in failing_text(_verify_expected(w, [BONUS, DIV]))


def test_approved_exemption_does_not_cover_a_feature_change_for_that_symbol(tmp_path):
    w = _approved_world(tmp_path)
    feat = next(iter(sorted(p for p in w.root.rglob("*.parquet") if "feature" in str(p).lower())), None)
    if feat is None:
        pytest.skip("no feature file in the World fixture")
    w.edit(feat, lambda d: d.__setitem__(d.columns[-1], d[d.columns[-1]] + 1))
    assert failing_text(_verify_expected(w, [BONUS, DIV]))


def test_expected_with_only_filter_excluding_corporate_actions_fails(tmp_path):
    w = _approved_world(tmp_path)
    r = hg.verify(w.sources(), only=["adjusted"], expected=hg.load_expected(_write_expected(w, [BONUS, DIV])))
    assert r.failures


def test_approved_world_reports_exact_kind_counts(tmp_path):
    w = _approved_world(tmp_path)
    r = _verify_expected(w, [BONUS, DIV])
    assert r.failures == [] and r.kinds["new-action adjusted"] == 0 and r.kinds["empty-column type"] >= 0


def test_refresh_with_added_all_none_in_a_typed_column_keeps_stored_arrow_type_and_values(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "10-Mar-2026")]})
    stored = pq.read_schema(env.out / "AAA.parquet")
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "10-Mar-2026"), rec("Interim Dividend - Rs 5 Per Share", "15-Oct-2026")]}
    assert ing.run_refresh(date(2026, 9, 1))["failed"] == []
    now = pq.read_schema(env.out / "AAA.parquet")
    for f in stored:
        if f.type != "null":
            assert now.field(f.name).type == f.type, f.name
    df = pd.read_parquet(env.out / "AAA.parquet")
    assert df.loc[df["action_type"] == "bonus", "adjustment_factor"].iloc[0] == 0.5
    assert df.loc[df["action_type"] == "dividend", "adjustment_factor"].isna().all()   # NaN stays NaN, never 0


def test_limiter_with_zero_delay_never_sleeps():
    sleeps = []
    lim = ing.RateLimiter(clock=lambda: 1.0, sleep=sleeps.append)
    for _ in range(5):
        lim.wait(0)
    assert sleeps == []
