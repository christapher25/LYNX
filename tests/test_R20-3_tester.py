"""TESTER edge cases for R20-3 (corporate-actions correction). No network: fetch faked."""
import json

import pandas as pd
import pytest

from scripts import apply_corp_action_correction as cor
from tests.test_apply_corp_action_correction import (BONUS, BONUS_E, DIV, DIV_E, OLD, approved, entry, env,  # noqa: F401
                                                     rec, run, seed)


def test_partial_run_then_rerun_completes_without_touching_done_symbols(env):
    seed(env, "AAA"); seed(env, "BBB")
    env.fake.records = {("AAA", 2026): OLD + [DIV], ("BBB", 2026): RuntimeError("HTTP 500")}
    entries = [DIV_E, entry("BBB", "2026-09-15", "dividend")]
    assert run(env, entries) == 1
    a_bytes = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD + [DIV], ("BBB", 2026): OLD + [DIV]}
    assert run(env, entries) == 0
    assert (env.out / "AAA.parquet").read_bytes() == a_bytes
    assert len(pd.read_parquet(env.out / "BBB.parquet")) == 2


def test_entries_spanning_two_calendar_years_fetch_both_years(env):
    seed(env, "AAA")
    d25 = rec("Interim Dividend - Rs 4 Per Share", "20-Dec-2025")
    env.fake.records = {("AAA", 2025): [d25], ("AAA", 2026): OLD + [DIV]}
    assert run(env, [entry("AAA", "2025-12-20", "dividend"), DIV_E]) == 0
    assert {y for _, y in env.fake.calls} >= {2025, 2026}
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 3


def test_write_failure_leaves_no_tmp_file_and_original_bytes(env, monkeypatch):
    seed(env, "AAA")
    p = env.out / "AAA.parquet"; before = p.read_bytes()
    env.fake.records = {("AAA", 2026): OLD + [DIV]}
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, "write_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert run(env, [DIV_E]) == 1
    assert p.read_bytes() == before and not list(env.out.glob("*.tmp"))


def test_ex_date_exactly_on_cutoff_is_accepted(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [rec("Interim Dividend - Rs 5 Per Share", "29-Sep-2026")]}
    assert run(env, [entry("AAA", "2026-09-29", "dividend")]) == 0


@pytest.mark.parametrize("payload", [
    {"actions": [DIV_E]},                                                   # no history_cutoff
    {"history_cutoff": "2026-09-29", "actions": "nope"},
    {"history_cutoff": "2026-13-01", "actions": [DIV_E]},
    [DIV_E],                                                                # top-level list
])
def test_malformed_top_level_exits_2_without_fetch(env, payload):
    seed(env, "AAA")
    p = env.tmp / "bad.json"; p.write_text(json.dumps(payload), encoding="utf-8")
    assert cor.main(["--approved", str(p)]) == 2 and env.fake.calls == []


@pytest.mark.parametrize("factor", ["NaN", "Infinity", "true"])
def test_non_finite_or_boolean_factor_exits_2(env, factor):
    seed(env, "AAA")
    p = env.tmp / "bad.json"
    p.write_text('{"history_cutoff":"2026-09-29","actions":[{"symbol":"AAA","ex_date":"2026-09-15",'
                 f'"action_type":"bonus","adjustment_factor":{factor}}}]}}', encoding="utf-8")
    assert cor.main(["--approved", str(p)]) == 2 and env.fake.calls == []


def test_missing_approved_file_exits_2(env):
    assert cor.main(["--approved", str(env.tmp / "nope.json")]) == 2


def test_same_day_different_factor_bonus_is_not_confused(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [BONUS, rec("Bonus 2:1", "28-Sep-2026")]}
    assert run(env, [BONUS_E]) == 0
    new = pd.read_parquet(env.out / "AAA.parquet")
    assert list(new.loc[new["action_type"] == "bonus", "adjustment_factor"]) == [0.5]


def test_dry_run_on_already_present_exits_0_and_writes_nothing(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [DIV]}
    assert run(env, [DIV_E]) == 0
    before = (env.out / "AAA.parquet").read_bytes()
    assert run(env, [DIV_E], "--dry-run") == 0
    assert (env.out / "AAA.parquet").read_bytes() == before
