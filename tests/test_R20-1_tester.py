"""TESTER edge cases for R20-1 refresh mode. No network: fetch is faked (reuses the author's fixtures)."""
from datetime import date

import pandas as pd

from data import ingest_corporate_actions as ing
from tests.test_ingest_corporate_actions import Fake, env, rec, seed, OLD_2026  # noqa: F401


def _frame(env, sym="AAA"):
    return pd.read_parquet(env.out / f"{sym}.parquet")


def test_row_dated_exactly_on_refresh_from_is_added(env):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "01-Sep-2026")]}
    assert ing.run_refresh(date(2026, 9, 1))["gained"]["AAA"] == [("2026-09-01", "bonus")]


def test_refresh_from_equal_to_today_is_allowed(env):
    seed(env, "AAA", {2026: OLD_2026})
    assert ing.run_refresh(date(2026, 11, 10))["failed"] == []


def test_empty_fetch_for_a_delisted_symbol_leaves_file_untouched_and_ok(env):
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {}
    s = ing.run_refresh(date(2026, 9, 1))
    assert s["failed"] == [] and s["gained"] == {}
    assert (env.out / "AAA.parquet").read_bytes() == before


def test_row_with_no_ex_date_is_never_added(env):
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "-")]}
    assert ing.run_refresh(date(2026, 9, 1))["gained"] == {}
    assert (env.out / "AAA.parquet").read_bytes() == before


def test_same_row_twice_in_one_fetch_is_added_once(env):
    seed(env, "AAA", {2026: OLD_2026})
    dup = rec("Interim Dividend - Rs 5 Per Share", "15-Oct-2026")
    env.fake.records = {("AAA", 2026): OLD_2026 + [dup, dict(dup)]}
    ing.run_refresh(date(2026, 9, 1))
    df = _frame(env)
    assert len(df) == len(df.drop_duplicates(["symbol", "ex_date", "raw_subject"]))


def test_two_reworded_bonus_rows_new_in_one_fetch_are_not_both_added(env):
    # nothing existing on that day; NSE returns the same bonus twice with different wording -> prices would be adjusted twice
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "15-Oct-2026"), rec("Bonus Issue 1:1", "15-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    df = _frame(env)
    assert (df["action_type"] == "bonus").sum() <= 1


def test_refresh_is_resumable_after_one_symbol_failed(env):
    seed(env, "AAA", {2026: OLD_2026}); seed(env, "BBB", {2026: OLD_2026})
    new = rec("Bonus 1:1", "15-Oct-2026")
    env.fake.records = {("AAA", 2026): OLD_2026 + [new], ("BBB", 2026): RuntimeError("x")}
    assert ing.run_refresh(date(2026, 9, 1))["failed"] == ["BBB"]
    env.fake.records = {("AAA", 2026): OLD_2026 + [new], ("BBB", 2026): OLD_2026 + [new]}
    s = ing.run_refresh(date(2026, 9, 1))
    assert s["failed"] == [] and list(s["gained"]) == ["BBB"] and len(_frame(env, "AAA")) == 2
