"""TESTER edge cases for R21-2 (corporate-action reparse proposal). Synthetic trees, injected parser."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import propose_corp_action_reparse as prp
from tests.test_propose_corp_action_reparse import GOOD, FakeParser, one, world  # noqa: F401


def test_new_factor_zero_does_not_crash_the_run(world):
    world.add("AAA", [one(world)], GOOD)
    try:
        code = world.run(FakeParser({"split x": 0.0}))
    except ZeroDivisionError:
        pytest.fail("a zero factor from the parser crashed the proposal with ZeroDivisionError")
    assert code in (0, 1)


def test_infinite_close_does_not_crash_the_write(world):
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", float("inf"))])
    try:
        code = world.run(FakeParser({"split x": 0.1}))
    except ValueError:
        pytest.fail("an infinite close crashed the JSON write")
    assert code in (0, 1)


def test_nan_close_on_the_previous_session_is_no_price(world):
    world.add("AAA", [one(world)], [("2026-03-09", float("nan")), ("2026-03-10", 100)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "bad_close"


def test_missing_ex_date_row_is_listed_as_no_price_and_does_not_crash(world):
    world.add("AAA", [one(world)], GOOD)
    p = world.actions / "AAA.parquet"
    df = pd.read_parquet(p); df["ex_date"] = pd.NaT; df.to_parquet(p)
    assert world.run(FakeParser({"split x": 0.1})) == 0
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "no_ex_date" and c["ex_date"] is None


def test_gap_of_exactly_ten_days_is_priced_and_eleven_is_not(world):
    world.add("AAA", [one(world, ex_date="2026-03-20")], [("2026-03-10", 1000), ("2026-03-20", 101)])
    world.run(FakeParser({"split x": 0.1}))
    assert world.changes()[0]["check"] == "consistent"
    world.add("BBB", [one(world, ex_date="2026-03-21")], [("2026-03-10", 1000), ("2026-03-21", 101)])
    world.run(FakeParser({"split x": 0.1}))
    b = [c for c in world.changes() if c["symbol"] == "BBB"][0]
    assert b["no_price_reason"] == "gap_gt_10_days"


def test_check_band_boundary_exactly_25_percent_is_consistent(world):
    # observed 0.125 vs new 0.1 -> ratio/new - 1 = 0.25
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", 125)])
    world.run(FakeParser({"split x": 0.1}))
    assert world.changes()[0]["check"] == "consistent"


def test_two_stored_rows_same_day_both_get_the_same_day_flag(world):
    rows = [one(world, raw_subject="split x"), one(world, raw_subject="bonus y", action_type="bonus", factor=0.5)]
    world.add("AAA", rows, GOOD)
    world.run(FakeParser({"split x": 0.1, "bonus y": 0.5}))
    (c,) = world.changes()
    assert c["same_day"] is True


def test_empty_actions_directory_writes_an_empty_proposal(world):
    assert world.run(FakeParser({})) == 0
    assert world.result()["changes"] == [] and world.result()["totals"]["rows"] == 0


def test_unreadable_actions_file_exits_1_and_writes_nothing(world):
    (world.actions / "BAD.parquet").write_bytes(b"not parquet")
    assert world.run(FakeParser({})) == 1
    assert not (world.out / "proposed_changes.json").exists()


def test_output_json_is_strict_json(world):
    world.add("AAA", [one(world)], GOOD)
    world.run(FakeParser({"split x": 0.1}))
    json.loads((world.out / "proposed_changes.json").read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(c))


def test_combined_factor_includes_an_unchanged_same_day_bonus(world):
    rows = [one(world, raw_subject="split x"), one(world, raw_subject="bonus y", action_type="bonus", factor=0.5)]
    world.add("AAA", rows, [("2026-03-09", 1000), ("2026-03-10", 51)])        # observed 0.051 ~ 0.1 * 0.5
    world.run(FakeParser({"split x": 0.1, "bonus y": 0.5}))
    (c,) = world.changes()
    assert c["combined_factor"] == pytest.approx(0.05) and c["check"] == "inconsistent" and c["check_combined"] == "consistent"


def test_lose_row_alone_is_checked_against_one_when_combined(world):
    world.add("AAA", [one(world, raw_subject="bonus x", action_type="bonus", factor=0.5)], [("2026-03-09", 100), ("2026-03-10", 99)])
    world.run(FakeParser({"bonus x": None}))
    (c,) = world.changes()
    assert c["combined_factor"] is None and c["check_combined"] == "consistent"


def test_bad_factor_row_is_no_price_for_both_checks_and_json_is_strict(world):
    world.add("AAA", [one(world)], GOOD)
    world.run(FakeParser({"split x": -0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["check_combined"] == "no_price" and c["no_price_reason"] == "bad_factor"
