"""Tests for scripts/propose_corp_action_reparse.py: read-only proposal list of corporate-action factor changes.

Synthetic tmp_path trees, an injected parser (so no dependency on today's parser), synthetic adjusted-price files.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import propose_corp_action_reparse as prp

EX = "2026-03-10"


def actions_frame(symbol, rows):
    """rows: dicts with ex_date, raw_subject and optionally factor, action_type, face_value, extra stored fields."""
    n = len(rows)
    df = pd.DataFrame({
        "symbol": [symbol] * n,
        "ex_date": pd.to_datetime([r["ex_date"] for r in rows]),
        "action_type": [r.get("action_type", "split") for r in rows],
        "raw_subject": [r["raw_subject"] for r in rows],
        "face_value": pd.Series([r.get("face_value", "10") for r in rows], dtype=object),
        "ratio_new": pd.Series([r.get("ratio_new") for r in rows], dtype=object),
        "parse_confidence": pd.Series([r.get("parse_confidence", "high") for r in rows], dtype=object),
        "adjustment_factor": pd.Series([r.get("factor") for r in rows], dtype=object),
    })
    return df


def prices_frame(symbol, closes):
    """closes: list of (date string, close)."""
    return pd.DataFrame({"SYMBOL": symbol, "SERIES": "EQ", "DATE": pd.to_datetime([d for d, _ in closes]),
                         "CLOSE": [float(c) for _, c in closes]})


class FakeParser:
    """mapping: raw_subject -> prp.Reparse (or None -> Reparse(None))."""

    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def __call__(self, subject, face_value):
        self.calls.append((subject, face_value))
        got = self.mapping.get(subject)
        return got if isinstance(got, prp.Reparse) else prp.Reparse(got, {})


class World:
    def __init__(self, root: Path):
        self.root = root
        self.actions, self.adjusted, self.out = root / "actions", root / "adjusted", root / "out"
        self.actions.mkdir()
        self.adjusted.mkdir()

    def add(self, symbol, rows, closes=None):
        actions_frame(symbol, rows).to_parquet(self.actions / f"{symbol}.parquet")
        if closes is not None:
            prices_frame(symbol, closes).to_parquet(self.adjusted / f"{symbol}.parquet")

    def run(self, parser, **kw):
        code = prp.main(["--out-dir", str(self.out)], reparse=parser, actions_dir=self.actions, adjusted_dir=self.adjusted,
                        head="deadbeef", **kw)
        return code

    def result(self):
        return json.loads((self.out / "proposed_changes.json").read_text(encoding="utf-8"))

    def changes(self):
        return self.result()["changes"]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def one(world, symbol="AAA", **row):
    """One stored split row dated EX with no factor."""
    base = {"ex_date": EX, "raw_subject": "split x"}
    base.update(row)
    return base


GOOD = [("2026-03-09", 1000), ("2026-03-10", 101), ("2026-03-11", 100)]


# --------------------------------------------------------------------------------------------------- T1 / T2
def test_T1_gain_with_a_matching_price_drop_is_consistent(world):
    world.add("AAA", [one(world)], GOOD)
    assert world.run(FakeParser({"split x": 0.1})) == 0
    (c,) = world.changes()
    assert (c["kind"], c["check"]) == ("gain", "consistent")
    assert c["old_factor"] is None and c["new_factor"] == 0.1
    assert c["observed_ratio"] == pytest.approx(0.101)
    assert (c["prev_date"], c["prev_close"], c["ex_session_date"], c["ex_close"]) == ("2026-03-09", 1000.0, "2026-03-10", 101.0)
    assert c["same_day"] is False and c["symbol"] == "AAA" and c["ex_date"] == EX and c["raw_subject"] == "split x"


def test_T2_gain_with_no_price_drop_is_inconsistent(world):
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", 990)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert (c["kind"], c["check"]) == ("gain", "inconsistent")


# --------------------------------------------------------------------------------------------------- T3
def test_T3_lose_is_consistent_when_the_price_did_not_drop_by_the_old_factor(world):
    world.add("AAA", [one(world, raw_subject="bonus x", action_type="bonus", factor=0.5)], [("2026-03-09", 100), ("2026-03-10", 99)])
    world.run(FakeParser({"bonus x": None}))
    (c,) = world.changes()
    assert (c["kind"], c["check"]) == ("lose", "consistent") and c["old_factor"] == 0.5 and c["new_factor"] is None


def test_T3_lose_is_inconsistent_when_the_price_really_halved(world):
    world.add("AAA", [one(world, raw_subject="bonus x", action_type="bonus", factor=0.5)], [("2026-03-09", 100), ("2026-03-10", 50)])
    world.run(FakeParser({"bonus x": None}))
    (c,) = world.changes()
    assert (c["kind"], c["check"]) == ("lose", "inconsistent")


# --------------------------------------------------------------------------------------------------- T4
def test_T4_no_adjusted_file_is_no_price(world):
    world.add("AAA", [one(world)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "no_adjusted_file" and c["observed_ratio"] is None


def test_T4_ex_date_after_the_last_session_is_no_price(world):
    world.add("AAA", [one(world, ex_date="2026-04-01")], [("2026-03-30", 100), ("2026-03-31", 100)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "no_ex_session"


def test_T4_ex_date_before_the_first_session_is_no_price(world):
    world.add("AAA", [one(world, ex_date="2026-01-01")], [("2026-03-30", 100), ("2026-03-31", 100)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "no_prev_session"


def test_T4_a_gap_of_more_than_ten_days_is_no_price(world):
    world.add("AAA", [one(world, ex_date="2026-03-15")], [("2026-03-01", 1000), ("2026-03-31", 100)])   # 30 days between sessions
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "no_price" and c["no_price_reason"] == "gap_gt_10_days"


def test_T4_a_gap_of_exactly_ten_days_is_still_checked(world):
    world.add("AAA", [one(world, ex_date="2026-03-15")], [("2026-03-05", 1000), ("2026-03-15", 100)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == "consistent"


# --------------------------------------------------------------------------------------------------- T5
def test_T5_prev_is_strictly_before_the_ex_date_and_ex_session_is_on_it(world):
    world.add("AAA", [one(world)], GOOD)
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["prev_date"] == "2026-03-09" and c["ex_session_date"] == "2026-03-10"


def test_T5_a_non_session_ex_date_uses_the_last_session_before_and_the_first_after(world):
    world.add("AAA", [one(world, ex_date="2026-03-14")],            # a Saturday
              [("2026-03-12", 900), ("2026-03-13", 1000), ("2026-03-16", 101)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert (c["prev_date"], c["ex_session_date"]) == ("2026-03-13", "2026-03-16") and c["check"] == "consistent"


# --------------------------------------------------------------------------------------------------- T6
def test_T6_unchanged_factor_is_counted_not_listed_and_value_to_value_is_a_change(world):
    world.add("AAA", [
        one(world, ex_date="2026-03-02", raw_subject="same", factor=0.5, action_type="bonus"),
        one(world, ex_date="2026-03-03", raw_subject="float noise", factor=0.1 + 0.2, action_type="split"),
        one(world, ex_date="2026-03-04", raw_subject="plain dividend", action_type="dividend"),
        one(world, ex_date=EX, raw_subject="to other", factor=0.5, action_type="bonus"),
    ], GOOD)
    world.run(FakeParser({"same": 0.5, "float noise": 0.3, "plain dividend": None, "to other": 0.25}))
    result = world.result()
    assert [c["raw_subject"] for c in result["changes"]] == ["to other"]
    assert result["changes"][0]["kind"] == "change"
    assert result["totals"]["rows"] == 4 and result["totals"]["unchanged_factor"] == 3


def test_T6_unchanged_factor_with_another_parsed_field_differing_is_counted_separately(world):
    world.add("AAA", [one(world, raw_subject="x", factor=0.5, action_type="bonus", ratio_new="1")], GOOD)
    world.run(FakeParser({"x": prp.Reparse(0.5, {"action_type": "bonus", "ratio_new": "2", "parse_confidence": "high"})}))
    result = world.result()
    assert result["changes"] == [] and result["totals"]["unchanged_factor"] == 1
    assert result["totals"]["unchanged_factor_other_field_differs"] == 1


# --------------------------------------------------------------------------------------------------- T7
def test_T7_same_day_flag_marks_every_proposed_change_that_shares_its_day_with_a_factor_bearing_row(world):
    world.add("AAA", [one(world, raw_subject="split a"), one(world, raw_subject="bonus a", action_type="bonus", factor=0.5)], GOOD)
    world.add("BBB", [one(world, raw_subject="split b")], GOOD)
    world.run(FakeParser({"split a": 0.1, "bonus a": 0.5, "split b": 0.1}))
    by = {c["symbol"]: c for c in world.changes()}
    assert by["AAA"]["same_day"] is True and by["BBB"]["same_day"] is False


def test_T7_a_dividend_on_the_same_day_does_not_set_the_flag(world):
    world.add("AAA", [one(world, raw_subject="split a"), one(world, raw_subject="div", action_type="dividend")], GOOD)
    world.run(FakeParser({"split a": 0.1, "div": None}))
    (c,) = world.changes()
    assert c["same_day"] is False


# --------------------------------------------------------------------------------------------------- T8
def test_T8_the_parser_gets_exactly_the_arguments_the_ingester_passes(world):
    world.add("AAA", [one(world, raw_subject="s1", face_value="5"), one(world, ex_date="2026-03-02", raw_subject="s2", face_value=None)], GOOD)
    parser = FakeParser({})
    world.run(parser)
    assert sorted(parser.calls, key=lambda c: c[0]) == [("s1", "5"), ("s2", None)]


def test_T8_the_default_reparse_is_the_ingesters_parse_then_factor():
    from data.corporate_action_parser import adjustment_factor, parse_subject
    for subject, fv in [("Bonus 1:1", None), ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", "2"),
                        ("Dividend - Rs 6 Per Share", "10")]:
        got = prp.default_reparse(subject, fv)
        want = adjustment_factor(parse_subject(subject, fv))
        assert got.factor == want
    assert prp.default_reparse("Bonus 1:1", None).factor == 0.5
    assert prp.default_reparse("Bonus 1:1", None).fields["action_type"] == "bonus"


def test_T8_ingester_source_calls_parse_subject_with_the_record_and_face_value():
    # guards the premise of T8: if the ingester ever changes how it calls the parser, this proposal script must follow
    src = (Path(prp.__file__).resolve().parent.parent / "data" / "ingest_corporate_actions.py").read_text(encoding="utf-8")
    assert 'parse_subject(subject, face_val)' in src and 'adjustment_factor(parsed)' in src


# --------------------------------------------------------------------------------------------------- T9
def _tree_hashes(*dirs):
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for d in dirs for p in sorted(d.rglob("*")) if p.is_file()}


def test_T9_writes_only_the_two_outputs_and_leaves_the_data_directories_untouched(world):
    world.add("AAA", [one(world)], GOOD)
    before = _tree_hashes(world.actions, world.adjusted)
    world.run(FakeParser({"split x": 0.1}))
    assert _tree_hashes(world.actions, world.adjusted) == before
    assert sorted(p.name for p in world.out.iterdir()) == ["proposed_changes.json", "proposed_changes.md"]


def test_T9_the_json_header_and_ordering(world):
    world.add("BBB", [one(world, raw_subject="b")], GOOD)
    world.add("AAA", [one(world, ex_date="2026-03-10", raw_subject="a2"), one(world, ex_date="2026-03-09", raw_subject="a1")],
              [("2026-03-06", 10), ("2026-03-09", 1), ("2026-03-10", 1)])
    world.run(FakeParser({"a1": 0.5, "a2": 0.5, "b": 0.1}))
    result = world.result()
    assert result["history_cutoff"] == "2026-09-29" and result["generated_from_commit"] == "deadbeef"
    assert [(c["symbol"], c["ex_date"]) for c in result["changes"]] == [("AAA", "2026-03-09"), ("AAA", EX), ("BBB", EX)]


def test_T9_markdown_has_totals_and_lists_inconsistent_and_no_price_first(world):
    world.add("AAA", [one(world, raw_subject="ok")], GOOD)
    world.add("BBB", [one(world, raw_subject="bad")], [("2026-03-09", 1000), ("2026-03-10", 990)])
    world.add("CCC", [one(world, raw_subject="nop")])
    world.run(FakeParser({"ok": 0.1, "bad": 0.1, "nop": 0.1}))
    md = (world.out / "proposed_changes.md").read_text(encoding="utf-8")
    assert "gain" in md and "consistent" in md and "2026" in md and "split" in md                  # totals tables
    assert md.index("BBB") < md.index("AAA") and md.index("CCC") < md.index("AAA")                  # worst first in the line table
    for sym in ("AAA", "BBB", "CCC"):
        assert sym in md                                                                             # every line is there


# --------------------------------------------------------------------------------------------------- T10
def test_T10_a_duplicate_key_exits_1_and_names_it_and_writes_nothing(world, capsys):
    from loguru import logger
    world.add("AAA", [one(world), one(world)], GOOD)
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="ERROR")
    try:
        code = world.run(FakeParser({"split x": 0.1}))
    finally:
        logger.remove(sink)
    assert code == 1 and any("AAA" in m and "split x" in m and EX in m for m in msgs)
    assert not world.out.exists() or list(world.out.iterdir()) == []


def test_T10_an_unreadable_file_exits_1_and_names_it(world):
    from loguru import logger
    (world.actions / "ZZZ.parquet").write_bytes(b"not a parquet file")
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="ERROR")
    try:
        code = world.run(FakeParser({}))
    finally:
        logger.remove(sink)
    assert code == 1 and any("ZZZ.parquet" in m for m in msgs)


def test_exit_code_is_0_even_when_everything_is_inconsistent(world):
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", 990)])
    assert world.run(FakeParser({"split x": 0.1})) == 0


def test_an_unreadable_adjusted_file_exits_1_and_names_it(world):
    from loguru import logger
    world.add("AAA", [one(world)], GOOD)
    (world.adjusted / "AAA.parquet").write_bytes(b"garbage")
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="ERROR")
    try:
        code = world.run(FakeParser({"split x": 0.1}))
    finally:
        logger.remove(sink)
    assert code == 1 and any("AAA.parquet" in m for m in msgs)


def test_module_never_writes_under_data_or_calls_the_network():
    src = Path(prp.__file__).read_text(encoding="utf-8")
    for banned in ("requests", "urllib", "to_parquet", ".unlink(", "os.remove", "shutil"):
        assert banned not in src


# --------------------------------------------------------------------------------------------------- Round 1 (1): non-finite closes
def _strict_json(world):
    return json.loads((world.out / "proposed_changes.json").read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(c))


@pytest.mark.parametrize("closes", [
    [("2026-03-09", float("nan")), ("2026-03-10", 100)],
    [("2026-03-09", 1000), ("2026-03-10", float("nan"))],
    [("2026-03-09", 1000), ("2026-03-10", float("inf"))],
    [("2026-03-09", float("inf")), ("2026-03-10", 100)],
    [("2026-03-09", 0.0), ("2026-03-10", 100)],
])
def test_R1_a_non_finite_or_zero_close_is_no_price_bad_close_and_the_run_completes(world, closes):
    world.add("AAA", [one(world)], closes)
    assert world.run(FakeParser({"split x": 0.1})) == 0
    (c,) = _strict_json(world)["changes"]
    assert c["check"] == "no_price" and c["no_price_reason"] == "bad_close"
    assert c["observed_ratio"] is None and c["check_combined"] == "no_price"
    for k in ("prev_close", "ex_close"):                                   # null where not finite, a number otherwise
        assert c[k] is None or np.isfinite(c[k])


def test_R1_the_finite_close_of_a_bad_pair_is_still_written(world):
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", float("inf"))])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["prev_close"] == 1000.0 and c["ex_close"] is None


# --------------------------------------------------------------------------------------------------- Round 1 (2): bad factors
@pytest.mark.parametrize("bad", [0.0, -0.5, float("inf"), float("nan")])
def test_R1_a_zero_negative_or_non_finite_new_factor_is_no_price_bad_factor(world, bad):
    world.add("AAA", [one(world)], GOOD)
    assert world.run(FakeParser({"split x": bad})) == 0
    cs = _strict_json(world)["changes"]
    if bad != bad:                       # a NaN factor from a parser is treated as no factor at all: nothing to propose
        assert cs == []
        return
    (c,) = cs
    assert c["check"] == "no_price" and c["no_price_reason"] == "bad_factor" and c["observed_ratio"] is None
    assert c["check_combined"] == "no_price"


# --------------------------------------------------------------------------------------------------- Round 1 (3): combined same-day factor
def test_R1_split_with_a_same_day_bonus_gain_is_consistent_when_combined(world):
    world.add("AAA", [one(world, raw_subject="split x"), one(world, raw_subject="bonus y", action_type="bonus")],
              [("2026-03-09", 1000), ("2026-03-10", 104)])
    world.run(FakeParser({"split x": 0.2, "bonus y": 0.5}))                 # the bonus is itself a gain (stored without a factor)
    by = {c["raw_subject"]: c for c in world.changes()}
    s = by["split x"]
    assert s["check"] == "inconsistent"                                       # `check` is exactly what it was
    assert s["combined_factor"] == pytest.approx(0.1) and s["check_combined"] == "consistent"
    b = by["bonus y"]
    assert b["combined_factor"] == pytest.approx(0.1) and b["check_combined"] == "consistent"


def test_R1_the_combined_factor_uses_the_new_factor_of_an_unchanged_same_day_row(world):
    world.add("AAA", [one(world, raw_subject="split x"), one(world, raw_subject="bonus y", action_type="bonus", factor=0.5)],
              [("2026-03-09", 1000), ("2026-03-10", 104)])
    world.run(FakeParser({"split x": 0.2, "bonus y": 0.5}))
    (c,) = world.changes()                                                    # only the split is a proposed change
    assert c["combined_factor"] == pytest.approx(0.1) and c["check_combined"] == "consistent"


def test_R1_a_lone_split_has_combined_equal_to_its_factor_and_the_same_check(world):
    world.add("AAA", [one(world)], GOOD)
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["combined_factor"] == c["new_factor"] == 0.1 and c["check_combined"] == c["check"] == "consistent"


def test_R1_a_lone_split_that_is_inconsistent_stays_inconsistent_combined(world):
    world.add("AAA", [one(world)], [("2026-03-09", 1000), ("2026-03-10", 990)])
    world.run(FakeParser({"split x": 0.1}))
    (c,) = world.changes()
    assert c["check"] == c["check_combined"] == "inconsistent"


def test_R1_a_lose_row_with_nothing_else_on_the_day_has_a_null_combined_factor_compared_with_one(world):
    world.add("AAA", [one(world, raw_subject="bonus x", action_type="bonus", factor=0.5)], [("2026-03-09", 100), ("2026-03-10", 99)])
    world.run(FakeParser({"bonus x": None}))
    (c,) = world.changes()
    assert c["kind"] == "lose" and c["combined_factor"] is None and c["check_combined"] == "consistent"


def test_R1_a_lose_row_beside_a_same_day_factor_uses_that_factor(world):
    world.add("AAA", [one(world, raw_subject="bonus x", action_type="bonus", factor=0.5), one(world, raw_subject="split y")],
              [("2026-03-09", 1000), ("2026-03-10", 101)])
    world.run(FakeParser({"bonus x": None, "split y": 0.1}))
    by = {c["raw_subject"]: c for c in world.changes()}
    assert by["bonus x"]["combined_factor"] == pytest.approx(0.1) and by["bonus x"]["check_combined"] == "consistent"
    assert by["bonus x"]["check"] == "inconsistent"                          # alone it would be compared with 1


def test_R1_rows_on_other_days_do_not_enter_the_combined_factor(world):
    world.add("AAA", [one(world, raw_subject="split x"), one(world, ex_date="2026-03-02", raw_subject="bonus y", action_type="bonus", factor=0.5)],
              GOOD)
    world.run(FakeParser({"split x": 0.1, "bonus y": 0.5}))
    (c,) = world.changes()
    assert c["combined_factor"] == 0.1 and c["same_day"] is False


def test_R1_the_markdown_has_the_new_columns_a_second_totals_table_and_sorts_by_combined(world):
    world.add("AAA", [one(world, raw_subject="ok")], GOOD)
    world.add("ZZZ", [one(world, raw_subject="bad")], [("2026-03-09", 1000), ("2026-03-10", 990)])
    world.add("AAB", [one(world, raw_subject="split c"), one(world, raw_subject="bonus c", action_type="bonus")],
              [("2026-03-09", 1000), ("2026-03-10", 104)])
    world.run(FakeParser({"ok": 0.1, "bad": 0.1, "split c": 0.2, "bonus c": 0.5}))
    md = (world.out / "proposed_changes.md").read_text(encoding="utf-8")
    assert "combined" in md and "check_combined" in md
    assert "By kind and check_combined" in md and "By kind and check" in md
    # AAB is `check` inconsistent but combined consistent: it must NOT sort ahead of the genuinely inconsistent ZZZ (alphabetically later)
    assert md.index("ZZZ") < md.index("AAB") and md.index("AAB") < md.index("AAA")
    totals = world.result()["totals"]
    assert totals["by_kind_check_combined"]["gain/consistent"] == 3 and totals["by_kind_check_combined"]["gain/inconsistent"] == 1
    assert totals["by_kind_check"]["gain/inconsistent"] == 3                       # `check` is unchanged
