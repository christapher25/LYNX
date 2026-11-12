"""TESTER edge cases for R20-9 (data_refresh --downstream). Children faked; no data, no network."""
import json

import pytest

from scripts import run_data_refresh as rdr
from tests.test_run_data_refresh import DOWNSTREAM_ORDER, capture, chain, ckpt, fake, sandbox  # noqa: F401

ARGS = ["--downstream", "--start", "2026-10-01"]


def _write(sandbox, text):
    p = sandbox / "data" / "processed" / "data_refresh_checkpoint.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("junk", ["", "not json", "[]", "null", '{"key": 1}', '{"key": {"start": "2026-10-01", "today": "2026-11-10"}, "completed": "adjust_prices"}',
                                  '{"key": {"start": "2026-10-01", "today": "2026-11-10"}, "completed": [1, null, {"x": 1}]}'])
def test_corrupt_or_odd_checkpoint_restarts_the_chain_and_never_crashes(chain, sandbox, junk):
    _write(sandbox, junk)
    assert rdr.main(ARGS) == 0
    assert chain.steps() == DOWNSTREAM_ORDER


def test_checkpoint_with_unknown_step_names_skips_nothing_real(chain, sandbox):
    _write(sandbox, json.dumps({"key": {"start": "2026-10-01", "today": "2026-11-10"}, "completed": ["not_a_step", "universe"]}))
    assert rdr.main(ARGS) == 0
    assert chain.steps() == [s for s in DOWNSTREAM_ORDER if s != "universe"]


def test_first_step_failure_exits_with_its_code_and_nothing_else_runs(chain, sandbox):
    chain.fail = {"data.ingest_corporate_actions": 9}
    assert rdr.main(ARGS) == 9
    assert chain.steps() == ["corporate_actions"]
    assert ckpt(sandbox) is None or ckpt(sandbox)["completed"] == []


def test_guard_failure_leaves_the_checkpoint_complete_but_a_rerun_runs_only_the_guard(chain, sandbox):
    chain.fail = {"scripts.history_guard": 1}
    assert rdr.main(ARGS) == 1
    assert ckpt(sandbox)["completed"] == DOWNSTREAM_ORDER[:-1]
    chain.calls.clear()
    assert rdr.main(ARGS) == 1
    assert chain.steps() == ["history_guard"]


def test_failed_step_is_never_recorded_as_completed(chain, sandbox):
    chain.fail = {"features.blocks.fno": 4}
    rdr.main(ARGS)
    assert "fno" not in ckpt(sandbox)["completed"]


def test_orphan_lines_from_several_children_are_all_in_the_summary(chain):
    from tests.test_run_data_refresh import ORPHAN
    chain.output = {"features.blocks.delivery": [ORPHAN], "features.blocks.circuits": [ORPHAN.replace("delivery", "circuits")]}
    msgs, sink = capture("WARNING")
    try:
        rdr.main(ARGS)
    finally:
        from loguru import logger; logger.remove(sink)
    text = "\n".join(msgs)
    assert "features/delivery" in text and "features/circuits" in text


def test_downstream_with_days_uses_refresh_from_the_computed_start(chain):
    assert rdr.main(["--downstream", "--days", "7"]) == 0
    first = chain.calls[0]
    assert first[2] == "data.ingest_corporate_actions" and first[-1] == "2026-11-03"


def test_no_downstream_step_command_goes_through_a_shell(chain):
    rdr.main(ARGS)
    assert all(isinstance(c, list) and all(isinstance(a, str) for a in c) for c in chain.calls)
