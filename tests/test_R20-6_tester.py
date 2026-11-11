"""TESTER edge cases for R20-6 (paper-log backfill wrapper). Child runner and lookups faked; no data, no model."""
import numpy as np
import pandas as pd
import pytest

from scripts import run_paper_log_backfill as bf
from tests.test_run_paper_log_backfill import LATEST_EQ, WEEKS, World, dates_called, env, jsonl  # noqa: F401


def test_child_that_cannot_start_is_failed_stops_and_is_recorded(env, monkeypatch):
    def boom(args):
        raise OSError("no such interpreter")
    monkeypatch.setattr(bf, "_run_child", boom)
    assert bf.main(["--from", "2026-09-30"]) == 1
    lines = jsonl(env)
    assert [l["status"] for l in lines] == ["failed"] and lines[0]["child_exit_code"] is None


def test_unknown_latest_equity_session_fails_the_week(env, monkeypatch):
    for unknown in (None, pd.NaT):
        monkeypatch.setattr(bf, "_latest_equity", lambda u=unknown: u)
        env.install(World())
        assert bf.main(["--from", "2026-09-30"]) == 1


def test_one_row_not_late_among_several_fails(env, monkeypatch):
    w = env.install(World())
    monkeypatch.setattr(bf, "_decision_rows", lambda d: pd.DataFrame({"decision_date": [pd.Timestamp(d)] * 3,
                                                                       "late": [True, False, True]}))
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert dates_called(w) == WEEKS[:1]


def test_late_flag_missing_value_is_not_treated_as_late(env, monkeypatch):
    env.install(World())
    monkeypatch.setattr(bf, "_decision_rows", lambda d: pd.DataFrame({"decision_date": [pd.Timestamp(d)],
                                                                       "late": pd.array([np.nan], dtype=object)}))
    assert bf.main(["--from", "2026-09-30"]) == 1          # NaN must stay "unknown", never silently True


def test_row_need_not_be_late_when_no_later_session_exists(env, monkeypatch):
    monkeypatch.setattr(bf, "_latest_equity", lambda: pd.Timestamp(WEEKS[0]))
    env.install(World(late_value=False))
    monkeypatch.setattr(bf, "_latest_complete", lambda s: pd.Timestamp(WEEKS[0]))
    assert bf.main(["--from", "2026-09-30"]) == 0


def test_to_before_from_is_nothing_to_do(env):
    assert bf.main(["--from", "2026-10-09", "--to", "2026-10-01"]) == 0 and env.world.calls == []


def test_to_equal_to_a_week_date_includes_it(env):
    assert bf.main(["--from", "2026-09-30", "--to", WEEKS[1]]) == 0
    assert dates_called(env.world) == WEEKS[:2]


def test_dry_run_exits_0_even_with_would_run_weeks_and_writes_no_jsonl(env):
    assert bf.main(["--from", "2026-09-30", "--dry-run"]) == 0
    assert env.world.calls == [] and jsonl(env) == []


def test_jsonl_accumulates_across_reruns_refused_then_logged(env):
    w = env.install(World(exit_codes={WEEKS[1]: 3}))
    assert bf.main(["--from", "2026-09-30"]) == 1
    w.exit_codes = {}
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert [(l["decision_date"], l["status"]) for l in jsonl(env)] == [
        (WEEKS[0], "logged"), (WEEKS[1], "refused"), (WEEKS[1], "logged"), (WEEKS[2], "logged")]


def test_logged_dates_handles_an_empty_decisions_log(monkeypatch):
    monkeypatch.setattr(bf, "_decisions", lambda: pd.DataFrame())
    assert bf._logged_dates() == set()


def test_none_late_flag_with_later_session_fails(env, monkeypatch):
    env.install(World())
    monkeypatch.setattr(bf, "_decision_rows", lambda d: pd.DataFrame({"decision_date": [pd.Timestamp(d)],
                                                                       "late": pd.array([None], dtype=object)}))
    assert bf.main(["--from", "2026-09-30"]) == 1


def test_numpy_bool_and_python_bool_late_flags_pass(env, monkeypatch):
    env.install(World())
    monkeypatch.setattr(bf, "_decision_rows", lambda d: pd.DataFrame({"decision_date": [pd.Timestamp(d)] * 2,
                                                                       "late": pd.array([True, np.True_], dtype=object)}))
    assert bf.main(["--from", "2026-09-30"]) == 0


def test_code_head_failure_does_not_break_a_run(env, monkeypatch):
    monkeypatch.setattr(bf.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    env.install(World())
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert all(l["code_head"] is None for l in jsonl(env))
