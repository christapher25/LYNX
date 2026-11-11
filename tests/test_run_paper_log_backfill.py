"""R20-6: tests for scripts/run_paper_log_backfill.py. The child runner and the paper-log lookups are faked; no real data, no model."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import run_paper_log as rpl
from scripts import run_paper_log_backfill as bf

WEEKS = ["2026-10-02", "2026-10-09", "2026-10-16"]
LATEST_EQ = pd.Timestamp("2026-11-09")        # later than every test week, so every row must be late


class World:
    """Fake paper log: `logged` = dates in the decisions log; a child call 'logs' its date unless told otherwise."""

    def __init__(self, logged=(), exit_codes=None, log_on_success=True, late_value=True):
        self.logged = {pd.Timestamp(d) for d in logged}
        self.late = {pd.Timestamp(d): late_value for d in logged}
        self.exit_codes = exit_codes or {}
        self.log_on_success = log_on_success
        self.late_value = late_value
        self.calls = []

    def child(self, args):
        self.calls.append(list(args))
        d = pd.Timestamp(args[args.index("--decision-date") + 1])
        rc = self.exit_codes.get(str(d.date()), 0)
        if rc == 0 and self.log_on_success:
            self.logged.add(d)
            self.late[d] = self.late_value
        return rc

    def rows(self, date):
        d = pd.Timestamp(date)
        if d not in self.logged:
            return pd.DataFrame({"decision_date": [], "late": []})
        return pd.DataFrame({"decision_date": [d, d], "late": [self.late[d]] * 2})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "LOG_DIR", tmp_path / "logs")
    sessions = pd.DatetimeIndex(pd.to_datetime(["2026-09-25", *WEEKS, "2026-11-09"]))
    monkeypatch.setattr(bf, "_sessions", lambda: sessions)
    monkeypatch.setattr(bf, "_weekly", lambda s: pd.DatetimeIndex(pd.to_datetime(["2026-09-25", *WEEKS])))
    monkeypatch.setattr(bf, "_latest_complete", lambda s: pd.Timestamp(WEEKS[-1]))
    monkeypatch.setattr(bf, "_latest_equity", lambda: LATEST_EQ)
    world = World()

    def install(w):
        monkeypatch.setattr(bf, "_run_child", w.child)
        monkeypatch.setattr(bf, "_logged_dates", lambda: set(w.logged))
        monkeypatch.setattr(bf, "_decision_rows", w.rows)
        return w

    install(world)
    return type("Env", (), {"world": world, "install": staticmethod(install), "tmp": tmp_path})


def jsonl(env):
    p = env.tmp / "logs" / "paper_log_backfill.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def dates_called(world):
    return [c[c.index("--decision-date") + 1] for c in world.calls]


# T1 -----------------------------------------------------------------------------------------------
def test_t1_runs_each_week_in_order_with_allow_late(env):
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert dates_called(env.world) == WEEKS
    for c in env.world.calls:
        assert c[:3] == [sys.executable, "-m", "scripts.run_paper_log"]
        assert c[3:] == ["--decision-date", c[4], "--allow-late"]
    lines = jsonl(env)
    assert [l["decision_date"] for l in lines] == WEEKS
    assert all(l["status"] == "logged" and l["child_exit_code"] == 0 for l in lines)
    assert all("allow-late" in l["statement"] and "068593b" in l["statement"] and "2026-09-29" in l["statement"] for l in lines)
    assert all(l["started_utc"] and l["ended_utc"] for l in lines)


# T2 -----------------------------------------------------------------------------------------------
def test_t2_already_logged_week_is_not_run(env):
    env.install(World(logged=[WEEKS[1]]))
    assert bf.main(["--from", "2026-09-30"]) == 0
    w = bf._run_child.__self__
    assert dates_called(w) == [WEEKS[0], WEEKS[2]]
    assert len(jsonl(env)) == 2


# T3 -----------------------------------------------------------------------------------------------
def test_t3_refusal_stops_and_rerun_resumes(env):
    w = env.install(World(exit_codes={WEEKS[1]: 3}))
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert dates_called(w) == WEEKS[:2]                      # third never attempted
    st = {l["decision_date"]: l for l in jsonl(env)}
    assert st[WEEKS[0]]["status"] == "logged"
    assert st[WEEKS[1]]["status"] == "refused" and st[WEEKS[1]]["child_exit_code"] == 3
    assert WEEKS[2] not in st
    w.exit_codes.clear()                                       # the data got fixed; rerun
    w.calls.clear()
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert dates_called(w) == WEEKS[1:]                        # resumes at the refused week, skips the logged one


# T4 -----------------------------------------------------------------------------------------------
def test_t4_exit0_without_rows_is_failed(env):
    w = env.install(World(log_on_success=False))
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert dates_called(w) == WEEKS[:1]
    assert jsonl(env)[-1]["status"] == "failed"


def test_t4_exit0_with_late_false_is_failed(env):
    w = env.install(World(late_value=False))                   # a later session (2026-11-09) exists
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert dates_called(w) == WEEKS[:1]
    assert jsonl(env)[-1]["status"] == "failed"


# T5 -----------------------------------------------------------------------------------------------
def test_t5_dry_run_runs_nothing_and_lists_weeks(env):
    w = env.install(World(logged=[WEEKS[0]]))
    msgs = []
    from loguru import logger
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        assert bf.main(["--from", "2026-09-30", "--dry-run"]) == 0
    finally:
        logger.remove(sink)
    assert w.calls == []
    assert jsonl(env) == [] and not (env.tmp / "logs").exists()
    text = "\n".join(msgs)
    assert f"{WEEKS[0]} already_logged" in text
    assert f"{WEEKS[1]} would_run" in text and f"{WEEKS[2]} would_run" in text


# T6 -----------------------------------------------------------------------------------------------
def test_t6_from_after_latest_complete_is_nothing_to_do(env):
    assert bf.main(["--from", "2026-12-01"]) == 0
    assert env.world.calls == [] and jsonl(env) == []


def test_t6_to_limits_the_list(env):
    assert bf.main(["--from", "2026-09-30", "--to", "2026-10-09"]) == 0
    assert dates_called(env.world) == WEEKS[:2]


def test_t6_from_is_inclusive(env):
    assert bf.main(["--from", "2026-10-09"]) == 0
    assert dates_called(env.world) == WEEKS[1:]


# T7 -----------------------------------------------------------------------------------------------
def test_t7_weeks_follow_the_paper_log_calendar_function(env, monkeypatch):
    monkeypatch.undo()        # drop the fakes, then patch only the paper-log's own functions
    sessions = pd.DatetimeIndex(pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-09"]))
    seen = {}
    monkeypatch.setattr(rpl, "trading_sessions", lambda: list(sessions))
    monkeypatch.setattr(rpl, "_weekly", lambda s: (seen.setdefault("s", s), pd.DatetimeIndex(pd.to_datetime(["2026-10-05"])))[1])
    monkeypatch.setattr(rpl.P, "latest_complete_decision_date", lambda s, now: pd.Timestamp("2026-10-09"))
    assert [str(d.date()) for d in bf.decision_weeks(pd.Timestamp("2026-09-30"), None)] == ["2026-10-05"]
    assert list(seen["s"]) == list(sessions)


def test_t7_calendar_failure_is_an_error_before_anything_runs(env, monkeypatch):
    monkeypatch.setattr(bf, "_latest_complete", lambda s: pd.NaT)
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert env.world.calls == []


def test_t7_calendar_exception_is_exit_1_not_a_traceback(env, monkeypatch):
    def boom(s):
        raise RuntimeError("calendar unreadable")
    monkeypatch.setattr(bf, "_weekly", boom)
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert env.world.calls == []


# T8 -----------------------------------------------------------------------------------------------
def test_t8_never_writes_under_the_paper_log_dir(env, monkeypatch, tmp_path):
    pl = tmp_path / "paper_log"
    pl.mkdir()
    (pl / "decisions.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(rpl, "LOG_DIR", pl)
    before = {p: p.stat().st_mtime_ns for p in pl.rglob("*")}
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert {p: p.stat().st_mtime_ns for p in pl.rglob("*")} == before
    assert (tmp_path / "logs" / "paper_log_backfill.jsonl").exists()


def test_default_lookup_reads_the_paper_log_decisions_file(tmp_path, monkeypatch):
    pl = tmp_path / "paper_log"
    pl.mkdir()
    (pl / "decisions.jsonl").write_text(
        json.dumps({"decision_date": "2026-10-02", "symbol": "AAA", "horizon": 20, "late": True}) + "\n", encoding="utf-8")
    monkeypatch.setattr(rpl, "LOG_DIR", pl)
    assert bf._logged_dates() == {pd.Timestamp("2026-10-02")}
    assert len(bf._decision_rows("2026-10-02")) == 1
    assert len(bf._decision_rows("2026-10-09")) == 0


def test_source_conventions():
    src = open(bf.__file__, encoding="utf-8").read()
    assert "print(" not in src and "append_decisions" not in src and "unlink" not in src


# Round 1: a missing late flag is missing, never True; the jsonl records the checkout's git HEAD ------------------------
@pytest.mark.parametrize("missing", [float("nan"), None])
def test_round1_missing_late_flag_is_not_treated_as_late(env, missing):
    w = env.install(World(late_value=missing))                 # a later session (2026-11-09) exists
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert dates_called(w) == WEEKS[:1]
    assert jsonl(env)[-1]["status"] == "failed"


def test_round1_one_missing_late_flag_among_true_rows_fails(env, monkeypatch):
    env.install(World())
    d = pd.Timestamp(WEEKS[0])
    monkeypatch.setattr(bf, "_decision_rows",
                        lambda date: pd.DataFrame({"decision_date": [d, d, d], "late": [True, float("nan"), True]}))
    assert bf.main(["--from", "2026-09-30"]) == 1
    assert jsonl(env)[-1]["status"] == "failed"


def test_round1_every_jsonl_record_carries_code_head(env, monkeypatch):
    monkeypatch.setattr(bf, "_code_head", lambda: "abc1234")
    assert bf.main(["--from", "2026-09-30"]) == 0
    assert [l["code_head"] for l in jsonl(env)] == ["abc1234"] * 3


def test_round1_code_head_is_none_when_git_is_unavailable(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(bf.subprocess, "run", boom)
    assert bf._code_head() is None
