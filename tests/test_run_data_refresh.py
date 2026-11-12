"""Tests for scripts/run_data_refresh.py: the one-at-a-time ingester wrapper.

No network and no real subprocess: subprocess.run inside the module is replaced by a fake that records each call.
Manifests, data directories and the log directory are redirected to tmp_path.
"""
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from loguru import logger

from scripts import run_data_refresh as rdr

MODULE_SOURCE = Path(rdr.__file__).read_text(encoding="utf-8")
REPO_ROOT = Path(rdr.__file__).resolve().parent.parent

STEP_ORDER = [
    "bhavcopy", "bhavdata", "fno_bhavcopy", "fno_ban", "sec_list",
    "price_band_hits", "indices", "fii_dii", "announcements",
]
TODAY = date(2026, 11, 10)


class FakeRun:
    """Records calls. `fail_on`: module whose first call exits nonzero. `hook(cmd)` runs inside each call."""

    def __init__(self, fail_on=None, code=3, hook=None):
        self.calls, self.kwargs = [], []
        self.fail_on, self.code, self.hook = fail_on, code, hook
        self.order_log = []

    def __call__(self, cmd, *args, **kwargs):
        self.order_log.append(("start", cmd[2]))
        self.calls.append(list(cmd))
        self.kwargs.append(kwargs)
        if self.hook:
            self.hook(cmd)
        self.order_log.append(("end", cmd[2]))
        rc = self.code if (self.fail_on and cmd[2] == self.fail_on) else 0
        return subprocess.CompletedProcess(cmd, rc)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Redirect manifests, data dirs and logs; pin 'today'."""
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    dirs = {name: tmp_path / "data" / name for name in STEP_ORDER}
    for d in dirs.values():
        d.mkdir(parents=True)
    monkeypatch.setattr(rdr, "MANIFEST_DIR", manifests)
    monkeypatch.setattr(rdr, "DATA_DIRS", dirs)
    monkeypatch.setattr(rdr, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(rdr, "_today", lambda: TODAY)
    return tmp_path


@pytest.fixture
def fake(monkeypatch):
    f = FakeRun()
    monkeypatch.setattr(rdr.subprocess, "run", f)
    return f


def modules(calls):
    return [c[2] for c in calls]


def write_manifest(sandbox_dir, step, records):
    path = sandbox_dir / "manifests" / f"{step}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            day, status = rec[0], rec[1]
            error = rec[2] if len(rec) > 2 else ("boom" if status == "failed" else "")
            f.write(json.dumps({"trade_date": day, "status": status, "rows": 0, "bytes_downloaded": 0,
                                "format_era": "", "error": error}) + "\n")


def capture(level="WARNING"):
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level=level)
    return msgs, sink


# ---- order, flags, range ----------------------------------------------------------------------
def test_every_call_is_python_dash_m_with_repo_cwd_and_timeout(fake):
    assert rdr.main([]) == 0
    for c, kw in zip(fake.calls, fake.kwargs):
        assert c[0] == sys.executable and c[1] == "-m" and c[2].startswith("data.ingest_")
        assert Path(kw["cwd"]) == REPO_ROOT
        assert kw["timeout"] == 2 * 3600


def test_order_and_weekend_passes(fake):
    assert rdr.main([]) == 0
    expected = []
    for name in STEP_ORDER:
        mod = "data.ingest_" + name
        expected += [mod] if name in ("fii_dii", "announcements") else [mod, mod]
    assert modules(fake.calls) == expected


def test_weekday_pass_precedes_weekends_pass_same_range(fake):
    rdr.main(["--only", "indices", "--start", "2026-10-01"])
    weekday, weekend = fake.calls
    assert "--weekends" not in weekday and "--weekends" in weekend
    assert [a for a in weekend if a != "--weekends"] == weekday


def test_default_range_is_21_days(fake):
    rdr.main(["--only", "bhavcopy"])
    assert all(c[3:5] == ["--days", "21"] for c in fake.calls)


def test_days_and_start_forwarded(fake):
    rdr.main(["--only", "fno_ban", "--days", "7"])
    assert all(c[3:5] == ["--days", "7"] for c in fake.calls)
    fake.calls.clear()
    rdr.main(["--only", "fno_ban", "--start", "2026-10-05"])
    assert all(c[3:5] == ["--start", "2026-10-05"] for c in fake.calls)


@pytest.mark.parametrize("bad", [["--days", "0"], ["--days", "-3"], ["--days", "x"], ["--start", "2026-13-40"],
                                 ["--start", "yesterday"], ["--start", "2026-10-01", "--days", "5"],
                                 ["--step-timeout-hours", "0"]])
def test_bad_arguments_rejected_before_anything_runs(fake, bad):
    with pytest.raises(SystemExit):
        rdr.main(bad)
    assert fake.calls == []


def test_fii_dii_takes_no_arguments(fake):
    rdr.main(["--only", "fii_dii", "--days", "9"])
    assert fake.calls == [[sys.executable, "-m", "data.ingest_fii_dii"]]


def test_announcements_has_range_but_no_weekends_flag(fake):
    rdr.main(["--only", "announcements", "--days", "10"])
    assert fake.calls == [[sys.executable, "-m", "data.ingest_announcements", "--days", "10"]]


def test_only_runs_just_that_step(fake):
    for name in STEP_ORDER:
        fake.calls.clear()
        assert rdr.main(["--only", name]) == 0
        assert set(modules(fake.calls)) == {"data.ingest_" + name}


def test_unknown_only_step_rejected(fake):
    with pytest.raises(SystemExit):
        rdr.main(["--only", "nonsense"])
    assert fake.calls == []


def test_custom_step_timeout_is_passed(fake):
    rdr.main(["--only", "sec_list", "--step-timeout-hours", "0.5"])
    assert all(kw["timeout"] == 1800 for kw in fake.kwargs)


# ---- failures -----------------------------------------------------------------------------------
def test_stops_at_first_failure_and_names_step(monkeypatch):
    f = FakeRun(fail_on="data.ingest_fno_bhavcopy", code=7)
    monkeypatch.setattr(rdr.subprocess, "run", f)
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main([])
    finally:
        logger.remove(sink)
    assert rc == 7
    assert modules(f.calls)[-1] == "data.ingest_fno_bhavcopy"
    assert "data.ingest_fno_ban" not in modules(f.calls)
    assert any("fno_bhavcopy" in m for m in msgs)


def test_weekend_pass_exit_code_propagates_and_skips_later_steps(monkeypatch):
    calls = []

    def run(cmd, *a, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 9 if "--weekends" in cmd else 0)

    monkeypatch.setattr(rdr.subprocess, "run", run)
    assert rdr.main([]) == 9
    assert modules(calls) == ["data.ingest_bhavcopy"] * 2


def test_timeout_fails_the_step_nonzero_and_names_it(monkeypatch):
    def run(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, k["timeout"])

    monkeypatch.setattr(rdr.subprocess, "run", run)
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--only", "bhavdata"])
    finally:
        logger.remove(sink)
    assert rc != 0 and any("bhavdata" in m and "timed out" in m for m in msgs)


def test_keyboard_interrupt_is_a_clean_nonzero_exit(monkeypatch):
    def run(cmd, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(rdr.subprocess, "run", run)
    assert rdr.main(["--only", "bhavdata"]) == 130


def test_one_subprocess_at_a_time_by_order(fake):
    rdr.main([])
    log = fake.order_log
    assert [k for k, _ in log] == ["start", "end"] * (len(log) // 2)  # never two starts without an end between


# ---- HIGH 1: manifest `failed` dates fail the step ---------------------------------------------
def test_failed_date_in_range_fails_the_step(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-05", "ok"), ("2026-11-06", "failed")])
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--only", "bhavcopy"])
    finally:
        logger.remove(sink)
    assert rc != 0
    assert any("bhavcopy" in m and "2026-11-06" in m for m in msgs)


def test_failed_then_ok_later_is_not_a_failure(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-06", "failed"), ("2026-11-06", "ok")])
    assert rdr.main(["--only", "bhavcopy"]) == 0


def test_failed_date_outside_the_range_is_ignored(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2020-07-13", "failed")])
    assert rdr.main(["--only", "bhavcopy", "--days", "21"]) == 0


def test_range_boundaries_follow_start(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-01", "failed")])
    assert rdr.main(["--only", "bhavcopy", "--start", "2026-10-02"]) == 0
    assert rdr.main(["--only", "bhavcopy", "--start", "2026-10-01"]) != 0


@pytest.mark.parametrize("status", ["empty", "holiday", "ok"])
def test_empty_holiday_ok_are_not_failures(fake, sandbox, status):
    write_manifest(sandbox, "bhavcopy", [("2026-11-06", status)])
    assert rdr.main(["--only", "bhavcopy"]) == 0


def test_every_manifest_step_is_checked(fake, sandbox):
    for name in STEP_ORDER:
        if name == "fii_dii":
            continue
        write_manifest(sandbox, name, [("2026-11-06", "failed")])
        assert rdr.main(["--only", name]) != 0, name
        (sandbox / "manifests" / f"{name}.jsonl").unlink()


def test_failure_stops_before_later_steps(fake, sandbox):
    write_manifest(sandbox, "bhavdata", [("2026-11-06", "failed")])
    assert rdr.main([]) != 0
    assert "data.ingest_fno_bhavcopy" not in modules(fake.calls)


def test_announcements_chunk_key_overlapping_the_range_counts(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-10-25_2026-11-09", "failed")])
    assert rdr.main(["--only", "announcements"]) != 0


def test_missing_manifest_is_not_a_failure(fake):
    assert rdr.main(["--only", "indices"]) == 0


def test_never_claims_complete_when_a_manifest_failed(fake, sandbox):
    write_manifest(sandbox, "sec_list", [("2026-11-06", "failed")])
    msgs, sink = capture("INFO")
    try:
        rdr.main(["--only", "sec_list"])
    finally:
        logger.remove(sink)
    assert not any("data refresh complete" in m.lower() for m in msgs)


def test_currency_line_logged_per_step(fake, sandbox):
    (sandbox / "data" / "bhavcopy" / "2026-11-06.parquet").write_bytes(b"")
    (sandbox / "data" / "bhavcopy" / "2026-11-09.parquet").write_bytes(b"")
    (sandbox / "data" / "announcements" / "announcements_2026-10-01_2026-11-09.parquet").write_bytes(b"")
    msgs, sink = capture("INFO")
    try:
        rdr.main(["--only", "bhavcopy"])
        rdr.main(["--only", "announcements"])
    finally:
        logger.remove(sink)
    assert any("bhavcopy" in m and "newest on disk 2026-11-09" in m for m in msgs)
    assert any("announcements" in m and "newest on disk 2026-11-09" in m for m in msgs)


def test_currency_line_when_directory_empty(fake):
    msgs, sink = capture("INFO")
    try:
        rdr.main(["--only", "sec_list"])
    finally:
        logger.remove(sink)
    assert any("sec_list" in m and "no files on disk" in m for m in msgs)


# ---- MED 2: fii_dii ------------------------------------------------------------------------------
def test_fii_dii_crash_fails_the_run(monkeypatch):
    f = FakeRun(fail_on="data.ingest_fii_dii", code=2)
    monkeypatch.setattr(rdr.subprocess, "run", f)
    assert rdr.main(["--only", "fii_dii"]) == 2


def test_fii_dii_no_file_exit_zero_passes_with_warning(fake):
    msgs, sink = capture("WARNING")
    try:
        rc = rdr.main(["--only", "fii_dii"])
    finally:
        logger.remove(sink)
    assert rc == 0
    assert any("fii_dii" in m and "DII" in m and "lost" in m for m in msgs)


def test_fii_dii_file_for_today_means_no_warning(fake, sandbox):
    (sandbox / "data" / "fii_dii" / f"{TODAY.isoformat()}.parquet").write_bytes(b"")
    msgs, sink = capture("WARNING")
    try:
        assert rdr.main(["--only", "fii_dii"]) == 0
    finally:
        logger.remove(sink)
    assert not any("fii_dii" in m for m in msgs)


def test_fii_dii_file_for_last_session_means_no_warning(fake, sandbox):
    (sandbox / "data" / "bhavcopy" / "2026-11-09.parquet").write_bytes(b"")
    (sandbox / "data" / "fii_dii" / "2026-11-09.parquet").write_bytes(b"")
    msgs, sink = capture("WARNING")
    try:
        assert rdr.main(["--only", "fii_dii"]) == 0
    finally:
        logger.remove(sink)
    assert not any("fii_dii" in m for m in msgs)


def test_fii_dii_stale_file_warns(fake, sandbox):
    (sandbox / "data" / "bhavcopy" / "2026-11-09.parquet").write_bytes(b"")
    (sandbox / "data" / "fii_dii" / "2026-09-24.parquet").write_bytes(b"")
    msgs, sink = capture("WARNING")
    try:
        assert rdr.main(["--only", "fii_dii"]) == 0
    finally:
        logger.remove(sink)
    assert any("fii_dii" in m for m in msgs)


# ---- MED 5: file log ------------------------------------------------------------------------------
def test_a_file_log_is_written_under_the_log_dir(fake, sandbox):
    rdr.main(["--only", "indices"])
    logs = list((sandbox / "logs").glob("data_refresh*.log"))
    assert logs and "indices" in logs[0].read_text(encoding="utf-8")


def test_file_log_sink_is_removed_after_the_run(fake, sandbox):
    rdr.main(["--only", "indices"])
    before = [p.stat().st_size for p in (sandbox / "logs").glob("data_refresh*.log")]
    logger.info("after the run")
    after = [p.stat().st_size for p in (sandbox / "logs").glob("data_refresh*.log")]
    assert before == after


# ---- idempotence / hygiene --------------------------------------------------------------------------
def test_rerun_is_idempotent_and_stateless(fake):
    rdr.main(["--start", "2026-10-01"])
    first = [list(c) for c in fake.calls]
    fake.calls.clear()
    rdr.main(["--start", "2026-10-01"])
    assert fake.calls == first


def test_module_never_deletes():
    assert not re.search(r"os\.remove|os\.unlink|\.unlink\(|rmtree|os\.rmdir|\.rmdir\(|os\.removedirs", MODULE_SOURCE)


def test_no_print_and_uses_loguru():
    assert not re.search(r"^\s*print\(", MODULE_SOURCE, re.M)
    assert "loguru" in MODULE_SOURCE


def test_no_point_in_time_store_step():
    assert [s.name for s in rdr.STEPS] == STEP_ORDER


def test_docstring_documents_announcements_days_semantics():
    doc = rdr.__doc__
    assert "announcements" in doc and "yesterday" in doc


# ---- torn manifest line ---------------------------------------------------------------------------------
def test_torn_last_manifest_line_is_skipped_with_a_warning(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-05", "ok")])
    with (sandbox / "manifests" / "bhavcopy.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"trade_date": "2026-11-06", "sta')
    msgs, sink = capture("WARNING")
    try:
        assert rdr.main(["--only", "bhavcopy"]) == 0
    finally:
        logger.remove(sink)
    assert any("bhavcopy" in m and "line" in m for m in msgs)


def test_earlier_failed_record_still_fails_when_last_line_is_torn(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-06", "failed")])
    with (sandbox / "manifests" / "bhavcopy.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"trade_date": "2026-11-07", "sta')
    assert rdr.main(["--only", "bhavcopy"]) != 0


def test_undecodable_bytes_in_manifest_do_not_crash(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-05", "ok")])
    with (sandbox / "manifests" / "bhavcopy.jsonl").open("ab") as f:
        f.write(b'\xff\xfe{"trade_date"\n')
    assert rdr.main(["--only", "bhavcopy"]) == 0


# ---- stale re-serve on a non-session is excused (never anything else) ---------------------------------------
def stale(day, earlier):
    return f"file dated {earlier}, requested {day}"


def run_bhavdata(sandbox, start="2026-10-01"):
    msgs, sink = capture("INFO")
    try:
        rc = rdr.main(["--only", "bhavdata", "--start", start])
    finally:
        logger.remove(sink)
    return rc, msgs


def test_stale_reserve_on_equity_holiday_is_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    rc, msgs = run_bhavdata(sandbox)
    assert rc == 0
    assert any("expected stale re-serve on a non-session" in m and "2026-10-02" in m for m in msgs)


def test_stale_reserve_on_weekend_with_equity_holiday_is_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-04", "holiday"), ("2026-10-03", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-04", "failed", stale("2026-10-04", "2026-10-02")),
                                         ("2026-10-03", "failed", stale("2026-10-03", "2026-10-02"))])
    rc, msgs = run_bhavdata(sandbox)
    assert rc == 0
    assert any("expected stale re-serve on a non-session" in m and "2026-10-04" in m and "2026-10-03" in m for m in msgs)


def test_mutant_weekend_with_no_equity_record_fails(fake, sandbox):
    write_manifest(sandbox, "bhavdata", [("2026-10-04", "failed", stale("2026-10-04", "2026-10-02"))])
    assert run_bhavdata(sandbox)[0] != 0


@pytest.mark.parametrize("equity_status", ["failed", "empty", "ok"])
@pytest.mark.parametrize("day", ["2026-10-04", "2026-10-02"])  # a Sunday and a weekday
def test_mutant_equity_not_holiday_fails_on_weekend_and_weekday(fake, sandbox, equity_status, day):
    write_manifest(sandbox, "bhavcopy", [(day, equity_status)])
    write_manifest(sandbox, "bhavdata", [(day, "failed", stale(day, "2026-10-01"))])
    assert run_bhavdata(sandbox)[0] != 0


def test_a_step_never_excuses_itself_bhavcopy_weekend_stale_fails(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-04", "failed", stale("2026-10-04", "2026-10-02"))])
    assert rdr.main(["--only", "bhavcopy", "--start", "2026-10-01"]) != 0


def test_bhavcopy_step_stale_fails_even_if_a_holiday_record_followed_elsewhere(fake, sandbox):
    # the bhavcopy step's own manifest can only say failed OR holiday for a date; a failed last record stays failed
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert rdr.main(["--only", "bhavcopy", "--start", "2026-10-01"]) != 0


@pytest.mark.parametrize("step", ["fno_bhavcopy", "indices", "price_band_hits"])
def test_same_excuse_applies_through_other_steps(fake, sandbox, step):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, step, [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert rdr.main(["--only", step, "--start", "2026-10-01"]) == 0
    (sandbox / "manifests" / "bhavcopy.jsonl").unlink()
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "ok")])
    assert rdr.main(["--only", step, "--start", "2026-10-01"]) != 0


# ---- today and later are not checked (not yet published) -------------------------------------------------------
def test_failed_today_is_not_checked_and_logged_as_not_yet_published(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [(TODAY.isoformat(), "failed", "404 not yet")])
    msgs, sink = capture("INFO")
    try:
        rc = rdr.main(["--only", "bhavcopy"])
    finally:
        logger.remove(sink)
    assert rc == 0
    assert sum("not yet published, not checked" in m and TODAY.isoformat() in m for m in msgs) == 1


def test_failed_yesterday_still_fails(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-09", "failed", "timeout")])
    assert rdr.main(["--only", "bhavcopy"]) != 0


def test_failed_after_today_is_not_checked(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-11", "failed", "x")])
    assert rdr.main(["--only", "bhavcopy"]) == 0


def test_today_pending_does_not_hide_a_real_failure(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [(TODAY.isoformat(), "failed", "x"), ("2026-11-09", "failed", "timeout")])
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--only", "bhavcopy"])
    finally:
        logger.remove(sink)
    assert rc != 0 and "2026-11-09" in " ".join(msgs)


def test_weekend_with_equity_ok_record_fails(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-04", "ok")])
    write_manifest(sandbox, "bhavdata", [("2026-10-04", "failed", stale("2026-10-04", "2026-10-02"))])
    assert run_bhavdata(sandbox)[0] != 0


def test_mutant_equity_file_exists_for_the_date_fails(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    (sandbox / "data" / "bhavcopy" / "2026-10-02.parquet").write_bytes(b"")
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert run_bhavdata(sandbox)[0] != 0


def test_mutant_equity_says_ok_on_a_weekday_fails(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "ok")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert run_bhavdata(sandbox)[0] != 0


def test_weekday_with_no_equity_record_at_all_fails(fake, sandbox):
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert run_bhavdata(sandbox)[0] != 0


def test_equity_last_record_decides_not_an_earlier_holiday(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday"), ("2026-10-02", "ok")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01"))])
    assert run_bhavdata(sandbox)[0] != 0


@pytest.mark.parametrize("error", [
    "connection reset", "boom", "file dated 2026-10-01 requested 2026-10-02", "parsed to zero rows",
    "file dated 2026-10-02, requested 2026-10-02",   # equal
    "file dated 2026-10-03, requested 2026-10-02",   # later
    "file dated 2026-10-01, requested 2026-10-05",   # requested date differs from the manifest key
    "",
])
def test_mutant_other_error_text_or_non_earlier_file_date_fails(fake, sandbox, error):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", error)])
    assert run_bhavdata(sandbox)[0] != 0


def test_one_real_failure_among_excused_ones_still_fails_and_names_only_the_real_one(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01")),
                                         ("2026-10-06", "failed", "timeout")])
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--only", "bhavdata", "--start", "2026-10-01"])
    finally:
        logger.remove(sink)
    assert rc != 0
    err = " ".join(msgs)
    assert "2026-10-06" in err and "2026-10-02" not in err


def test_excuse_applies_to_the_last_record_only(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01")),
                                         ("2026-10-02", "failed", "timeout")])
    assert run_bhavdata(sandbox)[0] != 0


# ---- Round 1: pending uses the FIRST date of the key; an impossible calendar date fails cleanly ------------------------
@pytest.mark.parametrize("key", ["2026-11-09_2026-11-10",    # starts yesterday, ends today
                                 "2026-11-08_2026-11-10",
                                 "2026-11-09_2026-11-12"])   # starts in the past, ends after today
def test_round1_failed_range_key_starting_before_today_still_fails(fake, sandbox, key):
    write_manifest(sandbox, "announcements", [(key, "failed", "timeout")])
    assert rdr.main(["--only", "announcements"]) != 0


def test_round1_failed_range_key_starting_after_today_is_not_checked(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-11-11_2026-11-12", "failed", "x")])   # outside [start, today]
    assert rdr.main(["--only", "announcements"]) == 0


@pytest.mark.parametrize("key", ["2026-11-10_2026-11-10", "2026-11-10_2026-11-12"])
def test_round1_failed_range_key_starting_today_is_pending(fake, sandbox, key):
    write_manifest(sandbox, "announcements", [(key, "failed", "404 not yet")])
    msgs, sink = capture("INFO")
    try:
        rc = rdr.main(["--only", "announcements"])
    finally:
        logger.remove(sink)
    assert rc == 0
    assert sum("not yet published, not checked" in m and key in m for m in msgs) == 1


def test_round1_invalid_calendar_date_in_a_failed_key_fails_cleanly(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-13-45", "failed", "garbled key")])
    msgs, sink = capture("WARNING")
    try:
        rc = rdr.main(["--only", "bhavcopy"])
    finally:
        logger.remove(sink)
    assert rc != 0
    assert any("2026-13-45" in m for m in msgs)


def test_round1_invalid_calendar_range_key_fails_cleanly(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-11-01_2026-02-30", "failed", "garbled key")])
    assert rdr.main(["--only", "announcements"]) != 0


def test_round1_equity_manifest_is_read_once_per_step(fake, sandbox, monkeypatch):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday"), ("2026-10-03", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01")),
                                         ("2026-10-03", "failed", stale("2026-10-03", "2026-10-02"))])
    reads = []
    real = rdr.last_records
    monkeypatch.setattr(rdr, "last_records", lambda p: (reads.append(Path(p).name), real(p))[1])
    assert run_bhavdata(sandbox)[0] == 0
    assert reads.count("bhavcopy.jsonl") == 1


# ================================================================================================
# R20-9: --downstream = the rebuild chain after the ingesters, then the history guard
# ================================================================================================
import os  # noqa: E402

DOWNSTREAM_ORDER = ["corporate_actions", "adjust_prices", "universe", "delivery", "pledging", "circuits", "fno", "flows",
                    "breadth", "history_guard"]
GUARD_ARGS = ["verify", "--skip-db", "--expected-changes", "docs/agent-team/corrections/2026-11-10_precutoff_corporate_actions.json",
              "--expected-fills", "docs/agent-team/corrections/2026-11-12_guard_expected_fills.json"]
EXPECTED_CMDS = {                                          # step -> (module, args), exactly as the brief lists them
    "corporate_actions": ("data.ingest_corporate_actions", ["--refresh-from", "2026-10-01"]),
    "adjust_prices": ("data.adjust_prices", []),
    "universe": ("data.build_universe", []),
    "delivery": ("features.blocks.delivery", ["--no-resume", "--list-orphans-only"]),
    "pledging": ("features.blocks.pledging", ["--no-resume", "--list-orphans-only"]),
    "circuits": ("features.blocks.circuits", ["--no-resume", "--list-orphans-only"]),
    "fno": ("features.blocks.fno", ["--no-resume", "--list-orphans-only"]),
    "flows": ("features.blocks.flows", []),
    "breadth": ("features.blocks.breadth", []),
    "history_guard": ("scripts.history_guard", GUARD_ARGS),
}
ORPHAN = ("12:00:00 | WARNING | x:y:42 - data/processed/features/delivery: 3 orphan symbol file(s) not in the universe "
          "(--list-orphans-only, NOT deleted): ['AAA', 'BBB', 'CCC']")


class FakeChild:
    """Replaces rdr._run_child. `fail`: module -> exit code; `output`: module -> list of output lines; `seq` shared with FakeRun."""

    def __init__(self, fail=None, output=None, seq=None):
        self.calls, self.fail, self.output = [], fail or {}, output or {}
        self.seq = seq if seq is not None else []

    def __call__(self, cmd, timeout):
        module = cmd[2]
        self.calls.append(list(cmd))
        self.seq.append(("down", module))
        return self.fail.get(module, 0), list(self.output.get(module, []))

    def steps(self):
        by_module = {m: s for s, (m, _) in EXPECTED_CMDS.items()}
        return [by_module[c[2]] for c in self.calls]


@pytest.fixture
def chain(fake, sandbox, monkeypatch):
    monkeypatch.setattr(rdr, "CHECKPOINT_PATH", sandbox / "data" / "processed" / "data_refresh_checkpoint.json")
    seq = []
    fake.hook = lambda cmd: seq.append(("ingest", cmd[2]))
    child = FakeChild(seq=seq)
    monkeypatch.setattr(rdr, "_run_child", child)
    child.seq = seq
    return child


def ckpt(sandbox):
    p = sandbox / "data" / "processed" / "data_refresh_checkpoint.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# T1 -----------------------------------------------------------------------------------------------
def test_t1_downstream_runs_a_to_j_in_order_with_exact_args_after_the_ingesters(chain):
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == DOWNSTREAM_ORDER
    for cmd, step in zip(chain.calls, DOWNSTREAM_ORDER):
        module, args = EXPECTED_CMDS[step]
        assert cmd == [sys.executable, "-m", module, *args], step
    kinds = [k for k, _ in chain.seq]
    assert kinds == sorted(kinds, key=lambda k: k != "ingest")           # every ingester call comes before every downstream call
    assert kinds.count("down") == 10 and kinds.count("ingest") > 0


def test_t1_without_the_flag_no_downstream_step_runs(chain):
    assert rdr.main(["--start", "2026-10-01"]) == 0
    assert chain.calls == []


def test_t1_corporate_actions_refresh_from_follows_a_days_range(chain):
    assert rdr.main(["--downstream", "--days", "10"]) == 0
    assert chain.calls[0][-2:] == ["--refresh-from", "2026-10-31"]       # TODAY (2026-11-10) minus 10 days


def test_t1_children_are_argument_lists_never_a_shell(chain):
    rdr.main(["--downstream", "--start", "2026-10-01"])
    assert all(isinstance(c, list) and all(isinstance(a, str) for a in c) for c in chain.calls)
    assert "shell=True" not in MODULE_SOURCE and "os.system" not in MODULE_SOURCE


# T2 -----------------------------------------------------------------------------------------------
def test_t2_failed_ingester_means_no_downstream(monkeypatch, sandbox):
    monkeypatch.setattr(rdr, "CHECKPOINT_PATH", sandbox / "ck.json")
    f = FakeRun(fail_on="data.ingest_fno_bhavcopy", code=7)
    monkeypatch.setattr(rdr.subprocess, "run", f)
    child = FakeChild()
    monkeypatch.setattr(rdr, "_run_child", child)
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 7
    assert child.calls == []


def test_t2_manifest_failure_in_an_ingester_means_no_downstream(chain, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-11-09", "failed", "timeout")])
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) != 0
    assert chain.calls == []


# T3 -----------------------------------------------------------------------------------------------
def test_t3_failed_step_stops_the_chain_and_a_rerun_resumes(chain, sandbox):
    chain.fail = {"features.blocks.circuits": 3}
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 3
    assert chain.steps() == DOWNSTREAM_ORDER[:6]                          # g..j never ran
    assert ckpt(sandbox)["completed"] == DOWNSTREAM_ORDER[:5]             # a..e done, f not
    chain.calls.clear()
    chain.fail = {}
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == ["circuits", "fno", "flows", "breadth", "history_guard"]


def test_t3_failure_names_the_step(chain):
    chain.fail = {"data.adjust_prices": 5}
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--downstream", "--start", "2026-10-01"])
    finally:
        logger.remove(sink)
    assert rc == 5 and any("adjust_prices" in m for m in msgs)
    assert chain.steps() == ["corporate_actions", "adjust_prices"]


# T4 -----------------------------------------------------------------------------------------------
def test_t4_different_start_ignores_the_checkpoint(chain):
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    chain.calls.clear()
    assert rdr.main(["--downstream", "--start", "2026-10-02"]) == 0
    assert chain.steps() == DOWNSTREAM_ORDER


def test_t4_restart_downstream_ignores_the_checkpoint(chain):
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    chain.calls.clear()
    assert rdr.main(["--downstream", "--restart-downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == DOWNSTREAM_ORDER


def test_t4_same_key_second_run_skips_the_rebuild_but_not_the_guard(chain):
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    chain.calls.clear()
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == ["history_guard"]


def test_t4_a_new_day_is_a_new_key(chain, monkeypatch):
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    chain.calls.clear()
    monkeypatch.setattr(rdr, "_today", lambda: date(2026, 11, 11))
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == DOWNSTREAM_ORDER


# T5 -----------------------------------------------------------------------------------------------
def test_t5_guard_runs_even_when_the_checkpoint_lists_it(chain, sandbox):
    (sandbox / "data" / "processed").mkdir(parents=True, exist_ok=True)
    rdr.CHECKPOINT_PATH.write_text(json.dumps({"key": {"start": "2026-10-01", "today": "2026-11-10"},
                                               "completed": DOWNSTREAM_ORDER}), encoding="utf-8")
    assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    assert chain.steps() == ["history_guard"]


def test_t5_guard_failure_is_a_non_zero_exit_with_a_clear_message(chain):
    chain.fail = {"scripts.history_guard": 1}
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--downstream", "--start", "2026-10-01"])
    finally:
        logger.remove(sink)
    assert rc == 1
    assert any("history guard FAILED: stop, do not use the rebuilt data until the owner decides" in m for m in msgs)


def test_t5_guard_is_never_recorded_as_completed(chain, sandbox):
    rdr.main(["--downstream", "--start", "2026-10-01"])
    assert "history_guard" not in ckpt(sandbox)["completed"]


# T6 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("module", ["delivery", "pledging", "circuits", "fno"])
def test_t6_every_per_symbol_block_carries_list_orphans_only(chain, module):
    rdr.main(["--downstream", "--start", "2026-10-01"])
    cmd = [c for c in chain.calls if c[2] == f"features.blocks.{module}"][0]
    assert "--list-orphans-only" in cmd and "--no-resume" in cmd


def test_t6_no_per_symbol_block_command_can_be_built_without_the_flag():
    cmds = rdr.downstream_commands(date(2026, 10, 1))
    blocks = [c for _, c in cmds if c[2].split(".")[-1] in {"delivery", "pledging", "circuits", "fno"}
              and c[2].startswith("features.blocks.")]
    assert len(blocks) == 4 and all("--list-orphans-only" in c for c in blocks)
    assert not any("--prune" in " ".join(c) or "--delete" in " ".join(c) for _, c in cmds)


# T7 -----------------------------------------------------------------------------------------------
def test_t7_only_with_downstream_is_a_usage_error(fake, chain):
    with pytest.raises(SystemExit) as e:
        rdr.main(["--only", "bhavcopy", "--downstream"])
    assert e.value.code == 2
    assert fake.calls == [] and chain.calls == []


def test_t7_restart_downstream_needs_downstream(fake, chain):
    with pytest.raises(SystemExit) as e:
        rdr.main(["--restart-downstream"])
    assert e.value.code == 2 and chain.calls == []


# T8 -----------------------------------------------------------------------------------------------
def _summary_tail(msgs):
    text = "\n".join(msgs)
    assert "summary" in text
    return text[text.rindex("summary") - 1:]


def test_t8_orphan_lines_and_the_guard_line_appear_in_the_summary(chain):
    chain.output = {"features.blocks.delivery": ["something", ORPHAN],
                    "features.blocks.circuits": [ORPHAN.replace("delivery", "circuits").replace("3 orphan", "7 orphan")],
                    "scripts.history_guard": ["noise", "history guard: verified 12 datasets, 0 differences"]}
    msgs, sink = capture("INFO")
    try:
        assert rdr.main(["--downstream", "--start", "2026-10-01"]) == 0
    finally:
        logger.remove(sink)
    tail = _summary_tail(msgs)
    assert "3 orphan symbol file(s)" in tail and "7 orphan symbol file(s)" in tail
    assert "history guard: verified 12 datasets, 0 differences" in tail


def test_t8_summary_lists_status_and_seconds_per_step(chain):
    msgs, sink = capture("INFO")
    try:
        rdr.main(["--downstream", "--start", "2026-10-01"])
    finally:
        logger.remove(sink)
    tail = _summary_tail(msgs)
    for step in DOWNSTREAM_ORDER:
        assert step in tail
    assert "ok" in tail and "total" in tail.lower()


def test_t8_the_wrapper_deletes_nothing():
    assert "unlink(" not in MODULE_SOURCE and "rmtree" not in MODULE_SOURCE
    assert "os.remove" not in MODULE_SOURCE


# T9 -----------------------------------------------------------------------------------------------
def test_t9_checkpoint_is_written_atomically(chain, sandbox, monkeypatch):
    real = os.replace
    n = {"i": 0}

    def flaky(src, dst):
        n["i"] += 1
        if n["i"] == 2:
            raise OSError("crash between steps")
        return real(src, dst)

    monkeypatch.setattr(rdr.os, "replace", flaky)
    rdr.main(["--downstream", "--start", "2026-10-01"])
    data = ckpt(sandbox)                                                  # a valid file, holding the steps before the crash
    assert data["completed"][0] == "corporate_actions"
    assert not list((sandbox / "data" / "processed").glob("*.tmp"))        # no half-written leftover


def test_t9_checkpoint_holds_the_run_key(chain, sandbox):
    rdr.main(["--downstream", "--start", "2026-10-01"])
    assert ckpt(sandbox)["key"] == {"start": "2026-10-01", "today": "2026-11-10"}


def test_child_timeout_exits_124_and_names_the_step(chain, monkeypatch):
    monkeypatch.setattr(rdr, "_run_child", lambda cmd, timeout: (124, []))
    msgs, sink = capture("ERROR")
    try:
        rc = rdr.main(["--downstream", "--start", "2026-10-01"])
    finally:
        logger.remove(sink)
    assert rc == 124 and any("corporate_actions" in m for m in msgs)


# the real child runner, on tiny harmless python children (no pipeline is run) ------------------------------------
def test_run_child_keeps_orphan_lines_the_last_line_and_the_exit_code():
    code = ("import sys\n"
            "print('progress 1')\n"
            "print(" + repr(ORPHAN) + ")\n"
            "print('progress 2', file=sys.stderr)\n"
            "print('verdict: ok')\n"
            "sys.exit(0)\n")
    rc, kept = rdr._run_child([sys.executable, "-c", code], 30)
    assert rc == 0
    assert kept[0] == ORPHAN and kept[-1] == "verdict: ok"
    assert "progress 1" not in kept


def test_run_child_passes_a_non_zero_exit_code_through():
    rc, kept = rdr._run_child([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], 30)
    assert rc == 3 and kept[-1] == "boom"


def test_run_child_timeout_kills_the_child_and_returns_124():
    rc, _ = rdr._run_child([sys.executable, "-c", "import time; time.sleep(60)"], 1)
    assert rc == 124
