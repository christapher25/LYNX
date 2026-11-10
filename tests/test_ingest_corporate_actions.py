"""Tests for data/ingest_corporate_actions.py refresh mode (--refresh-from). No network: fetch_symbol_year/get_session are faked."""
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data import ingest_corporate_actions as ing

SOURCE = Path(ing.__file__).read_text(encoding="utf-8")
TODAY = date(2026, 11, 10)


def rec(subject, ex, isin="INE000A01010"):
    return {"subject": subject, "exDate": ex, "recDate": ex, "bcStartDate": "-", "bcEndDate": "-", "isin": isin,
            "series": "EQ", "faceVal": "10"}


class Fake:
    """records[(symbol, year)] -> list of raw records, or an Exception instance to raise."""

    def __init__(self, records=None):
        self.records = records or {}
        self.calls = []

    def __call__(self, session, symbol, year):
        self.calls.append((symbol, year))
        value = self.records.get((symbol, year), [])
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = tmp_path / "corporate_actions"
    out.mkdir()
    monkeypatch.setattr(ing, "OUTPUT_DIR", out)
    monkeypatch.setattr(ing, "MANIFEST_PATH", tmp_path / "manifests" / "corporate_actions.jsonl")
    monkeypatch.setattr(ing, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(ing, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ing, "get_session", lambda: object())
    monkeypatch.setattr(ing, "_today", lambda: TODAY)
    fake = Fake()
    monkeypatch.setattr(ing, "fetch_symbol_year", fake)
    return type("Env", (), {"out": out, "fake": fake, "tmp": tmp_path})


def seed(env, symbol, records_by_year):
    """Write an existing per-symbol file the way default mode would."""
    fake = Fake({(symbol, y): r for y, r in records_by_year.items()})
    saved = ing.fetch_symbol_year
    ing.fetch_symbol_year = fake
    try:
        frame = ing.ingest_symbol(object(), symbol, min(records_by_year), max(records_by_year))
    finally:
        ing.fetch_symbol_year = saved
    ing.write_parquet_atomic(frame, env.out / f"{symbol}.parquet")
    return pd.read_parquet(env.out / f"{symbol}.parquet")   # as stored, so dtype checks are against the file


def manifest_records(env):
    path = ing.MANIFEST_PATH
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


OLD_2026 = [rec("Interim Dividend - Rs 6 Per Share", "29-Apr-2026")]


def test_new_rows_are_appended_and_old_rows_preserved(env):
    old = seed(env, "AAA", {2025: [rec("Bonus 1:1", "10-Mar-2025")], 2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Interim Dividend - Rs 5 Per Share", "15-Oct-2026")]}
    summary = ing.run_refresh(date(2026, 9, 1))
    new = pd.read_parquet(env.out / "AAA.parquet")
    assert len(new) == len(old) + 1
    # every existing row is present unchanged (compared value by value, same order among themselves)
    kept = new[new["raw_subject"].isin(old["raw_subject"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, old.sort_values("ex_date").reset_index(drop=True), check_dtype=True)
    assert new["ex_date"].is_monotonic_increasing
    assert summary["gained"]["AAA"] == [("2026-10-15", "dividend")]


def test_duplicate_is_not_doubled_and_file_untouched_when_nothing_new(env):
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD_2026}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert (env.out / "AAA.parquet").read_bytes() == before
    assert summary["gained"] == {}
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 1


def test_existing_row_wins_a_tie(env):
    seed(env, "AAA", {2026: [rec("Interim Dividend - Rs 6 Per Share", "29-Apr-2026", isin="OLDISIN")]})
    env.fake.records = {("AAA", 2026): [rec("Interim Dividend - Rs 6 Per Share", "29-Apr-2026", isin="NEWISIN"),
                                        rec("Bonus 1:1", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    df = pd.read_parquet(env.out / "AAA.parquet")
    assert set(df.loc[df["raw_subject"].str.startswith("Interim"), "isin"]) == {"OLDISIN"}
    assert len(df) == 2


def test_failed_fetch_leaves_the_file_untouched_and_is_recorded_failed(env):
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): ValueError("HTTP 403")}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert (env.out / "AAA.parquet").read_bytes() == before
    assert summary["failed"] == ["AAA"]
    assert [r["status"] for r in manifest_records(env)] == ["failed"]


def test_a_failed_year_among_several_aborts_the_symbol_even_if_another_year_has_new_rows(env):
    seed(env, "AAA", {2025: [rec("Bonus 1:1", "10-Mar-2025")], 2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2025): [rec("Bonus 1:1", "10-Mar-2025"), rec("Interim Dividend - Rs 1 Per Share", "11-Dec-2025")],
                        ("AAA", 2026): ValueError("HTTP 500")}
    summary = ing.run_refresh(date(2025, 6, 1))
    assert (env.out / "AAA.parquet").read_bytes() == before
    assert summary["failed"] == ["AAA"]


def test_one_symbols_failure_does_not_stop_the_others(env):
    seed(env, "AAA", {2026: OLD_2026})
    seed(env, "BBB", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): ValueError("HTTP 403"),
                        ("BBB", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert summary["failed"] == ["AAA"] and list(summary["gained"]) == ["BBB"]
    assert len(pd.read_parquet(env.out / "BBB.parquet")) == 2


def test_symbols_without_a_file_are_ignored(env):
    env.fake.records = {("NEW", 2026): [rec("Bonus 1:1", "20-Oct-2026")]}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert not (env.out / "NEW.parquet").exists()
    assert env.fake.calls == [] and summary["requests"] == 0


def test_requests_are_symbols_times_years_in_range(env):
    seed(env, "AAA", {2026: OLD_2026})
    seed(env, "BBB", {2026: OLD_2026})
    summary = ing.run_refresh(date(2024, 5, 1))   # 2024, 2025, 2026
    assert sorted(env.fake.calls) == sorted((s, y) for s in ("AAA", "BBB") for y in (2024, 2025, 2026))
    assert summary["requests"] == 6
    assert ing.expected_refresh_requests(date(2024, 5, 1)) == 6


def test_only_year_chunks_covering_refresh_from_to_today_are_requested(env):
    seed(env, "AAA", {2026: OLD_2026})
    ing.run_refresh(date(2026, 9, 1))
    assert env.fake.calls == [("AAA", 2026)]


def test_refresh_from_in_the_future_is_rejected(env):
    with pytest.raises(ValueError):
        ing.run_refresh(date(2027, 1, 1))


def test_new_row_dated_before_refresh_from_is_not_added_and_is_warned(env):
    from loguru import logger
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "05-Feb-2026")]}   # earlier than refresh_from, absent from file
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        summary = ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert (env.out / "AAA.parquet").read_bytes() == before and summary["gained"] == {}
    assert any("AAA" in m and "before" in m for m in msgs)


def test_gained_symbols_are_logged_with_ex_date_and_type(env):
    from loguru import logger
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert any("AAA" in m and "2026-10-20" in m and "bonus" in m for m in msgs)


def test_manifest_records_ok_with_rows_for_gained_and_zero_for_unchanged(env):
    seed(env, "AAA", {2026: OLD_2026})
    seed(env, "BBB", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")], ("BBB", 2026): OLD_2026}
    ing.run_refresh(date(2026, 9, 1))
    by = {r["trade_date"]: r for r in manifest_records(env)}
    assert by["AAA"]["status"] == "ok" and by["AAA"]["rows"] == 1 and by["AAA"]["format_era"] == "refresh"
    assert by["BBB"]["status"] == "ok" and by["BBB"]["rows"] == 0


def test_rerun_is_idempotent(env):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    once = (env.out / "AAA.parquet").read_bytes()
    summary = ing.run_refresh(date(2026, 9, 1))
    assert (env.out / "AAA.parquet").read_bytes() == once and summary["gained"] == {}


def test_atomic_write_is_used_only_when_there_are_new_rows(env, monkeypatch):
    seed(env, "AAA", {2026: OLD_2026})
    writes = []
    real = ing.write_parquet_atomic
    monkeypatch.setattr(ing, "write_parquet_atomic", lambda df, path: (writes.append(path), real(df, path))[1])
    env.fake.records = {("AAA", 2026): OLD_2026}
    ing.run_refresh(date(2026, 9, 1))
    assert writes == []
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    assert len(writes) == 1


# ---- default mode unchanged / hygiene ----------------------------------------------------------------
def test_default_process_symbol_still_skips_an_existing_file(env):
    seed(env, "AAA", {2026: OLD_2026})
    assert ing.process_symbol("AAA", 2026, 2026) == {"symbol": "AAA", "status": "skipped"}
    assert env.fake.calls == []


def test_default_process_symbol_still_writes_a_new_symbol(env):
    env.fake.records = {("NEW", 2026): [rec("Bonus 1:1", "20-Oct-2026")]}
    r = ing.process_symbol("NEW", 2026, 2026)
    assert r["status"] == "ok" and (env.out / "NEW.parquet").exists()


def test_default_ingest_symbol_still_swallows_a_failed_year(env):
    env.fake.records = {("NEW", 2025): ValueError("HTTP 403"), ("NEW", 2026): [rec("Bonus 1:1", "20-Oct-2026")]}
    df = ing.ingest_symbol(object(), "NEW", 2025, 2026)
    assert len(df) == 1


def test_module_has_a_single_main_definition():
    assert len(re.findall(r"^def main\(", SOURCE, re.M)) == 1


def test_module_never_deletes():
    assert not re.search(r"os\.remove|os\.unlink|\.unlink\(|rmtree|os\.rmdir|\.rmdir\(", SOURCE)


def test_cli_rejects_bad_refresh_from_before_any_work():
    out = subprocess.run([sys.executable, "-m", "data.ingest_corporate_actions", "--refresh-from", "2026-13-40"],
                         capture_output=True, text=True, timeout=120, cwd=str(Path(ing.__file__).resolve().parent.parent))
    assert out.returncode == 2 and "refresh-from" in out.stderr


def test_cli_help_mentions_refresh_from():
    out = subprocess.run([sys.executable, "-m", "data.ingest_corporate_actions", "--help"], capture_output=True, text=True,
                         timeout=120, cwd=str(Path(ing.__file__).resolve().parent.parent))
    assert out.returncode == 0 and "--refresh-from" in out.stdout


# ---- review round 2: reworded duplicates, missing-from-fetch warning, write failure, dtype pin ----------------
def _capture(level="WARNING"):
    from loguru import logger
    msgs = []
    return msgs, logger.add(lambda m: msgs.append(str(m)), level=level), logger


def test_reworded_factor_bearing_duplicate_is_not_appended_and_is_warned(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): [rec("Bonus Issue 1:1", "05-Oct-2026")]}   # same ex_date/type/factor, new wording
    msgs, sink, logger = _capture()
    try:
        summary = ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert (env.out / "AAA.parquet").read_bytes() == before and summary["gained"] == {}
    assert any("possible reworded duplicate" in m and "Bonus 1:1" in m and "Bonus Issue 1:1" in m for m in msgs)


def test_a_different_factor_on_the_same_day_is_appended(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Bonus 2:1", "05-Oct-2026")]}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert summary["gained"]["AAA"] == [("2026-10-05", "bonus")]
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 2


def test_two_different_dividends_on_one_day_are_both_kept(env):
    # no adjustment factor: the reworded-duplicate rule must not swallow a genuinely separate dividend
    seed(env, "AAA", {2026: [rec("Interim Dividend - Rs 6 Per Share", "05-Oct-2026")]})
    env.fake.records = {("AAA", 2026): [rec("Interim Dividend - Rs 6 Per Share", "05-Oct-2026"),
                                        rec("Interim Dividend - Rs 3 Per Share", "05-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 2


def test_existing_action_missing_from_fetch_warns_but_does_not_fail(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): []}   # an empty 200 / rate-limited answer
    msgs, sink, logger = _capture()
    try:
        summary = ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert summary["failed"] == [] and (env.out / "AAA.parquet").read_bytes() == before
    assert any("existing action missing from fetch" in m and "AAA" in m for m in msgs)


def test_missing_from_fetch_warns_when_fetch_has_other_rows(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    env.fake.records = {("AAA", 2026): [rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")]}
    msgs, sink, logger = _capture()
    try:
        ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert any("existing action missing from fetch" in m for m in msgs)
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 2


def test_no_missing_warning_when_everything_reappears_or_is_before_refresh_from(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Feb-2026"), rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")]})
    env.fake.records = {("AAA", 2026): [rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")]}   # Feb row is before refresh_from
    msgs, sink, logger = _capture()
    try:
        ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert not any("missing from fetch" in m for m in msgs)


def test_reworded_existing_row_is_not_reported_missing(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    env.fake.records = {("AAA", 2026): [rec("Bonus Issue 1:1", "05-Oct-2026")]}
    msgs, sink, logger = _capture()
    try:
        ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert not any("missing from fetch" in m for m in msgs)


def test_write_failure_marks_that_symbol_failed_and_the_run_continues(env, monkeypatch):
    seed(env, "AAA", {2026: OLD_2026})
    seed(env, "BBB", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")],
                        ("BBB", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    real = ing.write_parquet_atomic

    def flaky(df, path):
        if Path(path).stem == "AAA":
            raise OSError("disk full")
        return real(df, path)

    monkeypatch.setattr(ing, "write_parquet_atomic", flaky)
    summary = ing.run_refresh(date(2026, 9, 1))
    assert summary["failed"] == ["AAA"] and list(summary["gained"]) == ["BBB"]
    assert (env.out / "AAA.parquet").read_bytes() == before
    by = {r["trade_date"]: r for r in manifest_records(env)}   # manifest flushed despite the failure
    assert by["AAA"]["status"] == "failed" and "disk full" in by["AAA"]["error"] and by["BBB"]["status"] == "ok"


def test_cli_exit_code_is_1_when_a_write_fails(env, monkeypatch):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")]}
    monkeypatch.setattr(ing, "write_parquet_atomic", lambda df, path: (_ for _ in ()).throw(OSError("disk full")))
    assert ing.cli(["--refresh-from", "2026-09-01"]) == 1


def _column_hashes(frame):
    return {c: pd.util.hash_pandas_object(frame[c], index=False).tolist() for c in frame.columns}


def test_dtypes_and_values_of_existing_rows_are_pinned_through_a_merge_with_an_all_none_row(env):
    # the added row has record_date "-" (None) and no adjustment factor: the classic way concat silently turns datetimes to object
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026"), rec("Interim Dividend - Rs 6 Per Share", "06-Oct-2026")]})
    old = pd.read_parquet(env.out / "AAA.parquet")
    odd = rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")
    odd["recDate"] = "-"
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Interim Dividend - Rs 6 Per Share", "06-Oct-2026"), odd]}
    ing.run_refresh(date(2026, 9, 1))
    new = pd.read_parquet(env.out / "AAA.parquet")
    assert len(new) == 3
    head = new.iloc[:len(old)].reset_index(drop=True)
    assert head.dtypes.to_dict() == old.dtypes.to_dict()                    # no dtype drift on any column
    assert new.dtypes.to_dict() == old.dtypes.to_dict()
    assert _column_hashes(head) == _column_hashes(old)                      # every existing value identical, column by column
    assert list(new.columns) == list(old.columns)


def test_append_preserves_existing_rows_strictly(env):
    old = seed(env, "AAA", {2025: [rec("Bonus 1:1", "10-Mar-2025")], 2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 2:1", "15-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    new = pd.read_parquet(env.out / "AAA.parquet")
    kept = new[new["raw_subject"].isin(old["raw_subject"])].reset_index(drop=True)
    stored_old = old.sort_values("ex_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, stored_old, check_dtype=True)


# ---- NOT-added detail: ex_date, type, factor ----------------------------------------------------------------
def _run_capture(env, refresh_from=date(2026, 9, 1)):
    from loguru import logger
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        summary = ing.run_refresh(refresh_from)
    finally:
        logger.remove(sink)
    return summary, msgs


def test_not_added_warning_lists_ex_date_type_and_factor(env):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "05-Feb-2026"),
                                                    rec("Interim Dividend - Rs 3 Per Share", "06-Feb-2026")]}
    summary, msgs = _run_capture(env)
    line = next(m for m in msgs if "NOT added" in m and "AAA" in m)
    assert "'2026-02-05 bonus factor=0.5'" in line
    assert "'2026-02-06 dividend'" in line and "dividend factor" not in line   # no factor text when the factor is null


def test_summary_counts_skipped_by_type_and_lists_those_with_a_factor(env):
    seed(env, "AAA", {2026: OLD_2026})
    seed(env, "BBB", {2026: OLD_2026})
    env.fake.records = {
        ("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "05-Feb-2026"), rec("Interim Dividend - Rs 3 Per Share", "06-Feb-2026")],
        ("BBB", 2026): OLD_2026 + [rec("Interim Dividend - Rs 2 Per Share", "07-Feb-2026")],
    }
    summary, msgs = _run_capture(env)
    assert summary["skipped_by_type"] == {"bonus": 1, "dividend": 2}
    assert summary["skipped_with_factor"] == [("AAA", "2026-02-05", "bonus", 0.5)]
    final = [m for m in msgs if "skipped" in m and "refresh_from" in m or "pre-refresh" in m]
    text = " ".join(final)
    assert "bonus" in text and "dividend" in text and "AAA" in text and "2026-02-05" in text


def test_summary_with_nothing_skipped_is_empty_and_says_so(env):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026}
    summary, msgs = _run_capture(env)
    assert summary["skipped_by_type"] == {} and summary["skipped_with_factor"] == []
    assert any("pre-refresh" in m and "0" in m for m in msgs)


def test_skipped_rows_are_still_not_added(env):
    seed(env, "AAA", {2026: OLD_2026})
    before = (env.out / "AAA.parquet").read_bytes()
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "05-Feb-2026")]}
    _run_capture(env)
    assert (env.out / "AAA.parquet").read_bytes() == before


# ---- --symbols restricts the refresh ---------------------------------------------------------------------------
def test_symbols_restricts_the_refresh_to_the_named_files(env):
    for sym in ("AAA", "BBB", "CCC"):
        seed(env, sym, {2026: OLD_2026})
    env.fake.records = {(s, 2026): OLD_2026 + [rec("Bonus 1:1", "20-Oct-2026")] for s in ("AAA", "BBB", "CCC")}
    summary = ing.run_refresh(date(2026, 9, 1), symbols=["BBB"])
    assert env.fake.calls == [("BBB", 2026)]
    assert list(summary["gained"]) == ["BBB"] and summary["requests"] == 1
    assert len(pd.read_parquet(env.out / "AAA.parquet")) == 1 and len(pd.read_parquet(env.out / "BBB.parquet")) == 2


def test_symbols_requests_are_named_symbols_times_years(env):
    for sym in ("AAA", "BBB", "CCC"):
        seed(env, sym, {2026: OLD_2026})
    assert ing.expected_refresh_requests(date(2025, 1, 1), symbols=["AAA", "CCC"]) == 4
    assert ing.expected_refresh_requests(date(2025, 1, 1)) == 6


def test_duplicate_symbol_names_are_refreshed_once(env):
    seed(env, "AAA", {2026: OLD_2026})
    ing.run_refresh(date(2026, 9, 1), symbols=["AAA", "AAA"])
    assert env.fake.calls == [("AAA", 2026)]


def test_a_named_symbol_without_a_file_is_an_error_before_any_fetch(env):
    seed(env, "AAA", {2026: OLD_2026})
    with pytest.raises(ValueError, match="NOPE"):
        ing.run_refresh(date(2026, 9, 1), symbols=["AAA", "NOPE"])
    assert env.fake.calls == []


def test_cli_symbols_without_a_file_exits_2_and_fetches_nothing(env):
    seed(env, "AAA", {2026: OLD_2026})
    with pytest.raises(SystemExit) as exc:
        ing.cli(["--refresh-from", "2026-09-01", "--symbols", "AAA", "NOPE"])
    assert exc.value.code == 2 and env.fake.calls == []


def test_cli_symbols_requires_refresh_from(env):
    with pytest.raises(SystemExit) as exc:
        ing.cli(["--symbols", "AAA"])
    assert exc.value.code == 2


def test_cli_symbols_accepts_several_and_runs_only_them(env):
    for sym in ("AAA", "BBB", "CCC"):
        seed(env, sym, {2026: OLD_2026})
    assert ing.cli(["--refresh-from", "2026-09-01", "--symbols", "AAA", "CCC"]) == 0
    assert sorted(env.fake.calls) == [("AAA", 2026), ("CCC", 2026)]


def test_cli_help_mentions_symbols():
    out = subprocess.run([sys.executable, "-m", "data.ingest_corporate_actions", "--help"], capture_output=True, text=True,
                         timeout=120, cwd=str(Path(ing.__file__).resolve().parent.parent))
    assert out.returncode == 0 and "--symbols" in out.stdout


# ---- review round 3 (Round 1 fixes): reworded duplicates inside ONE fetch ----------------------------------
def test_two_reworded_bonus_rows_new_in_one_fetch_append_only_one_and_warn(env):
    seed(env, "AAA", {2026: OLD_2026})
    env.fake.records = {("AAA", 2026): OLD_2026 + [rec("Bonus 1:1", "05-Oct-2026"), rec("Bonus Issue 1:1", "05-Oct-2026")]}
    msgs, sink, logger = _capture()
    try:
        summary = ing.run_refresh(date(2026, 9, 1))
    finally:
        logger.remove(sink)
    assert summary["gained"]["AAA"] == [("2026-10-05", "bonus")]
    stored = pd.read_parquet(env.out / "AAA.parquet")
    assert int((stored["action_type"] == "bonus").sum()) == 1
    assert sum("possible reworded duplicate NOT added" in m for m in msgs) == 1


def test_fetch_failed_in_default_ingest_is_a_log_warning_not_a_print(env, capsys):
    env.fake.records = {("AAA", 2026): RuntimeError("x")}
    import requests
    env.fake.records = {("AAA", 2026): requests.RequestException("boom")}
    msgs, sink, logger = _capture()
    try:
        ing.ingest_symbol(object(), "AAA", 2026, 2026)
    finally:
        logger.remove(sink)
    assert any("FETCH FAILED" in m and "boom" in m for m in msgs)
    assert "FETCH FAILED" not in capsys.readouterr().out


# ---- R20-2 (b): the refresh keeps each file's stored Arrow schema --------------------------------------------
import pyarrow as pa
import pyarrow.parquet as pq


def _schema(path):
    return pq.read_schema(path)


def _retype(path, **types):
    """Rewrite a stored file with some columns cast to the given Arrow types (an old file written by another pandas)."""
    table = pq.read_table(path)
    for name, typ in types.items():
        i = table.schema.get_field_index(name)
        table = table.set_column(i, name, table.column(name).cast(typ))
    pq.write_table(table, path, compression="snappy")


def test_b_T9_typed_columns_keep_their_exact_stored_type(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    path = env.out / "AAA.parquet"
    _retype(path, isin=pa.string())            # stored as plain string; pandas inference would say large_string
    before = _schema(path)
    assert before.field("adjustment_factor").type == pa.float64() and before.field("isin").type == pa.string()
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    after = _schema(path)
    assert len(pd.read_parquet(path)) == 2
    assert after.field("isin").type == pa.string()                              # exact stored type kept
    assert after.field("adjustment_factor").type == pa.float64()                # added row is all-None there
    assert after.names == before.names


def test_b_T10_null_typed_column_stays_null_when_added_rows_are_null_there(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    path = env.out / "AAA.parquet"
    assert _schema(path).field("bc_start_date").type == pa.null()
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Bonus 2:1", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    assert len(pd.read_parquet(path)) == 2 and _schema(path).field("bc_start_date").type == pa.null()


def test_b_T11_null_typed_column_takes_the_type_of_an_added_value_history_stays_null(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    path = env.out / "AAA.parquet"
    assert _schema(path).field("dividend_amount").type == pa.null()
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Interim Dividend - Rs 3 Per Share", "20-Oct-2026")]}
    ing.run_refresh(date(2026, 9, 1))
    df = pd.read_parquet(path)
    assert _schema(path).field("dividend_amount").type == pa.float64()
    assert df["dividend_amount"].isna().tolist() == [True, False] and df["dividend_amount"].iloc[1] == 3.0


def test_b_T12_uncastable_added_value_fails_the_symbol_and_leaves_the_file_untouched(env):
    seed(env, "AAA", {2026: [rec("Bonus 1:1", "05-Oct-2026")]})
    path = env.out / "AAA.parquet"
    _retype(path, dividend_amount=pa.int32())      # stored typed; a fractional dividend cannot be cast without loss
    before = path.read_bytes()
    env.fake.records = {("AAA", 2026): [rec("Bonus 1:1", "05-Oct-2026"), rec("Interim Dividend - Rs 3.5 Per Share", "20-Oct-2026")]}
    summary = ing.run_refresh(date(2026, 9, 1))
    assert summary["failed"] == ["AAA"] and summary["gained"] == {}
    assert path.read_bytes() == before and not list(env.out.glob("*.tmp"))


# ---- R20-2 (d): one process-wide NSE request rate -------------------------------------------------------------
import threading


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.sleeps.append(d)
        self.t += d


class _Resp:
    status_code = 200

    def json(self):
        return []


class _Session:
    def get(self, *a, **k):
        return _Resp()


def test_d_T13_four_threads_never_start_two_requests_closer_than_request_delay(monkeypatch):
    clock, starts = FakeClock(), []
    monkeypatch.setattr(ing, "REQUEST_DELAY", 0.6)
    monkeypatch.setattr(ing, "_LIMITER", ing.RateLimiter(clock=clock, sleep=clock.sleep, on_start=lambda t: starts.append(t)))
    session = _Session()
    threads = [threading.Thread(target=lambda: [ing.fetch_symbol_year(session, "AAA", 2026) for _ in range(3)]) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    starts.sort()
    assert len(starts) == 12
    assert all(b - a >= 0.6 - 1e-9 for a, b in zip(starts, starts[1:]))


def test_d_T14_a_single_thread_waits_only_the_remainder(monkeypatch):
    clock = FakeClock()
    lim = ing.RateLimiter(clock=clock, sleep=clock.sleep)
    lim.wait(0.6)                      # first request: no wait
    assert clock.sleeps == []
    clock.t += 0.2                     # 0.2 s of other work
    lim.wait(0.6)
    assert clock.sleeps == [pytest.approx(0.4)]          # the remainder, not a full extra delay
    clock.t += 5.0                     # long idle: no wait at all
    lim.wait(0.6)
    assert len(clock.sleeps) == 1


def test_d_session_creation_goes_through_the_limiter_too(monkeypatch):
    calls = []
    monkeypatch.setattr(ing, "_LIMITER", ing.RateLimiter(clock=FakeClock(), sleep=lambda d: None, on_start=calls.append))
    monkeypatch.setattr(ing.requests.Session, "get", lambda self, *a, **k: _Resp())
    ing.get_session()
    assert len(calls) == 1


def test_d_ingest_symbol_has_no_per_worker_sleep_any_more(env, monkeypatch):
    slept = []
    monkeypatch.setattr(ing.time, "sleep", lambda d: slept.append(d))
    env.fake.records = {("AAA", 2025): [rec("Bonus 1:1", "05-Mar-2025")], ("AAA", 2026): requests_error()}
    ing.ingest_symbol(object(), "AAA", 2025, 2026)           # one good year, one failed year: both paths used to sleep
    assert slept == []


def requests_error():
    import requests
    return requests.RequestException("boom")


def test_d_refresh_time_estimate_is_requests_times_delay_without_dividing_by_workers(env, monkeypatch):
    for sym in ("AAA", "BBB", "CCC", "DDD"):
        seed(env, sym, {2026: OLD_2026})
    monkeypatch.setattr(ing, "REQUEST_DELAY", 60)
    monkeypatch.setattr(ing, "_LIMITER", ing.RateLimiter(clock=FakeClock(), sleep=lambda d: None))
    env.fake.records = {(s, 2026): OLD_2026 for s in ("AAA", "BBB", "CCC", "DDD")}
    _, msgs = _run_capture(env)
    assert any("est. 4 min" in m for m in msgs)
