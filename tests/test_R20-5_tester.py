"""TESTER edge cases for R20-5 (wrapper manifest check). No network, no subprocess."""
import pytest

from scripts import run_data_refresh as rdr
from tests.test_run_data_refresh import (TODAY, capture, fake, sandbox, stale, write_manifest)  # noqa: F401


def test_failed_chunk_that_ends_today_but_covers_earlier_days_still_fails(fake, sandbox):
    # item 3 must not hide a failure of earlier dates: the chunk starts well before today
    write_manifest(sandbox, "announcements", [("2026-10-25_2026-11-10", "failed", "timeout")])
    assert rdr.main(["--only", "announcements"]) != 0


def test_failed_chunk_that_ends_after_today_but_starts_in_the_past_still_fails(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-11-05_2026-11-12", "failed", "timeout")])
    assert rdr.main(["--only", "announcements"]) != 0


def test_failed_chunk_starting_today_is_not_checked(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-11-10_2026-11-12", "failed", "x")])
    assert rdr.main(["--only", "announcements"]) == 0


def test_invalid_calendar_date_in_a_failed_key_does_not_crash_the_wrapper(fake, sandbox):
    write_manifest(sandbox, "bhavdata", [("2026-13-45", "failed", "boom")])
    try:
        rc = rdr.main(["--only", "bhavdata"])
    except ValueError as e:                                       # a traceback is not a clean wrapper exit
        pytest.fail(f"wrapper crashed on a malformed key: {e}")
    assert isinstance(rc, int)


def test_stale_error_for_a_range_key_is_never_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "announcements", [("2026-10-02_2026-10-05", "failed", stale("2026-10-02", "2026-10-01"))])
    assert rdr.main(["--only", "announcements", "--start", "2026-10-01"]) != 0


def test_stale_file_date_equal_to_requested_is_not_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-02"))])
    assert rdr.main(["--only", "bhavdata", "--start", "2026-10-01"]) != 0


def test_stale_file_dated_later_than_requested_is_not_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-05"))])
    assert rdr.main(["--only", "bhavdata", "--start", "2026-10-01"]) != 0


def test_stale_error_naming_a_different_requested_date_is_not_excused(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-03", "2026-10-01"))])
    assert rdr.main(["--only", "bhavdata", "--start", "2026-10-01"]) != 0


def test_range_key_with_invalid_first_date_fails_cleanly_and_is_never_pending(fake, sandbox):
    write_manifest(sandbox, "announcements", [("2026-02-30_2026-11-12", "failed", "x")])
    assert rdr.main(["--only", "announcements"]) not in (0, None)


def test_valid_single_day_key_equal_today_with_invalid_sibling_still_fails(fake, sandbox):
    # today's key stays pending, the malformed sibling must still fail the step
    write_manifest(sandbox, "bhavdata", [(TODAY.isoformat(), "failed", "x"), ("2026-13-45", "failed", "x")])
    assert rdr.main(["--only", "bhavdata"]) != 0


def test_excused_stale_and_malformed_key_together_still_fail(fake, sandbox):
    write_manifest(sandbox, "bhavcopy", [("2026-10-02", "holiday")])
    write_manifest(sandbox, "bhavdata", [("2026-10-02", "failed", stale("2026-10-02", "2026-10-01")),
                                          ("2026-13-45", "failed", "x")])
    assert rdr.main(["--only", "bhavdata", "--start", "2026-10-01"]) != 0
