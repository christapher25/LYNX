"""TESTER edge cases for R20-10 (announcements range-chunk handling). Fake children, tmp dirs; no network."""
from datetime import date

from scripts import run_data_refresh as rdr
from tests.test_run_data_refresh import (ANN, OVERLAP, ann_calls, capture, chunk, fake, run_ann, sandbox,  # noqa: F401
                                         today12, write_manifest)


def test_chunk_ending_today_means_already_current(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-12")
    rc, text = run_ann()
    assert rc == 0 and ann_calls(fake) == [] and "already current" in text


def test_reversed_chunk_name_is_ignored_with_a_warning_and_does_not_set_E(fake, sandbox, today12):
    chunk(sandbox, "2026-11-20", "2026-11-01")                        # reversed
    chunk(sandbox, "2026-10-01", "2026-11-05")
    rc, text = run_ann(start="2026-11-03")
    assert rc == 0
    call = ann_calls(fake)[0]
    assert call[call.index("--start") + 1] == "2026-11-06"


def test_start_inside_a_hole_before_E_moves_to_E_plus_1_as_briefed(fake, sandbox, today12):
    # pins the briefed behaviour: the days 10-20..10-31 have no chunk and are NOT re-requested (see the report, finding 1)
    chunk(sandbox, "2026-10-01", "2026-10-10")
    chunk(sandbox, "2026-11-01", "2026-11-09")
    rc, _ = run_ann(start="2026-10-20")
    call = ann_calls(fake)[0]
    assert call[call.index("--start") + 1] == "2026-11-10"


def test_same_day_range_key_covered_by_a_chunk_is_excused(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-10_2026-11-10", "failed", OVERLAP)])
    assert run_ann()[0] == 0


def test_range_key_with_reversed_dates_is_not_excused(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-11_2026-11-10", "failed", OVERLAP)])
    assert run_ann()[0] != 0


def test_range_key_with_invalid_calendar_date_is_not_excused_and_does_not_crash(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-05_2026-13-45", "failed", OVERLAP)])
    assert run_ann()[0] != 0


def test_leading_whitespace_in_the_error_text_is_not_excused(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-09_2026-11-11", "failed", " " + OVERLAP)])
    assert run_ann()[0] != 0


def test_excuse_for_one_key_does_not_hide_another_real_failure(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-09_2026-11-11", "failed", OVERLAP), ("2026-11-10", "failed", "HTTP 503")])
    assert run_ann()[0] != 0


def test_overlap_excuse_needs_the_chunk_to_cover_the_whole_range_not_just_its_ends(fake, sandbox, today12):
    chunk(sandbox, "2026-11-01", "2026-11-05")
    chunk(sandbox, "2026-11-09", "2026-11-11")
    write_manifest(sandbox, ANN, [("2026-11-04_2026-11-10", "failed", OVERLAP)])
    assert run_ann()[0] != 0
