"""TESTER edge cases for R20-8 (breadth comparison). Synthetic parquet only."""
import numpy as np
import pandas as pd
import pytest

from scripts import compare_breadth as cb
from tests.test_compare_breadth import BASE, COLS, frame, run, text


def test_empty_old_file_is_not_a_silent_pass(tmp_path):
    code, _ = run(tmp_path, frame([]), frame(BASE))
    assert code != 0                       # an old file with no rows has no "old last date": must not report 0 differences


def test_nat_date_row_in_new_file_does_not_hide_from_the_stop_rule(tmp_path):
    new = frame(BASE + [("2024-01-08", 58.0, 1503, False)])
    new.loc[len(new)] = [pd.NaT, 1.0, 1.0, False]
    code, _ = run(tmp_path, frame(BASE), new)
    assert code != 0                       # a NaT-dated row is neither "before" nor "after": it must not pass silently


def test_missing_required_column_exits_2(tmp_path):
    bad = frame(BASE).drop(columns=["universe_n_evaluable"])
    assert run(tmp_path, frame(BASE), bad)[0] == 2


def test_unreadable_old_file_exits_2(tmp_path):
    p = tmp_path / "junk.parquet"; p.write_bytes(b"not parquet")
    n = tmp_path / "new.parquet"; frame(BASE).to_parquet(n)
    assert cb.main(["--old", str(p), "--new", str(n), "--out", str(tmp_path / "o.md")]) == 2
    assert not (tmp_path / "o.md").exists()


def test_new_file_shorter_than_old_drops_old_last_row_only_is_not_a_stop(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE[:-1]))
    assert code == 0 and "HISTORY DIFFERENCES: 0" in text(out).splitlines()[0]


def test_change_on_the_day_before_old_last_date_is_a_stop(tmp_path):
    new = frame(BASE); new.loc[2, "universe_pct_above_200dma"] = 57.0000001
    code, out = run(tmp_path, frame(BASE), new)
    assert code == 1 and "(STOP)" in text(out).splitlines()[0] and "2024-01-04" in text(out)


def test_inf_equals_inf_and_inf_to_value_is_changed(tmp_path):
    old = frame(BASE); old.loc[0, "universe_pct_above_200dma"] = np.inf
    assert run(tmp_path, old, old.copy())[0] == 0
    new = old.copy(); new.loc[0, "universe_pct_above_200dma"] = 55.0
    assert run(tmp_path, old, new)[0] == 1


def test_n_evaluable_int_vs_float_same_value_is_not_a_difference(tmp_path):
    new = frame(BASE); new["universe_n_evaluable"] = new["universe_n_evaluable"].astype("float64")
    old = frame(BASE).copy(); old["universe_n_evaluable"] = old["universe_n_evaluable"].fillna(0).astype("int64")
    old.loc[3, "universe_n_evaluable"] = 0
    new.loc[3, "universe_n_evaluable"] = 0.0
    assert run(tmp_path, old, new)[0] == 0


def test_timestamps_with_time_of_day_still_match_by_value(tmp_path):
    old = frame(BASE); new = frame(BASE)
    old["date"] = old["date"].astype("datetime64[ms]")
    assert run(tmp_path, old, new)[0] == 0
