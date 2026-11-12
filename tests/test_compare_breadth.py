"""R20-8: tests for scripts/compare_breadth.py (synthetic parquet files only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import compare_breadth as cb

COLS = ["date", "universe_pct_above_200dma", "universe_n_evaluable", "is_missing_breadth"]


def frame(rows):
    """rows: (date, pct, n, missing)"""
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    df["universe_pct_above_200dma"] = df["universe_pct_above_200dma"].astype(float)
    df["universe_n_evaluable"] = df["universe_n_evaluable"].astype(float)
    df["is_missing_breadth"] = df["is_missing_breadth"].astype(bool)
    return df


BASE = [
    ("2024-01-02", 55.0, 1500, False),
    ("2024-01-03", 56.5, 1501, False),
    ("2024-01-04", 57.0, 1502, False),
    ("2024-01-05", np.nan, np.nan, True),      # the old last date: missing-flagged
]


def run(tmp_path, old, new, *extra):
    po, pn, out = tmp_path / "old.parquet", tmp_path / "new.parquet", tmp_path / "out" / "cmp.md"
    old.to_parquet(po)
    new.to_parquet(pn)
    code = cb.main(["--old", str(po), "--new", str(pn), "--out", str(out), *extra])
    return code, out


def text(out):
    return out.read_text(encoding="utf-8")


def csv(out):
    return pd.read_csv(out.with_suffix(".csv"))


# T1 -------------------------------------------------------------------
def test_t1_identical_files(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE))
    assert code == 0
    assert text(out).splitlines()[0] == "HISTORY DIFFERENCES: 0"
    assert len(csv(out)) == 0


# T2 -------------------------------------------------------------------
def test_t2_extension_and_old_last_row_gaining_a_value_are_not_history(tmp_path):
    new = frame(BASE[:3] + [("2024-01-05", 58.0, 1503, False), ("2024-01-08", 59.0, 1504, False)])
    code, out = run(tmp_path, frame(BASE), new)
    assert code == 0
    t = text(out)
    assert t.splitlines()[0] == "HISTORY DIFFERENCES: 0"
    rows = csv(out)
    assert sorted(rows["date"]) == ["2024-01-05", "2024-01-08"]
    assert set(rows["class"]) == {"changed", "added"}
    assert "2024-01-05" in t and "2024-01-08" in t


# T3 -------------------------------------------------------------------
def test_t3_changed_value_before_old_last_date_stops(tmp_path):
    new = frame(BASE)
    new.loc[1, "universe_pct_above_200dma"] = 60.25
    code, out = run(tmp_path, frame(BASE), new)
    assert code == 1
    t = text(out)
    assert t.splitlines()[0] == "HISTORY DIFFERENCES: 1 (STOP)"
    assert "2024-01-03" in t and "56.5" in t and "60.25" in t
    row = csv(out).iloc[0]
    assert row["date"] == "2024-01-03" and row["class"] == "changed" and row["before_old_last_date"]


# T4 / T5 --------------------------------------------------------------
def test_t4_added_date_before_old_last_date_stops(tmp_path):
    old = frame(BASE[:2] + BASE[3:])                                # no 2024-01-04
    code, out = run(tmp_path, old, frame(BASE))                     # new has it: added before the old last date
    assert code == 1
    row = csv(out).iloc[0]
    assert row["class"] == "added" and row["date"] == "2024-01-04"


def test_t4_added_date_after_old_last_date_is_fine(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE + [("2024-01-06", 50.0, 1400, False)]))   # a weekend session
    assert code == 0
    assert csv(out).iloc[0]["class"] == "added"


def test_t5_removed_date_stops(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE[:1] + BASE[2:]))
    assert code == 1
    row = csv(out).iloc[0]
    assert row["class"] == "removed" and row["date"] == "2024-01-03"


def test_removed_old_last_date_is_not_a_history_difference(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE[:3]))
    assert code == 0
    assert set(csv(out)["class"]) == {"removed"}


# T6 -------------------------------------------------------------------
def test_t6_nan_equals_nan(tmp_path):
    code, _ = run(tmp_path, frame(BASE), frame(BASE))
    assert code == 0
    mid = frame(BASE[:2] + [("2024-01-04", np.nan, np.nan, True)] + BASE[3:])      # NaN row BEFORE the old last date
    code, out = run(tmp_path, mid, mid.copy())
    assert code == 0 and csv(out).empty


def test_t6_nan_to_value_is_changed_and_flag_flip_alone_is_changed(tmp_path):
    old = frame(BASE[:2] + [("2024-01-04", np.nan, np.nan, True)] + BASE[3:])
    new = frame(BASE)
    code, out = run(tmp_path, old, new)
    assert code == 1
    assert csv(out).iloc[0]["date"] == "2024-01-04"
    old2, new2 = frame(BASE), frame(BASE)
    new2.loc[0, "is_missing_breadth"] = True                       # only the flag differs
    code, out = run(tmp_path, old2, new2)
    assert code == 1
    assert "is_missing_breadth" in csv(out).iloc[0]["columns_differing"]


# T7 -------------------------------------------------------------------
def test_t7_tiny_float_difference_is_changed_but_not_counted_as_material(tmp_path):
    new = frame(BASE)
    new.loc[0, "universe_pct_above_200dma"] = 55.0 + 1e-12
    code, out = run(tmp_path, frame(BASE), new)
    assert code == 1                                               # exact comparison decides
    t = text(out)
    assert "changed dates with abs pct difference > 1e-9: 0" in t
    assert "maximum abs pct difference" in t


# T8 -------------------------------------------------------------------
def test_t8_duplicate_date_exits_2(tmp_path):
    dup = frame(BASE + [("2024-01-04", 1.0, 1.0, False)])
    code, out = run(tmp_path, frame(BASE), dup)
    assert code == 2 and not out.exists()
    code, out = run(tmp_path, dup, frame(BASE))
    assert code == 2


# T9 -------------------------------------------------------------------
def test_t9_writes_only_out_and_csv(tmp_path):
    before_snapshot = None
    po, pn = tmp_path / "old.parquet", tmp_path / "new.parquet"
    frame(BASE).to_parquet(po)
    frame(BASE).to_parquet(pn)
    snap = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "res" / "r.md"
    assert cb.main(["--old", str(po), "--new", str(pn), "--out", str(out)]) == 0
    now = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert now - set(snap) == {out, out.with_suffix(".csv")}
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == snap[p] for p in snap)


def test_report_gives_old_last_dates_counts_and_year_split(tmp_path):
    new = frame(BASE[:3] + [("2024-01-05", 58.0, 1503, False)])
    new.loc[0, "universe_n_evaluable"] = 1499.0
    code, out = run(tmp_path, frame(BASE), new)
    t = text(out)
    assert "old last date: 2024-01-05" in t
    assert "old last non-missing date: 2024-01-04" in t
    assert "2024: 1" in t                                          # counts by year (history differences)
    assert "same: 2" in t


def test_dates_compare_across_datetime_units(tmp_path):
    old = frame(BASE)
    new = frame(BASE)
    old["date"] = old["date"].astype("datetime64[ms]")
    new["date"] = new["date"].astype("datetime64[us]")
    code, _ = run(tmp_path, old, new)
    assert code == 0


def test_source_conventions():
    src = open(cb.__file__, encoding="utf-8").read()
    assert "print(" not in src and "isclose" not in src and "unlink" not in src


# Round 1 (tester): the STOP gate must not fail open --------------------------------------------------------------
def test_round1_nat_date_row_in_new_file_exits_2_and_writes_nothing(tmp_path):
    new = frame(BASE)
    new.loc[len(new)] = [pd.NaT, 1.0, 1.0, False]
    code, out = run(tmp_path, frame(BASE), new)
    assert code == 2 and not out.exists() and not out.with_suffix(".csv").exists()


def test_round1_nat_date_row_in_old_file_exits_2(tmp_path):
    old = frame(BASE)
    old.loc[len(old)] = [pd.NaT, 1.0, 1.0, False]
    code, out = run(tmp_path, old, frame(BASE))
    assert code == 2 and not out.exists()


def test_round1_empty_old_file_exits_2_not_a_silent_pass(tmp_path):
    code, out = run(tmp_path, frame(BASE).iloc[0:0], frame(BASE))
    assert code == 2 and not out.exists()


def test_round1_empty_new_file_is_every_old_date_removed_and_stops(tmp_path):
    code, out = run(tmp_path, frame(BASE), frame(BASE).iloc[0:0])
    assert code == 1
    assert text(out).splitlines()[0].startswith("HISTORY DIFFERENCES: 3 (STOP)")
