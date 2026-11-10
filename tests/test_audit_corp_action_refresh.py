"""Tests for scripts/audit_corp_action_refresh.py (read-only audit of a corporate-actions refresh)."""
import hashlib
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import audit_corp_action_refresh as audit

SINCE = date(2026, 9, 30)


def write(directory, symbol, rows):
    cols = ["symbol", "ex_date", "raw_subject", "action_type", "adjustment_factor"]
    frame = pd.DataFrame([(symbol, pd.Timestamp(d), subj, t, f) for d, subj, t, f in rows], columns=cols)
    frame.to_parquet(directory / f"{symbol}.parquet")


def test_reworded_duplicate_with_a_factor_is_reported(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "Bonus 1:1", "bonus", 0.5), ("2026-10-05", "Bonus Issue 1:1", "bonus", 0.5)])
    dups = audit.find_possible_duplicates(tmp_path, SINCE)
    assert len(dups) == 1 and dups[0]["symbol"] == "AAA" and set(dups[0]["subjects"]) == {"Bonus 1:1", "Bonus Issue 1:1"}


def test_different_factor_or_type_or_date_is_not_a_duplicate(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "Bonus 1:1", "bonus", 0.5), ("2026-10-05", "Bonus 2:1", "bonus", 1 / 3)])
    write(tmp_path, "BBB", [("2026-10-05", "Bonus 1:1", "bonus", 0.5), ("2026-10-06", "Bonus 1:1 again", "bonus", 0.5)])
    write(tmp_path, "CCC", [("2026-10-05", "Split", "split", 0.5), ("2026-10-05", "Bonus 1:1", "bonus", 0.5)])
    assert audit.find_possible_duplicates(tmp_path, SINCE) == []


def test_null_factor_rows_are_never_duplicates(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "Div 6", "dividend", np.nan), ("2026-10-05", "Div 3", "dividend", np.nan)])
    assert audit.find_possible_duplicates(tmp_path, SINCE) == []


def test_rows_before_since_are_ignored(tmp_path):
    write(tmp_path, "AAA", [("2026-09-29", "Bonus 1:1", "bonus", 0.5), ("2026-09-29", "Bonus Issue 1:1", "bonus", 0.5)])
    assert audit.find_possible_duplicates(tmp_path, SINCE) == []


def test_factor_rounding_matches_near_equal_factors(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "a", "split", 0.5), ("2026-10-05", "b", "split", 0.5 + 1e-15)])
    assert len(audit.find_possible_duplicates(tmp_path, SINCE)) == 1


def test_factor_bearing_counts_per_symbol_since_the_date(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "Bonus 1:1", "bonus", 0.5), ("2026-11-01", "Split", "split", 0.5),
                            ("2026-10-06", "Div", "dividend", np.nan), ("2026-09-01", "Bonus", "bonus", 0.5)])
    write(tmp_path, "BBB", [("2026-10-06", "Div", "dividend", np.nan)])
    counts = audit.factor_bearing_counts(tmp_path, SINCE)
    assert counts == {"AAA": {"bonus": 1, "split": 1}}


def test_audit_never_modifies_data_files(tmp_path):
    write(tmp_path, "AAA", [("2026-10-05", "Bonus 1:1", "bonus", 0.5), ("2026-10-05", "Bonus Issue 1:1", "bonus", 0.5)])
    before = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    audit.find_possible_duplicates(tmp_path, SINCE)
    audit.factor_bearing_counts(tmp_path, SINCE)
    assert before == {p: hashlib.md5(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


def test_parse_not_added_lines(tmp_path):
    log = tmp_path / "x.err.log"
    log.write_text(
        "2026-11-10 13:58:57.181 | WARNING  | __main__:refresh_symbol:342 - AAATECH: 1 new action(s) dated before 2026-09-30 "
        "(or with no ex_date) were NOT added: ['2026-09-23']\n"
        "2026-11-10 13:59:00.398 | WARNING  | __main__:refresh_symbol:342 - XYZ: 2 new action(s) dated before 2026-09-30 "
        "(or with no ex_date) were NOT added: ['2026-09-22', 'NaT']\n"
        "2026-11-10 13:59:01.000 | INFO     | other line\n", encoding="utf-8")
    rows = audit.parse_not_added(log)
    assert rows == [{"symbol": "AAATECH", "count": 1, "dates": ["2026-09-23"]},
                    {"symbol": "XYZ", "count": 2, "dates": ["2026-09-22", "NaT"]}]


def test_parse_not_added_tolerates_a_missing_file(tmp_path):
    assert audit.parse_not_added(tmp_path / "nope.log") == []


def test_not_added_rows_can_be_grouped_by_type_only_when_the_log_has_it():
    rows = [{"symbol": "A", "count": 1, "dates": ["2026-09-23"]}]
    assert audit.group_by_type(rows) is None   # these log lines carry no action type: say so, never guess


def test_module_is_read_only_and_uses_loguru():
    src = Path(audit.__file__).read_text(encoding="utf-8")
    assert not re.search(r"os\.remove|os\.unlink|\.unlink\(|rmtree|to_parquet|write_text|write_bytes|open\([^)]*['\"]w", src)
    assert not re.search(r"^\s*print\(", src, re.M) and "loguru" in src
