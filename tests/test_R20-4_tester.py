"""TESTER edge cases for R20-4 (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from scripts import report_unexplained_moves as rep
from tests.test_report_unexplained_moves import (CUTOFF, _adj_frame, _candidates, _log, _write_actions,
                                                 _write_adjusted)

D = pd.to_datetime


def _cands(rows):
    return pd.DataFrame(rows, columns=["decision_date", "symbol", "liquidity_rank"]).assign(
        decision_date=lambda d: D(d["decision_date"]))


def test_empty_current_list_still_produces_exposure_and_report(tmp_path):
    cur = pd.DataFrame({"symbol": [], "date": D([]), "ret": []})
    c = _cands([("2020-01-03", "A", 5)])
    assert len(rep.top200_exposure(cur, c)) == 0
    text, out = rep.build_report(cur, {"A": 0}, CUTOFF, c, tmp_path / "nope.parquet", 0)
    assert "current total: 0" in text and "AASTHA in M1 file: no" in text


def test_single_row_frame_has_no_move():
    assert len(rep.unexplained_moves("X", _adj_frame([100.0]))) == 0
    assert len(rep.unexplained_moves("X", _adj_frame([]))) == 0


def test_nan_close_never_makes_a_move():
    assert len(rep.unexplained_moves("X", _adj_frame([100, np.nan, 101]))) == 0


def test_move_after_last_decision_date_uses_only_previous():
    cur = pd.DataFrame({"symbol": ["A"], "date": D(["2026-12-01"]), "ret": [0.5]})
    c = _cands([("2026-01-02", "A", 100)])
    r = rep.top200_exposure(cur, c).iloc[0]
    assert r["bracket_2013"] and pd.isna(r["next_dd"]) and pd.isna(r["next_rank"])


def test_move_on_a_decision_date_brackets_with_that_date_as_previous():
    cur = pd.DataFrame({"symbol": ["A"], "date": D(["2020-01-10"]), "ret": [0.5]})
    c = _cands([("2020-01-10", "A", 400), ("2020-01-17", "A", 100)])
    r = rep.top200_exposure(cur, c).iloc[0]
    assert r["prev_dd"] == D("2020-01-10") and r["bracket_2013"]      # next side qualifies


def test_bracket_requires_the_qualifying_side_itself_to_be_2013_or_later():
    # prev decision is 2012 with rank 10 (does not count), next decision is 2013 with rank 300 -> not counted
    cur = pd.DataFrame({"symbol": ["A"], "date": D(["2012-12-31"]), "ret": [0.5]})
    c = _cands([("2012-12-28", "A", 10), ("2013-01-04", "A", 300)])
    assert not rep.top200_exposure(cur, c).iloc[0]["bracket_2013"]


def test_rank_exactly_200_counts_and_201_does_not():
    cur = pd.DataFrame({"symbol": ["A", "B"], "date": D(["2020-01-10"] * 2), "ret": [0.5, 0.5]})
    c = _cands([("2020-01-03", "A", 200), ("2020-01-03", "B", 201)])
    r = rep.top200_exposure(cur, c)
    assert list(r["bracket_2013"]) == [True, False]


def test_duplicate_candidate_rows_are_an_error():
    cur = pd.DataFrame({"symbol": ["A"], "date": D(["2020-01-10"]), "ret": [0.5]})
    c = _cands([("2020-01-03", "A", 5), ("2020-01-03", "A", 6)])
    with pytest.raises(rep.ReportError):
        rep.top200_exposure(cur, c)


def test_logged_row_for_symbol_without_adjusted_file_exits_1(tmp_path):
    adj, act = tmp_path / "adj", tmp_path / "act"
    _write_adjusted(adj, "AAA", D(["2026-09-01", "2026-09-02"]), [100, 101])
    log = tmp_path / "l.log"; _log(log, [("ZZZ", "2026-10-05", "bonus")])
    with pytest.raises(rep.ReportError):
        rep.scan_symbols(adj, act, CUTOFF, rep.parse_refresh_logs([log]))


def test_same_new_line_in_two_logs_is_never_silently_approximated(tmp_path):
    adj, act = tmp_path / "adj", tmp_path / "act"
    _write_adjusted(adj, "AAA", D(["2026-09-01", "2026-09-02"]), [100, 101])
    _write_actions(act, "AAA", [("2026-10-05", "bonus", 0.5)])
    l1, l2 = tmp_path / "1.log", tmp_path / "2.log"
    _log(l1, [("AAA", "2026-10-05", "bonus")]); _log(l2, [("AAA", "2026-10-05", "bonus")])
    with pytest.raises(rep.ReportError):                 # second removal finds nothing -> exit 1, not a silent skip
        rep.scan_symbols(adj, act, CUTOFF, rep.parse_refresh_logs([l1, l2]))


def test_move_exactly_on_cutoff_is_in_current_le_cutoff_not_after(tmp_path):
    adj, act = tmp_path / "adj", tmp_path / "act"
    _write_adjusted(adj, "AAA", D(["2026-09-28", "2026-09-29", "2026-09-30"]), [100, 150, 150])
    cur, pre, _ = rep.scan_symbols(adj, act, CUTOFF, [])
    c = _cands([("2020-01-03", "AAA", 5)])
    text, _ = rep.build_report(cur, pre, CUTOFF, c, tmp_path / "x.parquet", 0)
    assert "current total <= cutoff: 1" in text and "after cutoff: 0" in text


def test_cli_missing_log_file_is_not_a_silent_success(tmp_path):
    with pytest.raises((FileNotFoundError, OSError, SystemExit)):
        rep.main(["--refresh-logs", str(tmp_path / "missing.log"), "--out", str(tmp_path / "o.md"),
                  "--adjusted-dir", str(tmp_path), "--actions-dir", str(tmp_path)])
