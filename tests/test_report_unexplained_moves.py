"""R20-4: tests for scripts/report_unexplained_moves.py (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.adjust_prices import apply_backward_adjustment
from scripts import report_unexplained_moves as rep

CUTOFF = pd.Timestamp("2026-09-29")


def _prices(symbol, dates, closes):
    c = np.asarray(closes, dtype=float)
    d = pd.DataFrame({"SYMBOL": symbol, "SERIES": "EQ", "DATE": pd.to_datetime(dates)})
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREV_CLOSE"]:
        d[col] = c
    d["VOLUME"] = 1000
    return d


def _write_adjusted(adj_dir, symbol, dates, closes, actions=None):
    p = _prices(symbol, dates, closes)
    a = apply_backward_adjustment(
        p, actions if actions is not None else
        pd.DataFrame(columns=["ex_date", "adjustment_factor", "action_type"]))
    adj_dir.mkdir(parents=True, exist_ok=True)
    a.to_parquet(adj_dir / f"{symbol}.parquet")
    return a


def _write_actions(act_dir, symbol, rows):
    """rows: (ex_date, action_type, factor-or-None)"""
    act_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "symbol": symbol,
        "ex_date": pd.to_datetime([r[0] for r in rows]),
        "action_type": [r[1] for r in rows],
        "adjustment_factor": [r[2] for r in rows],
    }).to_parquet(act_dir / f"{symbol}.parquet")


def _adj_frame(closes, applied=None, start="2020-01-01"):
    n = len(closes)
    return pd.DataFrame({
        "DATE": pd.date_range(start, periods=n, freq="D"),
        "ADJ_CLOSE": np.asarray(closes, dtype=float),
        "ADJUSTMENT_APPLIED": applied if applied is not None else [False] * n,
    })


# T1 ------------------------------------------------------------------
def test_t1_band_and_flag():
    f = _adj_frame([100, 130])
    assert len(rep.unexplained_moves("X", f)) == 1
    f2 = _adj_frame([100, 130], applied=[True, True])
    assert len(rep.unexplained_moves("X", f2)) == 0
    f3 = _adj_frame([100, 120.0])          # exactly 20%: not strict >
    assert len(rep.unexplained_moves("X", f3)) == 0
    f4 = _adj_frame([100])                 # first row NaN return
    assert len(rep.unexplained_moves("X", f4)) == 0


# helpers for the end-to-end runs ----------------------------------------
def _candidates(path, rows):
    pd.DataFrame(rows, columns=["decision_date", "symbol", "liquidity_rank"]).assign(
        decision_date=lambda d: pd.to_datetime(d["decision_date"])
    ).to_parquet(path)


def _log(path, lines):
    path.write_text("\n".join(
        f"2026-11-10 13:58:49.584 | INFO     | __main__:run_refresh:382 - {s}: NEW action ex_date {d} type {t}"
        for s, d, t in lines) + "\n", encoding="utf-8")


def _run(tmp_path, logs, extra=()):
    out = tmp_path / "out" / "rep.md"
    argv = ["--refresh-logs", *map(str, logs),
            "--cutoff", "2026-09-29", "--refresh-from", "2026-09-30",
            "--candidates", str(tmp_path / "cands.parquet"),
            "--m1-candidates", str(tmp_path / "m1.parquet"),
            "--adjusted-dir", str(tmp_path / "adj"),
            "--actions-dir", str(tmp_path / "act"),
            "--out", str(out), *extra]
    return rep.main(argv), out


def _base(tmp_path):
    _candidates(tmp_path / "cands.parquet", [("2019-01-04", "ZZZ", 300.0)])
    _candidates(tmp_path / "m1.parquet", [("2019-01-04", "ZZZ", 300.0)])
    _log(tmp_path / "log.txt", [])
    return [tmp_path / "log.txt"]


# T2 / T3 -------------------------------------------------------------
def test_t2_pre_refresh_differs(tmp_path):
    dates = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]
    closes = [100, 100, 130, 130]            # raw 30% jump on 2026-09-28
    # split logged NEW, ex_date after the cutoff: it flags every earlier row as adjusted,
    # which hides the pre-cutoff raw jump in the current file only.
    acts = pd.DataFrame({"ex_date": [pd.Timestamp("2026-10-05")],
                         "adjustment_factor": [1.0], "action_type": ["split"]})
    _write_adjusted(tmp_path / "adj", "SPL", dates, closes, acts)
    _write_actions(tmp_path / "act", "SPL", [("2026-10-05", "split", 1.0)])
    logs = _base(tmp_path)
    _log(logs[0], [("SPL", "2026-10-05", "split")])
    code, out = _run(tmp_path, logs)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "pre-refresh total: 1" in text
    assert "current total: 0" in text
    assert "SPL" in text.split("Symbols where pre-refresh and current differ")[1]


def test_t3_logged_row_missing_exits_1(tmp_path):
    _write_adjusted(tmp_path / "adj", "SPL", ["2026-09-24", "2026-09-25"], [100, 100])
    _write_actions(tmp_path / "act", "SPL", [("2026-09-01", "dividend", None)])
    logs = _base(tmp_path)
    _log(logs[0], [("SPL", "2026-11-11", "dividend")])
    code, out = _run(tmp_path, logs)
    assert code == 1
    assert not out.exists()


# T4 ------------------------------------------------------------------
def test_t4_after_cutoff_only(tmp_path):
    # jump on the cutoff date itself, and one the day after
    dates = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]
    _write_adjusted(tmp_path / "adj", "AAA", dates, [100, 100, 150, 225])
    logs = _base(tmp_path)
    code, out = _run(tmp_path, logs)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "current total: 2" in text.lower() or "Current total: 2" in text
    assert "after cutoff: 1" in text


# T5 ------------------------------------------------------------------
def _bracket_count(tmp_path, cand_rows, move_date, dates, closes):
    _write_adjusted(tmp_path / "adj", "AAA", dates, closes)
    _candidates(tmp_path / "cands.parquet", cand_rows)
    _candidates(tmp_path / "m1.parquet", cand_rows)
    _log(tmp_path / "log.txt", [])
    code, out = _run(tmp_path, [tmp_path / "log.txt"])
    assert code == 0
    t = out.read_text(encoding="utf-8")
    line = [l for l in t.splitlines() if l.startswith("iii-bracket >=2013:")][0]
    return int(line.split(":")[1].strip().split()[0])


MOVE = (["2019-01-01", "2019-01-02"], [100, 150])


def test_t5_bracket(tmp_path):
    # rank 150 at previous decision week
    assert _bracket_count(tmp_path, [("2018-12-28", "AAA", 150.0), ("2019-01-04", "AAA", 400.0)],
                          None, *MOVE) == 1


def test_t5_bracket_rank_250_both(tmp_path):
    assert _bracket_count(tmp_path, [("2018-12-28", "AAA", 250.0), ("2019-01-04", "AAA", 250.0)],
                          None, *MOVE) == 0


def test_t5_bracket_nan(tmp_path):
    assert _bracket_count(tmp_path, [("2018-12-28", "AAA", np.nan), ("2019-01-04", "AAA", np.nan)],
                          None, *MOVE) == 0


def test_t5_bracket_before_2013(tmp_path):
    assert _bracket_count(tmp_path, [("2011-12-30", "AAA", 100.0), ("2012-01-06", "AAA", 100.0)],
                          None, ["2012-01-01", "2012-01-02"], [100, 150]) == 0


def test_t5_bracket_before_first_decision_uses_next(tmp_path):
    assert _bracket_count(tmp_path, [("2019-01-04", "AAA", 100.0)],
                          None, *MOVE) == 1


# T6 ------------------------------------------------------------------
def _any_count(tmp_path, cand_rows, dates, closes):
    _write_adjusted(tmp_path / "adj", "AAA", dates, closes)
    _candidates(tmp_path / "cands.parquet", cand_rows)
    _candidates(tmp_path / "m1.parquet", cand_rows)
    _log(tmp_path / "log.txt", [])
    code, out = _run(tmp_path, [tmp_path / "log.txt"])
    assert code == 0
    t = out.read_text(encoding="utf-8")
    line = [l for l in t.splitlines() if l.startswith("iii-any >=2013:")][0]
    return int(line.split(":")[1].strip().split()[0])


def test_t6_any(tmp_path):
    d, c = ["2011-03-01", "2011-03-02"], [100, 150]
    assert _any_count(tmp_path, [("2019-05-03", "AAA", 190.0), ("2019-05-10", "AAA", 400.0)], d, c) == 1


def test_t6_any_only_2012(tmp_path):
    d, c = ["2011-03-01", "2011-03-02"], [100, 150]
    assert _any_count(tmp_path, [("2012-05-04", "AAA", 190.0), ("2019-05-10", "AAA", 400.0)], d, c) == 0


# T7 ------------------------------------------------------------------
def test_t7_aastha_absent_and_present(tmp_path):
    _write_adjusted(tmp_path / "adj", "AAA", ["2019-01-01", "2019-01-02"], [100, 100])
    logs = _base(tmp_path)
    code, out = _run(tmp_path, logs)
    assert code == 0
    t = out.read_text(encoding="utf-8")
    assert "AASTHA rank<=200 (all dates): never" in t
    assert "AASTHA in M1 file: no" in t
    _candidates(tmp_path / "cands.parquet", [("2018-03-02", "AASTHA", 180.0)])
    _candidates(tmp_path / "m1.parquet", [("2018-03-02", "AASTHA", 180.0)])
    code, out = _run(tmp_path, logs)
    t = out.read_text(encoding="utf-8")
    assert "2018-03-02" in t.split("AASTHA rank<=200 (all dates):")[1].splitlines()[0]
    assert "AASTHA in M1 file: yes" in t


# T8 ------------------------------------------------------------------
def test_t8_writes_only_out_and_csv(tmp_path):
    _write_adjusted(tmp_path / "adj", "AAA", ["2019-01-01", "2019-01-02"], [100, 150])
    logs = _base(tmp_path)

    def snapshot():
        return {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    before = snapshot()
    code, out = _run(tmp_path, logs)
    assert code == 0
    after = snapshot()
    new = set(after) - set(before)
    assert new == {out, out.with_suffix(".csv")}
    assert all(after[p] == before[p] for p in before)


def test_parse_refresh_logs(tmp_path):
    _log(tmp_path / "a.txt", [("AAA", "2026-10-01", "dividend"), ("BBB", "2026-11-11", "split")])
    assert rep.parse_refresh_logs([tmp_path / "a.txt"]) == [
        ("AAA", pd.Timestamp("2026-10-01"), "dividend"),
        ("BBB", pd.Timestamp("2026-11-11"), "split")]
