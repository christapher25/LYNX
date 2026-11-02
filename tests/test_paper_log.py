"""
Forward paper log invariants (lead's brief, M3_PREREG section 1: the paper log is the only fresh test). Module under test: vanta.paper_log
(does not exist yet; every name is INVENTED, see tests/test_paper_log_toy.py for the proposed contract).

  PL1  append-only decisions   a second write for an existing decision date with different content is refused; the bytes of every earlier row are
                               identical after later appends (decisions and outcomes)
  PL2  separate outcomes       outcomes go to their own append-only file; the decisions file is never touched when outcomes are recorded;
                               an outcome is filled only when its label has matured (matured_at <= asof) and is finite; never twice
  PL3  provenance and list     every row carries decision date, model version, code version; ALL candidates of the date are recorded
                               (not just the published ones), each with its published flag
  PL4  stale input             input data that ends before the decision date (or whose date is unknown) is refused; nothing is written
  PL5  calibration             uses only decision rows whose outcome has matured by `asof` and lies inside the window; too few rows is recorded
                               as INSUFFICIENT with missing values, never zeros
  PL6  idempotence             re-running a finished decision date (identical content) or the same outcomes is a no-op: files byte-identical

Each check takes an `api` (vanta.paper_log or the toy) and a tmp dir. The real module skips until it exists; the toy passes and every broken toy
fails.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_paper_log_toy import toy_paper_log

TS = pd.Timestamp
D0, D1, D2 = TS("2026-11-06"), TS("2026-11-13"), TS("2026-11-20")           # Fridays
KW = dict(model_version="b1-hold@m3", code_version="abc1234")


def _cands(tag, n=6, n_pub=2):
    return pd.DataFrame({"symbol": [f"{tag}{i}" for i in range(n)], "score": [0.9 - 0.1 * i for i in range(n)],
                         "percentile": [1.0 - i / n for i in range(n)], "published": [i < n_pub for i in range(n)]})


def _labels(d, tag, ys, matured):
    return pd.DataFrame({"decision_date": d, "symbol": [f"{tag}{i}" for i in range(len(ys))], "y_net": ys, "matured_at": matured})


def _lines(p):
    p = Path(p)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def _bytes(p):
    p = Path(p)
    return p.read_bytes() if p.exists() else b""


# =============================================================================================================== PL1
def check_decisions_append_only(api, tmp):
    """PL1: a second write for an existing decision date WITH DIFFERENT CONTENT raises PaperLogError and leaves the file untouched; the bytes
    of the earlier dates are a prefix of the file after later dates (and after outcomes) are appended."""
    f = tmp / "decisions.jsonl"
    assert api.append_decisions(tmp, D0, _cands("A"), data_asof=D0, **KW) == "APPENDED"
    first = _bytes(f)
    assert first
    changed = _cands("A").assign(score=lambda d: d["score"] + 0.01)
    with pytest.raises(api.PaperLogError):
        api.append_decisions(tmp, D0, changed, data_asof=D0, **KW)
    assert _bytes(f) == first
    with pytest.raises(api.PaperLogError):
        api.append_decisions(tmp, D0, _cands("A", n=3), data_asof=D0, **KW)                   # fewer candidates is also a change
    assert _bytes(f) == first
    api.append_decisions(tmp, D1, _cands("B"), data_asof=D1, **KW)
    second = _bytes(f)
    assert second.startswith(first) and len(second) > len(first)
    api.append_outcomes(tmp, _labels(D0, "A", [0.01] * 6, D0 + pd.Timedelta(days=30)), D0 + pd.Timedelta(days=40))
    assert _bytes(f) == second                                                                 # recording outcomes never touches decisions
    api.append_decisions(tmp, D2, _cands("C"), data_asof=D2, **KW)
    assert _bytes(f).startswith(second)
    assert [r["decision_date"] for r in _lines(f)].count("2026-11-06") == 6


# =============================================================================================================== PL2
def check_outcomes_separate_and_mature_only(api, tmp):
    """PL2: outcomes are written to outcomes.jsonl (never into decisions.jsonl, whose rows carry no outcome field). A label whose matured_at is
    after `asof` is NOT recorded (a mutant filling an immature outcome is caught), nor is a NaN label; once `asof` reaches matured_at it is
    recorded exactly once; earlier outcome bytes stay a prefix."""
    api.append_decisions(tmp, D0, _cands("A", n=4), data_asof=D0, **KW)
    api.append_decisions(tmp, D1, _cands("B", n=4), data_asof=D1, **KW)
    dec_bytes = _bytes(tmp / "decisions.jsonl")
    lab = pd.concat([_labels(D0, "A", [0.02, -0.01, 0.03, np.nan], TS("2026-12-07")), _labels(D1, "B", [0.01, 0.01, 0.01, 0.01], TS("2026-12-14"))],
                    ignore_index=True)
    n = api.append_outcomes(tmp, lab, TS("2026-12-10"))
    out = _lines(tmp / "outcomes.jsonl")
    assert n == 3 and len(out) == 3 and {r["decision_date"] for r in out} == {"2026-11-06"}        # D1 not matured; NaN label not recorded
    assert all(np.isfinite(r["y_net"]) for r in out) and {r["symbol"] for r in out} == {"A0", "A1", "A2"}
    first = _bytes(tmp / "outcomes.jsonl")
    assert _bytes(tmp / "decisions.jsonl") == dec_bytes
    assert not any({"y_net", "outcome"} & set(r) for r in _lines(tmp / "decisions.jsonl"))
    n2 = api.append_outcomes(tmp, lab, TS("2026-12-15"))
    out2 = _lines(tmp / "outcomes.jsonl")
    assert n2 == 4 and len(out2) == 7 and _bytes(tmp / "outcomes.jsonl").startswith(first)         # D0's 3 are not duplicated
    assert len({(r["decision_date"], r["symbol"]) for r in out2}) == 7


# =============================================================================================================== PL3
def check_rows_carry_provenance_and_full_list(api, tmp):
    """PL3: every row carries the decision date, the model version and the code version; the FULL candidate list of the date is recorded (6
    candidates, only 2 published), each with its own symbol, score and published flag."""
    c = _cands("A", n=6, n_pub=2)
    api.append_decisions(tmp, D0, c, data_asof=D0, **KW)
    rows = _lines(tmp / "decisions.jsonl")
    assert len(rows) == 6 and [r["symbol"] for r in rows] == list(c["symbol"])
    assert all(r["decision_date"] == "2026-11-06" and r["model_version"] == "b1-hold@m3" and r["code_version"] == "abc1234" for r in rows)
    assert [bool(r["published"]) for r in rows] == [True, True, False, False, False, False]
    assert [r["score"] for r in rows] == pytest.approx(list(c["score"]))


# =============================================================================================================== PL4
def check_stale_input_refused(api, tmp):
    """PL4: input data that ends before the decision date (data_asof one day early) is refused with StaleDataError, an unknown data date is
    refused too, and in both cases nothing is written (no guessing, no partial file). data_asof equal to the decision date is accepted."""
    with pytest.raises(api.StaleDataError):
        api.append_decisions(tmp, D0, _cands("A"), data_asof=D0 - pd.Timedelta(days=1), **KW)
    with pytest.raises(api.StaleDataError):
        api.append_decisions(tmp, D0, _cands("A"), data_asof=None, **KW)
    assert not list(tmp.glob("*")), list(tmp.glob("*"))
    assert api.append_decisions(tmp, D0, _cands("A"), data_asof=D0, **KW) == "APPENDED"


# =============================================================================================================== PL5
def check_calibration_matured_only_and_insufficient_is_recorded(api, tmp):
    """PL5: three dates of four candidates. Outcomes recorded at 2026-12-15 (D0 matured 12-07, D1 12-14; D2 immature, never recorded).
    calibration_summary at asof 2026-12-10 uses only D0 (D1's outcome is on file but matured_at > asof): 4 rows, loser share 0.5, mean 0.005.
    At 12-15: 8 rows. window_weeks=5 drops D0 (decided before asof - 5 weeks): 4 rows, mean 0.01. Fewer rows than min_rows -> INSUFFICIENT with
    NaN (or None) values, never 0; an empty log is INSUFFICIENT with n 0."""
    empty = api.calibration_summary(tmp, TS("2026-12-15"), min_rows=1)
    assert empty["status"] == "INSUFFICIENT" and empty["n_matured"] == 0
    for d, tag in ((D0, "A"), (D1, "B"), (D2, "C")):
        api.append_decisions(tmp, d, _cands(tag, n=4), data_asof=d, **KW)
    lab = pd.concat([_labels(D0, "A", [0.02, -0.01, 0.03, -0.02], TS("2026-12-07")), _labels(D1, "B", [0.01] * 4, TS("2026-12-14")),
                     _labels(D2, "C", [-0.5] * 4, TS("2027-01-04"))], ignore_index=True)
    assert api.append_outcomes(tmp, lab, TS("2026-12-15")) == 8
    s = api.calibration_summary(tmp, TS("2026-12-10"), min_rows=4)
    assert s["status"] == "OK" and s["n_matured"] == 4 and s["loser_share"] == pytest.approx(0.5) and s["mean_y_net"] == pytest.approx(0.005)
    s = api.calibration_summary(tmp, TS("2026-12-15"), min_rows=4)
    assert s["status"] == "OK" and s["n_matured"] == 8 and s["mean_y_net"] == pytest.approx(0.06 / 8)
    s = api.calibration_summary(tmp, TS("2026-12-15"), window_weeks=5, min_rows=4)
    assert s["n_matured"] == 4 and s["mean_y_net"] == pytest.approx(0.01)
    s = api.calibration_summary(tmp, TS("2026-12-15"), min_rows=100)
    assert s["status"] == "INSUFFICIENT" and s["n_matured"] == 8
    for k in ("loser_share", "mean_y_net"):
        assert s[k] is None or np.isnan(s[k]), (k, s[k])
    s0 = api.calibration_summary(tmp, TS("2026-11-30"), min_rows=1)                           # nothing has matured yet at that date
    assert s0["status"] == "INSUFFICIENT" and s0["n_matured"] == 0


# =============================================================================================================== PL6
def check_idempotent_rerun(api, tmp):
    """PL6: running a finished decision date again with identical content returns NOOP and leaves decisions.jsonl byte-identical; running the
    same outcomes again records 0 rows and leaves outcomes.jsonl byte-identical."""
    api.append_decisions(tmp, D0, _cands("A", n=4), data_asof=D0, **KW)
    before = _bytes(tmp / "decisions.jsonl")
    assert api.append_decisions(tmp, D0, _cands("A", n=4), data_asof=D0, **KW) == "NOOP"
    assert _bytes(tmp / "decisions.jsonl") == before
    lab = _labels(D0, "A", [0.01, -0.01, 0.02, 0.03], TS("2026-12-07"))
    assert api.append_outcomes(tmp, lab, TS("2026-12-10")) == 4
    ob = _bytes(tmp / "outcomes.jsonl")
    assert api.append_outcomes(tmp, lab, TS("2026-12-10")) == 0 and api.append_outcomes(tmp, lab, TS("2026-12-20")) == 0
    assert _bytes(tmp / "outcomes.jsonl") == ob and _bytes(tmp / "decisions.jsonl") == before


# =============================================================================================================== harness
CHECKS = {
    "append_only": (check_decisions_append_only, ["overwrite", "dup_append", "rewrite_all", "decisions_mutated_by_outcomes"]),
    "outcomes": (check_outcomes_separate_and_mature_only, ["outcome_immature", "outcome_nan_zero", "outcome_dupes", "decisions_mutated_by_outcomes"]),
    "provenance": (check_rows_carry_provenance_and_full_list, ["only_published", "no_model_version", "no_code_version"]),
    "stale": (check_stale_input_refused, ["stale_ok", "stale_none_ok"]),
    "calibration": (check_calibration_matured_only_and_insufficient_is_recorded,
                    ["cal_immature", "cal_fabricate", "cal_ignore_window", "cal_count_unmatured_decisions"]),
    "idempotent": (check_idempotent_rerun, ["refuse_identical", "dup_append", "outcome_dupes"]),
}
IDS = list(CHECKS)


@pytest.mark.parametrize("name", IDS)
def test_paper_log_real(name, tmp_path):
    api = pytest.importorskip("vanta.paper_log")
    CHECKS[name][0](api, tmp_path)


@pytest.mark.parametrize("name", IDS)
def test_paper_log_toy_passes(name, tmp_path):
    CHECKS[name][0](toy_paper_log(), tmp_path)


MUTANTS = [(n, b) for n in IDS for b in CHECKS[n][1]]


@pytest.mark.parametrize("name,bug", MUTANTS, ids=[f"{n}-{b}" for n, b in MUTANTS])
def test_paper_log_mutant_fails(name, bug, tmp_path):
    with pytest.raises((Exception, pytest.fail.Exception)) as ei:
        CHECKS[name][0](toy_paper_log(bug), tmp_path)
    assert not isinstance(ei.value, (ImportError, NameError, TypeError, AttributeError)), repr(ei.value)


# ---------------------------------------------------------------------------------------------------------------- reviewer-m3 fixes
def test_real_concurrent_writer_refused(tmp_path):
    api = pytest.importorskip("vanta.paper_log")
    (tmp_path / "decisions.jsonl.lock").write_text("held")
    with pytest.raises(api.PaperLogError):
        api.append_decisions(tmp_path, D0, _cands("A"), data_asof=D0, **KW)
    assert not (tmp_path / "decisions.jsonl").exists()
    (tmp_path / "decisions.jsonl.lock").unlink()
    assert api.append_decisions(tmp_path, D0, _cands("A"), data_asof=D0, **KW) == "APPENDED"
    assert not (tmp_path / "decisions.jsonl.lock").exists()                    # released after a normal append


def test_paper_log_label_cache_keyed_by_label_source():
    src = Path(__file__).resolve().parents[1].joinpath("scripts", "run_paper_log.py").read_text(encoding="utf-8")
    assert "hold_label_source_fp()" in src.split("def label_cache")[1].split("def hold_labels")[0]
    assert 'CACHE_DIR / "labels_hold" / f"h{h}"' not in src
