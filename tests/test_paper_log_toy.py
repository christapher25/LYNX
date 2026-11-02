"""
Reference ("toy") implementation of the forward paper log contract (vanta.paper_log) with switchable deliberate bugs. NOT a VANTA module, no tests.
All names are INVENTED by the test writer (proposed to quant-m3 / the lead for docs/vanta/INTERFACES.md section 9):

  append_decisions(log_dir, decision_date, candidates, *, model_version, code_version, data_asof) -> "APPENDED" | "NOOP"
  append_outcomes(log_dir, labels, asof) -> int          labels: decision_date, symbol, y_net, matured_at
  calibration_summary(log_dir, asof, window_weeks=52, min_rows=30) -> dict(status, n_matured, loser_share, mean_y_net)
  PaperLogError, StaleDataError;  files log_dir/decisions.jsonl and log_dir/outcomes.jsonl (one JSON object per line)
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

TS = pd.Timestamp


class PaperLogError(RuntimeError):
    pass


class StaleDataError(PaperLogError):
    pass


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def _read(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def toy_paper_log(*bugs):
    bugs = frozenset(bugs)

    def append_decisions(log_dir, decision_date, candidates, *, model_version, code_version, data_asof):
        log_dir = Path(log_dir)
        d = TS(decision_date)
        if data_asof is None or pd.isna(data_asof):
            if "stale_none_ok" not in bugs:
                raise StaleDataError("data_asof is missing: refusing to guess")
        elif TS(data_asof) < d and "stale_ok" not in bugs:
            raise StaleDataError(f"input data ends {TS(data_asof).date()}, before decision date {d.date()}")
        cand = candidates if "only_published" not in bugs else candidates[candidates["published"].astype(bool)]
        rows = []
        for rec in cand.to_dict("records"):
            r = {k: _clean(v) for k, v in rec.items()}
            r.update(decision_date=d.date().isoformat(), model_version=model_version, code_version=code_version)
            if "no_model_version" in bugs:
                r.pop("model_version")
            if "no_code_version" in bugs:
                r.pop("code_version")
            rows.append(json.dumps(r, sort_keys=True))
        f = log_dir / "decisions.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        have = [ln for ln in existing if json.loads(ln)["decision_date"] == d.date().isoformat()]
        if have:
            if "overwrite" in bugs:
                keep = [ln for ln in existing if json.loads(ln)["decision_date"] != d.date().isoformat()]
                f.write_text("\n".join(keep + rows) + "\n", encoding="utf-8")
                return "APPENDED"
            if "dup_append" in bugs:
                with open(f, "ab") as fh:
                    fh.write(("\n".join(rows) + "\n").encode())
                return "APPENDED"
            if have == rows and "refuse_identical" not in bugs:
                return "NOOP"
            raise PaperLogError(f"decision date {d.date()} already recorded")
        if "rewrite_all" in bugs and existing:
            f.write_text("\n".join(json.dumps(json.loads(ln), sort_keys=True, indent=None, separators=(", ", ": ")) for ln in existing + rows)
                         + "\n", encoding="utf-8")
            return "APPENDED"
        with open(f, "ab") as fh:
            fh.write(("\n".join(rows) + "\n").encode())
        return "APPENDED"

    def append_outcomes(log_dir, labels, asof):
        log_dir = Path(log_dir)
        f = log_dir / "outcomes.jsonl"
        have = {(r["decision_date"], r["symbol"]) for r in _read(f)}
        out = []
        for rec in labels.to_dict("records"):
            key = (TS(rec["decision_date"]).date().isoformat(), rec["symbol"])
            y = rec["y_net"]
            if "outcome_immature" not in bugs and TS(rec["matured_at"]) > TS(asof):
                continue
            if not (isinstance(y, (int, float, np.floating)) and math.isfinite(float(y))):
                if "outcome_nan_zero" in bugs:
                    y = 0.0
                else:
                    continue
            if key in have and "outcome_dupes" not in bugs:
                continue
            have.add(key)
            out.append(json.dumps(dict(decision_date=key[0], symbol=key[1], y_net=float(y),
                                       matured_at=TS(rec["matured_at"]).date().isoformat()), sort_keys=True))
        if out:
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(f, "ab") as fh:
                fh.write(("\n".join(out) + "\n").encode())
            if "decisions_mutated_by_outcomes" in bugs:
                df = log_dir / "decisions.jsonl"
                if df.exists():
                    df.write_text("\n".join(json.dumps({**json.loads(ln), "outcome_seen": True}, sort_keys=True)
                                            for ln in df.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        return len(out)

    def calibration_summary(log_dir, asof, window_weeks=52, min_rows=30):
        log_dir = Path(log_dir)
        asof = TS(asof)
        dec = {(r["decision_date"], r["symbol"]) for r in _read(log_dir / "decisions.jsonl")}
        rows = [r for r in _read(log_dir / "outcomes.jsonl") if (r["decision_date"], r["symbol"]) in dec]
        if "cal_immature" not in bugs:
            rows = [r for r in rows if TS(r["matured_at"]) <= asof]
        if "cal_ignore_window" not in bugs:
            rows = [r for r in rows if TS(r["decision_date"]) >= asof - pd.Timedelta(weeks=window_weeks)]
        y = np.array([r["y_net"] for r in rows], float)
        if "cal_count_unmatured_decisions" in bugs:
            have = {(r["decision_date"], r["symbol"]) for r in rows}
            y = np.concatenate([y, np.zeros(len([k for k in dec if k not in have]))])
        n = len(y)
        if n < min_rows:
            if "cal_fabricate" in bugs:
                return dict(status="OK", n_matured=n, loser_share=0.0, mean_y_net=0.0)
            return dict(status="INSUFFICIENT", n_matured=n, loser_share=float("nan"), mean_y_net=float("nan"))
        return dict(status="OK", n_matured=n, loser_share=float((y <= 0).mean()), mean_y_net=float(y.mean()))

    return SimpleNamespace(append_decisions=append_decisions, append_outcomes=append_outcomes, calibration_summary=calibration_summary,
                           PaperLogError=PaperLogError, StaleDataError=StaleDataError)
