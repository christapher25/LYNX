"""Proposed corporate-action changes for the owner (read-only): which stored rows would get a different adjustment factor if
their raw_subject were parsed by today's parser, and does the ex-date price move agree with each new factor?

    python -m scripts.propose_corp_action_reparse [--out-dir PATH]

For every stored row of every per-symbol corporate-actions file the raw_subject is re-parsed exactly as the ingester does
(parse_subject(raw_subject, face_value) then adjustment_factor) and the new factor is compared with the stored one. A row is a
proposed change when the factor differs: `gain` (none -> value), `lose` (value -> none) or `change` (value -> other value, compared
after round(.., 12)). Rows with an unchanged factor are only counted (and, separately, how many of them differ in another parsed
field). Each proposed change carries price evidence from the symbol's adjusted file (unadjusted CLOSE): the last session strictly
before the ex-date, the first session on or after it, their ratio, and a `check`. The check only sorts the list for the owner's
reading (CHECK_BAND); it approves or rejects nothing. `combined_factor` is the product of the NEW factors of all of the symbol's rows on the same
ex-date (a split and a bonus on one day move the price together) and `check_combined` applies the same band to that product.

Outputs, under --out-dir only: proposed_changes.json and proposed_changes.md. Nothing else is written; the data directories
are only read. Exit 0 whenever the run completes; exit 1 on an unreadable file or a duplicate (symbol, ex_date, raw_subject) key,
which is named, and then nothing is written.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HISTORY_CUTOFF = "2026-09-29"
CHECK_BAND = 0.25          # |observed_ratio / new_factor - 1| <= band (gain/change); |observed_ratio - 1| <= band (lose)
MAX_GAP_DAYS = 10          # calendar days between the two sessions; more is `no_price`
KEY = ["ex_date", "raw_subject"]
PARSED_FIELDS = ["action_type", "ratio_new", "ratio_old", "face_value_from", "face_value_to", "dividend_amount",
                 "dividend_percent", "parse_confidence"]


class ProposalError(Exception):
    """An unreadable file or a duplicate key: the run stops, nothing is written."""


@dataclass
class Reparse:
    factor: float | None
    fields: dict = field(default_factory=dict)       # parser-derived fields to compare with the stored row (optional)


def default_reparse(subject: str, face_value) -> Reparse:
    """Exactly what data/ingest_corporate_actions.py does for a fetched record: parse_subject(subject, faceVal), then adjustment_factor."""
    from data.corporate_action_parser import adjustment_factor, parse_subject
    parsed = parse_subject(subject, face_value)
    return Reparse(adjustment_factor(parsed), {
        "action_type": parsed.action_type.value, "ratio_new": parsed.ratio_new, "ratio_old": parsed.ratio_old,
        "face_value_from": parsed.face_value_from, "face_value_to": parsed.face_value_to,
        "dividend_amount": parsed.dividend_amount, "dividend_percent": parsed.dividend_percent,
        "parse_confidence": parsed.parse_confidence})


def _null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT or (not isinstance(v, str) and pd.isna(v) is True)


def _factor(v) -> float | None:
    return None if _null(v) else float(v)


def _same(a, b) -> bool:
    if _null(a) and _null(b):
        return True
    if _null(a) or _null(b):
        return False
    try:
        return round(float(a), 12) == round(float(b), 12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _kind(old: float | None, new: float | None) -> str | None:
    if old is None and new is None:
        return None
    if old is None:
        return "gain"
    if new is None:
        return "lose"
    return None if round(old, 12) == round(new, 12) else "change"


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def _in_band(ratio: float, factor: float | None) -> bool:
    """factor None (nothing on the day carries a new factor) is compared with 1."""
    return abs(ratio / (1.0 if factor is None else factor) - 1) <= CHECK_BAND


def _evidence(prices: pd.DataFrame | None, ex_date: pd.Timestamp, kind: str, old, new, combined) -> dict:
    out = {"check": "no_price", "no_price_reason": None, "observed_ratio": None, "prev_date": None, "prev_close": None,
           "ex_session_date": None, "ex_close": None, "check_combined": "no_price"}
    if kind != "lose" and not (_finite(new) and new > 0):
        out["no_price_reason"] = "bad_factor"
        return out
    if prices is None:
        out["no_price_reason"] = "no_adjusted_file"
        return out
    if pd.isna(ex_date):
        out["no_price_reason"] = "no_ex_date"
        return out
    dates = prices["DATE"].to_numpy()
    ex64 = np.datetime64(ex_date)
    ex_idx = int(np.searchsorted(dates, ex64, side="left"))          # first session on or after the ex-date
    prev_idx = ex_idx - 1                                            # last session strictly before it
    if prev_idx < 0:
        out["no_price_reason"] = "no_prev_session"
        return out
    if ex_idx >= len(dates):
        out["no_price_reason"] = "no_ex_session"
        return out
    prev_date, ex_session = pd.Timestamp(dates[prev_idx]), pd.Timestamp(dates[ex_idx])
    prev_close, ex_close = float(prices["CLOSE"].iloc[prev_idx]), float(prices["CLOSE"].iloc[ex_idx])
    out.update(prev_date=str(prev_date.date()), prev_close=prev_close if math.isfinite(prev_close) else None,
               ex_session_date=str(ex_session.date()), ex_close=ex_close if math.isfinite(ex_close) else None)
    if (ex_session - prev_date).days > MAX_GAP_DAYS:
        out["no_price_reason"] = "gap_gt_10_days"
        return out
    if not (math.isfinite(prev_close) and prev_close > 0 and math.isfinite(ex_close)):
        out["no_price_reason"] = "bad_close"
        return out
    ratio = ex_close / prev_close
    if not math.isfinite(ratio):
        out["no_price_reason"] = "bad_close"
        return out
    out["observed_ratio"] = ratio
    out["check"] = "consistent" if (abs(ratio - 1) <= CHECK_BAND if kind == "lose" else _in_band(ratio, new)) else "inconsistent"
    if combined is None or (math.isfinite(combined) and combined > 0):
        out["check_combined"] = "consistent" if _in_band(ratio, combined) else "inconsistent"
    return out


def _read(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, **kw)
    except Exception as exc:                                         # noqa: BLE001
        raise ProposalError(f"unreadable file {path.name}: {type(exc).__name__}: {exc}") from exc


def _symbol(path: Path, adjusted_dir: Path, reparse: Callable, totals: Counter) -> list[dict]:
    symbol = path.stem
    df = _read(path)
    dup = df[df.duplicated(KEY, keep=False)]
    if len(dup):
        r = dup.iloc[0]
        raise ProposalError(f"duplicate key in {path.name}: ({symbol}, {str(r['ex_date'])[:10]}, {r['raw_subject']!r})")
    rows = []
    for rec in df.to_dict("records"):
        face = None if _null(rec.get("face_value")) else rec["face_value"]
        res = reparse(rec["raw_subject"], face)
        old = _factor(rec.get("adjustment_factor"))
        new = None if (res.factor is None or math.isnan(float(res.factor))) else float(res.factor)
        rows.append((rec, old, new, res))
        totals["rows"] += 1
    changes = []
    prices = None
    loaded = False
    by_day: dict = {}                                                 # ex_date -> product of the NEW factors of all rows that day
    for rec, _old, new, _res in rows:
        if new is not None and not pd.isna(rec["ex_date"]):
            by_day[pd.Timestamp(rec["ex_date"])] = by_day.get(pd.Timestamp(rec["ex_date"]), 1.0) * new
    for i, (rec, old, new, res) in enumerate(rows):
        kind = _kind(old, new)
        if kind is None:
            totals["unchanged_factor"] += 1
            if any(k in rec and not _same(rec[k], v) for k, v in res.fields.items()):
                totals["unchanged_factor_other_field_differs"] += 1
            continue
        if not loaded:
            apath = adjusted_dir / f"{symbol}.parquet"
            prices = _read(apath, columns=["DATE", "CLOSE"]).sort_values("DATE").reset_index(drop=True) if apath.exists() else None
            loaded = True
        ex_date = pd.Timestamp(rec["ex_date"])
        same_day = any(j != i and not pd.isna(ex_date) and pd.Timestamp(o[0]["ex_date"]) == ex_date and (o[1] is not None or o[2] is not None)
                       for j, o in enumerate(rows))
        combined = None if pd.isna(ex_date) else by_day.get(ex_date)
        ev = _evidence(prices, ex_date, kind, old, new, combined)
        changes.append({"symbol": symbol, "ex_date": None if pd.isna(ex_date) else str(ex_date.date()),
                        "action_type": rec.get("action_type"), "raw_subject": rec["raw_subject"],
                        "old_factor": old, "new_factor": new if _finite(new) else None, "kind": kind,
                        "combined_factor": combined if _finite(combined) else None, **ev, "same_day": bool(same_day)})
    return changes


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:                                                # noqa: BLE001
        return "unknown"


def _markdown(result: dict) -> str:
    t, ch = result["totals"], result["changes"]
    lines = ["# Proposed corporate-action changes (read-only proposal; nothing is applied)", "",
             f"history cutoff {result['history_cutoff']}; commit {result['generated_from_commit']}; check band {CHECK_BAND}; "
             f"max gap {MAX_GAP_DAYS} days.", "",
             f"rows scanned {t['rows']} in {t['symbols']} files; unchanged factor {t['unchanged_factor']} "
             f"(of which another parsed field differs: {t['unchanged_factor_other_field_differs']}); proposed changes {t['proposed']}.", "",
             "## By kind and check", "", "| kind | consistent | inconsistent | no_price | total |", "|---|---|---|---|---|"]
    for kind in ("gain", "lose", "change"):
        row = [t["by_kind_check"].get(f"{kind}/{c}", 0) for c in ("consistent", "inconsistent", "no_price")]
        lines.append(f"| {kind} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    lines += ["", "## By kind and check_combined (the check against the product of all new factors on the ex-date)", "",
              "| kind | consistent | inconsistent | no_price | total |", "|---|---|---|---|---|"]
    for kind in ("gain", "lose", "change"):
        row = [t["by_kind_check_combined"].get(f"{kind}/{c}", 0) for c in ("consistent", "inconsistent", "no_price")]
        lines.append(f"| {kind} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    lines += ["", "## By action type", "", "| action_type | n |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(t["by_action_type"].items())]
    lines += ["", "## By year of ex-date", "", "| year | n |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(t["by_year"].items())]
    order = sorted(ch, key=lambda c: (0 if c["check_combined"] in ("inconsistent", "no_price") else 1 if c["same_day"] else 2,
                                      c["symbol"], c["ex_date"] or ""))
    lines += ["", "## Every proposed change (check_combined inconsistent or no_price first, then same_day, then the rest)", "",
              "| symbol | ex_date | type | kind | old | new | combined | check | check_combined | ratio | prev (date, close) | ex session (date, close) | same_day | subject |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def f(v, nd=6):
        return "" if v is None else f"{v:.{nd}g}"
    for c in order:
        reason = f" ({c['no_price_reason']})" if c["no_price_reason"] else ""
        lines.append(f"| {c['symbol']} | {c['ex_date']} | {c['action_type']} | {c['kind']} | {f(c['old_factor'])} | {f(c['new_factor'])} | "
                     f"{f(c['combined_factor'])} | {c['check']}{reason} | {c['check_combined']} | {f(c['observed_ratio'])} | {c['prev_date'] or ''} {f(c['prev_close'])} | "
                     f"{c['ex_session_date'] or ''} {f(c['ex_close'])} | {'YES' if c['same_day'] else ''} | {str(c['raw_subject']).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def run(out_dir: Path, actions_dir: Path, adjusted_dir: Path, reparse: Callable = default_reparse, head: str | None = None) -> dict:
    totals: Counter = Counter()
    changes: list[dict] = []
    files = sorted(actions_dir.glob("*.parquet"))
    for path in files:                                               # one symbol at a time
        changes += _symbol(path, adjusted_dir, reparse, totals)
    changes.sort(key=lambda c: (c["symbol"], c["ex_date"] or ""))
    by_kind_check = Counter(f"{c['kind']}/{c['check']}" for c in changes)
    by_kind_check_combined = Counter(f"{c['kind']}/{c['check_combined']}" for c in changes)
    result = {
        "history_cutoff": HISTORY_CUTOFF,
        "generated_from_commit": head or _git_head(),
        "totals": {"rows": totals["rows"], "symbols": len(files), "unchanged_factor": totals["unchanged_factor"],
                   "unchanged_factor_other_field_differs": totals["unchanged_factor_other_field_differs"],
                   "proposed": len(changes), "by_kind_check": dict(by_kind_check),
                   "by_kind_check_combined": dict(by_kind_check_combined),
                   "by_action_type": dict(Counter(str(c["action_type"]) for c in changes)),
                   "by_year": dict(Counter((c["ex_date"] or "none")[:4] for c in changes))},
        "changes": changes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposed_changes.json").write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    (out_dir / "proposed_changes.md").write_text(_markdown(result), encoding="utf-8")
    return result


def main(argv=None, *, reparse: Callable = default_reparse, actions_dir: Path | None = None, adjusted_dir: Path | None = None,
         head: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only proposal of corporate-action factor changes for the owner.")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <config.LOG_DIR>/row21")
    args = parser.parse_args(argv)
    import config
    out_dir = args.out_dir or (config.LOG_DIR / "row21")
    try:
        result = run(out_dir, actions_dir or config.CORPORATE_ACTIONS_DIR, adjusted_dir or config.ADJUSTED_PRICES_DIR, reparse, head)
    except ProposalError as exc:
        logger.error("{}", exc)
        logger.error("proposal NOT written")
        return 1
    t = result["totals"]
    logger.info("scanned {:,} rows in {:,} files: {:,} unchanged factor, {:,} proposed changes; check {}; check_combined {}",
                t["rows"], t["symbols"], t["unchanged_factor"], t["proposed"], t["by_kind_check"], t["by_kind_check_combined"])
    logger.info("written: {}", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
