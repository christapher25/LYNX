"""
R20-3: owner-approved corporate-actions data correction.

Adds exactly the owner-approved actions (ex_date up to the history cutoff) that NSE recorded after the per-symbol
files were written, and nothing else. Every appended row comes from the ingester's own fetch of that symbol; a row is
never built from the approved list alone. A symbol where any entry cannot be matched is left untouched and marked failed.

  python -m scripts.apply_corp_action_correction --approved docs/agent-team/corrections/2026-11-10_precutoff_corporate_actions.json [--dry-run]

Exit codes: 0 all entries added or already present; 1 something failed; 2 the approved list is invalid (no network call made).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR
from data import ingest_corporate_actions as ing

SUMMARY_DIR = LOG_DIR / "corrections"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_KEYS = ("symbol", "ex_date", "action_type", "adjustment_factor")


class ApprovedListError(Exception):
    """The approved list is malformed; the CLI exits 2 before any network call."""


def _parse_date(text, what: str) -> pd.Timestamp:
    if not isinstance(text, str) or not _DATE_RE.match(text):
        raise ApprovedListError(f"{what}: not YYYY-MM-DD: {text!r}")
    try:
        return pd.Timestamp(text)
    except ValueError as e:
        raise ApprovedListError(f"{what}: invalid date {text!r}") from e


def load_approved(path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ApprovedListError(f"cannot read approved list: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        raise ApprovedListError("approved list must be an object with an 'actions' list")
    cutoff = _parse_date(data.get("history_cutoff"), "history_cutoff")
    if not data["actions"]:
        raise ApprovedListError("approved list is empty")
    out, seen = [], set()
    for i, raw in enumerate(data["actions"]):
        if not isinstance(raw, dict) or any(k not in raw for k in _KEYS):
            raise ApprovedListError(f"entry {i}: needs keys {_KEYS}")
        sym, typ, fac = raw["symbol"], raw["action_type"], raw["adjustment_factor"]
        if not isinstance(sym, str) or not sym.strip() or not isinstance(typ, str) or not typ.strip():
            raise ApprovedListError(f"entry {i}: symbol and action_type must be non-empty strings")
        ex = _parse_date(raw["ex_date"], f"entry {i} ex_date")
        if fac is not None and (isinstance(fac, bool) or not isinstance(fac, (int, float)) or not math.isfinite(fac)):
            raise ApprovedListError(f"entry {i}: adjustment_factor must be a number or null")
        if ex > cutoff:
            raise ApprovedListError(f"entry {i}: ex_date {ex.date()} is after history_cutoff {cutoff.date()}")
        key = (sym, ex, typ, None if fac is None else round(float(fac), 12))
        if key in seen:
            raise ApprovedListError(f"entry {i}: duplicate of an earlier entry {key}")
        seen.add(key)
        out.append({"symbol": sym, "ex_date": ex, "action_type": typ,
                    "adjustment_factor": None if fac is None else float(fac)})
    return out


def _typed_table(stored: pa.Schema, merged: pd.DataFrame) -> pa.Table:
    """Keep the file's stored schema: a stored non-null type is kept (casts of nulls are fine); a null-typed column takes
    the added value's type only when the merged column has a value, otherwise it stays null-typed."""
    t = pa.Table.from_pandas(merged, preserve_index=False)
    cols = []
    for f in stored:
        col = t[f.name]
        if f.type != pa.null():
            col = col.cast(f.type)
        elif col.null_count == len(col):
            col = col.cast(pa.null())
        cols.append(col)
    return pa.Table.from_arrays(cols, names=stored.names)


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _process_symbol(session, symbol: str, entries: list[dict], dry_run: bool) -> dict:
    """Returns {symbol, entries: [{...,status,reason}], ignored, status}. status: ok | failed."""
    path = ing.OUTPUT_DIR / f"{symbol}.parquet"
    results = [dict(e, status="failed", reason="") for e in entries]

    def fail_all(reason):
        for r in results:
            r["status"], r["reason"] = "failed", reason
        return {"symbol": symbol, "status": "failed", "entries": results, "ignored": 0}

    if not path.exists():
        return fail_all("no existing corporate-actions file (this script never creates one)")
    try:
        existing = pd.read_parquet(path)
        stored = pq.read_schema(path)
        years = [e["ex_date"].year for e in entries]
        fetched = ing.ingest_symbol(session, symbol, min(years), max(years), raise_on_fetch_error=True)
    except Exception as e:                                                   # noqa: BLE001
        return fail_all(f"read/fetch failed: {str(e)[:200]}")

    fetched_sem = {}
    if not fetched.empty:
        for row in fetched.itertuples():
            fetched_sem.setdefault(ing._semantic_key(row.ex_date, row.action_type, row.adjustment_factor), []).append(row)
    ex_keys = set(zip(existing["symbol"], pd.to_datetime(existing["ex_date"]), existing["raw_subject"]))
    ex_sem = {ing._semantic_key(r.ex_date, r.action_type, r.adjustment_factor) for r in existing.itertuples()}

    add_index, used = [], 0
    for r in results:
        sem = ing._semantic_key(r["ex_date"], r["action_type"], r["adjustment_factor"])
        matches = fetched_sem.get(sem, [])
        if len(matches) != 1:
            r["reason"] = f"{len(matches)} fetched rows match (need exactly 1)"
            continue
        used += 1
        row = matches[0]
        if (symbol, pd.Timestamp(row.ex_date), row.raw_subject) in ex_keys or \
                (sem[2] is not None and sem in ex_sem):
            r["status"] = "present"
            continue
        r["status"] = "add"
        add_index.append(row.Index)
    ignored = (0 if fetched.empty else len(fetched)) - used

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        for r in results:
            if r["status"] == "add":
                r["status"], r["reason"] = "failed", "not written: another entry of this symbol failed"
        return {"symbol": symbol, "status": "failed", "entries": results, "ignored": ignored}
    if add_index and not dry_run:
        try:
            added = fetched.loc[add_index][list(existing.columns)]
            merged = pd.concat([existing, added], ignore_index=True).sort_values("ex_date", kind="stable")
            _write_table_atomic(_typed_table(stored, merged.reset_index(drop=True)), path)
        except Exception as e:                                               # noqa: BLE001
            for r in results:
                if r["status"] == "add":
                    r["status"], r["reason"] = "failed", f"write failed: {str(e)[:200]}"
            return {"symbol": symbol, "status": "failed", "entries": results, "ignored": ignored}
    return {"symbol": symbol, "status": "ok", "entries": results, "ignored": ignored}


def run(entries: list[dict], dry_run: bool) -> dict:
    by_symbol = defaultdict(list)
    for e in entries:
        by_symbol[e["symbol"]].append(e)
    session = None
    symbols = []
    for symbol in sorted(by_symbol):
        if (ing.OUTPUT_DIR / f"{symbol}.parquet").exists() and session is None:
            session = ing.get_session()
        symbols.append(_process_symbol(session, symbol, by_symbol[symbol], dry_run))
    flat = [r for s in symbols for r in s["entries"]]
    count = lambda st: sum(r["status"] == st for r in flat)             # noqa: E731
    return {"dry_run": dry_run, "approved": len(entries), "added": count("add"), "present": count("present"),
            "failed": count("failed"), "ignored_fetched_rows": sum(s["ignored"] for s in symbols),
            "symbols": symbols}


def _jsonable(summary: dict) -> dict:
    def clean(r):
        return {**r, "ex_date": str(r["ex_date"])[:10]}
    return {**summary, "symbols": [{**s, "entries": [clean(r) for r in s["entries"]]} for s in summary["symbols"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approved", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        entries = load_approved(a.approved)
    except ApprovedListError as e:
        logger.error("approved list rejected: {}", e)
        return 2
    summary = run(entries, a.dry_run)
    verb = "would add" if a.dry_run else "added"
    for s in summary["symbols"]:
        for r in s["entries"]:
            if r["status"] == "add":
                logger.info("{} {}: {} {} factor={}", verb, s["symbol"], str(r["ex_date"])[:10], r["action_type"],
                            r["adjustment_factor"])
            elif r["status"] == "failed":
                logger.error("FAILED {} {} {}: {}", s["symbol"], str(r["ex_date"])[:10], r["action_type"], r["reason"])
    logger.info("summary: {} {}, present {}, failed {}, fetched rows ignored {}", verb, summary["added"],
                summary["present"], summary["failed"], summary["ignored_fetched_rows"])
    if not a.dry_run:
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = SUMMARY_DIR / f"precutoff_corporate_actions_{stamp}.json"
        out.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        logger.info("summary written to {}", out)
    ok = summary["failed"] == 0 and summary["added"] + summary["present"] == summary["approved"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
