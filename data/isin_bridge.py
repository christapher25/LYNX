"""
Stage 16 row 16 -- ISIN bridge: resolve an announcement through its ISIN to the symbol valid ON THE ANNOUNCEMENT DATE.

Why it exists: the NSE announcements API stamps the company's CURRENT symbol on historical records, so a renamed company's
old records carry a symbol that did not exist (or was a different company) on the day. The ISIN survives a rename. The
bhavcopy symbol master holds, per (symbol, ISIN), the first and last session it traded; the bridge picks the symbol whose
window contains the announcement date.

Rules (nothing is guessed):
  * exactly one distinct symbol has a window containing the date  -> that symbol
  * two or more                                                    -> unresolved, "bridge_isin_ambiguous"
  * none: ISIN unknown / date before the first window / after the last / between two windows / date unknown
                                                                    -> unresolved, with that specific reason
No slack is applied: a record dated before the ISIN's first listing stays unresolved.
"""

from __future__ import annotations

from datetime import date

BRIDGE_REASONS = (
    "bridge_isin_not_in_master", "bridge_no_date", "bridge_before_first_listing", "bridge_after_last_session",
    "bridge_gap_between_windows", "bridge_isin_ambiguous",
)
BRIDGE_METHOD = "isin_bridge"


def bridge_isin(isin: str, day: date | None, master) -> tuple[str | None, str | None]:
    """(symbol, None) when unambiguous, else (None, reason). `master` needs `symbols_for_isin(isin) -> [(sym, first, last)]`."""
    candidates = master.symbols_for_isin(isin)
    if not candidates:
        return None, "bridge_isin_not_in_master"
    if day is None:
        return None, "bridge_no_date"
    inside = {sym for sym, first, last in candidates if first <= day <= last}
    if len(inside) == 1:
        return next(iter(inside)), None
    if len(inside) > 1:
        return None, "bridge_isin_ambiguous"
    if day < min(first for _, first, _ in candidates):
        return None, "bridge_before_first_listing"
    if day > max(last for _, _, last in candidates):
        return None, "bridge_after_last_session"
    return None, "bridge_gap_between_windows"


def pit_window_ok(ticker: str, isin: str, day: date, master) -> bool:
    """Independent PIT audit of a bridged row: exactly ONE distinct symbol's window for this ISIN contains `day`, and it is `ticker`."""
    inside = {s for s, lo, hi in master.symbols_for_isin(isin) if lo <= day <= hi}
    return inside == {ticker}


# ---------------------------------------------------------------------------------------------------------------------
# Re-resolve existing monthly files (separate, resumable mode; no network)
#   python -m data.isin_bridge --re-resolve [--dir PATH] [--start YYYY-MM] [--end YYYY-MM] [--no-resume]
# Only rows that are unresolved for "symbol_not_in_master" / "symbol_outside_traded_window" and carry an ISIN are touched; every
# other row (and every other column) is left byte-for-byte as it was. A file is rewritten atomically only if a row changed, and
# its name is recorded in a checkpoint after each file, so an interrupted run resumes and a finished run is a no-op.
# ---------------------------------------------------------------------------------------------------------------------

ELIGIBLE_REASONS = ("symbol_not_in_master", "symbol_outside_traded_window")
CHECKPOINT_NAME = "_isin_bridge_checkpoint.json"


def re_resolve_frame(df, master):
    """Return (new_frame, n_bridged, n_reason_changed). Pure; input untouched."""
    import pandas as pd
    out = df.copy()
    mask = ((out["RESOLUTION_METHOD"] == "unresolved") & out["UNRESOLVED_REASON"].isin(ELIGIBLE_REASONS) & out["ISIN"].notna())
    if not mask.any():
        return out, 0, 0
    from data.entity_resolution import _clean_isin
    bridged = changed = 0
    for idx in out.index[mask]:
        isin = _clean_isin(out.at[idx, "ISIN"])
        if isin is None:
            continue
        sym, reason = bridge_isin(isin, pd.Timestamp(out.at[idx, "PUBLISHED_DATE"]).date() if pd.notna(out.at[idx, "PUBLISHED_DATE"]) else None, master)
        if sym is not None:
            out.at[idx, "TICKER"], out.at[idx, "RESOLUTION_METHOD"], out.at[idx, "UNRESOLVED_REASON"] = sym, BRIDGE_METHOD, None
            bridged += 1
        else:
            out.at[idx, "UNRESOLVED_REASON"] = reason
            changed += 1
    return out, bridged, changed


def re_resolve_dir(directory, master, *, resume: bool = True, start: str | None = None, end: str | None = None) -> dict:
    import json
    import os
    from pathlib import Path

    import pandas as pd
    from loguru import logger

    directory = Path(directory)
    ckpt_path = directory / CHECKPOINT_NAME
    fingerprint = master.fingerprint()
    done: set[str] = set()
    if resume and ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if saved.get("master") == fingerprint:
            done = set(saved["done"])
        else:
            logger.warning(f"master fingerprint changed ({saved.get('master')} -> {fingerprint}); ignoring checkpoint, reprocessing all files")
    totals = {"files": 0, "skipped": 0, "rewritten": 0, "bridged": 0, "reason_changed": 0}
    for f in sorted(directory.glob("announcements_*.parquet")):
        month = f.name[len("announcements_"):][:7]
        if (start and month < start) or (end and month > end):
            continue
        totals["files"] += 1
        if f.name in done:
            totals["skipped"] += 1
            continue
        df = pd.read_parquet(f)
        new, n_b, n_c = re_resolve_frame(df, master)
        if n_b or n_c:
            tmp = f.with_suffix(".parquet.tmp")
            new.to_parquet(tmp, index=False)
            os.replace(tmp, f)
            totals["rewritten"] += 1
        totals["bridged"] += n_b
        totals["reason_changed"] += n_c
        done.add(f.name)
        tmp_ck = ckpt_path.with_suffix(".tmp")
        tmp_ck.write_text(json.dumps({"done": sorted(done), "master": fingerprint}), encoding="utf-8")
        os.replace(tmp_ck, ckpt_path)
        logger.info(f"{f.name}: bridged={n_b} reason_changed={n_c}")
    logger.info(f"ISIN re-resolve complete: {totals}")
    return totals


def main() -> int:
    import argparse
    from pathlib import Path

    from loguru import logger

    from config import ANNOUNCEMENTS_DIR, BHAVCOPY_DIR
    from data.entity_resolution import build_symbol_master

    ap = argparse.ArgumentParser(description="Re-resolve unresolved announcement records through the ISIN bridge.")
    ap.add_argument("--re-resolve", action="store_true", required=True)
    ap.add_argument("--dir", type=str)
    ap.add_argument("--start", type=str, help="first month YYYY-MM")
    ap.add_argument("--end", type=str, help="last month YYYY-MM")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    directory = Path(args.dir) if args.dir else ANNOUNCEMENTS_DIR
    master = build_symbol_master(BHAVCOPY_DIR)
    re_resolve_dir(directory, master, resume=not args.no_resume, start=args.start, end=args.end)
    logger.info("done")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
