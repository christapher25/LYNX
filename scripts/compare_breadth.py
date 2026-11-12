"""
R20-8: read-only comparison of the breadth feature file before and after a rebuild, with the owner's STOP rule.

A date is a "history difference" when it is added, removed or changed AND earlier than the OLD LAST DATE (the maximum date of the
old file, missing-flagged rows included). Any history difference means exit 1 (STOP). Floats compare exactly (NaN equals NaN, no
tolerance): the exact comparison decides, the 1e-9 count is information only.

  python -m scripts.compare_breadth --old PATH [--new PATH] [--out PATH]

Exit 0: no history difference. 1: at least one (STOP). 2: unreadable input or duplicate dates. Writes only --out and its CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

VALUE_COLS = ["universe_pct_above_200dma", "universe_n_evaluable", "is_missing_breadth"]
PCT = "universe_pct_above_200dma"
MATERIAL = 1e-9
DEFAULT_NEW = config.BREADTH_FEATURES_DIR / "universe_breadth.parquet"
DEFAULT_OUT = config.LOG_DIR / "row20_breadth_compare.md"


class CompareError(Exception):
    """Unreadable input or duplicate dates; the CLI exits 2."""


def read_breadth(path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception as e:                                                       # noqa: BLE001
        raise CompareError(f"cannot read {path}: {e}") from e
    missing = [c for c in ["date", *VALUE_COLS] if c not in df.columns]
    if missing:
        raise CompareError(f"{path}: missing columns {missing}")
    df = df[["date", *VALUE_COLS]].copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    if df["date"].isna().any():
        # a NaT date compares False with everything, so such a row could never trip the STOP rule: the gate must not fail open
        raise CompareError(f"{path}: {int(df['date'].isna().sum())} row(s) with a missing (NaT) date")
    if df["date"].duplicated().any():
        raise CompareError(f"{path}: duplicate dates {df.loc[df['date'].duplicated(), 'date'].dt.date.tolist()[:5]}")
    return df.sort_values("date").reset_index(drop=True)


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def compare(old: pd.DataFrame, new: pd.DataFrame):
    """Returns (merged frame with class columns, old last date, old last non-missing date, info)."""
    old_last = old["date"].max()
    ok = old.loc[(~old["is_missing_breadth"].astype(bool)) & old[PCT].notna(), "date"]
    old_last_value = ok.max() if len(ok) else None
    m = old.merge(new, on="date", how="outer", suffixes=("_old", "_new"), indicator=True)
    m = m.sort_values("date").reset_index(drop=True)
    differing = pd.DataFrame({c: ~_same(m[f"{c}_old"], m[f"{c}_new"]) for c in VALUE_COLS})
    both = m["_merge"] == "both"
    m["class"] = np.where(m["_merge"] == "left_only", "removed",
                          np.where(m["_merge"] == "right_only", "added",
                                   np.where(differing.any(axis=1), "changed", "same")))
    m["before_old_last_date"] = m["date"] < old_last
    m["columns_differing"] = [";".join(c for c in VALUE_COLS if differing.at[i, c]) if both[i] else "" for i in m.index]
    both_pct = both & m[f"{PCT}_old"].notna() & m[f"{PCT}_new"].notna() & (m["class"] == "changed")
    diffs = (m.loc[both_pct, f"{PCT}_new"] - m.loc[both_pct, f"{PCT}_old"]).abs()
    info = {"material": int((diffs > MATERIAL).sum()), "max_abs": float(diffs.max()) if len(diffs) else 0.0}
    return m, old_last, old_last_value, info


def _fmt(v) -> str:
    if pd.isna(v):
        return "NaN"
    return repr(bool(v)) if isinstance(v, (bool, np.bool_)) else repr(float(v))


def _row_text(r) -> str:
    if r["class"] == "added":
        return "added: " + ", ".join(f"{c}={_fmt(r[f'{c}_new'])}" for c in VALUE_COLS)
    if r["class"] == "removed":
        return "removed: " + ", ".join(f"{c}={_fmt(r[f'{c}_old'])}" for c in VALUE_COLS)
    diff = [c for c in r["columns_differing"].split(";") if c]
    return "changed: " + "; ".join(f"{c} {_fmt(r[f'{c}_old'])} -> {_fmt(r[f'{c}_new'])}" for c in diff)


def build_report(m, old_last, old_last_value, info) -> tuple[str, int]:
    diff = m[m["class"] != "same"]
    hist = diff[diff["before_old_last_date"]]
    after = diff[~diff["before_old_last_date"]]
    n_hist = len(hist)
    verdict = f"HISTORY DIFFERENCES: {n_hist} (STOP)" if n_hist else "HISTORY DIFFERENCES: 0"
    last_value = old_last_value.date() if old_last_value is not None else "none"
    lines = [verdict, "", f"old last date: {old_last.date()}", f"old last non-missing date: {last_value}", "", "## Counts"]
    for cls in ("same", "added", "removed", "changed"):
        sub = m[m["class"] == cls]
        b = int(sub["before_old_last_date"].sum())
        lines.append(f"{cls}: {len(sub)} (before the old last date: {b}, on or after it: {len(sub) - b})")
    lines += ["", f"changed dates with abs pct difference > 1e-9: {info['material']}",
              f"maximum abs pct difference: {info['max_abs']!r}", "",
              "## History differences by year (dates before the old last date)"]
    by_year = hist.groupby(hist["date"].dt.year).size()
    lines += [f"- {y}: {n}" for y, n in by_year.items()] or ["- none"]
    lines += ["", f"## Every differing date before the old last date ({n_hist})"]
    lines += [f"- {r['date'].date()} {_row_text(r)}" for _, r in hist.iterrows()] or ["- none"]
    lines += ["", f"## Differing dates on or after the old last date ({len(after)})"]
    lines += [f"- {r['date'].date()} {_row_text(r)}" for _, r in after.iterrows()] or ["- none"]
    return "\n".join(lines) + "\n", n_hist


CSV_RENAME = {"universe_pct_above_200dma_old": "old_pct", "universe_pct_above_200dma_new": "new_pct",
              "universe_n_evaluable_old": "old_n", "universe_n_evaluable_new": "new_n",
              "is_missing_breadth_old": "old_missing", "is_missing_breadth_new": "new_missing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default=str(DEFAULT_NEW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    try:
        old, new = read_breadth(a.old), read_breadth(a.new)
        if old.empty:
            # no old rows means no old last date: every comparison would be False and the gate would pass on nothing
            raise CompareError(f"{a.old}: the old file has no rows, so there is nothing to compare against")
    except CompareError as e:
        logger.error("{}", e)
        return 2
    m, old_last, old_last_value, info = compare(old, new)
    text, n_hist = build_report(m, old_last, old_last_value, info)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    cols = ["date", "class", "before_old_last_date", "columns_differing",
            *[f"{c}_{s}" for c in VALUE_COLS for s in ("old", "new")]]
    m.loc[m["class"] != "same", cols].rename(columns=CSV_RENAME).to_csv(out.with_suffix(".csv"), index=False)
    counts = {c: int((m["class"] == c).sum()) for c in ("same", "added", "removed", "changed")}
    logger.info("{} | {} | wrote {}", text.splitlines()[0], counts, out)
    return 1 if n_hist else 0


if __name__ == "__main__":
    sys.exit(main())
