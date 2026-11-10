"""
R20-4: read-only report on the unexplained >20% moves in data/processed/adjusted.

Definition is adjust_prices': abs(ADJ_CLOSE.pct_change()) > MAX_UNEXPLAINED_MOVE
and ADJUSTMENT_APPLIED is False on that row. The band is imported, and every
frame's count is cross-checked against adjust_prices.validate_adjustment.

Reads data/ only; writes the Markdown report and a CSV next to it, nothing else.

  python -m scripts.report_unexplained_moves --refresh-logs LOG [LOG ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, PROJECT_ROOT
from data.adjust_prices import (
    ACTIONS_DIR,
    MAX_UNEXPLAINED_MOVE,
    OUTPUT_DIR,
    PRICE_COLS,
    apply_backward_adjustment,
    validate_adjustment,
)

TOP_N = 200
FROM_2013 = pd.Timestamp("2013-01-01")
FROM_2016 = pd.Timestamp("2016-01-01")
DEFAULT_CANDIDATES = (PROJECT_ROOT / "data" / "processed" / "m1b"
                      / "candidates_cliff200_2010-01-08_2026-09-30_4e4ed37cbbda.parquet")
DEFAULT_M1 = (PROJECT_ROOT / "data" / "processed" / "m1"
              / "candidates_cliff200_2010-01-08_2026-09-30.parquet")
NEW_RE = re.compile(r"(\S+): NEW action ex_date (\d{4}-\d{2}-\d{2}) type (\S+)")
_NO_ACTIONS = ["ex_date", "adjustment_factor", "action_type"]


class ReportError(Exception):
    """Raised when the inputs cannot be reconciled; the CLI exits 1."""


def parse_refresh_logs(paths) -> list[tuple[str, pd.Timestamp, str]]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
            m = NEW_RE.search(line)
            if m:
                out.append((m.group(1), pd.Timestamp(m.group(2)), m.group(3)))
    return out


def unexplained_moves(symbol: str, adjusted: pd.DataFrame) -> pd.DataFrame:
    """Rows of one symbol's frame (sorted by DATE) that are unexplained moves."""
    f = adjusted.sort_values("DATE", kind="stable").reset_index(drop=True)
    ret = f["ADJ_CLOSE"].pct_change()
    mask = (ret.abs() > MAX_UNEXPLAINED_MOVE) & (~f["ADJUSTMENT_APPLIED"].astype(bool))
    res = pd.DataFrame({"symbol": symbol, "date": f.loc[mask, "DATE"].to_numpy(),
                        "ret": ret[mask].to_numpy()})
    if len(res) != len(validate_adjustment(symbol, f)):
        raise ReportError(f"{symbol}: move definition drifted from adjust_prices.validate_adjustment")
    return res


def _pre_refresh_actions(symbol, removals, actions_dir: Path) -> pd.DataFrame:
    path = actions_dir / f"{symbol}.parquet"
    if not path.exists():
        if removals:
            raise ReportError(f"{symbol}: logged NEW actions but no actions file")
        return pd.DataFrame(columns=_NO_ACTIONS)
    df = pd.read_parquet(path).reset_index(drop=True)
    for ex, typ in removals:
        hit = df.index[(df["ex_date"] == ex) & (df["action_type"] == typ)]
        if len(hit) == 0:
            raise ReportError(f"{symbol}: logged NEW action {ex.date()} {typ} not in actions file")
        df = df.drop(hit[0])
    usable = df[df["adjustment_factor"].notna()]
    return usable[_NO_ACTIONS].sort_values("ex_date")


def scan_symbols(adjusted_dir, actions_dir, cutoff, new_rows):
    """One symbol at a time. Returns (current list, pre-refresh counts, removed total)."""
    removals: dict[str, list] = {}
    for s, ex, typ in new_rows:
        removals.setdefault(s, []).append((ex, typ))
    files = sorted(Path(adjusted_dir).glob("*.parquet"))
    have = {f.stem for f in files}
    missing = sorted(set(removals) - have)
    if missing:
        raise ReportError(f"logged symbols without an adjusted file: {missing[:5]}")
    cur_parts, pre_counts, removed = [], {}, 0
    cols = ["DATE", "ADJ_CLOSE", "ADJUSTMENT_APPLIED", *PRICE_COLS, "VOLUME"]
    for f in files:
        sym = f.stem
        df = pd.read_parquet(f, columns=cols)
        cur = unexplained_moves(sym, df[["DATE", "ADJ_CLOSE", "ADJUSTMENT_APPLIED"]])
        if len(cur):
            cur_parts.append(cur)
        prices = (df[df["DATE"] <= cutoff][["DATE", *PRICE_COLS, "VOLUME"]]
                  .sort_values("DATE", kind="stable").reset_index(drop=True))
        rem = removals.get(sym, [])
        acts = _pre_refresh_actions(sym, rem, Path(actions_dir))
        removed += len(rem)
        pre = apply_backward_adjustment(prices, acts)
        pre_counts[sym] = len(unexplained_moves(sym, pre))
    if removed != len(new_rows):
        raise ReportError(f"removed {removed} rows but logs record {len(new_rows)}")
    current = (pd.concat(cur_parts, ignore_index=True) if cur_parts
               else pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "ret": []}))
    return current, pre_counts, removed


def top200_exposure(current: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Adds bracket/any flags and the bracketing decision dates and ranks (NaN stays NaN)."""
    c = cands[["decision_date", "symbol", "liquidity_rank"]]
    if c.duplicated(["symbol", "decision_date"]).any():
        raise ReportError("candidates file has duplicate (symbol, decision_date)")
    ranks = c.set_index(["symbol", "decision_date"])["liquidity_rank"]
    dec = np.sort(c["decision_date"].unique())
    cur = current.copy()
    idx = np.searchsorted(dec, cur["date"].to_numpy(), side="right")
    dec_s = pd.Series(pd.to_datetime(dec))
    cur["prev_dd"] = dec_s.reindex(idx - 1).to_numpy()   # label -1 / len -> NaT
    cur["next_dd"] = dec_s.reindex(idx).to_numpy()
    for tag in ("prev", "next"):
        key = pd.MultiIndex.from_arrays([cur["symbol"], cur[f"{tag}_dd"]])
        cur[f"{tag}_rank"] = ranks.reindex(key).to_numpy()
    for label, thr in (("2013", FROM_2013), ("2016", FROM_2016)):
        cur[f"bracket_{label}"] = (((cur["prev_rank"] <= TOP_N) & (cur["prev_dd"] >= thr))
                                   | ((cur["next_rank"] <= TOP_N) & (cur["next_dd"] >= thr)))
        syms = set(c.loc[(c["liquidity_rank"] <= TOP_N) & (c["decision_date"] >= thr), "symbol"])
        cur[f"any_{label}"] = cur["symbol"].isin(syms)
    return cur


def aastha(cands: pd.DataFrame, m1_path: Path, symbol: str = "AASTHA") -> dict:
    hit = cands[(cands["symbol"] == symbol) & (cands["liquidity_rank"] <= TOP_N)]
    dates = sorted(hit["decision_date"].dt.strftime("%Y-%m-%d"))
    out = {"all": dates,
           "2013": [d for d in dates if d >= "2013-01-01"],
           "2016": [d for d in dates if d >= "2016-01-01"]}
    m1 = pd.read_parquet(m1_path, columns=["symbol"]) if Path(m1_path).exists() else None
    out["m1"] = bool(m1 is not None and (m1["symbol"] == symbol).any())
    return out


def _dates_or_never(dates):
    return ", ".join(dates) if dates else "never"


def _top_table(df: pd.DataFrame, flag: str) -> list[str]:
    sub = df[df[flag]].reindex(df[df[flag]]["ret"].abs().sort_values(ascending=False).index).head(20)
    lines = ["| symbol | date | return | prev decision | rank | next decision | rank |",
             "|---|---|---|---|---|---|---|"]
    for r in sub.itertuples():
        lines.append(f"| {r.symbol} | {r.date.date()} | {r.ret:.4f} | "
                     f"{'NaT' if pd.isna(r.prev_dd) else r.prev_dd.date()} | {r.prev_rank} | "
                     f"{'NaT' if pd.isna(r.next_dd) else r.next_dd.date()} | {r.next_rank} |")
    return lines


def build_report(current, pre_counts, cutoff, cands, m1_path, n_new) -> str:
    cur = top200_exposure(current, cands)
    le = cur[cur["date"] <= cutoff]
    cur_by_sym = le.groupby("symbol").size().to_dict()
    diff = {s: (pre_counts.get(s, 0), cur_by_sym.get(s, 0))
            for s in set(pre_counts) | set(cur_by_sym)
            if pre_counts.get(s, 0) != cur_by_sym.get(s, 0)}
    a = aastha(cands, m1_path)
    L = ["# R20-4 unexplained moves (read-only)", "",
         f"Band: abs(ADJ_CLOSE return) > {MAX_UNEXPLAINED_MOVE} and ADJUSTMENT_APPLIED False. "
         f"Cutoff {cutoff.date()}. NEW actions removed for pre-refresh state: {n_new}.", "",
         "## A. Current state", f"current total: {len(cur)}",
         f"current total <= cutoff: {len(le)}", "",
         "## B. Pre-refresh state (recomputed in memory, dates <= cutoff)",
         f"pre-refresh total: {sum(pre_counts.values())}", "",
         "Symbols where pre-refresh and current differ (symbol: pre-refresh / current, dates <= cutoff):"]
    L += [f"- {s}: {p} / {c}" for s, (p, c) in sorted(diff.items())] or ["- none"]
    L += ["", "## C. After cutoff", f"after cutoff: {int((cur['date'] > cutoff).sum())}", "",
          "## D. Top-200 exposure (current list)",
          f"iii-bracket >=2013: {int(cur['bracket_2013'].sum())} (>=2016: {int(cur['bracket_2016'].sum())})",
          f"iii-any >=2013: {int(cur['any_2013'].sum())} (>=2016: {int(cur['any_2016'].sum())})", "",
          "### 20 largest, iii-bracket >=2013", *_top_table(cur, "bracket_2013"), "",
          "### 20 largest, iii-any >=2013", *_top_table(cur, "any_2013"), "",
          "## E. AASTHA",
          f"AASTHA rank<=200 (all dates): {_dates_or_never(a['all'])}",
          f"AASTHA rank<=200 (>=2013): {_dates_or_never(a['2013'])}",
          f"AASTHA rank<=200 (>=2016): {_dates_or_never(a['2016'])}",
          f"AASTHA in M1 file: {'yes' if a['m1'] else 'no'}", ""]
    return "\n".join(L), cur


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-logs", nargs="+", required=True)
    ap.add_argument("--cutoff", default="2026-09-29")
    ap.add_argument("--refresh-from", default="2026-09-30")
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--m1-candidates", default=str(DEFAULT_M1))
    ap.add_argument("--adjusted-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--actions-dir", default=str(ACTIONS_DIR))
    ap.add_argument("--out", default=str(LOG_DIR / "row20_unexplained_moves.md"))
    a = ap.parse_args(argv)
    cutoff, refresh_from = pd.Timestamp(a.cutoff), pd.Timestamp(a.refresh_from)
    out = Path(a.out)
    try:
        new_rows = parse_refresh_logs(a.refresh_logs)
        early = [r for r in new_rows if r[1] < refresh_from]
        if early:
            logger.warning("{} logged NEW rows have ex_date < {}", len(early), refresh_from.date())
        logger.info("{} NEW action lines parsed", len(new_rows))
        current, pre_counts, _ = scan_symbols(a.adjusted_dir, a.actions_dir, cutoff, new_rows)
        cands = pd.read_parquet(a.candidates, columns=["decision_date", "symbol", "liquidity_rank"])
        text, cur = build_report(current, pre_counts, cutoff, cands, a.m1_candidates, len(new_rows))
    except ReportError as e:
        logger.error("{}", e)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    cur.to_csv(out.with_suffix(".csv"), index=False)
    logger.info("wrote {} and {}", out, out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
