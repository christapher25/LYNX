"""
Stage 10 (breadth) checkpoint verification, against the real files on disk.

  data/processed/features/breadth/universe_breadth.parquet  =  universe_pct_above_200dma over the Stage 5 PIT universe

Every check returns a list of problems and is run on the real output (must be empty) and on an in-memory copy with one bug
injected (must be non-empty); see scripts/verify_common.py.

  1. Structure          one row per trading session, exact columns, bool flag == value is NaN.
  2. Independent rebuild every session recomputed from the raw Stage 4 panel by a different algorithm (sliding-window
                         means in numpy, per symbol) and compared with the file: all sessions, not a sample.
  3. Loop-based spot check  25 sessions recomputed by plain Python loops (math.fsum, every member).
  4. Missing sessions   every flagged session is explained: warm-up, a weekend special session, the known 2020-07-13 gap,
                         or after the panel's last price date. A new unexplained gap fails.
  5. Plausibility       breadth is low at the 2008 and 2020 lows and high at the 2021 peak (catches an inverted sign).
  6. Truncation         rebuilding from data cut at 3 dates never changes an earlier row.
  7. Adjustment         rescaling earlier closes of 5 stocks (what a later split does) changes no earlier row.

Usage: python -m scripts.verify_stage10_breadth
"""

from __future__ import annotations

import math
import sys
from bisect import bisect_right
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from config import BREADTH_FEATURES_DIR
from data.trading_calendar import trading_sessions
from features.blocks import breadth
from scripts.verify_common import Reporter

WINDOW = breadth.BREADTH_WINDOW
KNOWN_MISSING_WEEKDAYS = {
    date(2020, 7, 13): "the Stage 3 bhavcopy for this trading day was rejected (two-digit year); DATA_NOTES.md §11",
}
PLAUSIBILITY = [("2008-11-20", "<=", 10.0, "global financial crisis low"), ("2020-03-23", "<=", 10.0, "COVID crash low"),
                ("2021-10-19", ">=", 70.0, "2021 peak")]          # measured 0.9 / 2.6 / 85.3
V = Reporter("Stage 10 breadth")


# ── independent expectations ─────────────────────────────────────────────────────────────────────────────────────────

def expected_breadth(panel: pd.DataFrame, members: dict[int, set[str]], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-symbol sliding-window means in numpy (a different algorithm from the block's pandas rolling), then counts per date."""
    n_eval = np.zeros(len(sessions))
    n_above = np.zeros(len(sessions))
    has_rows = np.zeros(len(sessions), dtype=bool)
    for symbol, g in panel.groupby("SYMBOL", sort=False):
        g = g.sort_values("DATE")
        dates = pd.DatetimeIndex(g["DATE"])
        idx = sessions.get_indexer(dates)
        has_rows[idx[idx >= 0]] = True
        close = g["ADJ_CLOSE"].to_numpy(float)
        close = np.where(close > 0, close, np.nan)
        if len(close) < WINDOW:
            continue
        mean = sliding_window_view(close, WINDOW).mean(axis=1)                     # NaN when any close in the window is missing
        at = np.arange(WINDOW - 1, len(close))
        years = dates.year.to_numpy()[at]
        member = np.fromiter((symbol in members.get(int(y), ()) for y in years), dtype=bool, count=len(at))
        ok = member & ~np.isnan(mean) & (idx[at] >= 0)
        above = ok & (close[at] > mean * (1.0 + breadth.EQUALITY_TOLERANCE))
        n_eval[idx[at][ok]] += 1
        n_above[idx[at][above]] += 1
    enough = n_eval >= breadth.MIN_EVALUABLE
    pct = np.where(enough, 100.0 * n_above / np.where(n_eval > 0, n_eval, 1), np.nan)
    out = pd.DataFrame({"date": sessions, "universe_pct_above_200dma": pct,
                        "universe_n_evaluable": np.where(has_rows, n_eval, np.nan)})
    out["is_missing_breadth"] = out["universe_pct_above_200dma"].isna()
    return out


def loop_pct(panel_by_symbol: dict, members: dict[int, set[str]], day: pd.Timestamp) -> float:
    """One session by plain loops: every member's last 200 closes averaged with math.fsum."""
    evaluable = above = 0
    for symbol in members.get(day.year, ()):
        dates, closes = panel_by_symbol.get(symbol, ((), ()))
        i = bisect_right(dates, day) - 1
        if i < WINDOW - 1 or dates[i] != day:
            continue
        window = closes[i - WINDOW + 1: i + 1]
        if not all(c > 0 for c in window):
            continue
        evaluable += 1
        above += closes[i] > math.fsum(window) / WINDOW * (1.0 + breadth.EQUALITY_TOLERANCE)
    return 100.0 * above / evaluable if evaluable >= breadth.MIN_EVALUABLE else float("nan")


# ── checks: each returns a list of problems ──────────────────────────────────────────────────────────────────────────

def structure_problems(out: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[str]:
    problems = []
    if list(out.columns) != ["date", *breadth.BREADTH_COLUMNS, "is_missing_breadth"]:
        return [f"columns are {list(out.columns)}"]
    if len(out) != len(sessions) or not (pd.DatetimeIndex(out["date"]) == sessions).all():
        problems.append(f"rows are not exactly the {len(sessions)} trading sessions")
    if out["is_missing_breadth"].dtype != bool:
        problems.append(f"is_missing_breadth is {out['is_missing_breadth'].dtype}, not a plain bool")
    pct = out["universe_pct_above_200dma"]
    if (out["is_missing_breadth"] != pct.isna()).any():
        problems.append(f"{int((out['is_missing_breadth'] != pct.isna()).sum())} rows where the flag disagrees with the value being NaN")
    if ((pct < 0) | (pct > 100)).any():
        problems.append("a percentage outside 0-100")
    if out.loc[pct.notna(), "universe_n_evaluable"].lt(breadth.MIN_EVALUABLE).any():
        problems.append("a value reported on fewer evaluable stocks than the minimum")
    return problems


def rebuild_problems(out: pd.DataFrame, want: pd.DataFrame) -> list[str]:
    cols = breadth.BREADTH_COLUMNS
    a, b = out.reset_index(drop=True), want.reset_index(drop=True)
    wrong = ~np.isclose(a[cols].to_numpy(float), b[cols].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)
    wrong |= a["is_missing_breadth"].to_numpy() != b["is_missing_breadth"].to_numpy()
    if not wrong.any():
        return []
    return [f"{int(wrong.sum())} of {len(a)} sessions differ from the independent rebuild, first at {a.loc[wrong, 'date'].iloc[0].date()}"]


def spot_problems(out: pd.DataFrame, panel_by_symbol: dict, members: dict[int, set[str]], seed: int = 7) -> list[str]:
    have = out[~out["is_missing_breadth"]].reset_index(drop=True)
    picks = np.random.default_rng(seed).choice(len(have), size=25, replace=False)
    problems = []
    for i in sorted(picks):
        day, got = have.loc[i, "date"], have.loc[i, "universe_pct_above_200dma"]
        want = loop_pct(panel_by_symbol, members, day)
        if not math.isclose(got, want, abs_tol=1e-9):
            problems.append(f"{day.date()}: file {got:.6f} vs loop {want:.6f}")
    return problems


def missing_problems(out: pd.DataFrame, last_price_date: pd.Timestamp) -> list[str]:
    """Every flagged session after the first valid one must be a weekend session, a documented gap, or beyond the price data."""
    first_valid = out.loc[~out["is_missing_breadth"], "date"].min()
    problems = []
    for day in out.loc[out["is_missing_breadth"] & (out["date"] > first_valid), "date"]:
        if day.weekday() >= 5 or day.date() in KNOWN_MISSING_WEEKDAYS or day > last_price_date:
            continue
        problems.append(f"{day.date()}: flagged missing with no documented reason")
    return problems


def plausibility_problems(out: pd.DataFrame) -> list[str]:
    series = out.set_index("date")["universe_pct_above_200dma"]
    problems = []
    for day, op, bound, label in PLAUSIBILITY:
        value = series.get(pd.Timestamp(day), float("nan"))
        ok = (value <= bound) if op == "<=" else (value >= bound)
        if not ok:
            problems.append(f"{day} ({label}): {value:.1f}%, expected {op} {bound:.0f}%")
    return problems


def truncation_problems(panel, members, sessions, out, fn=breadth.compute_breadth_features, n_cuts: int = 3) -> list[str]:
    modern = sessions[sessions >= "2000-01-01"]
    cuts = [modern[int(len(modern) * f)] for f in (0.4, 0.7, 0.92)][:n_cuts]
    problems = []
    for cut in cuts:
        part = fn(panel[panel["DATE"] <= cut], members, sessions[sessions <= cut])
        full = fn(panel, members, sessions) if fn is not breadth.compute_breadth_features else out
        a = full[full["date"] <= cut].reset_index(drop=True)
        changed = ~np.isclose(a[breadth.BREADTH_COLUMNS].to_numpy(float), part[breadth.BREADTH_COLUMNS].to_numpy(float),
                              rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)
        if changed.any():
            problems.append(f"cut at {cut.date()}: {int(changed.sum())} earlier rows changed when later data was removed")
    return problems


def adjustment_problems(panel, members, sessions, out) -> list[str]:
    """Rescale every close before d0 for 5 stocks by 0.37 (what a later split does to an adjusted panel); rows before d0 must not move."""
    counts = panel["SYMBOL"].value_counts()
    victims = list(counts[counts > 1500].index[:5])
    modern = sessions[sessions >= "2000-01-01"]
    d0 = modern[int(len(modern) * 0.6)]
    rescaled = panel.copy()
    rows = rescaled["SYMBOL"].isin(victims) & (rescaled["DATE"] < d0)
    rescaled.loc[rows, "ADJ_CLOSE"] *= 0.37
    after = breadth.compute_breadth_features(rescaled, members, sessions)
    keep = out["date"] < d0
    changed = ~np.isclose(out[keep][breadth.BREADTH_COLUMNS].to_numpy(float), after[keep][breadth.BREADTH_COLUMNS].to_numpy(float),
                          rtol=1e-9, atol=1e-9, equal_nan=True).all(axis=1)
    return [f"{int(changed.sum())} sessions before {d0.date()} changed when {len(victims)} stocks' earlier closes were rescaled"] if changed.any() else []


# ── main ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions())).normalize()
    out = pd.read_parquet(BREADTH_FEATURES_DIR / "universe_breadth.parquet")
    panel = breadth.load_adjusted_close_panel()
    members = breadth.load_universe_members()
    last_price_date = panel["DATE"].max()
    by_symbol = {s: (list(g["DATE"]), list(g["ADJ_CLOSE"])) for s, g in panel.sort_values(["SYMBOL", "DATE"]).groupby("SYMBOL", sort=False)}
    logger.info(f"{len(sessions)} sessions, {len(panel):,} price rows, {panel['SYMBOL'].nunique()} symbols, "
                f"universe years {min(members)}-{max(members)}, last price date {last_price_date.date()}")

    want = expected_breadth(panel, members, sessions)
    V.report("structure: one row per session, exact columns, flag == NaN, values in 0-100", structure_problems(out, sessions))
    V.report("independent numpy rebuild matches every session", rebuild_problems(out, want),
             f"({int((~out['is_missing_breadth']).sum())} sessions with data)")
    V.report("loop-based spot check: 25 sessions, every member, math.fsum", spot_problems(out, by_symbol, members))
    V.report("every missing session is explained (warm-up / weekend / 2020-07-13 / after the last price date)",
             missing_problems(out, last_price_date), f"({int(out['is_missing_breadth'].sum())} flagged in all)")
    V.report("plausibility: low at the 2008 and 2020 lows, high at the 2021 peak", plausibility_problems(out))
    V.report("truncation: removing later data changes no earlier row (3 cuts)", truncation_problems(panel, members, sessions, out))
    V.report("adjustment: rescaling earlier closes (a later split) changes no earlier row", adjustment_problems(panel, members, sessions, out))

    logger.info("--- injected bugs (in-memory; each must be caught) ---")
    inverted = out.copy()
    inverted["universe_pct_above_200dma"] = 100.0 - inverted["universe_pct_above_200dma"]
    V.expect_caught("sign inverted (share BELOW the average)", plausibility_problems(inverted))
    V.expect_caught("same bug, seen by the independent rebuild", rebuild_problems(inverted, want))

    nudged = out.copy()
    nudged.loc[out.index[~out["is_missing_breadth"]][3000], "universe_pct_above_200dma"] += 0.5
    V.expect_caught("one session off by 0.5 percentage points", rebuild_problems(nudged, want))

    zero_filled = out.copy()
    zero_filled[breadth.BREADTH_COLUMNS] = zero_filled[breadth.BREADTH_COLUMNS].fillna(0.0)
    V.expect_caught("missing sessions filled with 0", structure_problems(zero_filled, sessions) + rebuild_problems(zero_filled, want))

    early = out.copy()
    early[breadth.BREADTH_COLUMNS] = out[breadth.BREADTH_COLUMNS].shift(-1)
    early["is_missing_breadth"] = early["universe_pct_above_200dma"].isna()
    V.expect_caught("value visible one session early", rebuild_problems(early, want))

    unflagged = out.copy()
    unflagged.loc[out.index[out["is_missing_breadth"]][-1], "is_missing_breadth"] = False
    V.expect_caught("missing flag cleared on a NaN row", structure_problems(unflagged, sessions))

    gap_day = out.loc[~out["is_missing_breadth"], "date"].iloc[4000]
    hole = out.copy()
    hole.loc[hole["date"] == gap_day, ["universe_pct_above_200dma", "universe_n_evaluable", "is_missing_breadth"]] = [np.nan, np.nan, True]
    V.expect_caught(f"a real session silently blanked ({gap_day.date()}): an unexplained gap", missing_problems(hole, last_price_date))

    def newest_universe(p, m, s):
        return breadth.compute_breadth_features(p, {y: m[max(m)] for y in m}, s)
    survivor = newest_universe(panel, members, sessions)
    V.expect_caught("today's universe applied to every year (survivorship)", rebuild_problems(survivor, want))

    def full_sample_rank(p, m, s):
        result = breadth.compute_breadth_features(p, m, s)
        result["universe_pct_above_200dma"] = result["universe_pct_above_200dma"].rank(pct=True) * 100
        return result
    V.expect_caught("full-sample statistic (past rows move when data is added)",
                    truncation_problems(panel, members, sessions, out, fn=full_sample_rank, n_cuts=1))

    return V.finish()


if __name__ == "__main__":
    sys.exit(main())
