"""
features/blocks/fno.py

Stage 10 — India block, F&O sub-block (open interest, rollover, ban status).

Built from the raw F&O bhavcopy (data/raw/fno_bhavcopy) and the F&O ban lists (data/raw/fno_ban). One Parquet per
universe symbol, row-aligned 1:1 with the symbol's Stage 4 adjusted-price rows (same shape as delivery/pledging).

    fut_oi                 stock-futures open interest summed over all live expiries, Stage 4 split/bonus-adjusted
    fut_oi_chg_1d / _5d    fractional change of fut_oi vs the row 1 / 5 rows earlier (NaN if either end is missing)
    fut_oi_ratio_20d       fut_oi / trailing-20-row mean of fut_oi (needs MIN_OBS_20D observed days)
    fut_roll_next_share    OI(next expiry) / (OI(near) + OI(next)): the daily form of the rollover percentage
    fut_days_to_expiry     calendar days from the row date to the near expiry
    opt_pcr_oi             stock-options put OI / call OI, all live expiries
    fno_in_ban             1.0 / 0.0 when a usable ban list exists for the session, NaN otherwise
    fno_ban_new_entry      1.0 if banned today and observed not banned on the previous row (NaN if either is unavailable)
    fno_ban_days_20d       banned sessions in the trailing 20 rows (needs MIN_OBS_20D observed lists)
    is_missing_fno         no usable futures observation (not an F&O stock that day, or the day's file is absent)
    fno_day_unavailable    the day's F&O file itself does not exist (distinguishes an outage from "not an F&O stock")
    is_missing_opt         no usable put AND call open interest
    is_missing_ban         no usable ban list for the session

Missing is NaN plus a flag, never 0. fno_in_ban == 0.0 with is_missing_ban False says only "a list was published and this
symbol is not on it": it does NOT imply the symbol is in F&O (a cash-only stock gets 0.0 too). Use is_missing_fno for membership. NaN with the matching flag False is "undefined" (a zero denominator, no next
contract listed, warm-up), not missing data. A real zero (a published list that omits the symbol, a contract with no
open interest) is a fact and stays 0.

Decisions, each measured or forced by the archive:

1. SAME-SESSION CONVENTION. Row t uses the F&O file dated t and the ban list dated t. The nightly run is 19:00 IST and the
   bhavcopy lands about 18:00 (docs, nightly timeline), the same convention circuits uses for its hit lists. The raw
   manifests carry no publication timestamp, so data/publication.py cannot be applied; a list that is public earlier only
   makes this conservative. "Banned on the NEXT session" is deliberately not provided: when it becomes public is unmeasured.
2. KEYED ON THE EMBEDDED DATE, never on the filename. A ban list is usable for session s iff a file whose rows carry DATE ==
   s exists. NIL is one null-symbol row: usable and empty, so every symbol gets 0.0. Sessions with no file (2008-11-28, the
   two stale re-serves 2014-02-19 and 2014-08-25, everything before 2008-01-01) are unavailable: NaN plus is_missing_ban.
   Lists dated for no session (2008-11-27, 2014-02-22) have no spine row and are never joined.
3. 2003-10-01 STAYS EXCLUDED. No F&O file exists for it; no drop-the-zero-copy rule is applied. The day is
   fno_day_unavailable like any archive gap.
4. FUTSTK / OPTSTK ONLY, and only LIVE contracts (EXPIRY_DATE >= the row date: 2002-09-27 still lists contracts that expired
   the day before). The expiry day itself is still "near". Index and interest-rate/VIX instruments are not stock features.
5. OI IS SPLIT/BONUS-ADJUSTED with the same per-row Stage 4 factor as delivery (ADJ_VOLUME / VOLUME). NSE rescales open
   interest at an ex-date, so unadjusted levels jump overnight and fut_oi_chg would register a corporate action as
   positioning. Ratios inside one day (rollover share, PCR) need no adjustment.
6. RAW-TICKER JOIN. The repo has no symbol-change table, and renames (BHARTI -> BHARTIARTL, INFOSYSTCH -> INFY) are not
   guessed: an unmatched ticker is NaN plus flagged, never joined to a guessed successor.
7. Duplicate live contracts in one day's file raise: summing them would double-count, picking one would be a guess.

compute_fno_features is pure (no I/O). The I/O layer streams the bhavcopy one day at a time, aggregates each day to one
row per symbol, and only then builds the per-symbol panels, so memory holds ~1M summary rows, not 180M contracts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from config import FNO_BAN_DIR, FNO_BHAVCOPY_DIR, FNO_FEATURES_CHECKPOINT, FNO_FEATURES_DIR
from data.ingest_common import write_parquet_atomic
from features.blocks.common import prune_orphaned_symbol_files
from features.blocks.delivery import load_price_panel
from signals.indicators import list_universe_symbols

FUTURE_INSTRUMENT = "FUTSTK"
OPTION_INSTRUMENT = "OPTSTK"
CALL_TYPES = ("CE", "CA")             # CA/PA: American-style codes used for stock options until ~2011
PUT_TYPES = ("PE", "PA")
MIN_OBS_20D = 15                      # observed days required inside a trailing-20 window (same as delivery)
WARMUP_DAYS = 60                      # calendar days of history loaded before a range start (>> 20 rows)

FEATURE_COLUMNS = [
    "fut_oi", "fut_oi_chg_1d", "fut_oi_chg_5d", "fut_oi_ratio_20d", "fut_roll_next_share", "fut_days_to_expiry",
    "opt_pcr_oi", "fno_in_ban", "fno_ban_new_entry", "fno_ban_days_20d",
]
FLAG_COLUMNS = ["is_missing_fno", "fno_day_unavailable", "is_missing_opt", "is_missing_ban"]
OUTPUT_COLUMNS = ["symbol", "date", *FEATURE_COLUMNS, *FLAG_COLUMNS]
SUMMARY_COLUMNS = ["fut_oi_raw", "near_oi", "next_oi", "days_to_expiry", "put_oi", "call_oi"]
BHAV_COLUMNS = ["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE", "DATE", "OPEN_INT"]
REQUIRED_OBS_COLUMNS = {
    "date", "volume_adj_factor", "fno_day_available", "ban_available", "in_ban", *SUMMARY_COLUMNS,
}


# --------------------------------------------------------------------------- #
# Pure computation — no I/O. See tests/test_fno_features.py.
# --------------------------------------------------------------------------- #

def summarise_fno_day(day: pd.DataFrame, trade_date) -> pd.DataFrame:
    """
    One day's raw contract rows -> one row per symbol, indexed by symbol, columns SUMMARY_COLUMNS.

    fut_oi_raw   sum of OPEN_INT over the symbol's live FUTSTK contracts (NaN if every OPEN_INT is NaN)
    near_oi      OI of the earliest live expiry (the expiry day itself counts as near)
    next_oi      OI of the following expiry in the same file (NaN when none is listed)
    days_to_expiry  calendar days from trade_date to the near expiry
    put_oi / call_oi  live OPTSTK open interest by side (NaN when that side has no rows)
    """
    trade_date = pd.Timestamp(trade_date).normalize()
    expiry = pd.to_datetime(day["EXPIRY_DATE"])
    live = day[((expiry >= trade_date) & day["INSTRUMENT"].isin([FUTURE_INSTRUMENT, OPTION_INSTRUMENT])).to_numpy()]

    futures = live[live["INSTRUMENT"] == FUTURE_INSTRUMENT]
    options = live[live["INSTRUMENT"] == OPTION_INSTRUMENT]
    for name, frame, key in (("futures", futures, ["SYMBOL", "EXPIRY_DATE"]),
                             ("options", options, ["SYMBOL", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE"])):
        dup = frame.duplicated(key, keep=False)
        if dup.any():
            raise ValueError(f"{trade_date.date()}: {int(dup.sum())} duplicated live {name} contract rows "
                             f"(e.g. {frame.loc[dup, key].head(2).values.tolist()}); refusing to double-count or guess")

    out = pd.DataFrame(index=pd.Index(sorted(set(live["SYMBOL"])), name="symbol"), columns=SUMMARY_COLUMNS, dtype=float)
    if len(futures):
        total = futures.groupby("SYMBOL")["OPEN_INT"].sum(min_count=1)
        per_expiry = (futures.assign(EXPIRY_DATE=pd.to_datetime(futures["EXPIRY_DATE"]))
                      .groupby(["SYMBOL", "EXPIRY_DATE"])["OPEN_INT"].sum(min_count=1).reset_index()
                      .sort_values(["SYMBOL", "EXPIRY_DATE"]))
        rank = per_expiry.groupby("SYMBOL").cumcount()
        near = per_expiry[rank == 0].set_index("SYMBOL")
        nxt = per_expiry[rank == 1].set_index("SYMBOL")
        out.loc[total.index, "fut_oi_raw"] = total
        out.loc[near.index, "near_oi"] = near["OPEN_INT"]
        out.loc[near.index, "days_to_expiry"] = (near["EXPIRY_DATE"] - trade_date).dt.days
        out.loc[nxt.index, "next_oi"] = nxt["OPEN_INT"]
    if len(options):
        for column, codes in (("call_oi", CALL_TYPES), ("put_oi", PUT_TYPES)):
            side = options[options["OPTION_TYPE"].isin(codes)].groupby("SYMBOL")["OPEN_INT"].sum(min_count=1)
            out.loc[side.index, column] = side
    return out


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def compute_fno_features(obs: pd.DataFrame) -> pd.DataFrame:
    """
    F&O features for ONE symbol.

    obs : one row per price-panel date, ascending, columns
          {date, volume_adj_factor, fno_day_available, ban_available, in_ban, *SUMMARY_COLUMNS}.
          fno_day_available: the day's F&O file exists. ban_available: a usable ban list exists for the session.
          in_ban: the symbol is on that list (only meaningful where ban_available). SUMMARY_COLUMNS are NaN where the
          symbol has no row that day. Causal: row t depends only on rows <= t.
    """
    missing = REQUIRED_OBS_COLUMNS - set(obs.columns)
    if missing:
        raise ValueError(f"compute_fno_features: missing required columns {missing}")
    if len(obs) > 1 and not obs["date"].is_monotonic_increasing:
        raise ValueError("compute_fno_features: input must be sorted ascending by date")

    df = obs.reset_index(drop=True)
    day_ok = _bool(df["fno_day_available"])
    ban_ok = _bool(df["ban_available"])

    usable = day_ok & df["fut_oi_raw"].notna() & (df["volume_adj_factor"] > 0)
    oi = (df["fut_oi_raw"] * df["volume_adj_factor"]).where(usable)

    out = pd.DataFrame({"symbol": None, "date": df["date"]})
    out["fut_oi"] = oi
    for window in (1, 5):
        prior = oi.shift(window)
        out[f"fut_oi_chg_{window}d"] = oi / prior.where(prior > 0) - 1
    mean_20d = oi.rolling(20, min_periods=MIN_OBS_20D).mean()
    out["fut_oi_ratio_20d"] = oi / mean_20d.where(mean_20d > 0)

    denominator = df["near_oi"] + df["next_oi"]
    out["fut_roll_next_share"] = (df["next_oi"] / denominator.where(denominator > 0)).where(usable)
    out["fut_days_to_expiry"] = df["days_to_expiry"].where(usable)

    opt_ok = day_ok & df["put_oi"].notna() & df["call_oi"].notna()
    out["opt_pcr_oi"] = (df["put_oi"] / df["call_oi"].where(df["call_oi"] > 0)).where(opt_ok)

    banned = _bool(df["in_ban"]).astype(float).where(ban_ok)          # NaN, never 0, where there is no usable list
    previous = banned.shift(1)
    out["fno_in_ban"] = banned
    out["fno_ban_new_entry"] = ((banned == 1.0) & (previous == 0.0)).astype(float).where(banned.notna() & previous.notna())
    out["fno_ban_days_20d"] = banned.rolling(20, min_periods=MIN_OBS_20D).sum()

    out["is_missing_fno"] = ~usable
    out["fno_day_unavailable"] = ~day_ok
    out["is_missing_opt"] = ~opt_ok
    out["is_missing_ban"] = ~ban_ok
    return out


# --------------------------------------------------------------------------- #
# I/O layer — raw Parquet -> per-symbol feature Parquet.
# --------------------------------------------------------------------------- #

def _files_in_range(directory: Path, start, end) -> list[Path]:
    """Files whose NAME falls in [start, end]. Selection only: rows are always dated by their embedded DATE."""
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    return [p for p in sorted(directory.glob("*.parquet")) if lo <= pd.Timestamp(p.stem) <= hi]


def _embedded_date(frame: pd.DataFrame, path: Path) -> pd.Timestamp | None:
    if frame.empty:
        return None
    dates = pd.to_datetime(frame["DATE"]).dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError(f"{path.name}: rows carry {len(dates)} distinct dates ({sorted(str(d)[:10] for d in dates)})")
    return pd.Timestamp(dates[0])


def load_fno_summaries(start=None, end=None, directory: Path = FNO_BHAVCOPY_DIR) -> tuple[pd.DataFrame, set]:
    """
    Stream the F&O bhavcopy one day at a time. Returns (summaries, available_days): summaries has columns
    {symbol, date, *SUMMARY_COLUMNS}, one row per (symbol, date); available_days is the set of embedded dates of
    the files that exist and hold rows.
    """
    frames, available = [], set()
    renamed = []
    for path in tqdm(_files_in_range(directory, start, end), desc="F&O bhavcopy"):
        day = pq.read_table(path, columns=BHAV_COLUMNS).to_pandas()
        when = _embedded_date(day, path)
        if when is None:
            logger.warning(f"{path.name}: empty file, day treated as unavailable")
            continue
        if when.strftime("%Y-%m-%d") != path.stem:
            renamed.append((path.stem, when.strftime("%Y-%m-%d")))
        if when in available:
            raise ValueError(f"{path.name}: a second file carries {when.date()}; two files for one session")
        available.add(when)
        summary = summarise_fno_day(day, when).reset_index()
        summary.insert(1, "date", when)
        frames.append(summary)
    if renamed:
        logger.warning(f"{len(renamed)} F&O file(s) hold rows for a different date than their name (dated by rows): {renamed[:6]}")
    summaries = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=["symbol", "date", *SUMMARY_COLUMNS]))
    return summaries, available


def load_ban_lists(start=None, end=None, directory: Path = FNO_BAN_DIR) -> tuple[pd.DataFrame, set]:
    """
    Returns (bans, available_days): bans has {symbol, date} for every (session, banned symbol); available_days is the
    set of embedded dates with a usable list. A NIL list (one null-symbol row) is available and contributes no pairs.
    """
    pairs, available = [], set()
    for path in _files_in_range(directory, start, end):
        frame = pd.read_parquet(path, columns=["DATE", "SYMBOL"])
        when = _embedded_date(frame, path)
        if when is None:
            logger.warning(f"{path.name}: empty ban file, day treated as unavailable")
            continue
        if when in available:
            raise ValueError(f"{path.name}: a second ban file carries {when.date()}")
        available.add(when)
        named = frame["SYMBOL"].dropna().astype(str).str.strip()
        pairs.append(pd.DataFrame({"symbol": named[named != ""].to_numpy(), "date": when}))
    bans = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame({"symbol": [], "date": []})
    return bans.drop_duplicates(), available


def build_symbol_features(symbol: str, panel: pd.DataFrame, summaries: pd.DataFrame | None, fno_days: set,
                          ban_dates: set, ban_available: set) -> pd.DataFrame:
    """Left-join one symbol's F&O summaries and ban dates onto its price-panel dates and compute the features."""
    obs = panel[["date", "volume_adj_factor"]].copy()
    if summaries is None or summaries.empty:
        for column in SUMMARY_COLUMNS:
            obs[column] = np.nan
    else:
        obs = obs.merge(summaries[["date", *SUMMARY_COLUMNS]], on="date", how="left", validate="one_to_one")
    obs["fno_day_available"] = obs["date"].isin(fno_days)
    obs["ban_available"] = obs["date"].isin(ban_available)
    obs["in_ban"] = obs["date"].isin(ban_dates)
    features = compute_fno_features(obs)
    features["symbol"] = symbol
    return features[OUTPUT_COLUMNS]


def _load_checkpoint() -> set[str]:
    if FNO_FEATURES_CHECKPOINT.exists():
        return set(json.loads(FNO_FEATURES_CHECKPOINT.read_text())["completed_symbols"])
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    FNO_FEATURES_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    FNO_FEATURES_CHECKPOINT.write_text(json.dumps({
        "completed_symbols": sorted(completed),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))


def run_fno_pipeline(resume: bool = True, start=None, end=None, symbols: list[str] | None = None,
                     output_dir: Path | None = None, list_orphans_only: bool = False) -> dict:
    """
    Compute and write F&O features, one Parquet per symbol.

    Default: every universe symbol, full history, into FNO_FEATURES_DIR with a per-symbol checkpoint (rerun with
    resume=False whenever raw F&O files change). With start/end/symbols/output_dir the run is a scoped development run:
    it loads WARMUP_DAYS of history before `start` so the windows are exact, writes only rows in [start, end], never
    touches the checkpoint and never prunes. Scoped rows equal the same rows of a full run (tested).
    """
    scoped = any(x is not None for x in (start, end, symbols, output_dir))
    target = Path(output_dir) if output_dir is not None else FNO_FEATURES_DIR
    universe = list_universe_symbols()
    chosen = universe if symbols is None else [s for s in symbols if s in set(universe)]
    if not scoped:
        prune_orphaned_symbol_files(target, universe, list_only=list_orphans_only)
    completed = _load_checkpoint() if (resume and not scoped) else set()
    remaining = [s for s in chosen if s not in completed]

    load_start = pd.Timestamp(start) - timedelta(days=WARMUP_DAYS) if start is not None else None
    summaries, fno_days = load_fno_summaries(load_start, end)
    bans, ban_available = load_ban_lists(load_start, end)
    by_symbol = {s: g for s, g in summaries.groupby("symbol", sort=False)}
    banned_dates = {s: set(g["date"]) for s, g in bans.groupby("symbol", sort=False)}
    logger.info(f"F&O pipeline: {len(chosen):,} symbols ({len(completed):,} done, {len(remaining):,} remaining); "
                f"{len(fno_days):,} F&O days, {len(ban_available):,} ban lists, {len(summaries):,} symbol-day summaries")

    rows = rows_with_fno = rows_with_ban = 0
    target.mkdir(parents=True, exist_ok=True)
    for symbol in tqdm(remaining, desc="fno features"):
        panel = load_price_panel(symbol)
        if load_start is not None:
            panel = panel[panel["date"] >= load_start]
        if end is not None:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
        features = build_symbol_features(symbol, panel.reset_index(drop=True), by_symbol.get(symbol), fno_days,
                                         banned_dates.get(symbol, set()), ban_available)
        if start is not None:
            features = features[features["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        write_parquet_atomic(features, target / f"{symbol}.parquet")
        rows += len(features)
        rows_with_fno += int((~features["is_missing_fno"]).sum())
        rows_with_ban += int((~features["is_missing_ban"]).sum())
        if not scoped:
            completed.add(symbol)
            _save_checkpoint(completed)

    summary = {"symbols_written": len(remaining), "rows": rows, "rows_with_fno": rows_with_fno, "rows_with_ban_list": rows_with_ban}
    logger.success(f"F&O pipeline complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LYNX Stage 10 — F&O feature block")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoint, recompute all")
    parser.add_argument("--start", help="scoped run: first date to write (YYYY-MM-DD); needs --output-dir")
    parser.add_argument("--end", help="scoped run: last date to write")
    parser.add_argument("--symbols", nargs="*", help="scoped run: only these symbols")
    parser.add_argument("--output-dir", type=Path, help="scoped run: write here instead of FNO_FEATURES_DIR")
    parser.add_argument("--list-orphans-only", action="store_true", help="orphans are listed, not deleted; normal processing continues")
    args = parser.parse_args()
    if (args.start or args.end or args.symbols) and args.output_dir is None:
        parser.error("a scoped run (--start/--end/--symbols) must name --output-dir; it never overwrites the real block")
    run_fno_pipeline(resume=not args.no_resume, start=args.start, end=args.end, symbols=args.symbols, output_dir=args.output_dir,
                     list_orphans_only=args.list_orphans_only)
