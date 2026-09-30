"""
Measure how well features/blocks/circuit_inference.py recovers the true price band, on the years where the truth is known.

Truth: for each (EQ symbol, session t) from 2020-01-01, the band in the sec_list file dated the previous session (the rule the
circuits block uses, itself measured at 97.2% agreement with hit-implied bands). Inference sees only prices and hits before t.

For each method (hit / envelope / combined) and each year it reports
  coverage  share of rows with a numeric true band that received an estimate
  accuracy  share of covered rows whose estimate equals the true band
  under / over   share of covered rows estimated too low / too high
and the confusion matrix of the combined method. Nothing downstream may rely on an inferred band unless this says it is accurate:
the decision rule is written in DATA_NOTES.md §13 and applied by the circuits pipeline.

Usage: python -m scripts.measure_band_inference [--out CSV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import BHAVCOPY_DIR, NSDL_FPI_DIR, PRICE_BAND_HITS_DIR, SEC_LIST_DIR
from data.trading_calendar import trading_sessions
from features.blocks.circuit_inference import infer_bands

TRUTH_FROM = pd.Timestamp("2020-01-01")


def load_prices() -> pd.DataFrame:
    """Raw (unadjusted) EQ prices as the exchange printed them: PREV_CLOSE is the exchange's own reference price."""
    frames = []
    for path in sorted(Path(BHAVCOPY_DIR).glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["SYMBOL", "SERIES", "DATE", "PREV_CLOSE", "HIGH", "LOW"])
        frames.append(frame[frame["SERIES"] == "EQ"].drop(columns="SERIES"))
    prices = pd.concat(frames, ignore_index=True)
    return prices.drop_duplicates(["SYMBOL", "DATE"]).sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)


def load_hits() -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=["DATE", "SYMBOL", "SERIES", "HIT"]) for p in sorted(Path(PRICE_BAND_HITS_DIR).glob("*.parquet"))]
    hits = pd.concat(frames, ignore_index=True)
    return hits[(hits["SERIES"] == "EQ") & hits["SYMBOL"].notna()].rename(columns={"DATE": "date"})[["SYMBOL", "date", "HIT"]]


def load_truth() -> pd.DataFrame:
    """{SYMBOL, DATE (the SESSION the band applies to), TRUE_BAND (NaN for No Band), IS_NO_BAND}."""
    sessions = [pd.Timestamp(s) for s in trading_sessions()]
    following = {a: b for a, b in zip(sessions[:-1], sessions[1:])}
    frames = []
    for path in sorted(Path(SEC_LIST_DIR).glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["DATE", "SYMBOL", "SERIES", "BAND_PCT", "IS_NO_BAND"])
        frames.append(frame[frame["SERIES"] == "EQ"])
    files = pd.concat(frames, ignore_index=True)
    files["DATE"] = files["DATE"].map(following)                     # the file dated d describes session d+1
    files = files.dropna(subset=["DATE"]).rename(columns={"BAND_PCT": "TRUE_BAND"})
    return files[["SYMBOL", "DATE", "TRUE_BAND", "IS_NO_BAND"]]


def run(out_csv: Path | None = None) -> pd.DataFrame:
    prices, hits, truth = load_prices(), load_hits(), load_truth()
    logger.info(f"{len(prices):,} price rows, {len(hits):,} EQ hits, {len(truth):,} truth rows "
                f"({truth['DATE'].min().date()} -> {truth['DATE'].max().date()})")
    hits_by = {s: g for s, g in hits.groupby("SYMBOL", sort=False)}
    estimates = []
    for symbol, g in prices.groupby("SYMBOL", sort=False):
        rows = g.rename(columns={"DATE": "date", "PREV_CLOSE": "prev_close", "HIGH": "high", "LOW": "low"}).reset_index(drop=True)
        symbol_hits = hits_by.get(symbol, pd.DataFrame(columns=["SYMBOL", "date", "HIT"]))
        combined = infer_bands(rows, symbol_hits[["date", "HIT"]])
        only_env = infer_bands(rows, symbol_hits[["date", "HIT"]].iloc[0:0])
        estimates.append(pd.DataFrame({"SYMBOL": symbol, "DATE": rows["date"], "combined": combined["BAND_PCT"], "method": combined["METHOD"],
                                       "envelope": only_env["BAND_PCT"]}))
    est = pd.concat(estimates, ignore_index=True)
    est["hit_only"] = est["combined"].where(est["method"] == "hit")
    joined = truth.merge(est, on=["SYMBOL", "DATE"], how="inner")
    joined = joined[(joined["DATE"] >= TRUTH_FROM) & ~joined["IS_NO_BAND"] & joined["TRUE_BAND"].notna()]
    joined["year"] = joined["DATE"].dt.year

    rows = []
    for year, g in list(joined.groupby("year")) + [("all", joined)]:
        for name in ("hit_only", "envelope", "combined"):
            covered = g[g[name].notna()]
            rows.append({"year": year, "method": name, "rows": len(g), "coverage": len(covered) / len(g),
                         "accuracy": float((covered[name] == covered["TRUE_BAND"]).mean()) if len(covered) else np.nan,
                         "under": float((covered[name] < covered["TRUE_BAND"]).mean()) if len(covered) else np.nan,
                         "over": float((covered[name] > covered["TRUE_BAND"]).mean()) if len(covered) else np.nan})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    covered = joined[joined["combined"].notna()]
    print("\nconfusion of the combined method (rows: true band, columns: estimate), all years:")
    print(pd.crosstab(covered["TRUE_BAND"], covered["combined"]).to_string())
    print("\nby METHOD used inside 'combined':", covered.groupby("method").apply(lambda g: float((g["combined"] == g["TRUE_BAND"]).mean())).round(3).to_dict(),
          "| counts:", covered["method"].value_counts().to_dict())
    print("\ntrue-band mix over the measured rows:", joined["TRUE_BAND"].value_counts(normalize=True).round(3).to_dict())
    if out_csv:
        table.to_csv(out_csv, index=False)
        logger.info(f"wrote {out_csv}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(SEC_LIST_DIR) / "_inference_measurement.csv")
    run(ap.parse_args().out)
