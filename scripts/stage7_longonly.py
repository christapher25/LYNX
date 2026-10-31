"""
Stage 7 long-only diagnostic (queue row 15a). NOT a gate, no GO/NO-GO.

Reuses eval/momentum_baseline.py unchanged (12-1 signal, Stage 5 point-in-time universe per year, month-end adjusted
closes, Stage 6 cost model, skip-on-missing-price handling) and reports the long-only top decile next to the existing
long-short decile spread.

Two views of the same code path:
  legacy  every year's universe is run over EVERY month of history, exactly as eval/momentum_baseline.main() does
          (this is what produces the published 7,749 rebalances / 4.69%). Used only to reproduce the published figure.
  clean   month m is traded only with the universe of m's own year, 2005-2025, one row per month. All statistics are
          computed on this series.

Cost conventions:
  round-trip   the Stage 7 convention: every name pays a full round trip every month.
  turnover     only names entering or leaving the decile pay; cost = turnover_fraction x average round-trip cost.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT  # noqa: E402
from costs.model import round_trip_cost_delivery  # noqa: E402
from vanta.m1 import newey_west_se  # noqa: E402


def _load_stage7():
    spec = importlib.util.spec_from_file_location("stage7_momentum_baseline", ROOT / "eval" / "momentum_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage7 = _load_stage7()
NW_LAGS = 3
INDEX_DIR = PROJECT_ROOT / "data" / "raw" / "indices"


# ---- selection and one-month legs ---------------------------------------------------------------------------------
def top_decile(panel_month: pd.DataFrame) -> list[str]:
    ranked = panel_month.sort_values("MOMENTUM_1M_LAG", ascending=False)
    k = max(1, int(len(ranked) * stage7.DECILE))
    return ranked.head(k)["SYMBOL"].tolist()


def turnover_fraction(prev: list[str] | None, cur: list[str]) -> float:
    """Share of the current equal-weight book that must be bought (equally, of the old book sold)."""
    if prev is None:
        return 1.0
    return 1.0 - len(set(prev) & set(cur)) / len(cur)


def long_only_month(panel_month, next_prices, as_of: date, exit_date: date, prev_longs) -> dict | None:
    longs = top_decile(panel_month)
    entry_map = panel_month.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    exit_map = next_prices.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    rets, costs = [], []
    for s in longs:
        e, x = entry_map.get(s), exit_map.get(s)
        if not stage7._valid_price(e) or not stage7._valid_price(x):
            continue
        q = max(1, int(stage7.POSITION_SIZE_RUPEES / e))
        rets.append((x - e) / e)
        costs.append(round_trip_cost_delivery(e, x, q, as_of)["total_pct_of_value"])
    if not rets:
        return None
    gross, rt = float(np.mean(rets)), float(np.mean(costs))
    turn = turnover_fraction(prev_longs, longs)
    return {"gross": gross, "rt_cost": rt, "net_roundtrip": gross - rt, "turnover": turn,
            "net_turnover": gross - turn * rt, "n": len(rets), "n_dropped": len(longs) - len(rets), "longs": longs}


def long_short_month(panel_month, next_prices, as_of: date, exit_date: date) -> dict | None:
    """Same quantities as eval.momentum_baseline.backtest_month (kept as a thin wrapper so the reproduction is by
    construction the Stage 7 code, not a re-implementation)."""
    return stage7.backtest_month(panel_month, next_prices, as_of, exit_date)


def ew_universe_month(panel_month, next_prices) -> float | None:
    entry_map = panel_month.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    exit_map = next_prices.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    r = [(exit_map[s] - e) / e for s, e in entry_map.items()
         if s in exit_map and stage7._valid_price(e) and stage7._valid_price(exit_map[s])]
    return float(np.mean(r)) if r else None


def run_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly loop of eval.momentum_baseline.main() on one universe's monthly panel. Index = signal month-end."""
    signal = stage7.compute_momentum_signal(panel.copy())
    months = sorted(signal["DATE"].unique())
    rows, prev = {}, None
    for i in range(len(months) - 1):
        this_m = signal[signal["DATE"] == months[i]]
        next_m = panel[panel["DATE"] == months[i + 1]]
        if this_m.empty or next_m.empty:
            continue
        as_of, exit_d = pd.Timestamp(months[i]).date(), pd.Timestamp(months[i + 1]).date()
        ls = long_short_month(this_m, next_m, as_of, exit_d)
        if not ls:
            continue
        lo = long_only_month(this_m, next_m, as_of, exit_d, prev)
        if not lo:
            continue
        prev = lo["longs"]
        rows[pd.Timestamp(months[i])] = {
            "ls_gross": ls["gross_spread"], "ls_cost": ls["avg_round_trip_cost"], "ls_net": ls["net_spread"],
            "lo_gross": lo["gross"], "lo_rt_cost": lo["rt_cost"], "lo_net_rt": lo["net_roundtrip"],
            "lo_turnover": lo["turnover"], "lo_net_turn": lo["net_turnover"], "lo_n": lo["n"],
            "lo_dropped": lo["n_dropped"], "ew_gross": ew_universe_month(this_m, next_m), "longs": lo["longs"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def finalize_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute turnover across a concatenated series (universe changes at year boundaries count as turnover)."""
    df = df.sort_index().copy()
    prev, turns = None, []
    for longs in df["longs"]:
        turns.append(turnover_fraction(prev, longs))
        prev = longs
    df["lo_turnover"] = turns
    df["lo_net_turn"] = df["lo_gross"] - df["lo_turnover"] * df["lo_rt_cost"]
    return df


# ---- statistics -----------------------------------------------------------------------------------------------------
def max_drawdown(x: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(x, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1.0))


def _block_boot_mean(x: np.ndarray, n_boot: int, seed: int, block: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = -(-n // block)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = x[idx].mean()
    return out


def series_stats(x, n_boot: int = 2000, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    ann = x.mean() * 12
    vol = x.std(ddof=1) * np.sqrt(12)
    se = newey_west_se(x, NW_LAGS) * 12
    boot = _block_boot_mean(x, n_boot, seed) * 12
    return {"n": len(x), "ann_return": float(ann), "vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else float("nan"),
            "hit_rate": float((x > 0).mean()), "max_dd": max_drawdown(x),
            "cagr": float(np.prod(1 + x) ** (12 / len(x)) - 1),
            "nw_se": float(se), "nw_ci95": (float(ann - 1.96 * se), float(ann + 1.96 * se)),
            "boot_ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))}


# ---- Nifty 50 -------------------------------------------------------------------------------------------------------
def nifty50_monthly(index_dir: Path | None = None) -> pd.Series:
    """Month-end Nifty 50 close (last file of each month in the Stage 8 index archive; starts 2012-02-21)."""
    files = sorted((index_dir or INDEX_DIR).glob("*.parquet"))
    last = {}
    for f in files:
        try:
            last[f.stem[:7]] = f
        except Exception:
            continue
    out = {}
    for ym, f in last.items():
        d = pd.read_parquet(f, columns=["INDEX_NAME", "DATE", "CLOSE"])
        row = d[d["INDEX_NAME"].str.strip().str.lower() == "nifty 50"]
        if len(row):
            out[pd.Period(ym, "M")] = float(row["CLOSE"].iloc[0])
    return pd.Series(out).sort_index()


def align_benchmark(nif: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Index return earned by a strategy row indexed m: end of month m to end of month m+1, i.e. pct_change().shift(-1).
    Months absent from the index series are NaN, never zero."""
    fwd = nif.pct_change().shift(-1)
    return pd.Series([fwd.get(k, np.nan) for k in index.to_period("M")], index=index)


def clean_series(parts_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Month m comes only from the run of m's own year; one row per month. Duplicated months are an error."""
    clean = pd.concat([res[res.index.year == year] for year, res in sorted(parts_by_year.items())])
    if not clean.index.is_unique:
        raise ValueError("clean series has duplicated month-ends")
    return finalize_turnover(clean)


# ---- driver ---------------------------------------------------------------------------------------------------------
def run(start_year=2005, end_year=2025):
    legacy_parts, clean_parts = [], {}
    for year in range(start_year, end_year + 1):
        universe = stage7.load_universe_year(year)
        if not universe:
            continue
        panel = stage7.load_monthly_panel(universe)
        if panel.empty:
            continue
        res = run_panel(panel)
        legacy_parts.append(res)
        clean_parts[year] = res
        logger.info("{}: {} rows (clean {})", year, len(res), int((res.index.year == year).sum()))
    legacy = pd.concat(legacy_parts)
    clean = clean_series(clean_parts)
    return legacy, clean


def _fmt(s):
    return (f"{s['ann_return']:.2%} | {s['vol']:.2%} | {s['sharpe']:.2f} | {s['hit_rate']:.1%} | {s['max_dd']:.1%} | "
            f"[{s['nw_ci95'][0]:.2%}, {s['nw_ci95'][1]:.2%}] | [{s['boot_ci95'][0]:.2%}, {s['boot_ci95'][1]:.2%}]")


def report(legacy: pd.DataFrame, clean: pd.DataFrame, out: Path) -> str:
    L = []
    leg_ls = legacy["ls_net"].mean() * 12
    L += ["# Stage 7 long-only diagnostic (queue row 15a)", "",
          "Diagnostic only. Not judged; no GO/NO-GO. Generated by `scripts/stage7_longonly.py`.", "",
          "## 1. Reproduction of the published long-short figure", "",
          f"- Published (CLAUDE.md / Stage 7 re-run 2026-09-30): net annualised spread 4.69%, 7,749 rebalances.",
          f"- Reproduced here with eval/momentum_baseline.py's own functions: **{leg_ls:.2%}**, {len(legacy)} rebalances "
          f"(mean gross {legacy['ls_gross'].mean():.4%}/month, mean cost {legacy['ls_cost'].mean():.4%}, "
          f"net positive {(legacy['ls_net'] > 0).mean():.1%}).", "",
          "**Finding about the published number.** `eval/momentum_baseline.main()` runs each year's universe over every "
          "month in that universe's history (one pass per year, 21 passes), so the 7,749 'rebalances' are about 21 "
          "overlapping copies of roughly 370 months, not ~250 monthly rebalances. A later year's top-700 is applied to "
          "earlier months, which is survivorship-conditioned, and it also trades months before 2005 and after the year "
          "named. The statistics below use the clean series (month m traded with the universe of its own year, "
          "2005-2025). The long-short figure for that series is in the table.", "",
          "## 2. Results, clean series 2005-2025", ""]
    n_clean = len(clean)
    L += [f"{n_clean} monthly rebalances ({clean.index.min().date()} to {clean.index.max().date()}). Annualised = monthly mean x 12 "
          f"(Stage 7's convention); vol = monthly std x sqrt(12); Sharpe = ann. return / vol with risk-free 0. CIs are 95% on the "
          f"annualised mean: Newey-West (Bartlett, {NW_LAGS} lags) and a moving-block bootstrap (block 3, 2000 draws, seed 0). "
          "Max drawdown from compounded monthly net returns.", "",
          "| Series | Net ann. return | Vol | Sharpe | Hit rate | Max DD | NW CI | Boot CI |", "|---|---|---|---|---|---|---|---|"]
    series = {
        "Long-short decile, net (Stage 7 cost: full round trip monthly)": clean["ls_net"],
        "Long-only top decile, net (round trip monthly)": clean["lo_net_rt"],
        "Long-only top decile, net (cost on turnover)": clean["lo_net_turn"],
        "Long-only top decile, gross": clean["lo_gross"],
        "Equal-weight universe, gross": clean["ew_gross"],
        "Equal-weight universe, net (long-leg avg round-trip cost, every month)": clean["ew_gross"] - clean["lo_rt_cost"],
    }
    allstats = {}
    for name, x in series.items():
        allstats[name] = series_stats(x.to_numpy())
        L.append(f"| {name} | {_fmt(allstats[name])} |")
    # Excess over EW
    ex = (clean["lo_net_turn"] - (clean["ew_gross"] - clean["lo_rt_cost"]))
    s = series_stats(ex.to_numpy())
    L.append(f"| Long-only (turnover cost) minus EW universe net | {_fmt(s)} |")
    nif = nifty50_monthly()
    if len(nif):
        sub = align_benchmark(nif, clean.index).dropna()
        if len(sub):
            ns = series_stats(sub.to_numpy())
            lo_sub = clean.loc[sub.index, "lo_net_turn"]
            L.append(f"| Nifty 50 price index (no costs, no dividends; {sub.index.min().date()} on, {len(sub)} months) | {_fmt(ns)} |")
            L.append(f"| Long-only (turnover cost), same {len(sub)} months | {_fmt(series_stats(lo_sub.to_numpy()))} |")
    L += ["", "## 3. Turnover and breadth", "",
          f"- Long-only mean monthly one-way book turnover (names entering): {clean['lo_turnover'].mean():.1%}; "
          f"median {clean['lo_turnover'].median():.1%}.",
          f"- Mean long-leg round-trip cost: {clean['lo_rt_cost'].mean():.3%} of value; long-short (both legs) "
          f"{clean['ls_cost'].mean():.3%}; cost paid per month on turnover basis: "
          f"{(clean['lo_turnover'] * clean['lo_rt_cost']).mean():.3%}.",
          f"- Mean names per long book: {clean['lo_n'].mean():.1f}; names dropped for a missing entry/exit price "
          f"(NaN, never zero-filled): {int(clean['lo_dropped'].sum())} of {int(clean['lo_n'].sum() + clean['lo_dropped'].sum())} "
          f"selections ({clean['lo_dropped'].sum() / (clean['lo_n'].sum() + clean['lo_dropped'].sum()):.2%}).", "",
          "## 4. Cost assumptions and caveats", "",
          "- Costs: `costs.model.round_trip_cost_delivery` (STT, stamp, exchange, SEBI, flat brokerage Rs 20 per side, GST) "
          "with a Rs 100,000 position per name, rate table as of each rebalance date. Short leg uses `round_trip_cost_short` "
          "(unchanged Stage 7 code). No impact or slippage model, no bid-ask spread.",
          "- Turnover cost = fraction of the book entering the decile x the average round-trip cost (each traded name pays a buy "
          "and, eventually, a sell). Held names pay nothing and are assumed to keep equal weight without rebalancing cost.",
          "- Signal at month-end t uses closes <= t (12-month return, lagged one month). The position is entered at that same "
          "month-end close and held to the next month-end close. This is the Stage 7 convention, not 'next session': it "
          "assumes you can trade at the close that formed the signal, so returns are mildly optimistic.",
          "- Delisting handling is Stage 7's: a name with no valid next month-end price is skipped, not given its terminal value "
          "(the universe files carry `IS_DELISTED` and `TERMINAL_VALUE`, which Stage 7 does not use). This flatters both legs "
          "slightly; the dropped-name count above bounds it.",
          "- The universe is the Stage 5 point-in-time top-700 of the year, applied to all months of that calendar year.",
          "- Benchmark Nifty 50 is the price index (no dividends) and only exists from 2012-02 in the Stage 8 archive.",
          "- The equal-weight universe net line subtracts the long leg's average round-trip cost every month, as a "
          "proxy; the cost function was not run for all ~700 names.", ""]
    text = "\n".join(L)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def main():
    legacy, clean = run()
    text = report(legacy, clean, ROOT / "logs" / "stage7_longonly.md")
    logger.info("report written to {}", ROOT / "logs" / "stage7_longonly.md")
    logger.info("\n{}", text)


if __name__ == "__main__":
    main()
