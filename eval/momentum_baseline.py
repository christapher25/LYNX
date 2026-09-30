"""
Stage 7 — momentum baseline, net of real costs. The go/no-go gate.

12-1 momentum: rank stocks by 12-month return, skipping the most recent
month (avoids short-term reversal contamination). Long top decile, short
bottom decile, monthly rebalance, using the REAL adjusted prices, REAL
point-in-time universe, and REAL cost model built tonight — not
placeholders.

If this doesn't clear costs, per the architecture doc: widen the horizon
or tighten the universe, not add model capacity. That decision needs to
happen now, in week four, not in week twenty.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT
from costs.model import round_trip_cost_delivery, round_trip_cost_short

ADJUSTED_DIR = PROJECT_ROOT / "data" / "processed" / "adjusted"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "processed" / "universe"

DECILE = 0.10
SKIP_MONTHS = 1
LOOKBACK_MONTHS = 12

POSITION_SIZE_RUPEES = 100_000   # fixed capital per leg, not a fixed share
                                  # count — see prior fix: a fixed share
                                  # count let flat brokerage dominate low-
                                  # priced stocks.


def load_universe_year(year: int) -> set[str]:
    path = UNIVERSE_DIR / f"{year}.parquet"
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["SYMBOL"])["SYMBOL"])


def load_monthly_panel(symbols: set[str]) -> pd.DataFrame:
    """
    Month-end adjusted close for every symbol in the given set. Loading
    per-symbol here (not a full-archive scan) since this only touches the
    ~700 in-universe names per year, not all 3,132.

    NOTE: resample('ME').last() on a groupby produces NaN for any month
    where a symbol has zero trading days — suspensions, delistings,
    genuine gaps. This is expected, not corrupted data; the downstream
    consumer (backtest_month) must check for it explicitly.
    """
    frames = []
    for sym in symbols:
        path = ADJUSTED_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["DATE", "ADJ_CLOSE"])
        df["SYMBOL"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.set_index("DATE").groupby("SYMBOL")["ADJ_CLOSE"].resample("ME").last()
    return panel.reset_index()


def compute_momentum_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 momentum: 12-month return ending 1 month before the signal date.
    Computed per symbol, using only that symbol's own trailing history —
    no cross-symbol leakage possible by construction.
    """
    panel = panel.sort_values(["SYMBOL", "DATE"])
    panel["RET_12M"] = panel.groupby("SYMBOL")["ADJ_CLOSE"].pct_change(LOOKBACK_MONTHS)
    panel["MOMENTUM_1M_LAG"] = panel.groupby("SYMBOL")["RET_12M"].shift(SKIP_MONTHS)
    return panel.dropna(subset=["MOMENTUM_1M_LAG"])


def _valid_price(x) -> bool:
    """True only for a real, usable, positive price — catches None, NaN,
    and non-positive values in one place rather than three separate
    checks that are each individually easy to get subtly wrong."""
    return x is not None and pd.notna(x) and x > 0


def backtest_month(
    panel_month: pd.DataFrame, next_month_prices: pd.DataFrame,
    as_of: date, exit_date: date,
) -> dict | None:
    """One rebalance: rank by momentum, form decile portfolios. Long and
    short legs now use genuinely different cost functions — see
    costs/model.py's round_trip_cost_short for why they can't share one
    formula (stamp duty falls on a different leg; shorts carry an SLB
    lending fee longs never do)."""
    ranked = panel_month.sort_values("MOMENTUM_1M_LAG", ascending=False)
    n = len(ranked)
    decile_size = max(1, int(n * DECILE))

    longs = ranked.head(decile_size)["SYMBOL"].tolist()
    shorts = ranked.tail(decile_size)["SYMBOL"].tolist()
    if not longs or not shorts:
        return None

    price_map = next_month_prices.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    entry_map = panel_month.set_index("SYMBOL")["ADJ_CLOSE"].to_dict()
    holding_days = max(1, (exit_date - as_of).days)

    long_rets, long_costs = [], []
    for sym in longs:
        entry, exit_ = entry_map.get(sym), price_map.get(sym)
        if not _valid_price(entry) or not _valid_price(exit_):
            continue
        quantity = max(1, int(POSITION_SIZE_RUPEES / entry))
        gross_ret = (exit_ - entry) / entry
        cost = round_trip_cost_delivery(entry, exit_, quantity, as_of)["total_pct_of_value"]
        long_rets.append(gross_ret)
        long_costs.append(cost)

    short_rets, short_costs = [], []
    for sym in shorts:
        entry, exit_ = entry_map.get(sym), price_map.get(sym)
        if not _valid_price(entry) or not _valid_price(exit_):
            continue
        quantity = max(1, int(POSITION_SIZE_RUPEES / entry))
        gross_ret = -(exit_ - entry) / entry     # short: profit when price falls
        cost = round_trip_cost_short(
            entry, exit_, quantity, as_of, holding_days
        )["total_pct_of_value"]
        short_rets.append(gross_ret)
        short_costs.append(cost)

    if not long_rets or not short_rets:
        return None

    gross_spread = (sum(long_rets) / len(long_rets)) - (sum(short_rets) / len(short_rets)) * -1
    avg_cost = (sum(long_costs) / len(long_costs) + sum(short_costs) / len(short_costs))
    net_spread = gross_spread - avg_cost

    return {
        "date": as_of,
        "n_long": len(long_rets),
        "n_short": len(short_rets),
        "gross_spread": gross_spread,
        "avg_round_trip_cost": avg_cost,
        "net_spread": net_spread,
    }


def main(start_year: int = 2005, end_year: int = 2025) -> None:
    # Starts 2005, not the dataset's 1994 origin: STT was introduced
    # 2004-10-01 (Finance Act 2004, confirmed) and the cost model's
    # pre-STT rate entry carries unverified approximations for stamp
    # duty / exchange charges. 2005-2025 is already 21 years / ~250
    # rebalances — more than enough sample for this gate.
    results = []

    for year in range(start_year, end_year + 1):
        universe = load_universe_year(year)
        if not universe:
            continue

        panel = load_monthly_panel(universe)
        if panel.empty:
            continue

        signal = compute_momentum_signal(panel)

        months = sorted(signal["DATE"].unique())
        for i in range(len(months) - 1):
            this_month = signal[signal["DATE"] == months[i]]
            next_month = panel[panel["DATE"] == months[i + 1]]
            if this_month.empty or next_month.empty:
                continue

            result = backtest_month(
                this_month, next_month, months[i].date(), months[i + 1].date()
            )
            if result:
                results.append(result)

        print(f"{year}: {len([r for r in results if r['date'].year == year])} rebalances")

    if not results:
        print("\nNo results — check universe/adjusted data exists for requested years.")
        return

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"MOMENTUM BASELINE — {len(df)} monthly rebalances, {start_year}-{end_year}")
    print(f"{'='*60}")
    print(f"Mean gross spread (monthly):  {df['gross_spread'].mean():.4%}")
    print(f"Mean round-trip cost:         {df['avg_round_trip_cost'].mean():.4%}")
    print(f"Mean NET spread (monthly):    {df['net_spread'].mean():.4%}")
    print(f"Net spread annualized (x12):  {df['net_spread'].mean() * 12:.2%}")
    print(f"Share of months net positive: {(df['net_spread'] > 0).mean():.1%}")

    verdict = "GO" if df["net_spread"].mean() > 0 else "NO-GO"
    print(f"\nGATE VERDICT: {verdict}")
    if verdict == "NO-GO":
        print("Per architecture doc: widen horizon or tighten universe, "
              "not more model capacity. Decide before Stage 8.")


if __name__ == "__main__":
    main()