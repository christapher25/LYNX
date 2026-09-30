"""
Stage 6 — Indian transaction cost model. Built before any features, per
architecture doc: "the largest risk to this project is not that the
architecture fails, it is that a sophisticated system is built and then
found to have no edge net of Indian frictions."

Rates are date-versioned — STT changed materially within the sample
period, and applying today's rate to a 2005 backtest silently inflates
historical returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class CostRates:
    stt_delivery_buy: float
    stt_delivery_sell: float
    stt_intraday_sell: float       # buy side is 0 for intraday
    stamp_duty_buy: float          # buy side only, state-varying, using Maharashtra
    exchange_txn_charge: float
    sebi_turnover_fee: float
    gst_rate: float                # applied ON brokerage + exchange charges
    brokerage_flat: float          # per-order flat fee, discount-broker assumption


# STT rate changed 2023-10-01 (intraday sell STT: 0.025% -> unchanged,
# delivery unchanged) — kept as a versioned table since Indian STT rates
# have moved multiple times across the sample period; extend this table
# with real effective dates before trusting years before 2018.
_RATE_TABLE: list[tuple[date, CostRates]] = [
    # Pre-STT era. STT did not exist before this date — confirmed via
    # Finance Act 2004 (effective 2004-10-01), not a placeholder. Stamp
    # duty and exchange-charge rates for this era are NOT independently
    # verified and are approximated by carrying the 2004 rates backward —
    # flagged here rather than silently assumed correct.
    (date(1994, 1, 1), CostRates(
        stt_delivery_buy=0.0,
        stt_delivery_sell=0.0,
        stt_intraday_sell=0.0,
        stamp_duty_buy=0.00015,        # UNVERIFIED for pre-2004, approximated
        exchange_txn_charge=0.0000345, # UNVERIFIED for pre-2004, approximated
        sebi_turnover_fee=0.0000001,   # UNVERIFIED for pre-2004, approximated
        gst_rate=0.0,                  # GST itself didn't exist pre-2017;
                                        # pre-2004 used service tax instead —
                                        # zeroed rather than guessed
        brokerage_flat=20.0,
    )),
    (date(2004, 10, 1), CostRates(
        stt_delivery_buy=0.001,
        stt_delivery_sell=0.001,
        stt_intraday_sell=0.00025,
        stamp_duty_buy=0.00015,
        exchange_txn_charge=0.0000345,
        sebi_turnover_fee=0.0000001,
        gst_rate=0.18,
        brokerage_flat=20.0,
    )),
]


def get_rates(as_of: date) -> CostRates:
    """Most recent rate table entry with effective_date <= as_of."""
    applicable = [r for eff_date, r in _RATE_TABLE if eff_date <= as_of]
    if not applicable:
        raise ValueError(
            f"No cost rate table entry covers {as_of} — the rate table "
            f"starts {_RATE_TABLE[0][0]}. Extend _RATE_TABLE before "
            f"backtesting dates earlier than that."
        )
    return applicable[-1]


def round_trip_cost_delivery(
    entry_price: float, exit_price: float, quantity: int, as_of: date
) -> dict:
    """
    Full round-trip cost for a delivery (swing/long-term) trade.
    Returns a breakdown, not just a total — needed for the report to show
    the components weren't hand-waved.
    """
    rates = get_rates(as_of)
    buy_value = entry_price * quantity
    sell_value = exit_price * quantity

    stt = buy_value * rates.stt_delivery_buy + sell_value * rates.stt_delivery_sell
    stamp_duty = buy_value * rates.stamp_duty_buy
    exchange_charge = (buy_value + sell_value) * rates.exchange_txn_charge
    sebi_fee = (buy_value + sell_value) * rates.sebi_turnover_fee
    brokerage = rates.brokerage_flat * 2                  # one for buy, one for sell
    gst = (brokerage + exchange_charge) * rates.gst_rate

    total = stt + stamp_duty + exchange_charge + sebi_fee + brokerage + gst

    return {
        "stt": stt,
        "stamp_duty": stamp_duty,
        "exchange_charge": exchange_charge,
        "sebi_fee": sebi_fee,
        "brokerage": brokerage,
        "gst": gst,
        "total": total,
        "total_pct_of_value": total / buy_value if buy_value else None,
    }


def round_trip_cost_intraday(
    entry_price: float, exit_price: float, quantity: int, as_of: date
) -> dict:
    """
    Intraday round trip. STT applies sell-side only — this is the
    correction the architecture doc flags explicitly: intraday is CHEAPER
    per round trip than delivery in India, not more expensive.
    """
    rates = get_rates(as_of)
    buy_value = entry_price * quantity
    sell_value = exit_price * quantity

    stt = sell_value * rates.stt_intraday_sell               # sell-side only
    stamp_duty = buy_value * rates.stamp_duty_buy
    exchange_charge = (buy_value + sell_value) * rates.exchange_txn_charge
    sebi_fee = (buy_value + sell_value) * rates.sebi_turnover_fee
    brokerage = rates.brokerage_flat * 2
    gst = (brokerage + exchange_charge) * rates.gst_rate

    total = stt + stamp_duty + exchange_charge + sebi_fee + brokerage + gst

    return {
        "stt": stt,
        "stamp_duty": stamp_duty,
        "exchange_charge": exchange_charge,
        "sebi_fee": sebi_fee,
        "brokerage": brokerage,
        "gst": gst,
        "total": total,
        "total_pct_of_value": total / buy_value if buy_value else None,
    }


# SLB lending fee: 2% annualized is an ILLUSTRATIVE rate from a published
# worked example (IIFL), NOT a fixed tariff — India's SLB segment runs
# on a daily auction with no unified rate sheet, so real fees vary by
# stock demand and tenure. Flagged the same way the pre-2004 stamp-duty
# approximation is flagged: real, sourced, but not a verified constant.
SLB_LENDING_FEE_ANNUAL = 0.02
SLB_PROCESSING_MARKUP = 0.15     # broker markup ON the lending fee, not on trade value
SLB_PROCESSING_GST = 0.18        # GST on the processing markup


def round_trip_cost_short(
    entry_price: float, exit_price: float, quantity: int,
    as_of: date, holding_days: int,
) -> dict:
    """
    Short position via SLB (borrow-sell-cover-return) — genuinely
    different cost structure from a long trade, not the same formula
    reused:

    1. Stamp duty is buyer-side only. On a LONG, that's the entry leg.
       On a SHORT, the entry leg is a SALE (no stamp duty) and the exit
       leg (covering, i.e. buying back) IS the buy-side transaction —
       so stamp duty applies at COVER, not at open. Structural fact,
       not an approximation.
    2. STT applies to both legs same as delivery — STT is transaction-
       type based (buy/sell), not direction-based, so this part is
       unchanged from round_trip_cost_delivery.
    3. Securities lending fee, prorated by actual holding period —
       this is the cost a long position never incurs at all. Zero
       modeling of this (the original bug) understates short-leg cost
       by exactly this amount.
    """
    rates = get_rates(as_of)
    open_value = entry_price * quantity     # short-sale (entry)
    cover_value = exit_price * quantity     # buy-back (exit)

    stt = open_value * rates.stt_delivery_sell + cover_value * rates.stt_delivery_buy
    stamp_duty = cover_value * rates.stamp_duty_buy     # COVER leg only — the fix
    exchange_charge = (open_value + cover_value) * rates.exchange_txn_charge
    sebi_fee = (open_value + cover_value) * rates.sebi_turnover_fee
    brokerage = rates.brokerage_flat * 2

    lending_fee = open_value * SLB_LENDING_FEE_ANNUAL * (holding_days / 365)
    lending_processing = lending_fee * SLB_PROCESSING_MARKUP
    lending_gst = lending_processing * SLB_PROCESSING_GST

    trading_gst = (brokerage + exchange_charge) * rates.gst_rate

    total = (stt + stamp_duty + exchange_charge + sebi_fee + brokerage
             + trading_gst + lending_fee + lending_processing + lending_gst)

    return {
        "stt": stt,
        "stamp_duty": stamp_duty,
        "exchange_charge": exchange_charge,
        "sebi_fee": sebi_fee,
        "brokerage": brokerage,
        "trading_gst": trading_gst,
        "lending_fee": lending_fee,
        "lending_processing": lending_processing,
        "lending_gst": lending_gst,
        "total": total,
        "total_pct_of_value": total / open_value if open_value else None,
    }


if __name__ == "__main__":
    d = date(2024, 1, 1)
    print("Delivery, Rs.1000 x 100 shares, 2% move:")
    print(round_trip_cost_delivery(1000, 1020, 100, d))
    print("\nIntraday, same trade:")
    print(round_trip_cost_intraday(1000, 1020, 100, d))
    print("\nShort, same trade, 30-day hold:")
    print(round_trip_cost_short(1000, 1020, 100, d, holding_days=30))