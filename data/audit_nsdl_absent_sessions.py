"""
Stage 10 — exact completeness audit of the NSDL daily FPI series.

Every NSE trading session since 1999 that has no NSDL equity figure is either a day NSDL did not report (custodians
closed: a weekend Budget session, Muhurat trading, a bank holiday) or a day our data LOST. Row identities and monthly
totals cannot separate the two, and the monthly cross-check does not reach 1999-2001 at all. This audit does: it re-fetches
NSDL's own page for each affected month, lists the dates in the page's date cells (independently of the table parser),
and records for each such session whether NSDL lists it.

  absent_on_nsdl_page            NSDL has no row for the date: nothing was lost.
  on_page_but_missing_from_data  NSDL lists the date and we have no figure: a real loss (or a documented quarantine).

Positive control: every day we DO have in a fetched month must appear in that page's date cells. If one does not, the page
was misread and "absent" would prove nothing, so the audit aborts instead of writing a verdict.

Output: data/raw/nsdl_fpi/_absent_sessions.csv (session, status, checked_at_ist), read by scripts/verify_stage10_flows.py.
Re-run after new sessions without figures appear (the verifier names any it has not audited).

Usage: python data/audit_nsdl_absent_sessions.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NSDL_FPI_DIR
from data.trading_calendar import trading_sessions
from data.ingest_nsdl_fpi import FIRST_DAILY, HEADERS, fetch_month_html, page_dates    # page_dates is re-exported for callers/tests
from data.publication import IST

ABSENT = "absent_on_nsdl_page"
LOST = "on_page_but_missing_from_data"
OUTPUT_PATH = NSDL_FPI_DIR / "_absent_sessions.csv"

def classify(sessions_without_figure: list[date], page: set[date], have: set[date]) -> list[tuple[date, str]]:
    """Verdict per session for ONE month's page. Aborts if the page does not list a day we hold (positive control)."""
    unread = sorted(d for d in have if d not in page)
    if unread:
        raise ValueError(f"positive control failed: we hold {unread[0]} (and {len(unread) - 1} more) but the page's date cells do not list it")
    return [(s, LOST if s in page else ABSENT) for s in sessions_without_figure]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=3, min=3, max=60), reraise=True)
def _page_for(session: requests.Session, month_start: date, today: date) -> str:
    return fetch_month_html(session, month_start, today)


def run(delay: float = 0.6) -> pd.DataFrame:
    from features.blocks.flows import load_fpi_daily

    today = date.today()
    sessions = [pd.Timestamp(s).date() for s in trading_sessions()]
    have = {d.date() for d in pd.to_datetime(load_fpi_daily().dropna(subset=["net_cr"])["date"])}
    last_reported = max(have)
    missing = [s for s in sessions if FIRST_DAILY <= s <= last_reported and s not in have]
    months = sorted({(s.year, s.month) for s in missing})
    logger.info(f"{len(missing)} sessions in {len(months)} months have no NSDL figure (through {last_reported}); re-fetching those months")

    http = requests.Session()
    http.headers.update(HEADERS)
    verdicts: list[tuple[date, str]] = []
    for i, (year, month) in enumerate(months, 1):
        page = page_dates(_page_for(http, date(year, month, 1), today))
        verdicts += classify([s for s in missing if (s.year, s.month) == (year, month)], page,
                             {d for d in have if (d.year, d.month) == (year, month)})
        if i % 20 == 0:
            logger.info(f"  {i}/{len(months)} months")
        time.sleep(delay)

    stamp = datetime.now(IST).isoformat()
    table = pd.DataFrame(verdicts, columns=["session", "status"]).assign(checked_at_ist=stamp)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    counts = table["status"].value_counts().to_dict()
    logger.info(f"wrote {len(table)} verdicts to {OUTPUT_PATH}: {counts}")
    for row in table[table["status"] == LOST].itertuples():
        logger.warning(f"NSDL lists {row.session} but our data have no figure for it")
    return table


if __name__ == "__main__":
    run()
