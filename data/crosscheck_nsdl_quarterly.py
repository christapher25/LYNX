"""
Stage 10 — NSDL's own quarterly equity totals, as independent evidence about the daily FPI series.

The daily series (data/ingest_nsdl_fpi.py) is checked row by row at ingest (purchases - sales = net) and, for
2002-01 -> 2023-12, against NSDL's monthly totals. That leaves 2024 onward, the years that matter most for features,
covered by one month a year. NSDL's "Quarterly FPI Net Investments" page (QuarterlyWise.aspx) publishes calendar-year
tables of quarterly totals per asset class; its first numeric column is Equity. This module fetches them and writes
data/raw/nsdl_fpi/_crosscheck_quarterly.csv, which scripts/verify_stage10_flows.py compares with the daily sums.

Measured 2026-09-24: my quarterly sums of the daily figures differ from NSDL's quarterly Equity by 0.3 to 14.9 crore
(0.01-0.02%) on quarterly flows of 40,000-140,000 crore. The quarter still in progress differs by 0.6. Not an identity:
consistent with NSDL compiling the quarterly table from figures that were later revised on the daily page. The verifier's
tolerance is set from these measurements, and is a lost/duplicated-day detector, not a rounding-level identity.

The page is an ASP.NET form: GET returns the latest year; the year dropdown `ddl` posts back.

Usage: python data/crosscheck_nsdl_quarterly.py
"""

from __future__ import annotations

import html as html_lib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NSDL_FPI_DIR
from data.ingest_nsdl_fpi import HEADERS
from data.publication import IST

URL = "https://www.fpi.nsdl.co.in/web/Reports/QuarterlyWise.aspx"
OUTPUT_PATH = NSDL_FPI_DIR / "_crosscheck_quarterly.csv"
OUTPUT_COLS = ["year", "quarter", "nsdl_equity_cr", "up_to", "fetched_at_ist"]

_ROW = re.compile(r"Quarter\s+([1-4])\s*\([^)]*\)\s*(?:\*\*)?\s*(-?\d+|-)(?=\s)")
_TOTAL = re.compile(r"Total\s*-\s*(\d{4})\s+(-?\d+|-)(?=\s)")
_UP_TO = re.compile(r"\*\*\s*up to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})")


def _visible_text(html: str) -> str:
    stripped = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", stripped))) + " "


def _number(token: str) -> float:
    return float("nan") if token == "-" else float(token)


def parse_quarterly(html: str) -> pd.DataFrame:
    """
    One row per quarter on the page: year, quarter, nsdl_equity_cr (the first numeric column, Equity), up_to.
    `up_to` is set only for the quarter the footnote "** up to DD Mon YYYY" marks as still in progress; a `**` beside a
    quarter without that footnote is an unrelated note (Debt-FAR/MF/AIF coverage from Sep 2024).
    Raises ValueError if no quarter rows are found or the page's own Total row disagrees with the sum of the quarters.
    """
    text = _visible_text(html)
    total = _TOTAL.search(text)
    rows = [(int(m.group(1)), _number(m.group(2))) for m in _ROW.finditer(text)]
    if not rows or total is None:
        raise ValueError("no quarterly table found on the page")
    year, page_total = int(total.group(1)), _number(total.group(2))
    if sum(v for _, v in rows) != page_total:
        raise ValueError(f"quarter equities sum to {sum(v for _, v in rows)} but the page's total row says {page_total}: misparsed")
    up_to = _UP_TO.search(text)
    partial_date = pd.Timestamp(datetime.strptime(re.sub(r"\s+", " ", up_to.group(1)), "%d %b %Y")) if up_to else pd.NaT
    frame = pd.DataFrame(rows, columns=["quarter", "nsdl_equity_cr"])
    frame.insert(0, "year", year)
    frame["up_to"] = pd.NaT
    if up_to:
        marked = [int(m.group(1)) for m in re.finditer(r"Quarter\s+([1-4])\s*\([^)]*\)\s*\*\*", text)]
        frame.loc[frame["quarter"].isin(marked), "up_to"] = partial_date
    return frame


def _hidden_fields(html: str) -> dict[str, str]:
    return {name: html_lib.unescape(value)
            for name, value in re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html)}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
def _request(session: requests.Session, data: dict | None) -> str:
    response = session.post(URL, data=data, timeout=40) if data else session.get(URL, timeout=40)
    response.raise_for_status()
    return response.text


def fetch_all(delay: float = 1.0) -> pd.DataFrame:
    """Every year the dropdown offers: the default page is the latest year, the others are postbacks."""
    session = requests.Session()
    session.headers.update(HEADERS)
    first = _request(session, None)
    select = re.search(r'(?is)<select[^>]*id="ddl".*?</select>', first)
    years = re.findall(r'value="(\d{4})"', select.group(0)) if select else []
    frames = [parse_quarterly(first)]
    latest = int(frames[0]["year"].iloc[0])                       # the GET page is the newest year; the rest are postbacks
    for year in (y for y in years if int(y) != latest):
        time.sleep(delay)
        page = _request(session, {**_hidden_fields(first), "__EVENTTARGET": "ddl", "__EVENTARGUMENT": "", "ddl": year})
        frames.append(parse_quarterly(page))
    out = pd.concat(frames, ignore_index=True).sort_values(["year", "quarter"]).reset_index(drop=True)
    out["fetched_at_ist"] = datetime.now(IST).isoformat()
    return out[OUTPUT_COLS]


def run() -> pd.DataFrame:
    table = fetch_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    logger.info(f"wrote {len(table)} quarters ({table['year'].min()}-{table['year'].max()}) to {OUTPUT_PATH}")
    return table


if __name__ == "__main__":
    run()
