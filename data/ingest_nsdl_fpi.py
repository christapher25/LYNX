"""
Stage 10 — NSDL daily FPI (foreign portfolio investor) net investment, 1999 to present.

Source: NSDL's "Archive (Trends in FPI/FII Investments)" report, https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx
(probed 2026-09-24; see data/raw/DATA_NOTES.md §12). It is the deepest free daily FII series: the NSE FII/DII endpoint
serves only the latest day, and the earlier belief that no free multi-year history exists was wrong.

How the report works (all measured):
  * It is an ASP.NET postback form. POST the ViewState fields plus txtDate=hdnDate=<dd-Mon-yyyy> and
    __EVENTTARGET=btnSubmit1. The date is an UPPER BOUND: the page returns month-start THROUGH that date, so a full
    month needs the month's last day (the current month: today).
  * Daily data starts 1999-01-01. The single 1998-12-31 row is a cumulative "since inception" seed, not a day.
  * The page holds several tables (cash flows and a derivatives table); the cash table is the one whose header has
    "Gross Purchases" and "Debt/Equity". It uses rowspan and colspan: date, asset class and FX are written once per group.
  * Three column eras: no "Investment Route" column until ~2010 (rows Equity/Debt only); routes (Stock Exchange /
    Primary market & others / Sub-total / Total) after; more debt categories (General Limit, VRR, FAR, Hybrid) recently.
  * Publication: NSDL states the confirmed figures come from custodian reports covering trades "on and up to the
    previous trading day(s)", so day t's figure is public no earlier than t+1 (features/blocks/flows.py applies that).
  * Values can be revised after first publication and the archive serves current values only, so history here is the
    latest vintage. The two most recent months are re-fetched on every run for that reason.

Output: data/raw/nsdl_fpi/{YYYY-MM}.parquet, long format, one row per (date, asset, route), columns OUTPUT_COLS.
Negative figures are written "(1234.56)" in the source and parsed as negative; a blank cell is NaN, never 0.

Usage
-----
    python data/ingest_nsdl_fpi.py                    # resumable full backfill 1999-01 -> now
    python data/ingest_nsdl_fpi.py --refresh-months 3 # also re-fetch the last 3 months (revisions)
"""

from __future__ import annotations

import argparse
import calendar
import html as html_lib
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, MANIFEST_DIR, NSDL_FPI_DIR
from data.ingest_common import DayRecord, Manifest, get_logger, write_parquet_atomic

URL = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"
FIRST_DAILY = date(1999, 1, 1)         # 1998-12-31 is a cumulative seed row, not a day (probed 2026-09-24)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Referer": URL,
}
OUTPUT_COLS = ["DATE", "ASSET", "ROUTE", "GROSS_PURCHASES_CR", "GROSS_SALES_CR", "NET_CR", "NET_USD_MN", "FX_INR_PER_USD"]
IDENTITY_TOLERANCE_CR = 0.02           # net = purchases - sales for a two-decimal row (measured max gap 0.01)
IDENTITY_TOLERANCE_COARSE_CR = 0.11    # a ONE-decimal row (1999 through 2014-06-02): each figure can be off by 0.05 (measured max 0.10)
                                       # A "Total" row adds two separately rounded sub-totals, so its tolerance is doubled (measured max 0.20).
MANIFEST_PATH = MANIFEST_DIR / "nsdl_fpi.jsonl"

_COLUMN_KEYS = {
    "reportingdate": "DATE",
    "debt/equity": "ASSET",
    "investmentroute": "ROUTE",
    "grosspurchases": "GROSS_PURCHASES_CR",
    "grosssales": "GROSS_SALES_CR",
    "netinvestment(rscrore)": "NET_CR",
    "netinvestmentus($)million": "NET_USD_MN",
    "conversion": "FX_INR_PER_USD",
}


# ─── numbers ─────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_number(text) -> float:
    """'(6139.56)' -> -6139.56, '1,234.50' -> 1234.5; blank, '-' or 'NA' -> NaN. Never 0 for a missing cell."""
    if text is None:
        return np.nan
    cleaned = str(text).strip().replace(",", "")
    if cleaned in ("", "-", "NA", "N/A", "n/a", "na"):
        return np.nan
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return np.nan
    return -value if negative else value


def parse_fx(text) -> float:
    """'Rs.63.7615' / ' Rs. 42.48' -> the rupee-per-dollar rate; anything without a number -> NaN."""
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return float(match.group(0)) if match else np.nan


# ─── HTML table -> grid (rowspan and colspan resolved) ───────────────────────────────────────────────────────────

def _clean(cell_html: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", cell_html, flags=re.I)
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _span(attrs: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*['\"]?(\d+)", attrs, re.I)
    return max(1, int(match.group(1))) if match else 1


def _grid(table_html: str) -> list[list[str]]:
    """Expand a table into a rectangular grid: a cell with rowspan/colspan is repeated into every slot it covers."""
    grid: list[list[str]] = []
    carry: dict[int, list] = {}                                    # column -> [text, rows still to cover]
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, re.S | re.I):
        pending = [(_clean(inner), _span(attrs, "rowspan"), _span(attrs, "colspan"))
                   for _, attrs, inner in re.findall(r"<t([dh])\b([^>]*)>(.*?)</t\1>", row_html, re.S | re.I)]
        out: list[str] = []
        col = 0
        while pending or (carry and col <= max(carry)):
            if col in carry:
                text, remaining = carry[col]
                out.append(text)
                if remaining <= 1:
                    del carry[col]
                else:
                    carry[col][1] = remaining - 1
                col += 1
                continue
            if not pending:
                break
            text, rowspan, colspan = pending.pop(0)
            for k in range(colspan):
                out.append(text if k == 0 else "")
                if rowspan > 1:
                    carry[col] = [text if k == 0 else "", rowspan - 1]
                col += 1
        grid.append(out)
    return grid


def _column_map(header: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, name in enumerate(header):
        key = re.sub(r"\s+", "", name.lower())
        for prefix, column in _COLUMN_KEYS.items():
            if key.startswith(prefix):
                mapping.setdefault(column, index)
    return mapping


def parse_cash_table(page_html: str) -> pd.DataFrame:
    """Parse the FPI cash-flow table (not the derivatives table) of an NSDL daily-trends page into OUTPUT_COLS."""
    for table in re.findall(r"<table\b[^>]*>.*?</table>", page_html, re.S | re.I):
        if "gross purchases" not in table.lower() or "debt/equity" not in table.lower():
            continue
        grid = _grid(table)
        header_at = next((i for i, row in enumerate(grid) if row and "reporting" in row[0].lower() and "date" in row[0].lower()), None)
        if header_at is None:
            continue
        columns = _column_map(grid[header_at])
        if not {"DATE", "ASSET", "GROSS_PURCHASES_CR", "GROSS_SALES_CR", "NET_CR"} <= set(columns):
            continue
        records = []
        for row in grid[header_at + 1:]:
            if row and "reporting" in row[0].lower() and "date" in row[0].lower():
                break                       # 2003-~2014 pages are malformed HTML: the derivatives table is nested inside this one
            get = lambda name: row[columns[name]] if name in columns and columns[name] < len(row) else None
            try:
                day = datetime.strptime((get("DATE") or "").strip().title(), "%d-%b-%Y")
            except ValueError:
                continue                                              # title / note / spacer rows
            route = (get("ROUTE") or "").strip() or None
            asset = (get("ASSET") or "").strip()
            if route == "Total":                                      # its asset cell is carried down from the Debt rowspan
                asset = "Total"
            records.append({
                "DATE": pd.Timestamp(day), "ASSET": asset, "ROUTE": route,
                "GROSS_PURCHASES_CR": parse_number(get("GROSS_PURCHASES_CR")), "GROSS_SALES_CR": parse_number(get("GROSS_SALES_CR")),
                "NET_CR": parse_number(get("NET_CR")), "NET_USD_MN": parse_number(get("NET_USD_MN")),
                "FX_INR_PER_USD": parse_fx(get("FX_INR_PER_USD")),
            })
        return pd.DataFrame(records, columns=OUTPUT_COLS)
    raise ValueError("no FPI cash table found on the page (looked for 'Gross Purchases' and 'Debt/Equity')")


# Reading which DATES a page lists, independently of the row parser above (used by the completeness audit and the publication
# monitor). It must be scoped to the cash table: the page also carries FPI F&O activity whose date cells include days the cash
# table lacks (F&O traded on 1-Apr-2025; custodians were closed).
_DATE_CELL = re.compile(r"(?is)<td[^>]*>\s*(?:<[^/>][^>]*>\s*)*(\d{2}-[A-Za-z]{3}-\d{4})\s*(?:</?[^>]+>\s*)*</td>")
_REPORTING_DATE = re.compile(r"(?i)reporting(?:\s|<[^>]*>|&nbsp;)*date")


def _cash_region(page_html: str) -> str:
    """The cash-flow table only. Older pages are malformed and nest the derivatives table inside it, starting at a second
    "Reporting Date" header, where the region stops."""
    for table in re.findall(r"<table\b[^>]*>.*?</table>", page_html, re.S | re.I):
        low = table.lower()
        if "gross purchases" in low and "debt/equity" in low:
            first = _REPORTING_DATE.search(table)
            second = _REPORTING_DATE.search(table, first.end()) if first else None
            return table[: second.start()] if second else table
    return ""


def page_dates(page_html: str) -> set[date]:
    """Dates in the cash table's date cells. The form's query-date textbox is an <input>, never a cell, so it is not read."""
    return {datetime.strptime(token, "%d-%b-%Y").date() for token in _DATE_CELL.findall(_cash_region(page_html))}


def resolve_duplicate_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.Timestamp], int]:
    """
    Exact duplicate rows are redundant and collapse. A date whose copies DISAGREE is ambiguous (2000-02-28 appears twice
    with different figures; one entry is presumably really 2000-02-29, but which is unknowable) and is quarantined whole:
    dropped, reported, never resolved by picking a winner. Returns (frame, quarantined dates, exact duplicates collapsed).
    """
    work = df.assign(_route=df["ROUTE"].fillna(""))
    exact = work.duplicated(keep="first")
    collapsed = int(exact.sum())
    work = work[~exact]
    clashing = work.duplicated(["DATE", "ASSET", "_route"], keep=False)
    quarantined = sorted(pd.Timestamp(d) for d in work.loc[clashing, "DATE"].unique())
    work = work[~work["DATE"].isin(quarantined)]
    return work.drop(columns="_route").reset_index(drop=True), quarantined, collapsed


def identity_tolerances(df: pd.DataFrame) -> pd.Series:
    """
    Per-row rounding tolerance for net = purchases - sales, set by the PRECISION of that row's own figures. A row whose
    purchases/sales/net are all multiples of 0.1 was rounded to one decimal (1999 to 2014-06-02; net can differ from
    purchases - sales by 0.10), otherwise two decimals were kept (gap <= 0.01). Precision is a per-ROW property, not a
    per-month or per-route one: the route column arrived years before the second decimal did, and 2014-06 switches
    on the 3rd. A 'Total' row sums two rounded sub-totals, so it gets double (measured max gap 0.20 in the coarse era).
    """
    values = df[["GROSS_PURCHASES_CR", "GROSS_SALES_CR", "NET_CR"]]
    scaled = values * 10
    nonzero = values.notna() & (values != 0)                 # zeros carry no precision information (0.00 == 0.0)
    coarse = ((scaled - scaled.round()).abs() < 1e-6) | ~nonzero
    one_decimal = coarse.all(axis=1) & nonzero.any(axis=1)  # an all-zero row is judged at the fine tolerance
    tolerance = pd.Series(np.where(one_decimal, IDENTITY_TOLERANCE_COARSE_CR, IDENTITY_TOLERANCE_CR), index=df.index)
    return tolerance.where(df["ROUTE"] != "Total", tolerance * 2)


# ─── validation ──────────────────────────────────────────────────────────────────────────────────────────────────

def validate_month(df: pd.DataFrame, month_start: date) -> pd.DataFrame:
    if df.empty:
        raise ValueError("parsed to no rows")
    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    lo, hi = pd.Timestamp(month_start), pd.Timestamp(month_start.replace(day=last_day))
    outside = df[(df["DATE"] < lo) | (df["DATE"] > hi)]
    if len(outside):
        raise ValueError(f"{len(outside)} rows dated outside the month {month_start:%Y-%m}")
    key = df.assign(_route=df["ROUTE"].fillna("")).duplicated(["DATE", "ASSET", "_route"], keep=False)
    if key.any():
        raise ValueError(f"{int(key.sum())} duplicate (date, asset, route) rows")
    equity_total = df[(df["ASSET"] == "Equity") & (df["ROUTE"].isna() | (df["ROUTE"] == "Sub-total"))]
    missing_equity = sorted(set(df["DATE"]) - set(equity_total["DATE"]))
    if missing_equity:
        raise ValueError(f"{len(missing_equity)} dates have no equity total row, e.g. {missing_equity[0]:%Y-%m-%d}")
    if df["FX_INR_PER_USD"].isna().any():
        raise ValueError(f"{int(df['FX_INR_PER_USD'].isna().sum())} rows have no fx rate")
    tolerance = identity_tolerances(df)
    both = df.dropna(subset=["GROSS_PURCHASES_CR", "GROSS_SALES_CR", "NET_CR"])
    gap = (both["GROSS_PURCHASES_CR"] - both["GROSS_SALES_CR"] - both["NET_CR"]).abs()
    bad = gap > tolerance.loc[both.index]
    if bad.any():
        worst = both.loc[(gap - tolerance.loc[both.index]).idxmax()]
        raise ValueError(f"net != purchases - sales beyond the rounding tolerance on {int(bad.sum())} rows, "
                         f"e.g. {worst['DATE']:%Y-%m-%d} {worst['ASSET']} {worst['ROUTE'] or ''}".rstrip())
    return df


# ─── fetch ───────────────────────────────────────────────────────────────────────────────────────────────────────

def month_query_date(month_start: date, today: date) -> date:
    """The archive returns month-start THROUGH the queried date: a past month needs its last day, the current month today."""
    if (month_start.year, month_start.month) > (today.year, today.month):
        raise ValueError(f"{month_start:%Y-%m} is in the future")
    if (month_start.year, month_start.month) == (today.year, today.month):
        return today
    return month_start.replace(day=calendar.monthrange(month_start.year, month_start.month)[1])


def _hidden_fields(page: str) -> dict[str, str]:
    return {m.group(1): html_lib.unescape(m.group(2))
            for m in re.finditer(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page)}


def fetch_month_html(session: requests.Session, month_start: date, today: date) -> str:
    """The archive page for one month, exactly as served (month-start through the query date)."""
    query = month_query_date(month_start, today).strftime("%d-%b-%Y")
    landing = session.get(URL, timeout=30)
    landing.raise_for_status()
    form = _hidden_fields(landing.text)
    form.update({"__EVENTTARGET": "btnSubmit1", "__EVENTARGUMENT": "", "txtDate": query, "hdnDate": query})
    response = session.post(URL, data=form, timeout=60)
    response.raise_for_status()
    return response.text


def fetch_month(session: requests.Session, month_start: date, today: date) -> tuple[pd.DataFrame, list[pd.Timestamp], int] | None:
    """One month of daily rows, validated, with any quarantined dates; None when the archive holds nothing for it."""
    page = fetch_month_html(session, month_start, today)
    try:
        df = parse_cash_table(page)
    except ValueError:
        return None
    if df.empty:
        return None
    df, quarantined, collapsed = resolve_duplicate_dates(df)
    return validate_month(df, month_start), quarantined, collapsed


def _months(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield date(year, month, 1)
        month, year = (1, year + 1) if month == 12 else (month + 1, year)


def run(start: date = FIRST_DAILY, end: date | None = None, refresh_months: int = 2, delay: float = 0.6) -> dict:
    """Resumable monthly backfill. A month's file is skipped once written, except the newest `refresh_months`."""
    today = end or date.today()
    logger = get_logger("nsdl_fpi", LOG_DIR)
    manifest = Manifest(MANIFEST_PATH)
    NSDL_FPI_DIR.mkdir(parents=True, exist_ok=True)
    months = list(_months(max(start, FIRST_DAILY), today))
    refresh_from = months[max(0, len(months) - refresh_months)]
    counts = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    total_rows = 0
    session = requests.Session()
    session.headers.update(HEADERS)
    logger.info(f"nsdl_fpi: {len(months)} months {months[0]:%Y-%m} .. {months[-1]:%Y-%m}; refreshing from {refresh_from:%Y-%m}")
    for i, month in enumerate(months, 1):
        out = NSDL_FPI_DIR / f"{month:%Y-%m}.parquet"
        if out.exists() and month < refresh_from:
            counts["skipped"] += 1
            continue
        label = month.isoformat()
        try:
            fetched = None
            for attempt in range(5):
                try:
                    fetched = fetch_month(session, month, today)
                    break
                except (requests.RequestException, ValueError) as exc:
                    if isinstance(exc, ValueError) or attempt == 4:
                        raise
                    time.sleep(min(60, 5 * 3 ** attempt))         # 5, 15, 45, 60 s: rides out a DNS blip
        except Exception as exc:                                       # noqa: BLE001
            counts["failed"] += 1
            manifest.add(DayRecord(label, "failed", error=str(exc)[:300]))
            logger.warning(f"{month:%Y-%m}: FAILED - {exc}")
        else:
            if fetched is None:
                counts["empty"] += 1
                manifest.add(DayRecord(label, "empty"))
            else:
                df, quarantined, collapsed = fetched
                write_parquet_atomic(df, out)
                counts["ok"] += 1
                total_rows += len(df)
                note = ""
                if quarantined:
                    note = "quarantined ambiguous dates: " + ",".join(f"{d:%Y-%m-%d}" for d in quarantined)
                    logger.warning(f"{month:%Y-%m}: {note}")
                if collapsed:
                    note = (note + "; " if note else "") + f"collapsed {collapsed} exact duplicate rows"
                manifest.add(DayRecord(label, "ok", rows=len(df), error=note,
                                       format_era="route" if df["ROUTE"].notna().any() else "no-route"))
        if i % 12 == 0:
            manifest.flush()
            logger.info(f"[{i}/{len(months)}] {counts}")
        time.sleep(delay)
    manifest.flush()
    logger.info(f"nsdl_fpi complete: {counts} rows={total_rows:,}")
    return {**counts, "total_rows": total_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSDL daily FPI net investment (1999 -> now).")
    ap.add_argument("--start", type=str, help="first month YYYY-MM-DD (default 1999-01-01)")
    ap.add_argument("--refresh-months", type=int, default=2, help="re-fetch this many newest months (revisions)")
    ap.add_argument("--delay", type=float, default=0.6)
    args = ap.parse_args()
    run(start=date.fromisoformat(args.start) if args.start else FIRST_DAILY, refresh_months=args.refresh_months, delay=args.delay)


if __name__ == "__main__":
    main()
