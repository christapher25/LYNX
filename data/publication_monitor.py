"""
data/publication_monitor.py

Measure, do not assume, when each daily source publishes.

The India-block features are keyed on when a value became public (data/publication.py). For NSE's archive files the
`Last-Modified` header is a usable historical clock, but the NSE FII/DII API and NSDL's portal carry no such header, so
their real publication lag can only be observed live. This monitor polls every source for a target day D and records the
FIRST time each one shows D's data, plus the last poll that did not (so the true publication time is bracketed), and any
`Last-Modified` the server reports.

Output: one JSON line per source in PUBLICATION_LOG_DIR/first_seen.jsonl. It is append-only evidence: run it each
evening and the log accumulates a real distribution of publication times.

Usage:  python -m data.publication_monitor                       # today's date (IST), polls every 3 min
        python -m data.publication_monitor --date 2026-09-24 --until-next-day-hour 14
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PUBLICATION_LOG_DIR
from data.ingest_nsdl_fpi import HEADERS as NSDL_HEADERS, fetch_month_html, page_dates
from data.publication import IST

ARCHIVE_BASE = "https://nsearchives.nseindia.com"
ARCHIVE_SOURCES = {
    "nse_cm_bhavcopy_udiff": "/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
    "nse_fo_bhavcopy_udiff": "/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
    "nse_sec_bhavdata_full": "/products/content/sec_bhavdata_full_{dmy}.csv",
    "nse_pr_zip": "/archives/equities/bhavcopy/pr/PR{dmy2}.zip",
    "nse_sec_list": "/content/equities/sec_list_{dmy}.csv",
    "nse_cm_security": "/content/cm/NSE_CM_security_{dmy}.csv.gz",
    "nse_fo_ban": "/archives/fo/sec_ban/fo_secban_{dmy}.csv",
}
API_SOURCES = ["nse_fii_dii_react", "nse_fii_dii_nse", "nsdl_cash"]
ALL_SOURCES = list(ARCHIVE_SOURCES) + API_SOURCES

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "*/*",
}
NSE_FII_DII = "https://www.nseindia.com/api/{endpoint}"


def archive_url(name: str, day: date) -> str:
    path = ARCHIVE_SOURCES[name].format(ymd=f"{day:%Y%m%d}", dmy=f"{day:%d%m%Y}", dmy2=f"{day:%d%m%y}")
    return ARCHIVE_BASE + path


# ---- parsers ---------------------------------------------------------------------------------------------------

def dates_in_nse_fii_dii(payload) -> set[date]:
    """Dates carried by an NSE FII/DII API payload (a list of {category, date, ...}); anything else -> empty."""
    found: set[date] = set()
    if not isinstance(payload, list):
        return found
    for record in payload:
        text = record.get("date") if isinstance(record, dict) else None
        if not text:
            continue
        try:
            found.add(datetime.strptime(text.strip().title(), "%d-%b-%Y").date())
        except ValueError:
            continue
    return found


def dates_in_nsdl_page(html: str) -> set[date]:
    """
    Reporting dates in the CASH table of an NSDL daily-trends page. The first version read every date on the page, derivatives
    table included, so its 'first seen' times described the wrong table; the 2026-09-24 evidence (item nsdl_latest) is void.
    """
    return page_dates(html)


# ---- polling ---------------------------------------------------------------------------------------------------

def poll_once(state: dict, items: list[str], fetch: Callable[[str], tuple[bool, dict]], now: datetime) -> None:
    """
    Poll every item not yet seen. `fetch(item)` returns (present, detail). A raised error counts as "not seen yet",
    never as a sighting. The first sighting is recorded once and never overwritten; the most recent poll that found the
    item absent is kept as a lower bound, so the true publication time lies between it and the first sighting.
    """
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    for item in items:
        entry = state.setdefault(item, {"first_seen_ist": None, "last_absent_ist": None, "detail": {}, "last_error": None})
        if entry["first_seen_ist"] is not None:
            continue
        try:
            present, detail = fetch(item)
        except Exception as exc:                                              # noqa: BLE001
            entry["last_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        if present:
            entry["first_seen_ist"], entry["detail"] = now, detail
        else:
            entry["last_absent_ist"] = now


class LiveFetcher:
    """Real HTTP checks for each watched source."""

    def __init__(self, day: date):
        self.day = day
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._nse_ready = False
        self._nsdl = None

    def _nsdl_session(self) -> requests.Session:
        if self._nsdl is None:
            self._nsdl = requests.Session()
            self._nsdl.headers.update(NSDL_HEADERS)
        return self._nsdl

    def _nse_api(self, endpoint: str):
        if not self._nse_ready:
            self.session.get("https://www.nseindia.com", timeout=20)
            self._nse_ready = True
        r = self.session.get(NSE_FII_DII.format(endpoint=endpoint), timeout=25,
                             headers={"Referer": "https://www.nseindia.com/reports/fii-dii"})
        r.raise_for_status()
        return r.json()

    def __call__(self, item: str) -> tuple[bool, dict]:
        if item in ARCHIVE_SOURCES:
            r = self.session.get(archive_url(item, self.day), timeout=40, stream=True)
            try:
                if r.status_code != 200:
                    return False, {}
                stamp = r.headers.get("Last-Modified")
                modified = parsedate_to_datetime(stamp).astimezone(IST).isoformat() if stamp else None
                return True, {"last_modified": modified, "bytes": r.headers.get("Content-Length")}
            finally:
                r.close()
        if item == "nse_fii_dii_react":
            dates = dates_in_nse_fii_dii(self._nse_api("fiidiiTradeReact"))
        elif item == "nse_fii_dii_nse":
            dates = dates_in_nse_fii_dii(self._nse_api("fiidiiTradeNSE"))
        elif item == "nsdl_cash":
            # the Archive page for the target month, through today: the same page the ingester reads, cash table only
            page = fetch_month_html(self._nsdl_session(), self.day.replace(day=1), max(self.day, datetime.now(IST).date()))
            dates = dates_in_nsdl_page(page)
        else:
            raise KeyError(item)
        return self.day in dates, {"dates_visible": sorted(d.isoformat() for d in dates)[-3:]}


def _serialise(entry: dict) -> dict:
    out = dict(entry)
    for key in ("first_seen_ist", "last_absent_ist"):
        out[key] = out[key].isoformat() if out[key] else None
    return out


def run(day: date, interval: float, until: datetime, log_dir: Path = PUBLICATION_LOG_DIR,
        fetch: Callable | None = None, sleep: Callable[[float], None] = _time.sleep, clock=None) -> dict:
    """Poll until every source has been seen or `until` passes; append one JSON line per source at the end."""
    fetch = fetch or LiveFetcher(day)
    clock = clock or (lambda: datetime.now(IST))
    state: dict = {}
    while True:
        poll_once(state, ALL_SOURCES, fetch, clock())
        if all(state[i]["first_seen_ist"] for i in ALL_SOURCES) or clock() >= until:
            break
        sleep(interval)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "first_seen.jsonl").open("a", encoding="utf-8") as handle:
        for item in ALL_SOURCES:
            handle.write(json.dumps({"item": item, "target_date": day.isoformat(), **_serialise(state[item])}) + "\n")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Record when each daily source first publishes a given day's data.")
    ap.add_argument("--date", type=str, help="target day YYYY-MM-DD (default: today, IST)")
    ap.add_argument("--interval", type=float, default=180.0, help="seconds between polls (default 180)")
    ap.add_argument("--until-next-day-hour", type=int, default=14,
                    help="stop at this IST hour on the day after the target day (NSDL publishes next day)")
    args = ap.parse_args()
    day = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    until = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=IST).replace(hour=args.until_next_day_hour)
    state = run(day, args.interval, until)
    for item, entry in state.items():
        print(item, entry["first_seen_ist"], "(last absent:", entry["last_absent_ist"], ")")


if __name__ == "__main__":
    main()
