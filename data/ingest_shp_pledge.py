"""
Stage 10 — promoter encumbrance (pledging) from NSE shareholding-pattern (SHP) filings.

Sources (probed 2026-09-24, see data/raw/DATA_NOTES.md §14):

  1. NSE master list, one call per symbol:
         https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={SYMBOL}
     One row per filed pattern: period end (`date`), `broadcastDate` (a timestamp to the second, IST), `submissionDate`,
     revision fields, and the `xbrl` link. The oldest filings are period 30-SEP-2021 and 08-DEC-2021, both broadcast in one
     batch on 07-JAN-2022 02:55-03:25 (a re-broadcast: their true public time is not recoverable and the batch time is the
     conservative key); from the 31-DEC-2021 quarter the timestamps look genuine (an XBRL Last-Modified matched its
     broadcastDate to four seconds).  A revised filing REPLACES the original in this list, so the original values are lost.
  2. The filing's XBRL (nsearchives.nseindia.com/corporate/xbrl/SHP_*.xml), 100-1,000 KB, one context per shareholder. The
     promoter aggregate is the context whose CategoryOfShareholdersAxis member is ShareholdingOfPromoterAndPromoterGroupMember.
     Two fact families (three namespace years: 2020-09-30, 2022-09-30, 2025-10-31):
       A  PledgedOrEncumberedNumberOfShares          "pledged OR otherwise encumbered", one combined count
       B  NumberOfSharesEncumbered (total) = ...UnderPledged + ...UnderNonDisposalUndertaking + ...UnderOtherEncumbrances
     The families print percentages on different scales (99.99 vs 0.9999). The ratio is therefore computed from share COUNTS;
     the reported percentage is an integrity check only. "Encumbered" (pledge + NDU + other) is the definition comparable
     across both families; the pledge-only count exists in family B only.
  NOT used: /api/corporate-pledgedata. It is a snapshot regenerated daily (every one of 1,544 records carries today's
  broadcast time) and shows "pledged" shares for companies with no promoter (ICICI Bank, Coforge).

A filing with no promoter aggregate, or with 0 promoter shares, has NO promoter group: the ratio is undefined, not 0%.
A count that is absent while every flag says "nothing encumbered" is a company-declared zero (a real 0). Anything the parser
cannot reconcile (a count contradicting its own flag, components not summing to the total, encumbered above promoter holding, a
reported percentage that disagrees with the counts, an unknown schema) is rejected, never guessed.

Output: data/raw/shp/{SYMBOL}.parquet, one row per filing (see OUTPUT_COLS). A filing that failed to download or parse keeps its
master-list row with PARSE_ERROR set and null counts: missing, never zero.

Usage
-----
    python data/ingest_shp_pledge.py                       # all universe symbols (resumable), most-often-member first
    python data/ingest_shp_pledge.py --limit 20
    python data/ingest_shp_pledge.py --symbols VEDL,GMRAIRPORT
    python data/ingest_shp_pledge.py --shard 0/2           # run two shards in parallel (2 processes)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, PROCESSED_DATA_DIR, SHP_DIR, UNIVERSE_DIR
from data.ingest_common import write_parquet_atomic
from data.ingest_sec_list import _last_modified_utc
from data.publication import IST

MASTER_URL = "https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={symbol}"
API_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern",
}
MASTER_COLS = ["SYMBOL", "PERIOD_END", "BROADCAST_IST", "SUBMISSION_DATE", "IS_REVISED", "REVISION_DATE", "RECORD_ID", "PROMOTER_PCT", "XBRL_URL"]
PARSED_COLS = ["SCHEMA_FAMILY", "HAS_PROMOTER_GROUP", "PROMOTER_SHARES", "ENCUMBERED_SHARES", "PLEDGED_SHARES", "NDU_SHARES",
               "OTHER_ENCUMBERED_SHARES", "XBRL_PERIOD_END", "XBRL_LAST_MODIFIED_UTC", "PARSE_ERROR"]
OUTPUT_COLS = MASTER_COLS + PARSED_COLS
PROMOTER_MEMBER = "ShareholdingOfPromoterAndPromoterGroupMember"
CATEGORY_AXIS = "CategoryOfShareholdersAxis"
PERCENT_TOLERANCE = 0.05          # percentage points: the sources round to 2 (percent) or 4 (fraction) decimals


# ─── the master list ─────────────────────────────────────────────────────────────────────────────────────────────

def _ist_timestamp(text: str | None, label: str) -> str | None:
    if text is None or str(text).strip() in ("", "-"):
        return None
    try:
        return datetime.strptime(str(text).strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).isoformat()
    except ValueError:
        raise ValueError(f"unparseable {label} timestamp {text!r}") from None


def _day(text: str | None) -> pd.Timestamp:
    if text is None or str(text).strip() in ("", "-"):
        return pd.NaT
    return pd.Timestamp(datetime.strptime(str(text).strip()[:11], "%d-%b-%Y"))


def _xbrl_link(value: str | None) -> str | None:
    """A filing with no XBRL is written as the archive prefix plus a filename of '-' (.../corporate/xbrl/-), or as '-' alone."""
    text = (value or "").strip()
    return None if text in ("", "-") or text.endswith("/-") else text


def parse_master(records: list[dict], symbol: str) -> pd.DataFrame:
    """One row per filed pattern. Raises ValueError on another symbol's record, an unparseable timestamp or a duplicate."""
    rows = []
    for rec in records:
        if str(rec.get("symbol", "")).strip().upper() != symbol.upper():
            raise ValueError(f"record for symbol {rec.get('symbol')!r} in the list requested for {symbol}")
        promoter = pd.to_numeric(str(rec.get("pr_and_prgrp") or "").strip(), errors="coerce")
        rows.append({
            "SYMBOL": symbol.upper(), "PERIOD_END": _day(rec.get("date")), "BROADCAST_IST": _ist_timestamp(rec.get("broadcastDate"), "broadcast"),
            "SUBMISSION_DATE": _day(rec.get("submissionDate")), "IS_REVISED": str(rec.get("revisedData") or "N").strip().upper() == "Y",
            "REVISION_DATE": _ist_timestamp(rec.get("revisedDate"), "revision") if rec.get("revisedDate") else None,
            "RECORD_ID": str(rec.get("recordId") or ""), "PROMOTER_PCT": float(promoter) if pd.notna(promoter) else float("nan"),
            "XBRL_URL": _xbrl_link(rec.get("xbrl")),
        })
    out = pd.DataFrame(rows, columns=MASTER_COLS)
    if out.duplicated(["PERIOD_END", "RECORD_ID"]).any():
        raise ValueError("duplicate (period end, record id) rows in the master list")
    return out


# ─── the XBRL filing ─────────────────────────────────────────────────────────────────────────────────────────────

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _flag(flags: dict[str, bool], suffix_of: str) -> bool | None:
    hits = [v for k, v in flags.items() if k.endswith(suffix_of)]
    return hits[0] if hits else None


def parse_shp_xbrl(xml_text: str) -> dict:
    """
    The promoter-group encumbrance of one filing, from share counts. Returns SCHEMA_FAMILY ('A' | 'B'), PERIOD_END,
    HAS_PROMOTER_GROUP, PROMOTER_SHARES, ENCUMBERED_SHARES (pledged + NDU + other, or family A's combined count),
    PLEDGED_SHARES / NDU_SHARES / OTHER_ENCUMBERED_SHARES (family B only, else None).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML: {exc}") from None

    instants: list[date] = []
    promoter_ctx: list[str] = []
    for ctx in root.iter():
        if _local(ctx.tag) != "context":
            continue
        for el in ctx.iter():
            name = _local(el.tag)
            if name == "instant" and el.text:
                instants.append(date.fromisoformat(el.text.strip()))
            if name == "explicitMember" and _local(el.attrib.get("dimension", "").replace(":", "}")) == CATEGORY_AXIS \
                    and (el.text or "").strip().rsplit(":", 1)[-1] == PROMOTER_MEMBER:
                promoter_ctx.append(ctx.attrib["id"])

    facts: dict[str, dict[str, str]] = {}
    flags: dict[str, bool] = {}
    for el in root.iter():
        ref = el.attrib.get("contextRef")
        if ref is None:
            continue
        name, text = _local(el.tag), (el.text or "").strip()
        facts.setdefault(ref, {})[name] = text
        if name.startswith("WhetherAnySharesHeldByPromoters"):
            flags[name] = text.lower() == "true"

    all_names = {name for by in facts.values() for name in by}
    family_a = any("ArePledgeOrOtherwiseEncumbered" in n or n == "PledgedOrEncumberedNumberOfShares" for n in all_names)
    family_b = any("AreEncumberedUnder" in n or n.startswith("NumberOfSharesEncumbered") for n in all_names)
    if family_a == family_b:
        raise ValueError(f"unrecognised schema: neither (or both) of the pledge fact families found; {sorted(all_names)[:4]}")
    family = "A" if family_a else "B"
    if not instants:
        raise ValueError("no reporting period found in any context")
    result = {"SCHEMA_FAMILY": family, "PERIOD_END": max(set(instants), key=instants.count), "HAS_PROMOTER_GROUP": False,
              "PROMOTER_SHARES": None, "ENCUMBERED_SHARES": None, "PLEDGED_SHARES": None, "NDU_SHARES": None, "OTHER_ENCUMBERED_SHARES": None}

    if len(promoter_ctx) > 1:
        raise ValueError(f"more than one promoter aggregate context: {promoter_ctx}")
    if not promoter_ctx:
        return result                                                       # no promoter group in this filing
    agg = facts.get(promoter_ctx[0], {})
    if "NumberOfShares" not in agg:
        raise ValueError("the promoter aggregate has no NumberOfShares fact")
    promoter = float(agg["NumberOfShares"])
    result["PROMOTER_SHARES"] = promoter
    if promoter == 0:
        return result                                                       # 0 promoter shares: no promoter group, ratio undefined
    result["HAS_PROMOTER_GROUP"] = True

    def count(name: str) -> float | None:
        return float(agg[name]) if name in agg and agg[name] != "" else None

    if family == "A":
        flag = _flag(flags, "ArePledgeOrOtherwiseEncumberedForPromoterAndPromoterGroup")
        encumbered = count("PledgedOrEncumberedNumberOfShares")
        reported = count("PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares")
        if encumbered is None:
            if flag is False:
                encumbered = 0.0                                            # the company declared nothing is encumbered: a real zero
            else:
                raise ValueError("encumbered count absent while the pledge flag does not say 'false'")
        if flag is False and encumbered > 0:
            raise ValueError(f"pledge flag says nothing is encumbered but the count is {encumbered:,.0f}")
    else:
        flag_p = _flag(flags, "AreEncumberedUnderPledgedForPromoterAndPromoterGroup")
        flag_n = _flag(flags, "AreEncumberedUnderNonDisposalUndertakingForPromoterAndPromoterGroup")
        flag_o = _flag(flags, "AreEncumberedOtherThanByWayOfPledgeOrNDUForPromoterAndPromoterGroup")
        components = {}
        for key, fact, fl in (("PLEDGED_SHARES", "NumberOfSharesEncumberedUnderPledged", flag_p),
                              ("NDU_SHARES", "NumberOfSharesEncumberedUnderNonDisposalUndertaking", flag_n),
                              ("OTHER_ENCUMBERED_SHARES", "NumberOfSharesEncumberedUnderOtherEncumbrances", flag_o)):
            value = count(fact)
            if value is None:
                if fl is False:
                    value = 0.0                                             # a sub-fact is omitted when its flag is false
                else:
                    raise ValueError(f"{fact} absent while its flag does not say 'false'")
            elif fl is False and value > 0:
                raise ValueError(f"flag for {fact} says false but the count is {value:,.0f}")
            components[key] = value
        encumbered = count("NumberOfSharesEncumbered")
        if encumbered is None:
            encumbered = sum(components.values()) if all(f is False for f in (flag_p, flag_n, flag_o)) else None
            if encumbered is None:
                raise ValueError("total NumberOfSharesEncumbered absent while a flag says shares are encumbered")
        if abs(sum(components.values()) - encumbered) > 1:
            raise ValueError(f"components sum to {sum(components.values()):,.0f} but the total is {encumbered:,.0f}")
        reported = count("EncumberedSharesHeldAsPercentageOfTotalNumberOfShares")
        result.update(components)

    if encumbered > promoter:
        raise ValueError(f"encumbered shares {encumbered:,.0f} exceed the promoter holding {promoter:,.0f}")
    if reported is not None:
        ratio = 100.0 * encumbered / promoter
        if abs(ratio - reported) > PERCENT_TOLERANCE and abs(ratio - 100.0 * reported) > PERCENT_TOLERANCE:
            raise ValueError(f"counts imply {ratio:.2f}% but the filing reports a percentage of {reported}: misread")
    result["ENCUMBERED_SHARES"] = encumbered
    return result


# ─── driver: NSE API for the lists, the archive host for the XBRL ─────────────────────────────────────────────────

def _api_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(API_HEADERS)
    session.get("https://www.nseindia.com/", timeout=30)                    # cookies
    return session


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), retry=retry_if_exception_type(requests.RequestException), reraise=True)
def fetch_master(session: requests.Session, symbol: str) -> list[dict]:
    resp = session.get(MASTER_URL.format(symbol=symbol), timeout=40)
    if resp.status_code in (401, 403):
        raise requests.HTTPError(f"HTTP {resp.status_code}: session cookies expired")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"master list for {symbol} is not a list: {str(data)[:80]}")
    return data


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=40), retry=retry_if_exception_type(requests.RequestException), reraise=True)
def fetch_xbrl(session: requests.Session, url: str) -> tuple[str, str | None]:
    resp = session.get(url, timeout=90)
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig", errors="replace"), resp.headers.get("Last-Modified")


def universe_symbols_by_priority(first_year: int = 2021) -> list[str]:
    """Symbols that were ever in a Stage 5 universe from `first_year`, those in the most years first."""
    counts: dict[str, int] = {}
    for path in sorted(Path(UNIVERSE_DIR).glob("*.parquet")):
        if int(path.stem) >= first_year:
            for symbol in pd.read_parquet(path, columns=["SYMBOL"])["SYMBOL"]:
                counts[symbol] = counts.get(symbol, 0) + 1
    return sorted(counts, key=lambda s: (-counts[s], s))


def ingest_symbol(api: requests.Session, archive: requests.Session, symbol: str, delay: float) -> dict:
    """Master list + every filing's XBRL for one symbol. Rows already parsed in an earlier run are kept, not re-fetched."""
    path = SHP_DIR / f"{symbol}.parquet"
    master = parse_master(fetch_master(api, symbol), symbol)
    previous = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=OUTPUT_COLS)
    done = {(r.PERIOD_END, r.RECORD_ID): r for r in previous.itertuples() if r.PARSE_ERROR is None or pd.isna(r.PARSE_ERROR)} if len(previous) else {}
    rows, failed = [], 0
    for rec in master.itertuples():
        key = (rec.PERIOD_END, rec.RECORD_ID)
        if key in done:
            rows.append({c: getattr(done[key], c) for c in OUTPUT_COLS})
            continue
        base = {c: getattr(rec, c) for c in MASTER_COLS}
        blank = {c: None for c in PARSED_COLS}
        if rec.XBRL_URL is None or pd.isna(rec.XBRL_URL):
            rows.append({**base, **blank, "PARSE_ERROR": "no XBRL link in the master list"})      # nothing to fetch: missing, not a 404
            continue
        try:
            text, last_modified = fetch_xbrl(archive, rec.XBRL_URL)
            parsed = parse_shp_xbrl(text)
            if parsed["PERIOD_END"] != rec.PERIOD_END.date():
                raise ValueError(f"the filing is for {parsed['PERIOD_END']} but the list says {rec.PERIOD_END.date()}")
            rows.append({**base, **blank, **{k: parsed[k] for k in ("SCHEMA_FAMILY", "HAS_PROMOTER_GROUP", "PROMOTER_SHARES", "ENCUMBERED_SHARES",
                                                                   "PLEDGED_SHARES", "NDU_SHARES", "OTHER_ENCUMBERED_SHARES")},
                         "XBRL_PERIOD_END": pd.Timestamp(parsed["PERIOD_END"]), "XBRL_LAST_MODIFIED_UTC": _last_modified_utc(last_modified)})
        except (ValueError, requests.RequestException) as exc:
            failed += 1
            rows.append({**base, **blank, "PARSE_ERROR": str(exc)[:300]})
            logger.warning(f"{symbol} {rec.PERIOD_END.date()}: {str(exc)[:160]}")
        time.sleep(delay)
    out = pd.DataFrame(rows, columns=OUTPUT_COLS)
    for column in ("PROMOTER_SHARES", "ENCUMBERED_SHARES", "PLEDGED_SHARES", "NDU_SHARES", "OTHER_ENCUMBERED_SHARES"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["HAS_PROMOTER_GROUP"] = out["HAS_PROMOTER_GROUP"].astype("boolean")
    SHP_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(out, path)
    return {"filings": len(out), "failed": failed}


def run(symbols: list[str], shard: tuple[int, int] = (0, 1), delay: float = 0.4) -> dict:
    index, total = shard
    mine = symbols[index::total]
    checkpoint = PROCESSED_DATA_DIR / (f"shp_checkpoint_{index}of{total}.json" if total > 1 else "shp_checkpoint.json")
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    logger.add(LOG_DIR / f"shp_pledge_{index}of{total}.log", rotation="10 MB")
    api, archive = _api_session(), requests.Session()
    archive.headers.update({"User-Agent": API_HEADERS["User-Agent"], "Accept": "*/*"})
    started, requests_made = time.time(), 0
    for i, symbol in enumerate(mine, 1):
        if state.get(symbol, {}).get("failed") == 0 and symbol in state:
            continue
        try:
            try:
                state[symbol] = ingest_symbol(api, archive, symbol, delay)
            except requests.RequestException:
                api, requests_made = _api_session(), 0                      # stale cookies: renew once and retry the symbol
                state[symbol] = ingest_symbol(api, archive, symbol, delay)
        except (ValueError, requests.RequestException) as exc:
            state[symbol] = {"filings": 0, "failed": -1, "error": str(exc)[:200]}
            logger.error(f"{symbol}: {exc}")
        requests_made += state[symbol].get("filings", 0) + 1
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        os.replace(temporary, checkpoint)
        if requests_made > 150:                                              # the NSE API cookies go stale; renew them
            api, requests_made = _api_session(), 0
        if i % 10 == 0:
            rate = (time.time() - started) / i
            logger.info(f"[{i}/{len(mine)}] {symbol}: {state[symbol]} | {rate:.1f}s per symbol, eta {(len(mine) - i) * rate / 3600:.1f}h")
        time.sleep(delay)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest promoter encumbrance from NSE shareholding-pattern filings.")
    ap.add_argument("--symbols", type=str, help="comma-separated symbols (default: every Stage 5 universe symbol since 2021)")
    ap.add_argument("--limit", type=int, help="only the first N symbols of the priority order")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th symbol starting at i (run n processes)")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else universe_symbols_by_priority()
    if args.limit:
        symbols = symbols[:args.limit]
    index, total = (int(x) for x in args.shard.split("/"))
    run(symbols, (index, total), args.delay)


if __name__ == "__main__":
    main()
