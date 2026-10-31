"""
Stage 16 — entity resolution: map an announcement record to an NSE ticker, or say plainly that it cannot.

The rule this module exists to enforce: a ticker is NEVER guessed. Every record leaves `resolve` either with a ticker and
the method that produced it, or with no ticker, method "unresolved" and a reason. Ambiguity (two candidate tickers) is a
reason to refuse, not a reason to pick one.

Order of evidence, strongest first:

  1. SYMBOL   the exchange's own symbol tag, if the symbol traded (bhavcopy symbol master) near the publication date: some
              trading window must start no more than 30 days after, and end no more than 365 days before, it. A reused
              ticker's gap between two companies is refused ("symbol_outside_traded_window"). If the record
              also carries an ISIN that the master knows for OTHER symbols only, the two identifiers contradict each
              other and the record is refused ("symbol_isin_conflict"). An ISIN the master has never seen does not
              contradict anything (ISINs change on face-value splits). A symbol absent from the master is refused
              ("symbol_not_in_master") rather than rescued through the ISIN.
  2. ISIN     only when the symbol field is blank. One symbol ever carried it -> that symbol. Several (a rename keeps the
              ISIN) -> the one whose first/last traded window contains the publication date; zero or several matching
              windows -> "isin_ambiguous".
  3. NAME     only when symbol and ISIN are both unusable. Exact match of the normalised company name against a name
              index built from records that the exchange itself tagged with a symbol; exactly one symbol, present in
              the master -> that symbol. No fuzzy matching: a near-miss is a different company until proven otherwise.

The bhavcopy carries no company names, so the name index comes from the symbol-tagged records of the same chunk.

Point-in-time note: the symbol master is built from all bhavcopy on disk, so it knows symbols that began trading after a
given announcement. That decides only WHICH ticker a record names, not what was knowable when; the record's publication
time still keys it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

UNRESOLVED = "unresolved"
RESOLUTION_COLUMNS = ["TICKER", "RESOLUTION_METHOD", "UNRESOLVED_REASON"]
REASONS = ("symbol_isin_conflict", "symbol_not_in_master", "symbol_outside_traded_window", "isin_ambiguous", "name_ambiguous", "unmatched")

PRE_LISTING_SLACK = timedelta(days=30)        # an announcement may precede the first session (IPO / listing notices)
POST_DELISTING_SLACK = timedelta(days=365)    # delisting / merger notices follow the last session
NAN_ISIN_MAX_GAP_DAYS = 395                   # days with no ISIN (early bhavcopy) split into separate windows at a longer gap
ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")
_BLANKS = {"", "-", "NAN", "NONE", "NA"}
_SUFFIXES = {"limited": "ltd", "ltd": "ltd", "co": "co", "company": "co", "corporation": "corp", "corp": "corp",
             "pvt": "pvt", "private": "pvt", "and": "and"}


def normalise_name(name) -> str:
    """Lower-case, '&' -> 'and', punctuation dropped, legal-form words collapsed (Limited/Ltd, Company/Co...)."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    text = str(name).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(_SUFFIXES.get(tok, tok) for tok in text.split())


def _clean_symbol(value) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    return None if text in _BLANKS else text


def _clean_isin(value) -> str | None:
    text = _clean_symbol(value)
    return text if text and ISIN_RE.match(text) else None


class SymbolMaster:
    """Every (symbol, ISIN) pair seen in the bhavcopy with the first and last session it traded."""

    def __init__(self, symbols: set[str], by_isin: dict[str, list[tuple[str, date, date]]], isins_of: dict[str, set[str]],
                 windows: dict[str, list[tuple[date, date]]] | None = None):
        self._symbols, self._by_isin, self._isins_of = symbols, by_isin, isins_of
        self._windows = windows or {}

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SymbolMaster":
        symbols: set[str] = set()
        by_isin: dict[str, list[tuple[str, date, date]]] = {}
        isins_of: dict[str, set[str]] = {}
        windows: dict[str, list[tuple[date, date]]] = {}
        for sym, isin, first, last in frame[["SYMBOL", "ISIN", "FIRST_DATE", "LAST_DATE"]].itertuples(index=False):
            symbols.add(sym)
            windows.setdefault(sym, []).append((pd.Timestamp(first).date(), pd.Timestamp(last).date()))
            if isin is None or pd.isna(isin):
                continue
            by_isin.setdefault(isin, []).append((sym, pd.Timestamp(first).date(), pd.Timestamp(last).date()))
            isins_of.setdefault(sym, set()).add(isin)
        return cls(symbols, by_isin, isins_of, windows)

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._symbols

    def traded_near(self, symbol: str, day: date | None) -> bool:
        """Some trading window of `symbol` lies within the slack of `day`. An unknown date cannot be checked: False."""
        if day is None:
            return False
        return any(first - PRE_LISTING_SLACK <= day <= last + POST_DELISTING_SLACK
                   for first, last in self._windows.get(symbol, []))

    def symbols_for_isin(self, isin: str) -> list[tuple[str, date, date]]:
        return self._by_isin.get(isin, [])

    def symbol_count(self) -> int:
        return len(self._symbols)


def split_at_gaps(days: pd.Series | pd.DatetimeIndex, max_gap: int = NAN_ISIN_MAX_GAP_DAYS) -> list[tuple[date, date]]:
    """Sorted distinct days -> (first, last) windows, split wherever two consecutive days are more than max_gap apart."""
    ordered = pd.DatetimeIndex(sorted(set(pd.to_datetime(days))))
    if len(ordered) == 0:
        return []
    cuts = [0] + [i for i in range(1, len(ordered)) if (ordered[i] - ordered[i - 1]).days > max_gap] + [len(ordered)]
    return [(ordered[lo].date(), ordered[hi - 1].date()) for lo, hi in zip(cuts, cuts[1:])]


def build_symbol_master(bhavcopy_dir: Path) -> SymbolMaster:
    """
    Scan every bhavcopy Parquet (SYMBOL, ISIN columns only) into a SymbolMaster. One window per (symbol, ISIN); days with no
    ISIN (early bhavcopy has none) are split into separate windows at gaps over 395 days, so a reused ticker's idle years
    are not covered by one window spanning them.
    """
    files = sorted(Path(bhavcopy_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no bhavcopy files under {bhavcopy_dir}")
    parts = []
    for f in files:
        part = pd.read_parquet(f, columns=["SYMBOL", "ISIN"]).drop_duplicates()
        part["DAY"] = pd.Timestamp(f.stem)
        parts.append(part)
    allrows = pd.concat(parts, ignore_index=True)
    allrows["ISIN"] = allrows["ISIN"].astype("object").where(allrows["ISIN"].notna(), None)
    with_isin = allrows[allrows["ISIN"].notna()]
    frame = with_isin.groupby(["SYMBOL", "ISIN"])["DAY"].agg(FIRST_DATE="min", LAST_DATE="max").reset_index()
    no_isin = [(sym, None, pd.Timestamp(first), pd.Timestamp(last))
               for sym, days in allrows[allrows["ISIN"].isna()].groupby("SYMBOL")["DAY"]
               for first, last in split_at_gaps(days)]
    frame = pd.concat([frame, pd.DataFrame(no_isin, columns=["SYMBOL", "ISIN", "FIRST_DATE", "LAST_DATE"])], ignore_index=True)
    logger.info(f"symbol master: {frame['SYMBOL'].nunique():,} symbols over {len(files):,} bhavcopy sessions")
    return SymbolMaster.from_frame(frame)


def _resolve_one(symbol, isin, name, day, master: SymbolMaster, name_index: dict[str, set[str]]):
    sym, isin = _clean_symbol(symbol), _clean_isin(isin)
    when = None if day is None or pd.isna(day) else pd.Timestamp(day).date()

    if sym is not None:
        if not master.has_symbol(sym):
            return None, UNRESOLVED, "symbol_not_in_master"
        if not master.traded_near(sym, when):
            return None, UNRESOLVED, "symbol_outside_traded_window"
        if isin is not None:
            known = {s for s, _, _ in master.symbols_for_isin(isin)}
            if known and sym not in known:
                return None, UNRESOLVED, "symbol_isin_conflict"
        return sym, "symbol", None

    if isin is not None:
        candidates = master.symbols_for_isin(isin)
        distinct = {s for s, _, _ in candidates}
        if len(distinct) == 1:
            return next(iter(distinct)), "isin", None
        if len(distinct) > 1:
            inside = {s for s, first, last in candidates if when is not None and first <= when <= last}
            if len(inside) == 1:
                return next(iter(inside)), "isin", None
            return None, UNRESOLVED, "isin_ambiguous"

    key = normalise_name(name)
    if key:
        matches = name_index.get(key, set())
        if len(matches) > 1:
            return None, UNRESOLVED, "name_ambiguous"
        if len(matches) == 1:
            only = next(iter(matches))
            if master.has_symbol(only):
                return only, "name", None
    return None, UNRESOLVED, "unmatched"


def resolve(frame: pd.DataFrame, master: SymbolMaster, name_index: dict[str, set[str]]) -> pd.DataFrame:
    """
    Return a copy of `frame` with TICKER, RESOLUTION_METHOD, UNRESOLVED_REASON appended. Needs SYMBOL_RAW, ISIN,
    COMPANY_NAME and PUBLISHED_DATE columns. Row order is preserved.
    """
    out = frame.copy()
    tickers, methods, reasons = [], [], []
    for symbol, isin, name, day in frame[["SYMBOL_RAW", "ISIN", "COMPANY_NAME", "PUBLISHED_DATE"]].itertuples(index=False):
        t, m, r = _resolve_one(symbol, isin, name, day, master, name_index)
        tickers.append(t)
        methods.append(m)
        reasons.append(r)
    out["TICKER"] = pd.Series(tickers, index=out.index, dtype="string")
    out["RESOLUTION_METHOD"] = pd.Series(methods, index=out.index, dtype="string")
    out["UNRESOLVED_REASON"] = pd.Series(reasons, index=out.index, dtype="string")
    return out


def build_name_index(resolved: pd.DataFrame) -> dict[str, set[str]]:
    """Normalised company name -> every ticker the exchange itself tagged it with (symbol-method records only)."""
    index: dict[str, set[str]] = {}
    tagged = resolved[resolved["RESOLUTION_METHOD"] == "symbol"]
    for name, ticker in tagged[["COMPANY_NAME", "TICKER"]].itertuples(index=False):
        key = normalise_name(name)
        if key:
            index.setdefault(key, set()).add(ticker)
    return index
