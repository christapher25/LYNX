"""
Stage 10 — NSE F&O bhavcopy ingestion. Futures and options, one file per trading day.

Two formats, one normalized schema (measured by probing the live archive, 2026-09-24; see
data/raw/DATA_NOTES.md §8):

    legacy   2000-06-12 → 2024-07-05   15 columns + a trailing empty one, "fo{DD}{MON}{YYYY}bhav.csv.zip"
    UDiFF    ≤2024-07-05 → present     34 columns, "BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip"

The legacy format ends and UDiFF becomes the only one on the same date as the cash-market cutover
(NSE Circular 62424, 2024-07-08). Both were served for 2024-07-05 with identical contract counts,
which lets scripts/verify_stage10_fno.py prove the two parsers agree on real data.

Normalization decisions, each pinned by tests/test_fno_ingest.py and
tests/invariants/test_fno_format_equivalence.py:

  * Futures carry NO strike and NO option type (NaN). Legacy files write strike 0 / type "XX" for
    "not an option"; a strike of 0 is a false price level, so it is not carried into the output.
  * Traded value is in LAKHS of rupees in both eras (UDiFF reports exact rupees; divided by 100,000).
    Legacy files TRUNCATE it to 0.01 lakh (Rs 1,000): on the 2024-07-05 overlap day UDiFF >= legacy on
    every traded contract, by less than 0.01 lakh, so value has finer resolution after the cutover.
  * Instrument codes use the legacy names (FUTIDX/FUTSTK/OPTIDX/OPTSTK); UDiFF's IDF/STF/IDO/STO map to them.
  * Fields only UDiFF publishes (underlying price, lot size, number of trades) are NaN before it,
    never 0.
  * Option types keep their raw codes: CE/PE (European) and CA/PA (American-style stock options,
    the only kind until ~2011). Instruments outside the four equity-derivative products that
    the legacy archive also carries (FUTINT, UNDINT, FUTIVX) are ingested, not dropped.
  * Everything else is kept AS REPORTED, including the zero OPEN/HIGH/LOW that NSE writes for a
    contract that did not trade. Masking a zero price is the feature layer's job, as it is for
    equities in Stage 9; ingestion does not reinterpret the archive.

Source-file damage handled (legacy only; each is recorded in the day's manifest era, and anything not
matching a known shape still fails the day): repeated header lines and records fused onto one line
(repair_legacy_csv, lossless), a day's data pasted together more than once (collapse_identical_duplicates,
only when every copy is identical), and contracts left in the file after they expired (drop_expired_contracts).

Known limitation shared with Stage 3: only weekdays are requested, so weekend sessions (Muhurat
trading, Budget-day Saturdays) are not fetched.

Usage
-----
    python data/ingest_fno_bhavcopy.py                     # full backfill (resumable)
    python data/ingest_fno_bhavcopy.py --start 2024-07-01 --end 2024-07-12
    python data/ingest_fno_bhavcopy.py --days 30           # last 30 days
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FNO_BHAVCOPY_DIR, LOG_DIR, MANIFEST_DIR
from data.ingest_common import run_ingestion

# ─────────────────────────────────────────────────────────────────
# Constants — measured, see data/raw/DATA_NOTES.md §8
# ─────────────────────────────────────────────────────────────────

FIRST_AVAILABLE = date(2000, 6, 12)      # probed: 2000-06-09 -> 404, 2000-06-12 -> 200 (index futures launch)
UDIFF_CUTOVER = date(2024, 7, 8)         # NSE Circular 62424; UDiFF was also served for 2024-07-05

LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
    "{yyyy}/{MON}/fo{DD}{MON}{yyyy}bhav.csv.zip"
)
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

OUTPUT_COLS = [
    "SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE", "DATE",
    "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PRICE",
    "CONTRACTS", "VALUE_LAKHS", "OPEN_INT", "CHG_IN_OI",
    "UNDERLYING_PRICE", "LOT_SIZE", "NUM_TRADES",
]
CONTRACT_KEY = ["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE"]

# The four equity-derivative products, plus three that appear in the legacy archive and are kept
# as reported (dropping rows irreversibly at ingestion would be the wrong place to decide relevance;
# the feature layer filters by INSTRUMENT): FUTINT interest-rate futures (2003-2009), UNDINT their
# underlying rows (2003-2007), FUTIVX India VIX futures (2014-2018). Anything else fails the day.
KNOWN_INSTRUMENTS = {"FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK", "FUTINT", "UNDINT", "FUTIVX"}
UDIFF_INSTRUMENTS = {"IDF": "FUTIDX", "STF": "FUTSTK", "IDO": "OPTIDX", "STO": "OPTSTK"}
# CE/PE: European call/put. CA/PA: American-style call/put, the only stock-option codes until ~2011.
# Raw codes are kept so the distinction survives; call vs put is the first letter.
OPTION_TYPES = {"CE", "PE", "CA", "PA"}
# The legacy option-type column was "OPTION_TYP" in most years and "OPTIONTYPE" in some (e.g. 2003, 2005, 2007).
LEGACY_OPTION_TYPE_COLUMNS = ("OPTION_TYP", "OPTIONTYPE")
RUPEES_PER_LAKH = 100_000
# A contract that expired within this many days before the trade date is a stale row NSE left in the file
# (2002-09-27 still lists the 707 contracts that expired 2002-09-26); older than that means a date fault.
STALE_EXPIRY_WINDOW_DAYS = 7
FOOTER_NOTE = "*updated"                 # the only trailing non-data line seen in the archive (2012-05-14)

MANIFEST_PATH = MANIFEST_DIR / "fno_bhavcopy.jsonl"


# ─────────────────────────────────────────────────────────────────
# Repair of known source-file damage (legacy CSV only)
# ─────────────────────────────────────────────────────────────────

def repair_legacy_csv(text: str) -> tuple[str, dict]:
    """
    Fix the two kinds of damage found in NSE's 2002 legacy files, and nothing else.

      * A repeated copy of the header line mid-file (two files concatenated). It is not data; dropped.
      * Several records fused onto one line because a newline was lost. The files end every record with a
        trailing comma, and that comma is the only separator left, so a line of k fused records has
        15*k + 1 fields where 16 are expected; it is split back into its k records exactly.

    Any other line with the wrong number of fields raises, failing the day: a repair is only made where the
    damage has exactly the observed shape and the fix is lossless. The layout without a trailing comma
    (15 columns, e.g. 2001) has no known damage and is only checked, never repaired.

    Also drops a final line that is exactly '*Updated' (a re-publication marker NSE appended to the 2012-05-14 file).

    Returns (text, {"header_repeats": n, "fused_lines": m, "footers": k}); the text is returned untouched when all are 0.
    """
    report = {"header_repeats": 0, "fused_lines": 0, "footers": 0}
    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip()]
    if not lines:
        return text, report

    header = lines[0]
    expected = header.count(",") + 1
    trailing_comma = header.endswith(",")
    fields = expected - 1 if trailing_comma else expected
    repaired = [header]
    for position, line in enumerate(lines[1:], start=1):
        if line.strip() == header.strip():
            report["header_repeats"] += 1
            continue
        if position == len(lines) - 1 and line.strip().lower() == FOOTER_NOTE:
            report["footers"] += 1                   # a trailing '*Updated' note; not data
            continue
        n = line.count(",") + 1
        if n == expected:
            repaired.append(line)
        elif trailing_comma and n > expected and (n - 1) % fields == 0:
            parts = line.split(",")
            for k in range((n - 1) // fields):
                repaired.append(",".join(parts[k * fields:(k + 1) * fields]) + ",")
            report["fused_lines"] += 1
        else:
            raise ValueError(f"line with {n} fields where {expected} are expected: {line[:90]!r}")

    if not any(report.values()):
        return text, report
    return "\n".join(repaired) + "\n", report


# ─────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────

def _numeric(series: pd.Series) -> pd.Series:
    """Numeric or NaN. A blank or unparseable cell is missing, never zero."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def _dates(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), format=fmt, errors="coerce")


def _legacy_dates(series: pd.Series) -> pd.Series:
    """
    Legacy dates are '13-JUL-2020'. On 2020-07-13 NSE's equity bhavcopy wrote '13-Jul-20' (two-digit
    year) and Stage 3's strict parser lost the whole day, so rows that fail the four-digit parse get
    one retry with a two-digit year. Anything still unparseable stays NaT and fails validate(), which
    also checks the parsed date against the requested one, so the fallback cannot file a row under
    the wrong day.
    """
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d-%b-%Y", errors="coerce")
    retry = parsed.isna()
    if retry.any():
        parsed[retry] = pd.to_datetime(text[retry], format="%d-%b-%y", errors="coerce")
    return parsed


def _option_type(series: pd.Series) -> pd.Series:
    """CE / PE, else NaN. Legacy 'XX' (and any blank) means 'not an option'."""
    cleaned = series.astype(str).str.strip().str.upper()
    return cleaned.where(cleaned.isin(OPTION_TYPES))


def parse_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy format, 2000-06-12 → 2024-07-05."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]   # trailing-comma artifact

    type_column = next((c for c in LEGACY_OPTION_TYPE_COLUMNS if c in df.columns), None)
    if type_column is None:
        raise KeyError(f"no option-type column among {LEGACY_OPTION_TYPE_COLUMNS}; file has {list(df.columns)}")
    option_type = _option_type(df[type_column])
    out = pd.DataFrame({
        "SYMBOL": df["SYMBOL"].astype(str).str.strip(),
        "INSTRUMENT": df["INSTRUMENT"].astype(str).str.strip().str.upper(),
        "EXPIRY_DATE": _legacy_dates(df["EXPIRY_DT"]),
        "STRIKE": _numeric(df["STRIKE_PR"]).where(option_type.notna()),      # strike 0 on a future is not a strike
        "OPTION_TYPE": option_type,
        "DATE": _legacy_dates(df["TIMESTAMP"]),                              # '9-NOV-2001', '09-NOV-2001', '13-Jul-20'
        "OPEN": _numeric(df["OPEN"]),
        "HIGH": _numeric(df["HIGH"]),
        "LOW": _numeric(df["LOW"]),
        "CLOSE": _numeric(df["CLOSE"]),
        "SETTLE_PRICE": _numeric(df["SETTLE_PR"]),
        "CONTRACTS": _numeric(df["CONTRACTS"]),
        "VALUE_LAKHS": _numeric(df["VAL_INLAKH"]),
        "OPEN_INT": _numeric(df["OPEN_INT"]),
        "CHG_IN_OI": _numeric(df["CHG_IN_OI"]),
        "UNDERLYING_PRICE": np.nan,                                          # not published before UDiFF
        "LOT_SIZE": np.nan,
        "NUM_TRADES": np.nan,
    })
    return out[OUTPUT_COLS]


def parse_udiff(df: pd.DataFrame) -> pd.DataFrame:
    """UDiFF format, 2024-07-05 onward."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    code = df["FinInstrmTp"].astype(str).str.strip().str.upper()
    out = pd.DataFrame({
        "SYMBOL": df["TckrSymb"].astype(str).str.strip(),
        "INSTRUMENT": code.map(UDIFF_INSTRUMENTS).fillna(code),              # unknown codes surface in validate()
        "EXPIRY_DATE": _dates(df["XpryDt"], "%Y-%m-%d"),
        "STRIKE": _numeric(df["StrkPric"]),
        "OPTION_TYPE": _option_type(df["OptnTp"]),
        "DATE": _dates(df["TradDt"], "%Y-%m-%d"),
        "OPEN": _numeric(df["OpnPric"]),
        "HIGH": _numeric(df["HghPric"]),
        "LOW": _numeric(df["LwPric"]),
        "CLOSE": _numeric(df["ClsPric"]),
        "SETTLE_PRICE": _numeric(df["SttlmPric"]),
        "CONTRACTS": _numeric(df["TtlTradgVol"]),
        "VALUE_LAKHS": _numeric(df["TtlTrfVal"]) / RUPEES_PER_LAKH,           # UDiFF is in rupees
        "OPEN_INT": _numeric(df["OpnIntrst"]),
        "CHG_IN_OI": _numeric(df["ChngInOpnIntrst"]),
        "UNDERLYING_PRICE": _numeric(df["UndrlygPric"]),
        "LOT_SIZE": _numeric(df["NewBrdLotQty"]),
        "NUM_TRADES": _numeric(df["TtlNbOfTxsExctd"]),
    })
    return out[OUTPUT_COLS]


def collapse_identical_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Collapse rows that are exact copies of each other, and say how many were removed.

    Several NSE files (2003-04-17 twice over, 2003-04-23 six times over) are one day's data pasted together
    repeatedly, and every copy is identical on every column, so the extras are redundant, not extra
    information. Copies of one contract that DISAGREE mean the file cannot be trusted and raise: silently
    keeping one of two different answers would be a guess about which is right.
    """
    deduplicated = df.drop_duplicates()
    removed = len(df) - len(deduplicated)
    if deduplicated.duplicated(CONTRACT_KEY).any():
        n = int(deduplicated.duplicated(CONTRACT_KEY, keep=False).sum())
        raise ValueError(f"{n} rows are copies of the same contract that disagree on their values")
    return (deduplicated.reset_index(drop=True) if removed else df), removed


def drop_expired_contracts(df: pd.DataFrame, trade_date: date) -> tuple[pd.DataFrame, int]:
    """
    Drop rows for contracts that had already expired before `trade_date`, and say how many.

    NSE's 2002-09-27 file is a real trading day (it lists contracts the previous day lacks) yet still carries
    the September series that expired on the 26th, with volume and open interest, which cannot exist after
    expiry. Rejecting the day would discard ~1,400 valid contracts, so only the provably stale rows go.
    Two guards keep a systematic fault from being "repaired" into an empty file: a contract expired more than
    STALE_EXPIRY_WINDOW_DAYS before the trade date means the dates are wrong, not that a row is stale; and if
    half or more of the file would be dropped something is wrong with the file as a whole. Both raise.
    """
    day = pd.Timestamp(trade_date)
    expired = df["EXPIRY_DATE"] < day                    # NaT compares False; validate() rejects those separately
    n = int(expired.sum())
    if n == 0:
        return df, 0
    if (df.loc[expired, "EXPIRY_DATE"] < day - pd.Timedelta(days=STALE_EXPIRY_WINDOW_DAYS)).any():
        raise ValueError(f"contracts expired long before the trade date (>{STALE_EXPIRY_WINDOW_DAYS} days): a date fault, not stale rows")
    if n * 2 >= len(df):
        raise ValueError(f"{n} of {len(df)} rows expired before the trade date; half or more, refusing to drop them")
    return df.loc[~expired].reset_index(drop=True), n


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Structural checks before a file is accepted. Anything unexpected fails the day loudly (it lands
    in the manifest as 'failed' and is retried on the next run) rather than entering the archive.
    """
    if df.empty:
        raise ValueError("parsed to zero rows")

    if df["DATE"].isna().any():
        raise ValueError(f"{int(df['DATE'].isna().sum())} rows failed date parsing")
    if df["EXPIRY_DATE"].isna().any():
        raise ValueError(f"{int(df['EXPIRY_DATE'].isna().sum())} rows failed expiry date parsing")

    parsed_dates = df["DATE"].dt.date.unique()
    if len(parsed_dates) != 1:
        raise ValueError(f"expected one date, found {len(parsed_dates)}")
    if parsed_dates[0] != trade_date:
        raise ValueError(f"file dated {parsed_dates[0]}, requested {trade_date}")

    unknown = set(df["INSTRUMENT"]) - KNOWN_INSTRUMENTS
    if unknown:
        raise ValueError(f"unrecognised instrument type(s) {sorted(unknown)} - review before ingesting")

    for col in ("OPEN_INT", "CONTRACTS"):
        if (df[col] < 0).any():
            raise ValueError(f"negative {col} present")

    is_option = df["INSTRUMENT"].str.startswith("OPT")
    bad_options = is_option & (df["STRIKE"].isna() | (df["STRIKE"] <= 0) | df["OPTION_TYPE"].isna())
    if bad_options.any():
        raise ValueError(f"{int(bad_options.sum())} option rows lack a positive strike or a CE/PE type")
    bad_futures = ~is_option & (df["STRIKE"].notna() | df["OPTION_TYPE"].notna())
    if bad_futures.any():
        raise ValueError(f"{int(bad_futures.sum())} future rows carry a strike or option type")

    expired = df["EXPIRY_DATE"] < df["DATE"]
    if expired.any():
        raise ValueError(f"{int(expired.sum())} contracts expired before the trade date")

    duplicated = df.duplicated(CONTRACT_KEY, keep=False)
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate contract rows")

    return df


# ─────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────

def build_url(trade_date: date) -> tuple[str, str]:
    """Return (url, era_label) for the date's format generation."""
    if trade_date >= UDIFF_CUTOVER:
        return UDIFF_URL.format(yyyymmdd=trade_date.strftime("%Y%m%d")), "udiff"
    return (
        LEGACY_URL.format(
            yyyy=trade_date.strftime("%Y"),
            MON=trade_date.strftime("%b").upper(),
            DD=trade_date.strftime("%d"),
        ),
        "legacy",
    )


def fetch_day(session: requests.Session, trade_date: date) -> tuple[pd.DataFrame | None, str, int]:
    """
    Fetch one trading day. Returns (df, era, bytes). A None dataframe means no trading that day
    (the archive answers 404 for weekends outside sessions and exchange holidays).

    No minimum-size cutoff: the very first legacy files (2000-06-12, three futures rows) are 386
    bytes and perfectly valid. A 200 that is not a zip archive fails the day instead.
    """
    url, era = build_url(trade_date)
    resp = session.get(url, timeout=60)

    if resp.status_code == 404:
        return None, era, 0
    resp.raise_for_status()

    nbytes = len(resp.content)
    if not zipfile.is_zipfile(io.BytesIO(resp.content)):
        raise ValueError(f"HTTP 200 but not a zip archive ({nbytes} bytes)")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError("zip archive is empty")
        with archive.open(members[0]) as handle:
            payload = handle.read()

    if era == "udiff":
        parsed = parse_udiff(pd.read_csv(io.BytesIO(payload), low_memory=False))
    else:
        text, repairs = repair_legacy_csv(payload.decode("latin-1"))
        parsed = parse_legacy(pd.read_csv(io.StringIO(text), low_memory=False))
        parsed, dupes = collapse_identical_duplicates(parsed)
        parsed, expired = drop_expired_contracts(parsed, trade_date)
        if any(repairs.values()) or dupes or expired:   # the manifest keeps the audit trail of repaired days
            era = (f"legacy+repaired(header={repairs['header_repeats']},fused={repairs['fused_lines']},"
                   f"dupes={dupes},expired={expired},footer={repairs['footers']})")
    return validate(parsed, trade_date), era, nbytes


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NSE F&O bhavcopy (futures and options).")
    ap.add_argument("--start", type=str, help="start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="end date YYYY-MM-DD")
    ap.add_argument("--days", type=int, help="fetch only the last N days")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests (default 0.4)")
    ap.add_argument("--weekends", action="store_true",
                    help="request Saturdays and Sundays only (Budget-day, Muhurat and special sessions); "
                         "default is weekdays only")
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    if args.days:
        start = end - timedelta(days=args.days)
    elif args.start:
        start = date.fromisoformat(args.start)
    else:
        start = FIRST_AVAILABLE
    start = max(start, FIRST_AVAILABLE)

    run_ingestion(
        name="fno_bhavcopy",
        start=start,
        end=end,
        output_dir=FNO_BHAVCOPY_DIR,
        fetch_fn=fetch_day,
        manifest_path=MANIFEST_PATH,
        log_dir=LOG_DIR,
        request_delay=args.delay,
        days="weekends" if args.weekends else "weekdays",
    )


if __name__ == "__main__":
    main()
