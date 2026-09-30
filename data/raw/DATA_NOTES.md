# LYNX — Data Source Findings

**Stage 3 deliverable.** Every fact in this document was measured by
probing NSE's live archive, not assumed from documentation or blog
posts. Where a date matters to the project's evaluation design, the
probe that established it is recorded so it can be re-verified.

Last updated: Stage 10 (F&O ingestion, §10). Sections 1-9 are Stage 3.

---

## 1. Summary — what is available

| Dataset | Source file | First available | Coverage |
|---|---|---|---|
| **OHLCV prices** | bhavcopy | **3 Nov 1994** | ~31 years |
| **Delivery %** | `sec_bhavdata_full` | **30 Sep 2019** | ~6 years |
| **F&O / open interest** | F&O bhavcopy | **12 Jun 2000** | ~26 years (Stage 10) |
| **F&O ban list** | `fo_secban` | **1 Jan 2008** | ~18 years (Stage 10) |

**Implication for the walk-forward schedule:** price history supports
roughly **25 folds** under the 4y-train / 1y-validate / 1y-test rolling
scheme — more than the 21 years assumed in the v3 architecture plan.
Delivery percentage is a **per-fold-optional feature**, present in the
most recent ~4 folds only.

---

## 2. The July 2024 format cutover

NSE Circular No. 62424 (12 June 2024) discontinued the legacy bhavcopy
format effective **8 July 2024**, replacing it with CM-UDiFF (Unified
Distilled File Format).

This is not a rename. The column structure is entirely different, and
any ingestion code must branch on this date.

### Pre-cutover format (before 8 July 2024)

```
https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{MON}/cm{DD}{MON}{yyyy}bhav.csv.zip
```
Example: `.../2024/JUN/cm03JUN2024bhav.csv.zip`

**14 columns**, plain names:
```
SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE,
TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN, Unnamed: 13
```

Note the trailing `Unnamed: 13` — an artifact of a trailing comma in
the source file. Drop it on load.

### Post-cutover format (from 8 July 2024)

```
https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip
```
Example: `.../BhavCopy_NSE_CM_0_0_0_20240710_F_0000.csv.zip`

**34 columns**, abbreviated names:
```
TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, ISIN, TckrSymb,
SctySrs, XpryDt, FininstrmActlXpryDt, StrkPric, OptnTp, FinInstrmNm,
OpnPric, HghPric, LwPric, ClsPric, LastPric, PrvsClsgPric,
UndrlygPric, SttlmPric, OpnIntrst, ChngInOpnIntrst, TtlTradgVol,
TtlTrfVal, TtlNbOfTxsExctd, SsnId, NewBrdLotQty, Rmks,
Rsvd1, Rsvd2, Rsvd3, Rsvd4
```

### Column mapping between formats

| Old | New |
|---|---|
| `SYMBOL` | `TckrSymb` |
| `SERIES` | `SctySrs` |
| `OPEN` | `OpnPric` |
| `HIGH` | `HghPric` |
| `LOW` | `LwPric` |
| `CLOSE` | `ClsPric` |
| `LAST` | `LastPric` |
| `PREVCLOSE` | `PrvsClsgPric` |
| `TOTTRDQTY` | `TtlTradgVol` |
| `TOTTRDVAL` | `TtlTrfVal` |
| `TOTALTRADES` | `TtlNbOfTxsExctd` |
| `TIMESTAMP` | `TradDt` |
| `ISIN` | `ISIN` |

The new format also covers derivatives in the same file (hence
`XpryDt`, `StrkPric`, `OptnTp`), so equity rows must be filtered by
`FinInstrmTp == 'STK'`.

---

## 3. `sec_bhavdata_full` — the preferred backbone

```
https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv
```
Example: `.../sec_bhavdata_full_03062024.csv`

**Single stable format across the entire July 2024 cutover.** Verified
identical column structure on both 3 June 2024 (pre) and 10 July 2024
(post). Unlike the bhavcopy, this file needs only one parser.

**15 columns**, including full OHLCV *and* delivery:
```
SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,
NO_OF_TRADES, DELIV_QTY, DELIV_PER
```

This is the preferred source **where available** (2019-09-30 onward),
because it provides price and delivery data together in one stable
format. Older history must come from the bhavcopy files.

### Parser quirks — all three must be handled

**1. Leading spaces in column names.**
Every column except `SYMBOL` is prefixed with a space: `' SERIES'`,
`' DELIV_PER'`. Strip all column names on load.

**2. `-` denotes missing, not zero.**
`DELIV_QTY` and `DELIV_PER` use a literal `-` where no delivery data
exists (commonly `BE` series and other restricted securities).

> **This must never be zero-filled.** A stock with zero delivery is a
> different fact from a stock with no delivery data. Zero-filling
> corrupts the feature and is the exact class of silent bug the
> invariant tests exist to prevent. Parse to `NaN`, carry a missingness
> indicator.

**3. Date format is `03-Jun-2024`.**
Parse explicitly with `format="%d-%b-%Y"`. Do not rely on inference.

---

## 4. Delivery data availability — measured

**First available date: 30 September 2019.**

Probe results narrowing the boundary:

| Date | Result |
|---|---|
| 2 Jan 2018 | 404 |
| 1 Jul 2019 | 404 |
| 1 Aug 2019 | 404 |
| 2 Sep 2019 | 404 |
| 17 Sep 2019 | 404 |
| 24 Sep 2019 | 404 |
| 27 Sep 2019 | 404 |
| **30 Sep 2019** | **200 — 210,130 bytes** |
| 1 Oct 2019 | 200 |
| 3 Oct 2019 | 200 |
| Continuous thereafter | 200 |

### Consequence

Delivery percentage — the strongest India-specific feature in the
design — covers roughly **6 years**, not the full history.

This was anticipated. Per architecture plan §11.3 (*"Delivery data too
shallow"*, severity High), the mitigation is:

- Delivery is a **per-fold-optional feature**, not a required one
- Folds testing before Sept 2019 mask it with a missingness indicator
- Folds after Sept 2019 use it
- The difference becomes **ablation evidence**: does the delivery-enabled
  subset outperform? That is a cleaner test of the feature's value than
  any single-fold result would have been

The walk-forward schedule itself is **unaffected**, because it runs on
price history, which is far deeper.

---

## 5. Price history availability — measured

**First available date: 3 November 1994** — effectively NSE's equity
segment launch.

Annual probe, all returning 200:

| Date | Size |
|---|---|
| 3 Nov 1994 | 2,841 bytes |
| 2 Jan 1995 | 4,177 bytes |
| 2 Jan 1996 | 23,013 bytes |
| 2 Jan 1998 | 25,123 bytes |
| 3 Jan 2000 | 28,424 bytes |
| 1 Jan 2002 | 19,536 bytes |
| 3 Jan 2005 | 24,455 bytes |
| 2 Jan 2008 | 36,474 bytes |
| 2 Jan 2012 | 52,248 bytes |
| 1 Jan 2016 | 58,943 bytes |
| 2 Jan 2020 | 69,955 bytes |
| 3 Jun 2024 | 106,028 bytes |

### Note on universe expansion

File size grows ~37x across the sample, from 2.8 KB to 106 KB. This is
real universe growth, not a formatting change — NSE listed a handful of
securities in 1994 and thousands today.

**Early folds will have a materially smaller investable universe.** This
is expected and should be reported, not corrected for. It is one of the
things §8.4's era analysis is designed to surface.

### Note on microstructure comparability

Indian market structure in the 1990s — physical settlement, badla
financing, no derivatives, far fewer participants — does not resemble
the present market. The rolling-window design (architecture plan §7.2)
already handles this correctly: 1995 data trains the 1999 model and
never touches the 2024 model.

Deep history is therefore an **evaluation** benefit (more out-of-sample
folds), not a training benefit. Early-era performance differing from
recent-era performance is an expected finding, not a bug.

---

## 6. Stress windows covered

All events named in architecture plan §8.4 fall inside the available
price history:

| Event | Date | In range |
|---|---|---|
| Dot-com peak and unwind | 2000 | ✓ |
| Global financial crisis | 2008 | ✓ |
| Taper tantrum / rupee crisis | 2013 | ✓ |
| Demonetisation | Nov 2016 | ✓ |
| IL&FS / NBFC crisis | 2018 | ✓ |
| COVID crash | Mar 2020 | ✓ |
| Retail participation boom | 2021 | ✓ |

---

## 7. Access notes

**User-Agent header is required.** NSE returns errors for requests
without a browser-like `User-Agent`. All probes used
`{"User-Agent": "Mozilla/5.0"}`.

**Rate limiting.** Ingestion uses a 0.5s delay between requests. Do not
lower this. A full backfill is thousands of sequential requests against
a public archive; being aggressive risks a block.

**404 is expected and normal.** Weekends, exchange holidays, and dates
before a dataset's start all return 404. Ingestion must treat 404 as
"no trading day" rather than as an error.

**Intermittent DNS failures observed.** During probing,
`nsearchives.nseindia.com` occasionally failed to resolve, then
succeeded on retry moments later. Ingestion code includes retry with
exponential backoff for this reason.

---

## 8. Storage layout

```
data/raw/
├── delivery/          # sec_bhavdata_full, one .parquet per trading day
│   └── YYYY-MM-DD.parquet
├── bhavcopy/          # legacy + UDiFF price files
│   └── YYYY-MM-DD.parquet
├── fno_bhavcopy/      # not yet ingested
└── announcements/     # not yet ingested
```

Parquet rather than CSV: training reads thousands of these files
repeatedly, and Parquet's columnar format loads substantially faster
while taking less disk.

One file per trading day rather than one large file: makes ingestion
resumable, makes per-date point-in-time reads trivial, and means a
corrupted download affects one day rather than the whole archive.

---

## 9. Open items

- [x] F&O bhavcopy availability and format probed (Stage 10, see §10)
- [ ] Index data (NIFTY, BANKNIFTY, INDIAVIX) source not yet confirmed
- [ ] Corporate actions source not yet identified — required for Stage 4
- [ ] Historical index constituents — flagged in the architecture plan as
      manual assembly; not yet started
- [ ] Confirm whether the UDiFF file's `FinInstrmTp == 'STK'` filter is
      sufficient to isolate cash-segment equities
- ## Final ingestion coverage — CONFIRMED COMPLETE
Bhavcopy: 7,875 trading days, 1994-11-03 to 2026-09-04
Delivery: 1,819 candidate dates, 2019-09-30 to 2026-09-17, fully reconciled
  - 83 genuine NSE holidays/closures, spot-checked against official sources
  - 1 settlement-holiday edge case (2024-02-19, Chhatrapati Shivaji Maharaj
    Jayanti) — file existed but failed validation on a single bad row
    (DELIV_QTY > VOLUME), discarding valid data for ~700 other stocks.
    Candidate for a future fix: reject the row, not the whole file.


---

## 10. F&O data — measured (Stage 10, probed 2026-09-24)

Same rules as the rest of this file: every fact below was measured against the live archive.

### F&O bhavcopy (futures and options, one file per trading day)

| | URL | Coverage |
|---|---|---|
| Legacy | `https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{MON}/fo{DD}{MON}{yyyy}bhav.csv.zip` | 2000-06-12 → 2024-07-05 |
| UDiFF | `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip` | ≤ 2024-07-05 → present |

- **First available: 12 Jun 2000** (2000-06-09 → 404; 2000-06-12 → 200, NSE's first index-futures day). The first files
  hold three futures rows and are 386 bytes, which is why the ingester has no minimum-size cutoff.
- **The two formats overlap on 2024-07-05** and both hold 33,930 contracts. Parsed, they agree on every contract and
  every shared field exactly, except traded value: legacy TRUNCATES it to 0.01 lakh (Rs 1,000; measured on 12,531 traded
  contracts: UDiFF >= legacy on all of them, by less than 0.01 lakh, e.g. 1216.599997 -> 1216.59), UDiFF is exact rupees.
- **Legacy is not one format.** A survey of 112 real dates (2000-2026) found, beyond the four samples first inspected:
  - the option-type column is `OPTION_TYP` in most years but `OPTIONTYPE` in some (2003, 2005, 2007);
  - stock options carry `CA`/`PA` (American-style) codes until ~2011, `CE`/`PE` after; index options were always `CE`/`PE`;
  - instruments beyond the four equity products: `FUTINT` (interest-rate futures, 2003-2009), `UNDINT` (their underlying
    rows, option-type code `FF`, 2003-2007), `FUTIVX` (India VIX futures, 2014-2018);
  - 2020-07-13 writes dates with a two-digit year (`13-Jul-20`). The same glitch made Stage 3's equity ingester reject
    that day's 2,001-row bhavcopy (see §11).
- **Encoding differences a parser must normalize** (all pinned by tests): a future is `strike 0 / type XX` in legacy but
  blank in UDiFF; value is lakhs in legacy and rupees in UDiFF; expiry dates are `29-Jun-2000` vs `2024-08-29`; early
  files write single-digit days (`9-NOV-2001`); UDiFF instrument codes are `IDF/STF/IDO/STO` for
  `FUTIDX/FUTSTK/OPTIDX/OPTSTK`.
- **Zeros mean "did not trade".** A contract with no trades carries OPEN/HIGH/LOW = 0 (close and settlement are still
  computed). Ingestion keeps the file as reported; masking a zero price is the feature layer's job.
- **Options' `VALUE` is notional, not premium turnover** (measured on three AARTIIND rows: value ≈ 1.04-1.07 × strike ×
  quantity, and nowhere near premium × quantity). Do not read it as money spent on premium.
- **Weekend sessions exist and must be fetched.** Budget-day Saturdays (2015-02-28, 2020-02-01, 2025-02-01), special
  Saturday sessions (2024-01-20, 2024-03-02, 2024-05-18) and weekend Muhurat sessions (2009-10-17, 2016-10-30,
  2019-10-27, 2023-11-12) are all in the F&O archive (HTTP 200). A weekday-only crawl, which is what Stage 3 used, misses
  every one, and that breaks open-interest continuity: the next session's `CHG_IN_OI` is measured against the weekend
  session, so `OI(t) - OI(t-1)` from consecutive files no longer equals it (measured: 62.8% agreement on 2025-02-03, the
  Monday after the Budget Saturday, against 99.9%+ on ordinary pairs). Both F&O ingesters take `--weekends`.
  **Ingested 2026-09-24: 30 weekend F&O sessions (24 Saturdays, 6 Sundays)**, not just the 10 known: the earliest is
  2003-03-22, and 20 were previously unknown (2003-2006, 2010, 2012, 2013, 2014-03-22, 2020-11-14, 2026-02-01, ...).
  With them present, every one reconciles with both neighbouring sessions (60 pairs, 1,495,375 continuing contracts,
  99.9672% exact) and the UDiFF-era identity is 100.0000%. The equity bhavcopy and delivery crawls (Stage 3) were also weekdays-only, so they hold no weekend sessions either,
  apart from delivery's 2019-10-27, which is there only because a stale re-serve put that day's rows in the 2019-10-28
  file. Whether NSE serves the weekend equity files was not probed; it should be, since price and delivery windows
  span these sessions the same way.
- **Archive gaps.** 19 equity trading days have no F&O file: 2000-07-03..06, 2000-08-01..04, 2000-08-07..09,
  2000-09-04..08, 2000-10-09, 2013-10-09, 2021-03-30. The archive answers 404 on every reachable URL form, so they are
  gaps in NSE's archive, not ingestion failures; the cause was not established.
- **Source-file damage** (all pre-2013, handled and recorded per day in the manifest's `format_era`): a header line
  repeated mid-file; several records fused onto one line (a lost newline; the trailing comma is the separator);
  a whole day's data pasted 2x-6x with every copy identical; contracts left in the file after they expired (2002-09-27
  lists the 707 that expired on 09-26); and a trailing `*Updated` line (2012-05-14). One day, **2003-10-01**, is not
  recoverable without guessing: 57 far-month futures appear twice, once as an all-zero placeholder and once priced, and
  the copies disagree. It is left out.
- **NSE's own OI accounting holds:** `CHG_IN_OI(t) == OPEN_INT(t) - OPEN_INT(t-1)` for continuing contracts on
  consecutive sessions (weekend sessions included; verified 2026-09-24): 99.69% (2000-09), 99.97% (2010-19), 100.0000%
  (2020-24 and UDiFF), and 100% across the legacy→UDiFF cutover (33,930 contracts). The residue is source-data events
  (mergers such as RELPETRO/RIL adjust OI), not parsing.
- **The newest F&O day on disk is 2026-09-23.** The 2026-09-24 request was made before that day's file was published, so
  its manifest record says `holiday` (a provisional 404). Nothing reads that as final: a resume re-requests every date
  that has no file. Re-run `python data/ingest_fno_bhavcopy.py --days 3` after the evening publication.
- **Size:** ~0.7-0.9 MB of Parquet per recent day (~34,000 contracts), well under that before ~2010.

### F&O ban list (securities in the F&O ban period)

`https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{ddmmyyyy}.csv`

- **First available: 1 Jan 2008** (2007-12-31 → 404, 2008-01-01 → 200).
- **Only `/archives/fo/sec_ban/` works.** `/content/fo/fo_secban_*` is a 404 for every date tried.
- Files are 50-115 bytes of text. Header shapes seen: `Securities in Ban For Trade Date 11-JUN-2020:` followed by
  `1,BHEL` lines; `... 03-JUN-2024: NIL`; `Sr. No.,Securities in Ban For Trade Date 01-Jan-2008`; and, in 2008-2009,
  full month names (`02-June-2008`, `24-September-2008`) with a separator of `:`, ` :`, `,` (symbols on the next line),
  or `,Nil` on the same line, and `Nil`/`NIL` in either case. From 2012-09 some files also end each symbol line with a
  comma (`1,HDIL,`).
- **`NIL` is information.** It means the list was published and empty; a 404 means no file. Stored as one row with a null
  symbol (see `data/ingest_fno_ban.py`).
- Whether a date's list is knowable before that day's session (it is dated the trade date it applies to) is a
  point-in-time question for Stage 11/12, not settled here.
- **Padding.** From 2012 some files pad every line with empty CSV fields (`1,INDIACEM,,,`, header `...23-DEC-2015:,,,,`).
  Checked against raw bytes (2012-08-29, 09-21, 09-27, 2015-12-23): the padding is empty fields, nothing is hidden.
- **Ingested 2026-09-24:** 4,639 lists (4,618 weekday + 21 weekend), 2008-01-01 to 2026-09-24; 1,339 NIL days and 12,108
  (day, symbol) bans across 197 symbols. The 21 weekend lists are the 20 weekend F&O sessions since 2008 plus one that is
  not a session (below).
- **Days with NO usable list (real sessions; treat as "unavailable", NEVER as "nobody banned"):**
  - **2008-11-28**: the archive answers 404. The ban archive is off by one around the 26-Nov-2008 Mumbai attacks: NSE
    was shut on 2008-11-27 (equity and F&O bhavcopy both 404) yet the archive holds a NIL list for that closed day.
  - **2014-02-19**: NSE's URL serves a byte-identical copy of the 2014-03-19 list (header `19-Mar-2014`).
  - **2014-08-25**: NSE's URL serves the 2014-08-26 list (header `26-AUG-2014`).
  The header-date check is what catches the last two; it must never be loosened.
- **Lists that exist for a day with no session (harmless; key bans by session date, not by file date):** 2008-11-27
  (NIL, see above) and **2014-02-22**, a Saturday list naming AUROPHARMA that is consistent with its neighbours (banned
  2014-02-20, 21 and 24) but matches no equity or F&O session. It is the only weekend list that is not a session.
- **F&O archive-gap days that do have a ban list:** 2013-10-09 and 2021-03-30. The lists exist, which corroborates that
  these were real sessions whose F&O bhavcopy is simply absent from NSE's archive.
- The ban list for a day is published before that day's bhavcopy: the 2026-09-24 list existed while the F&O file for
  2026-09-24 did not.

---

## 11. Known gaps in earlier stages (found during Stage 10; not fixed here)

- **Bhavcopy 2020-07-13 is missing.** A genuine trading day. The file's dates use a two-digit year (`13-Jul-20`), so
  Stage 3's `%d-%b-%Y` parse rejected all 2,001 rows ("2001 rows failed date parsing"). The data is fine; the parser is
  not. The OHLCV panel, adjusted prices, indicators and delivery features have no row for that day, so every rolling
  window that spans it covers one more real trading day than its row count. Fix: accept the two-digit form (as
  `ingest_fno_bhavcopy._legacy_dates` now does), re-ingest that one day, then regenerate Stages 4, 5, 9 and 10.
  Heading above says "CONFIRMED COMPLETE"; that is true of every day except this one.
- **Delivery gaps.** Of the 84 days the delivery ingester recorded as failed, 82 are NSE re-serving the previous session's
  file on a holiday (correctly rejected). Two are real trading days with data lost: **2022-08-08** (CSV tokenizing
  error) and **2024-02-19** (one row with DELIV_QTY > VOLUME rejected the whole ~700-stock file; fix by rejecting the
  row, not the file). The delivery feature block flags both dates as missing.
- **Five delivery files hold another day's rows** (NSE stale re-serves, all pre-2019-11): the delivery loader dates every
  row by the date inside the file, never the filename (`tests/invariants/test_delivery_row_date.py`).
- **Delivery `DELIV_PER` contradicts its own `DELIV_QTY / VOLUME`** on five dates (2020-04-13, 2020-07-13, 2023-09-04,
  2025-10-21, 2026-09-11; 4,191 EQ rows, always upward). The delivery feature block treats those observations as missing.

---

## 12. FII/DII flows — measured (Stage 10, probed 2026-09-24)

Two blocks, each with its own shared missing flag, because their depth differs by 27 years:
`fii_flows.parquet` (NSDL, 1999 →) and `dii_flows.parquet` (NSE forward capture, from 2026-09). Both in
`data/processed/features/flows/`, one row per trading session, written by `features/blocks/flows.py`.

### FII — NSDL "Archive (Trends in FPI/FII Investments)" (`data/ingest_nsdl_fpi.py`)

- **Depth: daily from 1999-01-01** (the 1998-12-31 row is a cumulative "since inception" seed, not a day). 333 monthly
  files in `data/raw/nsdl_fpi/`. NSE's own FII/DII endpoint serves only the latest day, so NSDL is the deep source.
- **The page is an ASP.NET postback** (ViewState + `txtDate`/`hdnDate`, `__EVENTTARGET=btnSubmit1`). The query date is an
  UPPER bound: it returns month-start through that date. A page also holds a second (FPI F&O) table with its own date
  cells: never read dates from the whole page (1-Apr-2025 is in the derivatives table, not the cash table).
- **Three column eras:** no route column (1999 → ~2010: Equity/Debt rows only); routes Stock Exchange / Primary market &
  others / Sub-total / Total; then more debt categories (General Limit, VRR, FAR, Hybrid, AIFs, Mutual Funds). Equity total =
  the no-route row or the Sub-total row. Older pages are malformed HTML with the derivatives table nested inside.
- **Precision:** one decimal through 2014-06-02, two after. Row identity purchases − sales = net holds within 0.10 (one
  decimal) / 0.01 (two decimal); Total rows carry two rounded sub-totals (≤ 0.20).
- **Vintage:** the archive serves current values only, so history is the latest vintage (revisions are invisible). The two
  newest months are re-fetched on every run.
- **Publication: placement is conservative; the true time is not yet pinned.** NSDL states its confirmed figures come from
  custodian reports "on and up to the previous trading day(s)". Measured on 2026-09-24: at 15:31 IST the page's cash table
  held 23-Sep as its newest day, and by 20:38 IST it also held **24-Sep**, the same evening: NSDL publishes day t's cash
  figure on the evening of t, at some time not yet bracketed (this machine lost network 16:08-19:26 IST, so the monitor's
  `first_seen` of 19:26 is a censored interval, not a time). The block still places each figure on the first session
  strictly after t (assumed public by 10:00 IST, `FPI_PUBLICATION_TIME_IST`), which is safe against leakage but one session
  later than the evidence allows, and it is kept because (a) the instant is not shown to be before 19:00 and (b) the archive
  serves the current vintage only, so an evening-of-t figure that NSDL later revises cannot be reconstructed.
  **Correction:** the first monitor read every date on the NSDL page, derivatives table included, so its NSDL rows
  (`nsdl_latest`) describe the wrong table and are void; `nsdl_cash` (cash table only) replaced it.
- **Known, dated gap: `nsdl_cash` publication-timing evidence for target 2026-09-25 (confirmed 2026-09-30, superseding
  the "running for 2026-09-25" note above, which is now stale).** `data/publication_monitor.py` is not a daemon: `run()`
  polls every `--interval` seconds (default 180) until every source is seen or a hard deadline (`--until-next-day-hour`,
  default 14:00 IST the day AFTER the target day) passes, then appends one evidence line per source and exits — it is
  meant to be invoked once per evening, per target date. The `nsdl_cash` poll for target 2026-09-25 is recorded in
  `data/raw/_publication/first_seen.jsonl` (one line, `first_seen_ist: null`, `last_absent_ist: "2026-09-24T21:17:53+05:30"`,
  `last_error: "ConnectionError: ... www.fpi.nsdl.co.in ..."`): the process died from this machine's known network-outage
  pattern shortly after 21:17 IST on 2026-09-24 and never polled again; the file itself was not flushed/finalized until
  2026-09-26 20:00:31 (≈46.5 h later — the same "offline ~46 h" incident this section already documented, not a new one).
  Confirmed 2026-09-30: no `publication_monitor.py` process is running, and nothing — no Windows Scheduled Task, no
  Claude Code cron job — is configured to run or restart it; it simply died once and nothing has re-invoked it since.
  **This cannot be backfilled**, by the tool's own design: it measures live, in real time, when a source FIRST shows a
  target day's data, bracketed against its last poll that did not; re-running it for target 2026-09-25 today would only
  prove the figure is visible as of 2026-09-30, which says nothing about whether it was visible before the 2026-09-26
  14:00 IST deadline that already passed. A valid fresh measurement requires a NEW target date and a live run of up to
  ~24 h (polling until the next day's 14:00 IST cutoff) — not attempted 2026-09-30 (out of scope for that session; a
  monitor needs to run across the live event, not be launched speculatively). Until a future date's `nsdl_cash` evidence
  exists and resolves cleanly, `scripts/verify_stage10_flows.py`'s publication-timing check reports FAIL (not the
  softer PENDING it reported before this record existed) — this is the verifier correctly refusing to pass an unproven
  timing claim, not a code regression; every other flows check, including all 9 injected-bug regression tests, passes.
  **Separately, and confirmed on disk the same day:** the FII data itself (not just this timing evidence) is also only
  current through 2026-09-24 — the newest `data/raw/nsdl_fpi/2026-09.parquet` reporting row with a real Equity figure
  is 2026-09-24, and `fii_flows.parquet`'s last row is 2026-09-24. This is NOT a new or flows-specific stall: it is
  exactly in sync with `data/trading_calendar.py`'s global session calendar (the union of equity + F&O bhavcopy file
  dates), which is itself capped at 2026-09-24 because Stage 3's equity bhavcopy — and therefore Stage 4's adjusted
  prices, which every OTHER Stage 10 sub-block is row-aligned to — has not been re-ingested since **2026-09-04**; only
  the F&O bhavcopy (ingested through 2026-09-24) carries the union calendar that far. Relative to the calendar
  `fii_flows.parquet` is actually built against, it is fully current: every session through 2026-09-24 has a value or a
  correctly-flagged absence, none unexpectedly missing. Fixing this is a Stage 3 equity-bhavcopy re-ingestion, not a
  flows-block or monitor task, and was not attempted 2026-09-30 (out of scope for that session).
- **Sessions with no figure: 144 of 6,896 since 1999, all real.** `data/audit_nsdl_absent_sessions.py` re-fetches the month
  page for each and reads the cash table's date cells independently of the parser: 143 are absent from NSDL's own table
  (NSDL did not report them: 25 are Saturday/Sunday special sessions such as the Budget days 1-Feb-2025 and 1-Feb-2026,
  the rest are weekdays such as the Muhurat session 21-Oct-2025 and 1-Apr in 2025 and 2026; the reason NSDL skips each
  weekday was not established); the 144th, **2000-02-28**, is listed twice with conflicting figures and is quarantined,
  not guessed. Six NSDL reporting days are not NSE sessions. The daily value is NaN with `is_missing_fii_flows` True on
  those rows.
- **Rolling sums are over the last 5 / 20 REPORTED days,** not sessions. Counting the 144 no-report sessions as NaN slots
  blanked 47% of the 20-day sums since 2010; nothing is filled, a window is NaN only until k reported days exist.
- **Independent checks that agree:** NSDL's monthly totals (267 months 2002-01 → 2023-12 plus three later; mean gap 0.8,
  max 3.8 crore); NSDL's quarterly totals (`data/crosscheck_nsdl_quarterly.py`, 2024Q1 → now; 0.3–14.9 crore on quarters of
  40,000–140,000 crore, and 0.6 for the quarter in progress, consistent with the quarterly table being compiled from
  figures revised later). **NSDL's own financial-year table is the outlier for FY2009-10:** it says 30,877 crore while its
  monthly reports sum to 110,226 and my daily figures to 110,220.6; every other year agrees within 1.3 crore. The daily
  and monthly figures agree with each other, so the fault is in NSDL's FY table. `Yearwise.aspx` is blocked by NSDL's WAF.
- **One real zero:** 2003-05-09 (purchases 196.4, sales 196.5, NSDL prints net 0.0). Kept as 0.0; it is a source value.

### DII — NSE `fiidiiTradeReact` (`data/ingest_fii_dii.py`), forward capture only

- The endpoint serves ONLY the latest trading day and no history exists on it, so the DII series starts on the first day
  captured and can never be back-filled. Each capture stores `CAPTURED_AT_IST`; that timestamp, not the reporting date,
  defines when the value became public (the earliest capture of a day wins; a value with no timestamp is never used).
- **Captured so far: 2026-09-21** (file written before the timestamp column existed; the file's modification time,
  2026-09-22 07:38 IST, is used, a conservative upper bound on when it was public) **and 2026-09-24** (captured 19:34 IST,
  after that session's 19:00 run, so it lands on the next session). **2026-09-22 and 2026-09-23 were never captured and are
  lost.** `dii_flows.parquet` therefore has 1 populated row out of 7,919 sessions; treat DII as unavailable for modelling
  until months of captures accumulate.
- There is no scheduled capture: a day missed is a day lost. A Windows scheduled task running `python data/ingest_fii_dii.py`
  each weekday evening would fix that; none has been created.
- `fiidiiTradeReact` (combined NSE+BSE+MSEI) and `fiidiiTradeNSE` (NSE only) report different numbers; only Combined is used.

