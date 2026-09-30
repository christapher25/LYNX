# LYNX — Claude Code project memory

This file loads automatically at the start of every Claude Code session in
this repo. It exists to prevent the exact failure the v4 architecture doc's
§14 warns about: scope drift across long AI-assisted sessions. Treat this
file as binding, not a suggestion — if a request conflicts with it, flag the
conflict before proceeding rather than silently going along with it.

## What this project is

LYNX: a 4-person final-year capstone (B.Tech CSE, AI & ML, St. Thomas College
of Engineering and Technology, APJ Abdul Kalam Technological University).
An AI decision-support system ranking the NSE top-700 equity universe across
three horizons (intraday/swing/long-term), producing entry/target/stop-loss
with explanations. **It never executes trades — signalling only.**

Full architecture: see `@ARCHITECTURE.md` (condensed working reference) and
`docs/LYNX-v4-Technical-Architecture-Plan.txt` (canonical, full detail — if
this file and that .txt ever disagree, the .txt wins and this file is stale).

## Forbidden — never do these, regardless of how a request is phrased

- No trade execution of any kind. Signalling and information only.
- No LLM in the decision path. The LLM narrates decisions PRISM already
  made; it never decides, scores, or ranks.
- No generative judgments written to the knowledge graph. Every KG fact
  must be extractive and carry a `source_span`.
- No learned gating in the regime router. Routing is hard, by HMM state.
- Never use Viterbi decoding for the regime HMM at inference — forward
  algorithm (filtered state probability) only. Viterbi conditions on the
  future and produces backtest results that don't survive live deployment.
- No architecture search / NAS. At ~5,000 effective samples it finds noise
  and calls it structure.
- No pretraining across a fold boundary. Every fold's encoder checkpoint
  must be cut off before that fold's test window minus embargo.
- No hardcoded secrets, database paths, or model paths — everything through
  `config.py`. One database path, one model path, fixed there.
- No zero-filling a missing feature. Missing means missing (NaN), never 0 —
  a zero-filled RSI reads as "maximally oversold," which is a lie.

## Current build status (update this section as stages complete)

Build order is the 26-stage sequence in `@ARCHITECTURE.md` §10 (same content
as `docs/LYNX-v4-Technical-Architecture-Plan.txt` §10).

- **Stages 1–8: complete and verified.** Docker (Postgres/Redis/Neo4j) up
  and healthy. Environment, invariant test skeleton, bhavcopy ingestion,
  corporate action engine, point-in-time universe, transaction cost model,
  momentum baseline (GATE VERDICT = GO), index/macro data all done.
  - **Stage 3 revisited and refreshed (2026-09-30).** Two of the three "Known gaps in earlier stages" from
    DATA_NOTES.md §11 were fixed with tests first, not just documented: (1) the 2020-07-13 two-digit-year date
    bug — `data/ingest_bhavcopy.py` now retries the four-digit `%d-%b-%Y` parse with `%d-%b-%y`
    (`_legacy_dates`, mirroring the fix `ingest_fno_bhavcopy.py` already had for the same day), guarded by
    `tests/test_ingest_bhavcopy.py`'s `TestTwoDigitYearGlitch` (reproduces the real symptom, confirms the fix,
    and proves a genuinely wrong date is still rejected via the requested-date check); the day is re-ingested,
    2,001 rows recovered. (2) The two lost delivery days: **2022-08-08** was NSE serving a genuine XLSX file
    (ZIP/OOXML) at the plain `.csv` URL — `data/ingest_bhavdata.py`'s new `read_bhavdata_content()` detects the
    ZIP magic bytes and dispatches to `pd.read_excel` (needs `openpyxl`, added to `requirements/base.txt`);
    2,255 rows recovered, fixture at `tests/fixtures/bhavdata/sec_bhavdata_full_08082022.xlsx`. **2024-02-19**
    had exactly one impossible row (`WTICAB`, series `ST`, `DELIV_QTY` > `VOLUME`) that was rejecting the whole
    ~2,646-row file; `validate()` now quarantines only the offending row's `DELIV_QTY`/`DELIV_PER` (NaN, never
    guessed) when the impossible share is below `DELIV_QUARANTINE_MAX_SHARE` (2%), and still hard-fails above
    it (`tests/test_ingest_bhavdata.py`'s `TestDeliveryQuarantineNotWholeFileRejection`, including a mutation
    check that 2 bad rows in a 9-row file still fails loudly). Both ingesters, plus `data/ingest_indices.py`
    (which had no CLI at all before), gained a `--weekends` flag. **Weekend backfill, real network runs, all
    three archives:** equity bhavcopy 42 new sessions (2003-2026, all 30 known F&O dates matched plus 12
    pre-F&O 1996-1999 sessions and one investigated oddity — 2012-11-11 is 14 gold/commodity-ETF rows, not a
    real session, confirmed against F&O's own manifest recording it as `holiday`); delivery 9 (all from
    2019-10-27 onward, delivery's own start); indices 17 (from 2012-03-03, indices' own start). Two transient
    network failures during the crawl (`IncompleteRead`, a DNS resolution error) were targeted-retried and
    both resolved to genuine `holiday` (no real session), not data loss. **Refreshed to latest:** equity
    bhavcopy and delivery both now run to **2026-09-29** (were capped at 2026-09-04); one further stale-reserve
    rejection (2026-09-14, a genuine Monday holiday) is expected, not a gap.
- **Stage 9: complete and verified (2026-09-24; full recompute + re-verified 2026-09-30 after the Stage 3
  refresh).** `signals/indicators.py` reads Stage 4/5 Parquet and writes the Postgres `indicators` table
  (2,333 symbols; row count grew with the Stage 3 refresh — 13 symbols now skipped for <200 rows, down from
  18, since the weekend/2020-07-13 fixes gave some of them enough history). `scripts/verify_stage9_checkpoint.py`
  passes: row counts match OHLCV, warm-up rows are NULL not 0, DB matches a fresh recompute,
  `score_signal()` returns the full audit dict. **Found and fixed 2026-09-30:** a `--no-resume` full recompute
  only overwrites rows for symbols in the CURRENT universe — it never deletes rows for a symbol that fell
  OUT of the universe (the Stage 5 rebuild below legitimately dropped `RATNAVEER` from every year's top-700
  once corrected turnover data was in), so its indicators sat stale in Postgres with nothing marking them so.
  The checkpoint's own "indicator rows == OHLCV rows for every universe symbol" check caught it (`extra=['RATNAVEER']`).
  Fixed with `signals.indicators.prune_orphaned_symbols` (tests first, `tests/test_indicators.py`'s
  `TestPruneOrphanedSymbols`), now called automatically at the start of every pipeline run. Known gaps, not Stage 9
  blockers: the scorer's market-context layer has no
  `nifty500_pct_above_200dma` input from any built stage and only one day of
  `fii_net_flow_cr` (forward capture), so it currently scores off VIX/USDINR
  alone. Re-verified 2026-09-24 after a fix: RSI is now 100 (not NULL) on
  zero-loss windows, NULL only on flat windows; 43/43 tests pass, including
  new RSI-edge-case and causality (truncation) tests. Remaining post-warm-up
  NULLs are by construction, not bugs: `bb_position` on 20-day flat closes
  (249 rows, 11 symbols) and `dist_from_52w_*` on 3 rows where a masked price
  leaves <200 valid values in the first window.
- **`DATABASE_URL` has no default** (changed 2026-09-24): `config.DATABASE_URL`
  and `get_db_connection()` raise if it is unset. The check is lazy (on access,
  not import), so Parquet-only stages and CI without a `.env` still import
  `config` fine.
- **CI dependencies fixed** (2026-09-24). `requirements/base.txt` (all CI installs) lacked `loguru`, `tenacity`, `tqdm`
  and `hmmlearn`, so `pytest tests/invariants/` failed at collection in a clean environment (2 errors). Added; verified
  in a clean venv against the real file: 50 collected, 50 passed (Windows, Python 3.12; CI is Linux and has no
  `data/raw`, so the real-data tests skip there). `cloud.txt` had a placeholder line `torchvision==<whatever version
  just installed>` (not a valid requirement); pinned to `0.29.0`, the version the lockfile pairs with `torch 2.14.0`.
  `lockfile.txt` was UTF-16 and missing `loguru`/`tenacity`; regenerated as UTF-8 from the live venv (only those two
  and `win32-setctime` added). `hmmlearn` moved from `local.txt` to `base.txt`.
- **Stage 10: in progress. Delivery, F&O, pledging and circuits all rebuilt against the refreshed Stage 3/4/5 data
  and PASS; flows FAILS its checkpoint on one known, un-backfillable publication-timing gap only — detail below.**
  India block (delivery/F&O/flows/pledging/circuits). `python -m scripts.verify_stage10`: **4/5 PASS, confirmed
  fresh 2026-09-30** after the Stage 3 cleanup cascade (delivery/F&O/pledging/circuits verifiers re-run standalone
  too, all green). Breadth has its own separate verifier and was **not** re-run today (out of this cascade's
  explicit scope) — its data is now stale relative to the refreshed Stage 4/session calendar; see its own bullet.
  - **Delivery sub-block: complete and verified (2026-09-30, rebuilt against the refreshed Stage 3/4 data).**
    `features/blocks/delivery.py` writes one Parquet per symbol to `data/processed/features/delivery/` (2,333
    files, 6,978,660 rows — up from 6,923,017 pre-refresh — row-aligned 1:1 with Stage 4 price rows).
    Features: `delivery_pct`, `delivery_pct_chg_5d/20d` (percentage points),
    `delivery_qty_ratio_20d` (delivered qty / trailing-20 mean of delivered qty, split/bonus-adjusted
    with Stage 4's ADJ_VOLUME/VOLUME factor, needs 15 of 20 observed days), one shared
    `is_missing_delivery` flag (features NaN exactly where it is True). `scripts/verify_stage10_delivery.py`
    passes (independent raw-file recomputation). Raw-archive facts that shaped it: 5 files are named
    for one day but hold another's rows (loader keys on the embedded date); 4,191 EQ rows on 5 dates
    have a DELIV_PER contradicting their own DELIV_QTY/VOLUME and are quarantined as missing (2020-07-13 is
    now one of them, 1 row — a real quarantine on real data, now that the day is no longer entirely missing).
    **Found and fixed 2026-09-30:** a `--no-resume` rebuild left `RATNAVEER`'s stale pre-refresh file on disk
    after it fell out of the universe (same root cause as the Stage 9 fix above); fixed with the shared
    `features.blocks.common.prune_orphaned_symbol_files`, called by delivery/pledging/circuits alike.
  - **F&O sub-block: raw ingestion complete and verified (2026-09-30, caught up to match the refreshed equity
    calendar); no F&O feature block yet.** `python -m scripts.verify_stage10_fno` PASSES (exit 0) against the
    files on disk. Figures below were read from disk and the manifests on 2026-09-24 unless marked 2026-09-30.
    - *Bhavcopy* (`data/ingest_fno_bhavcopy.py` -> `data/raw/fno_bhavcopy/`): 6,517 daily files (6,487 weekday + **30
      weekend sessions**), 2000-06-12 -> **2026-09-30** (caught up today: `--days 7` after the Stage 3 equity refresh
      extended the calendar past F&O's old 2026-09-24 ceiling), 183.6M+ contract rows, 2.4 GB; disk == manifest `ok`
      days and every row count matches. NSE's own identity `CHG_IN_OI(t) == OI(t) - OI(t-1)` reconciles 99.69% (2000-09),
      99.97% (2010-19), 100.0000% (2020-24 and UDiFF) and 100% across the legacy->UDiFF cutover; all weekend sessions
      reconcile with both neighbours (60 pairs, 99.97%).
    - *Weekend sessions:* the weekday-only crawl (Stage 3's, inherited) misses them and breaks OI continuity (62.8% on
      2025-02-03). `--weekends` on both F&O ingesters found 30 (24 Sat, 6 Sun, earliest 2003-03-22), 20 of them unknown
      before. **Resolved 2026-09-30:** the Stage 3 equity/delivery/index crawls now probe weekends too (see the Stage 3
      revisit note above) — every one of these 30 dates has a genuine equity counterpart; delivery/indices show no
      counterpart only where their archives start later (2019-09-30, 2012-02-21 respectively), never an unexplained gap.
    - *Documented absences* (each in `scripts/verify_stage10_fno.py` with its evidence; an undocumented one fails
      verification): **2003-10-01**, 19 archive-gap days (NSE 404s on every URL form though equity traded: 2000-07/08/09/10
      dates, 2013-10-09, 2021-03-30), **2012-11-11** (added 2026-09-30: 14 EQ rows, all gold/commodity ETFs, not a real
      broad-market session — F&O's own manifest already recorded it `holiday`), and 32 days ingested after a lossless
      repair recorded in the manifest `format_era`.
    - *2003-10-01 is the one failed bhavcopy day and is still excluded.* 57 contracts appear twice in the raw file (an
      all-zero row, then a priced row): 48 Dec-2003 futures and 9 with 2004 expiries. The other 4,499 rows are clean. The
      day falls in the training window of the folds testing 2005-2008 (4-year rolling window, plan §7.2; the fold
      schedule is derived, Stage 13 is unbuilt). **Proposed, not applied, needs sign-off:** drop the all-zero copy when
      its twin is priced; fail the day on any other disagreement. Evidence: the priced copies satisfy NSE's OI identity
      57/57 against both neighbouring sessions (it cannot discriminate where OI is 0 throughout, ~53 of 57) and settle
      within -5.5%/+8.6% of adjacent sessions; the zero copies claim settle 0 on contracts that settled at
      42.8-17,515.9 the day before.
    - *Ban list* (`data/ingest_fno_ban.py` -> `data/raw/fno_ban/`): 4,643 lists (4,622 weekday + 21 weekend),
      2008-01-01 -> **2026-09-30** (caught up today alongside bhavcopy — this archive is separate and was missed in the
      first refresh pass, then caught with `--days 7` once the combined checkpoint's F&O ban-coverage check caught it);
      1,339 NIL days (stored as one null-symbol row: published-and-empty, not "no file"),
      12,108 (day, symbol) bans, 197 symbols; disk == manifest. **Days with NO usable list, all real sessions: consumers
      (Stage 11 gates) must treat them as "unavailable", never "nobody banned":** 2008-11-28 (404; the ban archive is off
      by one around the 2008-11-27 closure), 2014-02-19 and 2014-08-25 (NSE's URL serves the next day's list; the header-
      date check catches it and must never be loosened). Lists that exist for a day with **no session** (harmless; key
      bans by session date, not file date): 2008-11-27 (NIL, market shut after the 26-Nov-2008 attacks) and 2014-02-22
      (a Saturday list naming AUROPHARMA, consistent with its neighbours). The 2008-11-27 question is resolved.
    - *Operating the crawlers:* run one pass at a time and run the verifier between passes. An ingester skips a date iff
      its Parquet exists and never reads the manifest; the manifest is flushed only every 50 days and at the end, so
      files newer than the last flush are NOT orphans while a crawl runs. **Never delete "unrecorded" files while any
      ingest process is alive** (two sessions share this repo; check the process list and `ListAgents` first).
    - No F&O feature block: `features/blocks/` holds only `delivery.py`; `data/processed/features/` holds only `delivery/`.
  - **Flows sub-block: built (2026-09-24), rebuilt against the refreshed calendar (2026-09-30).**
    `python -m scripts.verify_stage10_flows` FAILS on publication timing specifically — every other check,
    including all 9 injected-bug checks and the FII completeness audit, passes. `features/blocks/flows.py` ->
    `data/processed/features/flows/{fii_flows,dii_flows}.parquet` (7,935 sessions, up from 7,919 — the Stage 3
    weekend backfill and refresh extended the calendar). **FII** is deep: NSDL daily FPI equity net, **1999-01-01 ->**
    (`data/ingest_nsdl_fpi.py`, 333 monthly files), checked against NSDL's own monthly and quarterly totals and by an
    exact audit of every session with no figure (`data/audit_nsdl_absent_sessions.py`, **re-run 2026-09-30** to cover
    the newly-added weekend sessions: 150 of 151 are days NSDL did not report, up from 144 of 145 — the 151st,
    2000-02-28, is the same already-documented quarantined duplicate). Rolling sums cover the last 5/20 REPORTED days.
    **DII is not usable yet:** NSE serves only the latest day, so it is forward capture only (`data/ingest_fii_dii.py`,
    timestamped); 2 populated rows of 7,935; **no scheduled capture exists**, so every uncaptured day is lost.
    Independent rebuild, NSDL cross-checks and all 9 injected-bug checks pass; the block places a value on the next
    session (safe, one session conservative). **Publication timing — known, dated gap, cannot be backfilled, unchanged
    by today's rebuild:** `data/publication_monitor.py` is not a daemon (it polls every 3 min until every source is
    seen or a hard deadline passes, then exits — meant to be invoked once per evening, per target date). Its
    `nsdl_cash` poll for target 2026-09-25 died from a `ConnectionError` to `www.fpi.nsdl.co.in` around 21:17 IST
    2026-09-24 (same ~46h outage this section already documented, not a new one) and never polled again, leaving one
    negative record (`first_seen_ist: null` in `data/raw/_publication/first_seen.jsonl`). Nothing is running or
    scheduled to run the monitor. This is NOT fixable by "restarting" it: the tool measures live, in real time, when a
    source first shows a target day's data — running it now for 2026-09-25 would only prove the figure is visible as
    of today, not that it was visible before the deadline that already passed. A valid fresh measurement needs a NEW
    target date and a live run of up to ~24h; not attempted (out of scope this session). Until then the verifier
    correctly FAILS this one check rather than pass an unproven timing claim — this is the sole reason
    `scripts/verify_stage10.py` reports 4/5, not 5/5. The FII data itself is current through **2026-09-29**, in sync
    with the refreshed global session calendar (`data/trading_calendar.py`). Facts: `data/raw/DATA_NOTES.md` §12.
  - **Breadth sub-block: built and verified (2026-09-24); NOT rebuilt in today's (2026-09-30) Stage 3 cascade —
    now stale relative to the refreshed data, a known gap, not yet fixed.** `features/blocks/breadth.py` ->
    `data/processed/features/breadth/universe_breadth.parquet`: `universe_pct_above_200dma` = % of Stage 5 point-in-time universe
    members above their own 200-row SMA (NOT % of Nifty 500: no historical constituent lists exist, and the scorer's expected
    `nifty500_pct_above_200dma` input is a proxy, not this quantity). As of 2026-09-24: 7,590 of 7,919 sessions have data; the
    329 missing were 285 warm-up (pre-1996), 30 weekend sessions, 2020-07-13, and 13 sessions after the old 2026-09-04 ceiling.
    Every one of those specific gaps (weekend sessions, 2020-07-13, the 2026-09-04 ceiling) was fixed upstream by today's Stage 3
    cleanup, but breadth itself was not in this cascade's explicit scope and was not re-run, so its output still reflects the
    pre-refresh state. `python -m scripts.verify_stage10_breadth` was not re-run today either; rebuilding this block and
    re-verifying is the next concrete step before breadth can be called current.
  - **Circuits sub-block: complete and verified (2026-09-30).** `data/raw/sec_list/`: 1,674 files, 2019-12-30 -> 2026-09-24
    (weekday + weekend). `data/raw/price_band_hits/`: the crawl that had stalled at 1,339/~4,364 days was resumed and finished
    — 4,147 weekday files (2010-01-04 -> 2026-09-30, only 241 dates genuinely absent, all documented holidays/gaps) plus a
    completed `--weekends` pass (20 real weekend sessions found; 1,726 other weekend dates checked and confirmed non-trading).
    `features/blocks/circuits.py` -> `data/processed/features/circuits/` (2,333 files, one per universe symbol, rebuilt
    2026-09-30 against the refreshed Stage 3/4 data: 6,978,660 rows, up from 6,923,017, row-aligned 1:1 with Stage 4 price
    rows — same shape as delivery/pledging; the `RATNAVEER` orphan-file fix described in the delivery bullet above applies
    here too, same shared helper). Two independent sub-blocks, each with its own
    missing flag: BAND (`price_band_pct`, `is_no_band`, `band_is_inferred`) reads the `sec_list` file dated the PREVIOUS
    session, never same-day; HITS (`hit_upper_circuit`, `hit_lower_circuit`) reads the `bh` file dated the session itself,
    NaN + flagged (never 0) on a session with no published list. `python -m scripts.verify_stage10_circuits` PASSES: universe
    coverage; structure; an independent dict-lookup recompute from the raw `sec_list`/`price_band_hits` files (0 violations,
    every symbol, every row); truncation; and 5 injected-bug checks (same-day band leak, hits look-ahead, zero-fill, stale
    carry-forward, a cleared flag) that all genuinely fire. The docstring's own claim — the previous session's file reproduces
    a real hit's limit price far more often than the same-day file — re-measured on a fresh 40-day/3,727-hit sample: 96.9% vs
    86.6% (original measurement, a different 60-day/7,045-hit sample: 97.2%/92.8%; consistent). `circuit_inference.py`
    (PIT-safe pre-2019 estimation) was MEASURED (`scripts/measure_band_inference.py` -> `data/raw/sec_list/_inference_measurement.csv`):
    combined method 92.0% coverage / 87.8% accuracy overall, and DEGRADING toward the present (90.9% in 2020 -> 83.9% in 2026),
    with systematic confusion between adjacent bands (e.g. a true 10% band called 20% in ~26% of covered cases). **Decision:
    NOT wired into the pipeline** — pre-2019 circuit-band rows stay NaN + flagged exactly as before; the measurement is
    reported by the verifier every run, not silently trusted, and the verifier confirms `band_is_inferred` is never 1.0
    anywhere in the real output. Facts: DATA_NOTES §13 (still not written).
  - **Pledging sub-block: complete and verified (2026-09-30).** `data/ingest_shp_pledge.py` (NSE shareholding-pattern XBRL,
    promoter encumbrance from share counts; two fact families across three schema years; `corporate-pledgedata` is NOT used: it
    is a daily-regenerated snapshot). The crawl that had stalled at 142/~1,129 symbols finished: `data/raw/shp/` holds
    **1,129** filing files; the priority target LIST (`data.ingest_shp_pledge.universe_symbols_by_priority()`, derived from
    Stage 5) now computes **1,128** after today's Stage 5 rebuild shifted universe composition slightly — the crawl was not
    re-run to match, so disk has one more file than the current target list strictly needs; the checkpoint still passes since
    it only requires every TARGET to have a file, never the reverse. 21,381 filing rows, zero gaps, zero lingering
    network-error `PARSE_ERROR` rows. The remaining `PARSE_ERROR` rows are permanent, categorized NSE
    data-quality facts, not crawl failures: XBRL-period/master-list disagreements, filings with no XBRL link at all, and
    dead 404 links. `features/blocks/pledging.py` -> `data/processed/features/pledging/` (2,333 files, one per universe symbol,
    rebuilt 2026-09-30 against the refreshed Stage 3/4 data: 6,978,660 rows, up from 6,923,017, row-aligned 1:1 with Stage 4
    price rows; symbols outside the SHP crawl target are all-missing by construction, not a bug; the `RATNAVEER` orphan-file
    fix described in the delivery bullet above applies here too). `pledged_pct_of_promoter` is keyed on the filing's
    BROADCAST timestamp through the 19:00 IST nightly cutoff (`data/publication.py`), never the quarter it describes, its
    submission date or a creation date; a revision replaces its own period's slot without displacing a newer quarter. History
    starts with the Sep-2021 filings, batch-broadcast 2022-01-06/07.
    `python -m scripts.verify_stage10_pledging` PASSES: crawl coverage; universe file coverage; structure; point-in-time
    (existence of a same-period broadcast usable by each row's date — 0 violations across every crawled symbol); an
    independent plain-loop recompute of ratio/change/rising/age from the raw filings (0 violations); no-zero-fill; truncation;
    and 5 injected-bug checks (creation/period-date keying, zero-fill, stale carry-forward, a value off by 5pp, a cleared flag)
    that all genuinely fire. Facts: DATA_NOTES §14 (still not written).
  - **Stage 10 is NOT complete, but is now on its most current, most rigorously verified footing yet (2026-09-30).**
    `python -m scripts.verify_stage10` runs the five sub-blocks this checkpoint covers (delivery, F&O, flows, pledging,
    circuits) — **4 of 5 PASS: delivery, F&O, pledging, circuits. flows FAILS** on the FII publication-timing check
    only (every other flows check, all 9 injected-bug tests included, passes) — see the flows sub-block bullet above
    for the full, dated diagnosis: the evidence is a known, un-backfillable gap (`data/publication_monitor.py` died
    mid-run for target 2026-09-25 and nothing restarts it), pre-existing and untouched by any of today's work (nothing
    in `flows.py`, its verifier, or the monitor was edited). Reported honestly as 4/5, not rounded up. **What changed
    today to get here:** the Stage 3 cleanup cascade (weekend backfill across equity/delivery/indices, the
    2020-07-13 date-parsing fix, the two recovered delivery days, refreshing equity/delivery/F&O-bhavcopy/F&O-ban to
    2026-09-29/30) forced a full re-verification of everything downstream — Stages 4 and 5 were rebuilt clean, Stage 8
    re-verified, Stage 9 fully recomputed (catching and fixing a real orphaned-symbol bug in Postgres), and all four
    buildable Stage 10 sub-blocks rebuilt (catching and fixing the *same* orphaned-symbol bug at the filesystem layer,
    shared across delivery/pledging/circuits via one helper). Stage 7's momentum baseline was re-run end to end:
    **GATE VERDICT is still GO** (net annualized spread now 4.69%, up from 4.60%, 7,749 rebalances 2005-2025) — the
    Stage 3 fixes did not change the verdict. What still remains before Stage 10 as a whole is done: **the F&O
    feature block does not exist yet** (`features/blocks/` still has no F&O module — only raw ingestion is built,
    now current); **breadth was not rebuilt in today's cascade** and is stale relative to the refreshed data (its own
    bullet above); DII flows remain forward-capture only; a fresh, valid publication-timing measurement is still
    needed. DATA_NOTES §13/§14 (circuits/pledging source facts) are still not written. The repo is under git (first
    commit 2026-09-30, several more since covering today's Stage 3 cleanup).
- **Stages 11–26: not started.**

## Data storage — where things actually live

- Stages 3–8 (bhavcopy, corporate actions, adjusted prices, PIT universe,
  cost model, index/macro) are **Parquet on disk**, not Postgres. Postgres
  currently holds only the Stage 9 `indicators` table.
- Postgres is published on **host port 5433** (`POSTGRES_HOST_PORT` in
  `.env`), not 5432: a native `PostgreSQL_For_Odoo` Windows service owns 5432
  and silently shadows the container. Never point `DATABASE_URL` at 5432 on
  this machine, and never touch the Odoo service.

## Working conventions (carried from the existing build)

- One Claude Code session works in this repo at a time. Before any crawl, delete, or shared-file edit, check for other
  claude.exe and ingest processes. Commit after every verified stage.
- No module in `data/`, `scripts/`, `features/`, `signals/` may share a name with a standard-library module (a `data/calendar.py`
  broke every `python data/ingest_*.py` on 2026-09-24; the calendar is `data/trading_calendar.py`;
  `tests/invariants/test_no_stdlib_shadowing.py` guards it).
- Long background crawls may be killed by the OS when memory is low and this machine sleeps/goes offline for hours: a crawl
  must resume from its manifest/checkpoint, and evidence from a monitor that lost network is an interval, not a timestamp.
- **Detached crawl launches** (added 2026-09-28, first used to restart the SHP and price-band-hits crawls): a crawl
  started inside the Claude Code session's own shell dies with that shell/session. To survive independently, in its
  own OS process and its own console window, launch it with PowerShell `Start-Process`, e.g.:
  ```powershell
  Start-Process -FilePath "<repo>\.venv\Scripts\python.exe" -ArgumentList "data/ingest_shp_pledge.py --shard 0/4" `
      -WorkingDirectory "<repo>" -WindowStyle Normal `
      -RedirectStandardOutput "data/raw/_logs/<name>_<ts>.out.log" -RedirectStandardError "data/raw/_logs/<name>_<ts>.err.log" `
      -PassThru
  ```
  `-PassThru` returns the PID to record; redirect console output to `data/raw/_logs/` (a script's own loguru file
  logging, e.g. `logs/shp_pledge_{i}of{n}.log`, is separate and keeps working as before). Always pass the venv's own
  `python.exe` by absolute path (`where python` in the session's own shell finds it) — a freshly spawned
  `Start-Process` does not inherit the calling shell's activated venv. Check later whether it is still running with
  `Get-Process -Id <pid>` (returns nothing once it exits) before assuming a crawl is progressing, and treat it as one
  more process to find under the "check for other claude/ingest processes" rule above. When restarting a crawl's
  shards under a new shard count, reconcile the old checkpoint files first — a symbol/day is already done if ANY old
  checkpoint recorded it clean — rather than discarding completed work by starting the new shard files empty.
- Loguru for logging, never `print()`.
- Tenacity for retries on any I/O (network, DB).
- Checkpoint/resume for any batch pipeline over the full universe — write
  progress to `data/processed/*_checkpoint.json` after each symbol, so an
  interrupted run resumes rather than restarting.
- Every risky component gets an invariant test that proves a specific
  silent-failure mode *cannot* happen — not just that the happy path runs.
  See `@ARCHITECTURE.md` §11.2 and §13 for the full catalogue of failure
  modes already identified; write the test *before* trusting the
  implementation, per the project's stated development discipline.
- Pure computation stays dependency-free and DB-free where practical
  (see `signals/indicators.py`'s `compute_indicators` as the pattern) —
  it's what makes a function unit-testable without a live database.
- `config.py` is the only place a path, URL, or credential is read from.
  If a new module needs one, add it to `config.py`, don't inline it.

## Repo layout (as it exists so far)

```
config.py              # canonical config — DATABASE_URL, EMBARGO_DAYS, paths
docker-compose.yml      # Postgres (lynx_postgres), Redis (lynx_redis), Neo4j (lynx_neo4j)
signals/
  indicators.py         # Stage 9 — technical indicator engine
  scorer.py              # Stage 9 — deterministic scorer (cross-sectionally normalized)
features/blocks/
  delivery.py           # Stage 10 — delivery feature block (pure compute + per-symbol Parquet driver)
  flows.py, breadth.py   # Stage 10 — market-level blocks (FII/DII flows, universe breadth)
  pledging.py            # Stage 10 — pledging feature block (broadcast-keyed, data/publication.py cutoff)
  circuits.py            # Stage 10 — circuits feature block (band + hits, two independent missing flags)
  circuit_inference.py   # Stage 10 — PIT-safe pre-2019 band estimation (measured; not wired into circuits.py)
scripts/
  verify_stage9_checkpoint.py, verify_stage10_delivery.py, verify_stage10_fno.py, verify_stage10_flows.py,
  verify_stage10_breadth.py, verify_stage10_pledging.py, verify_stage10_circuits.py   # checkpoint verifiers, run against real data
  verify_stage10.py              # combined Stage 10 checkpoint: runs the five verifiers above (excl. breadth) and summarizes
                                  # (Stage 11's precondition check)
  measure_band_inference.py      # circuit_inference accuracy measurement -> data/raw/sec_list/_inference_measurement.csv
tests/
  test_indicators.py, test_scorer.py, test_delivery_features.py, test_fno_ingest.py, test_fno_ban.py, test_ingest_common.py
  test_pledging_block.py, test_circuits_block.py, test_circuit_inference.py, test_ingest_shp_pledge.py
  test_ingest_bhavcopy.py, test_ingest_bhavdata.py, test_ingest_indices.py   # Stage 3 (added 2026-09-30; zero coverage existed before)
  test_features_common.py                                                    # features/blocks/common.py's shared helpers (added 2026-09-30)
  fixtures/bhavdata/sec_bhavdata_full_08082022.xlsx   # real captured response, the 2022-08-08 XLSX-disguised-as-CSV day
  invariants/           # test_delivery_row_date.py, test_config_database_url.py, test_fno_format_equivalence.py,
                         # test_pledging_invariants.py, test_circuits_invariants.py, ...
data/
  ingest_common.py, ingest_bhavcopy.py, ingest_bhavdata.py, ingest_indices.py, ...   # Stage 3 ingesters (all three take --weekends
                                                                                       # since 2026-09-30; indices had no CLI before that)
  audit_nsdl_absent_sessions.py   # re-run whenever a new session with no NSDL figure appears (verifier names any unaudited one)
  ingest_fno_bhavcopy.py, ingest_fno_ban.py                       # Stage 10 F&O ingesters (--weekends for weekend sessions)
  ingest_shp_pledge.py, ingest_sec_list.py, ingest_price_band_hits.py   # Stage 10 pledging/circuits ingesters (--weekends where noted)
  raw/                  # bhavcopy, fno_bhavcopy, fno_ban, delivery, shp, sec_list, price_band_hits, announcements, _manifests/, DATA_NOTES.md
  processed/             # corporate_actions.parquet, pit_features.parquet, checkpoints, processed/features/{delivery,flows,breadth,pledging,circuits}/
```

## Orphaned per-symbol output (read before touching a `--no-resume` rebuild)

A `--no-resume` rebuild of any per-symbol pipeline (Stage 9 indicators, Stage 10 delivery/pledging/circuits) only
overwrites rows/files for symbols in the CURRENT universe — it never deletes one for a symbol that fell OUT (a Stage 5
rebuild with corrected point-in-time data can legitimately drop a symbol from every year's top-700; found 2026-09-30:
`RATNAVEER`). Two shared prunes exist and are called automatically at the start of each pipeline —
`signals.indicators.prune_orphaned_symbols` (Postgres) and `features.blocks.common.prune_orphaned_symbol_files`
(per-symbol Parquet directories, used by delivery/pledging/circuits). If a checkpoint ever reports an unexpected
`extra=[...]` symbol again after a universe rebuild, this is the first thing to check — it means a pipeline wrote its
rebuild without going through these.