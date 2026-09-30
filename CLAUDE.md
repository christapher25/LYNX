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
- **Stage 9: complete and verified (2026-09-24).** `signals/indicators.py`
  reads Stage 4/5 Parquet and writes the Postgres `indicators` table
  (2,333 symbols, 6,920,547 rows; 18 symbols under 200 rows skipped by
  design). `scripts/verify_stage9_checkpoint.py` passes: row counts match
  OHLCV, warm-up rows are NULL not 0, DB matches a fresh recompute,
  `score_signal()` returns the full audit dict. Known gaps, not Stage 9
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
- **Stage 10: in progress. Delivery, pledging and circuits feature blocks complete and verified; F&O raw ingestion
  complete and verified but its feature block is not built; flows built but FAILING its checkpoint (FII
  publication-timing evidence gap, pre-existing and unrelated to today's work — detail below).** India block
  (delivery/F&O/flows/pledging/circuits). Sections marked 2026-09-30 below were checked against disk that day;
  sections still marked 2026-09-24 (delivery, F&O, breadth) were not re-ingested today but delivery and F&O were
  re-verified, unchanged, by the same-day `scripts/verify_stage10_checkpoint.py` run (breadth has its own separate
  verifier and was not re-run today).
  - **Delivery sub-block: complete and verified (2026-09-24).** `features/blocks/delivery.py` writes
    one Parquet per symbol to `data/processed/features/delivery/` (2,333 files, 6,923,017 rows,
    row-aligned 1:1 with Stage 4 price rows; 70.7% flagged missing = pre-2019 history).
    Features: `delivery_pct`, `delivery_pct_chg_5d/20d` (percentage points),
    `delivery_qty_ratio_20d` (delivered qty / trailing-20 mean of delivered qty, split/bonus-adjusted
    with Stage 4's ADJ_VOLUME/VOLUME factor, needs 15 of 20 observed days), one shared
    `is_missing_delivery` flag (features NaN exactly where it is True). `scripts/verify_stage10_delivery.py`
    passes (independent raw-file recomputation; re-run 2026-09-24 11:55, every check passes). Raw-archive facts that shaped it: 5 files are named
    for one day but hold another's rows (loader keys on the embedded date); 4,191 EQ rows on 5 dates
    have a DELIV_PER contradicting their own DELIV_QTY/VOLUME and are quarantined as missing.
  - **Upstream gap found, not fixed:** Stage 3's bhavcopy ingest FAILED for 2020-07-13 ("2001 rows
    failed date parsing"), so the OHLCV panel, indicators and delivery features have no row for that
    day (rolling windows span it). Bhavcopy also ends 2026-09-04 while delivery files run to 2026-09-16.
  - **F&O sub-block: raw ingestion complete and verified (2026-09-24); no F&O feature block yet.**
    `python -m scripts.verify_stage10_fno` PASSES (exit 0) against the files on disk. Every figure below was read from
    disk and the manifests on 2026-09-24, not carried forward from earlier notes.
    - *Bhavcopy* (`data/ingest_fno_bhavcopy.py` -> `data/raw/fno_bhavcopy/`): 6,517 daily files (6,487 weekday + **30
      weekend sessions**), 2000-06-12 -> 2026-09-23, 183.6M contract rows, 2.4 GB; disk == manifest `ok` days and every
      row count matches. NSE's own identity `CHG_IN_OI(t) == OI(t) - OI(t-1)` reconciles 99.69% (2000-09), 99.97% (2010-19),
      100.0000% (2020-24 and UDiFF) and 100% across the legacy->UDiFF cutover; all 30 weekend sessions reconcile with both
      neighbours (60 pairs, 99.97%). (Superseded: both F&O ingesters were caught up to 2026-09-24 after its 19:00 IST publication; the newest file is now
      2026-09-24.) When this was first written the newest day on disk was 2026-09-23: 2026-09-24 was requested before it was
      published (its manifest record says `holiday`, provisional); rerun `python data/ingest_fno_bhavcopy.py --days 3`.
    - *Weekend sessions:* the weekday-only crawl (Stage 3's, inherited) misses them and breaks OI continuity (62.8% on
      2025-02-03). `--weekends` on both F&O ingesters found 30 (24 Sat, 6 Sun, earliest 2003-03-22), 20 of them unknown
      before. The Stage 3 equity/delivery crawls are still weekday-only and NSE serving their weekend files is unprobed.
    - *Documented absences* (each in `scripts/verify_stage10_fno.py` with its evidence; an undocumented one fails
      verification): **2003-10-01**, 19 archive-gap days (NSE 404s on every URL form though equity traded: 2000-07/08/09/10
      dates, 2013-10-09, 2021-03-30), and 32 days ingested after a lossless repair recorded in the manifest `format_era`.
    - *2003-10-01 is the one failed bhavcopy day and is still excluded.* 57 contracts appear twice in the raw file (an
      all-zero row, then a priced row): 48 Dec-2003 futures and 9 with 2004 expiries. The other 4,499 rows are clean. The
      day falls in the training window of the folds testing 2005-2008 (4-year rolling window, plan §7.2; the fold
      schedule is derived, Stage 13 is unbuilt). **Proposed, not applied, needs sign-off:** drop the all-zero copy when
      its twin is priced; fail the day on any other disagreement. Evidence: the priced copies satisfy NSE's OI identity
      57/57 against both neighbouring sessions (it cannot discriminate where OI is 0 throughout, ~53 of 57) and settle
      within -5.5%/+8.6% of adjacent sessions; the zero copies claim settle 0 on contracts that settled at
      42.8-17,515.9 the day before.
    - *Ban list* (`data/ingest_fno_ban.py` -> `data/raw/fno_ban/`): 4,639 lists (4,618 weekday + 21 weekend),
      2008-01-01 -> 2026-09-24; 1,339 NIL days (stored as one null-symbol row: published-and-empty, not "no file"),
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
  - **Flows sub-block: built (2026-09-24); `python -m scripts.verify_stage10_flows` now FAILS (checked 2026-09-30, see
    below) on publication timing specifically — every other check, including all 9 injected-bug checks, still passes.**
    `features/blocks/flows.py` ->
    `data/processed/features/flows/{fii_flows,dii_flows}.parquet`. **FII** is deep: NSDL daily FPI equity net, **1999-01-01 ->**
    (`data/ingest_nsdl_fpi.py`, 333 monthly files), 1999-2026 checked against NSDL's own monthly and quarterly totals and by an
    exact audit of every session with no figure (`data/audit_nsdl_absent_sessions.py`: 143 of 144 are days NSDL did not report;
    the 144th, 2000-02-28, is a quarantined duplicate). Rolling sums cover the last 5/20 REPORTED days. **DII is not usable yet:**
    NSE serves only the latest day, so it is forward capture only (`data/ingest_fii_dii.py`, timestamped); 1 populated row of
    7,919; 2026-09-22 and 09-23 were never captured and are lost; **no scheduled capture exists**, so every uncaptured day is lost.
    Independent rebuild, NSDL cross-checks and all 9 injected-bug checks pass; the block places a value on the next session
    (safe, one session conservative). Publication timing: the monitor run meant to pin the true time (2026-09-24) got no
    evidence because this machine was offline ~46 h; a LATER monitor attempt (`data/raw/_publication/first_seen.jsonl`,
    last written 2026-09-26 20:00) got further — it ran, but died from a `ConnectionError` to `www.fpi.nsdl.co.in` (same
    outage pattern) before ever seeing target date 2026-09-25's figure, leaving one negative record
    (`first_seen_ist: null`). Since evidence now exists but proves nothing, the verifier's behaviour correctly changed from
    reporting PENDING (no evidence at all) to FAILING (an unresolved target date) — this is the verifier doing its job, not
    a regression. **Rerunning `python -m data.publication_monitor` for a fresh target date is the next step**; not done
    today (out of scope for this session, and a publication monitor needs to run live across the event, not be launched
    speculatively). Facts: `data/raw/DATA_NOTES.md` §12.
  - **Breadth sub-block: built and verified (2026-09-24).** `features/blocks/breadth.py` ->
    `data/processed/features/breadth/universe_breadth.parquet`: `universe_pct_above_200dma` = % of Stage 5 point-in-time universe
    members above their own 200-row SMA (NOT % of Nifty 500: no historical constituent lists exist, and the scorer's expected
    `nifty500_pct_above_200dma` input is a proxy, not this quantity). 7,590 of 7,919 sessions have data; the 329 missing are
    285 warm-up (pre-1996), 30 weekend sessions, 2020-07-13, and **13 sessions after 2026-09-04 because Stage 3's equity bhavcopy
    and Stage 4 prices end 2026-09-04** (the F&O and NSDL data run to 09-24). `python -m scripts.verify_stage10_breadth` passes.
  - **Circuits sub-block: complete and verified (2026-09-30).** `data/raw/sec_list/`: 1,674 files, 2019-12-30 -> 2026-09-24
    (weekday + weekend). `data/raw/price_band_hits/`: the crawl that had stalled at 1,339/~4,364 days was resumed and finished
    — 4,147 weekday files (2010-01-04 -> 2026-09-30, only 241 dates genuinely absent, all documented holidays/gaps) plus a
    completed `--weekends` pass (20 real weekend sessions found; 1,726 other weekend dates checked and confirmed non-trading).
    `features/blocks/circuits.py` -> `data/processed/features/circuits/` (2,333 files, one per universe symbol, 6,923,017 rows,
    row-aligned 1:1 with Stage 4 price rows — same shape as delivery/pledging). Two independent sub-blocks, each with its own
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
    is a daily-regenerated snapshot). The crawl that had stalled at 142/~1,129 symbols finished: `data/raw/shp/` holds all
    **1,129/1,129** target symbols (`data.ingest_shp_pledge.universe_symbols_by_priority()`), 21,381 filing rows, zero gaps,
    zero lingering network-error `PARSE_ERROR` rows. The 433 remaining `PARSE_ERROR` rows (2.0%) are permanent, categorized NSE
    data-quality facts, not crawl failures: 191 XBRL-period/master-list disagreements, 167 filings with no XBRL link at all, 75
    dead 404 links. `features/blocks/pledging.py` -> `data/processed/features/pledging/` (2,333 files, one per universe symbol,
    row-aligned 1:1 with Stage 4 price rows; symbols outside the 1,129-symbol SHP crawl target are all-missing by construction,
    not a bug). `pledged_pct_of_promoter` is keyed on the filing's BROADCAST timestamp through the 19:00 IST nightly cutoff
    (`data/publication.py`), never the quarter it describes, its submission date or a creation date; a revision replaces its own
    period's slot without displacing a newer quarter. History starts with the Sep-2021 filings, batch-broadcast 2022-01-06/07.
    `python -m scripts.verify_stage10_pledging` PASSES: crawl coverage; universe file coverage; structure; point-in-time
    (existence of a same-period broadcast usable by each row's date — 0 violations across all 1,129 crawled symbols); an
    independent plain-loop recompute of ratio/change/rising/age from the raw filings (0 violations); no-zero-fill (15,742 real
    0% disclosures in the raw filings, 822,645 feature rows at exactly 0%, both survive unflagged); truncation; and 5
    injected-bug checks (creation/period-date keying, zero-fill, stale carry-forward, a value off by 5pp, a cleared flag) that
    all genuinely fire. Facts: DATA_NOTES §14 (still not written).
  - **Stage 10 is NOT complete.** `python -m scripts.verify_stage10_checkpoint` runs the five sub-blocks this checkpoint
    covers (delivery, F&O, flows, pledging, circuits) — **4 of 5 PASS (2026-09-30): delivery, F&O, pledging, circuits.
    flows FAILS**, on the FII publication-timing check specifically (every other flows check, including all 9 injected-bug
    checks, passes). Cause, confirmed on disk: `data/publication_monitor.py`'s `nsdl_cash` evidence file
    (`data/raw/_publication/first_seen.jsonl`, last written 2026-09-26 20:00) holds exactly one record, for target date
    2026-09-25, with `first_seen_ist: null` and a `ConnectionError` to `www.fpi.nsdl.co.in` — the monitor died mid-run from
    the machine's known network-outage pattern before ever confirming whether that day's figure was on time, so the
    verifier correctly refuses to pass an unproven timing claim. This is a pre-existing gap, untouched by today's work
    (nothing in flows.py, its verifier, or the monitor was changed): **rerunning `python -m data.publication_monitor` for a
    fresh target date is needed to get real evidence** before this can pass; not attempted today (out of scope, and a
    monitor needs to run live across a publication event, not something to launch speculatively). Breadth has its own
    separate verifier, already passing (2026-09-24). What else remains before Stage 10 as a whole is done: **the F&O
    feature block does not exist yet** (`features/blocks/` still has no F&O module — only raw ingestion is built), and DII
    flows remain forward-capture only (one populated row). DATA_NOTES §13/§14 (circuits/pledging source facts) are still not
    written. The repo is now under git (first commit 2026-09-30).
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
  verify_stage10_checkpoint.py   # combined Stage 10 checkpoint: runs the five verifiers above (excl. breadth) and summarizes
  measure_band_inference.py      # circuit_inference accuracy measurement -> data/raw/sec_list/_inference_measurement.csv
tests/
  test_indicators.py, test_scorer.py, test_delivery_features.py, test_fno_ingest.py, test_fno_ban.py, test_ingest_common.py
  test_pledging_block.py, test_circuits_block.py, test_circuit_inference.py, test_ingest_shp_pledge.py
  invariants/           # test_delivery_row_date.py, test_config_database_url.py, test_fno_format_equivalence.py,
                         # test_pledging_invariants.py, test_circuits_invariants.py, ...
data/
  ingest_common.py, ingest_bhavcopy.py, ingest_bhavdata.py, ...   # Stage 3 ingesters
  ingest_fno_bhavcopy.py, ingest_fno_ban.py                       # Stage 10 F&O ingesters (--weekends for weekend sessions)
  ingest_shp_pledge.py, ingest_sec_list.py, ingest_price_band_hits.py   # Stage 10 pledging/circuits ingesters (--weekends where noted)
  raw/                  # bhavcopy, fno_bhavcopy, fno_ban, delivery, shp, sec_list, price_band_hits, announcements, _manifests/, DATA_NOTES.md
  processed/             # corporate_actions.parquet, pit_features.parquet, checkpoints, processed/features/{delivery,flows,breadth,pledging,circuits}/
```