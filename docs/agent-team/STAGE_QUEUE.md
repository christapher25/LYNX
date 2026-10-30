# Stage queue

The lead works through this list top to bottom under `docs/agent-team/STANDING_ORDERS.md`. A row can start when every row in its "Depends on" column is ticked. Rows with no dependency between them run in parallel. **STOP** rows need the user. Stage numbers and passing checks follow the v6 build order (section 15.1).

Tick a row when accepted: `[x] 2026-10-29 · a1b2c3d`.

| # | Done | Row | Depends on | Owner | Passing check | STOP? |
|---|---|---|---|---|---|---|
| 1 | [x] 2026-10-29 · 9a8519d | **Stage 10 F&O block**: run the `fno_features` job (full history), then accept and commit the fno files | — | lead (job runner) | `verify_stage10_fno_features` passes on the full output; `verify_stage10` shows the F&O check passing (flows stays a known limitation); invariants green | — |
| 2 | [x] 2026-10-29 · 9a8519d (user decision, flows lag confirmed) | **Decision: count Stage 10 as closed** with the FII/DII publication-timing gap documented as a known limitation, so VANTA implementation may start | 1 | user | User writes the decision in `logs/agent_team_decisions.md` and ticks this row | **STOP** |
| 3 | [x] 2026-10-29 · 4a4bb16 | **Gap-down rule**: add the lower-locked gap-down rule to `docs/vanta/VANTA_SPEC.md` and its invariant test | — | test-writer | New test passes against a toy implementation and fails against a broken one; invariants green | — |
| 4 | [x] 2026-10-29 · 807bc12 | **INTERFACES.md**: record every function name and signature the VANTA tests import | — | test-writer | File lists every import used in `tests/invariants/vanta/` | — |
| 5 | [x] 2026-10-29 · 717997d · PARTIAL: ban + circuit gates only (others need later-stage inputs or thresholds) | **Stage 11 override gates** and circuit/ban filters | 2 | data-engineer | Gates fire correctly on known historical examples (a verifier script); invariants green | — |
| 6 | [x] 2026-10-29 · 9b29c1b | **Stage 13A NEL label simulator** | 2, 3, 4 | quant-engineer | N1–N4, N6, N11 un-skipped and passing; coverage report by year; reviewer reconciles 20 sampled trades by hand | — |
| 7 | [x] 2026-10-29 · ed6abfe | **Stage 13B filtered regime posterior** (3-state HMM, forward recursion only) | 2, 4 | quant-engineer | N7 passes; posterior available from 2014-05-14 plus warm-up | — |
| 8 | [x] 2026-10-29 · e790cfb | **Stage 12 point-in-time feature store** | 5 | data-engineer | A read as of date t returns only what was knowable before t (invariant test) | — |
| 9 | [x] 2026-10-29 · d0eb0a5 | **Stage 13 walk-forward harness** with label purging | 8 | quant-engineer | A deliberately leaky configuration produces an implausible IC and is flagged; N11 holds | — |
| 10 | [x] 2026-10-29 · bc86d60 | **Stage 13C selection gate** (RW-CS with FCC) | 9, 6, 7 | quant-engineer | N5, N8–N10, N13 pass; on synthetic exchangeable data realised FDR ≤ q | — |
| 11a | [ ] | **M1 driver and tests** (`vanta/m1.py`, `scripts/run_m1.py`): weekly momentum top-10 -> NEL labels -> 52-week matured calibration -> gate; per-year FDR, publication rate, Newey-West, weekly block bootstrap, ungated and v4-reconstruction comparators; impact 0 and pessimistic floor; h=5 and 20; k_s/k_t chosen on 2010-2012 then frozen; GO on 2013+. Tests first | 10 | quant-engineer + test-writer | New M1 tests pass incl. mutants; short h=5 real-data harness check; invariants green; reviewer clean | — |
| 11b | [ ] | **M1 data prerequisites**: refresh index and USDINR data, rebuild breadth on the point-in-time universe (verifier), wire delisted terminal values into 13A, 5 seeded EM restarts in 13B, lower-circuit gate on for buys, liquidity-cliff gate (top 200 by trailing turnover) | 10 | data-engineer (+ quant for 13B/13A edits) | `verify_stage10_breadth` passes; delisted terminal-value test passes; invariants green | — |
| 11c | [ ] | **Add `m1` job to jobs.json and run it** (resumable, max_hours <= 8) | 11a, 11b | lead (job runner) | Job exits 0 for both cost runs and both horizons | — |
| 11d | [ ] | **M1 results report** against the five GO criteria (2013+ evaluation, 2010-2012 in-sample, pessimistic run judged, NO_FILL counts, not-applied gates listed, FDR is empirical not a guarantee) | 11c | lead | Report states each criterion with its number | **STOP** (GO/NO-GO is the user's call) |
| 12 | [ ] | **Publication monitor week** (`data/publication_monitor.py`, new target dates, user item 20), runs in parallel with 11 | — | lead | One valid first-seen record per target date per source | — |

After row 11 the run ends. The next queue (Stage 14 baselines onward) is written after the M1 decision.
