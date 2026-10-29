# Stage queue

The lead works through this list top to bottom under `docs/agent-team/STANDING_ORDERS.md`. A row can start when every row in its "Depends on" column is ticked. Rows with no dependency between them run in parallel. **STOP** rows need the user. Stage numbers and passing checks follow the v6 build order (section 15.1).

Tick a row when accepted: `[x] 2026-10-29 · a1b2c3d`.

| # | Done | Row | Depends on | Owner | Passing check | STOP? |
|---|---|---|---|---|---|---|
| 1 | [x] 2026-10-29 · 9a8519d | **Stage 10 F&O block**: run the `fno_features` job (full history), then accept and commit the fno files | — | lead (job runner) | `verify_stage10_fno_features` passes on the full output; `verify_stage10` shows the F&O check passing (flows stays a known limitation); invariants green | — |
| 2 | [x] 2026-10-29 · 9a8519d (user decision, flows lag confirmed) | **Decision: count Stage 10 as closed** with the FII/DII publication-timing gap documented as a known limitation, so VANTA implementation may start | 1 | user | User writes the decision in `logs/agent_team_decisions.md` and ticks this row | **STOP** |
| 3 | [x] 2026-10-29 · 4a4bb16 | **Gap-down rule**: add the lower-locked gap-down rule to `docs/vanta/VANTA_SPEC.md` and its invariant test | — | test-writer | New test passes against a toy implementation and fails against a broken one; invariants green | — |
| 4 | [x] 2026-10-29 · 807bc12 | **INTERFACES.md**: record every function name and signature the VANTA tests import | — | test-writer | File lists every import used in `tests/invariants/vanta/` | — |
| 5 | [ ] | **Stage 11 override gates** and circuit/ban filters | 2 | data-engineer | Gates fire correctly on known historical examples (a verifier script); invariants green | — |
| 6 | [x] 2026-10-29 · PENDING_HASH | **Stage 13A NEL label simulator** | 2, 3, 4 | quant-engineer | N1–N4, N6, N11 un-skipped and passing; coverage report by year; reviewer reconciles 20 sampled trades by hand | — |
| 7 | [x] 2026-10-29 · ed6abfe | **Stage 13B filtered regime posterior** (3-state HMM, forward recursion only) | 2, 4 | quant-engineer | N7 passes; posterior available from 2014-05-14 plus warm-up | — |
| 8 | [ ] | **Stage 12 point-in-time feature store** | 5 | data-engineer | A read as of date t returns only what was knowable before t (invariant test) | — |
| 9 | [ ] | **Stage 13 walk-forward harness** with label purging | 8 | quant-engineer | A deliberately leaky configuration produces an implausible IC and is flagged; N11 holds | — |
| 10 | [ ] | **Stage 13C selection gate** (RW-CS with FCC) | 9, 6, 7 | quant-engineer | N5, N8–N10, N13 pass; on synthetic exchangeable data realised FDR ≤ q | — |
| 11 | [ ] | **Stage 13M Milestone M1 run** (momentum scores through NEL and the gate) and its results report | 10 | lead (job runner) | Run completes; report states each pre-registered GO criterion with its number | **STOP** (GO/NO-GO is the user's call) |

After row 11 the run ends. The next queue (Stage 14 baselines onward) is written after the M1 decision.
