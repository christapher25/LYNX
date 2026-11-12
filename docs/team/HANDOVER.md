# LYNX tab-team handover (written by LEAD, 2026-11-13, on the owner's WRAP-UP AND STOP)

Team: LEAD, DEV-A, DEV-B, TESTER, SECURITY, VERIFIER, each in its own tab and worktree (set-up scripts in `tools/team/`).
All loops are stopped. No job is running. Nothing was pushed, no branch was deleted, no data was deleted.
The board as it stood at the stop is in `docs/team/board/` (BOARD.md, STATE.md, OWNER.md, DECISIONS-OWNER.md, `tasks/`, `handover/`). The full decision record is `logs/agent_team_decisions.md` (git-ignored, in the main checkout).

## 1. Progress

Overall build (30 stages = v6 stages 1-26 plus 13A, 13B, 13C, 13M): about 40% by effort. This is LEAD's estimate, not a measured figure.
- Done (17): stages 1-9, 10 (closed with the known flows publication-timing and DII gaps), 12, 13, 13A, 13B, 13C, 14, 16.
- In progress (2): stage 11 (ban and circuit gates only), 13M (M1, M1b, M2, M3 all NO-GO; the forward paper log is the only live evidence path).
- Not started (11): stage 15 and stages 17-26.
- By stage count that is 57%; it is weighted down because the unstarted stages are the heavy ones and the gate has no GO.

Queue rows (`docs/agent-team/STAGE_QUEUE.md`): 1-18 ticked except the publication-monitor rows 12 and 12a. Row 19 (forward paper log) has its tooling but no logged week. Row 20 (data catch-up) is done except one clean run of `data_refresh` and the paper-log backfill. Row 21 (split/bonus parser correction) is half done and waits for the owner.

## 2. What is merged (main, in order)

Every task below passed TESTER, SECURITY and VERIFIER on the head named, and LEAD ran the invariant suite on main afterwards (825 passed each time).

| Task | What | Reviewed head | On main at |
|---|---|---|---|
| R20-4 | Read-only report on unexplained >20% moves and top-200 exposure | f6deaac | 07cf8cc |
| R20-1 | Corporate-actions refresh mode (`--refresh-from`, `--symbols`), audit script | e781c8f | cafdf68 |
| R20-3 | Owner-approved correction script (AASTHA bonus + 112 pre-cutoff dividends) | 41cb950 | 606d73d |
| R20-5 | `run_data_refresh` follow-ups (manifest-failed check, stale re-serve excuse, range keys) | 162bccd | a10686b |
| R20-2 | History guard: empty-column tolerance, `--expected-changes`; refresh keeps stored schema; process-wide NSE rate limiter | 8cbe2db | 5b403b7 |
| R20-6 | Paper-log backfill wrapper (NOT run on real data) | abaeab5 | dd403e5 |
| (lead) | Team tools: `tools/team/`, `scripts/start_lead.ps1` | - | bdb9ee5 |
| R20-7 | History guard: `--expected-fills` (circuits late fills), `--exempt-dataset` | c40d643 | 3d02c5b |
| R20-8 | Breadth before/after comparison with the owner's STOP rule | d80327c | 28448dd |
| R21-1 | Parser fix: face-value splits in "Re"/short wordings get a factor; non-equity bonus gets none. No stored data changed | 2c9edf7 | 28448dd |
| R20-9 | `data_refresh --downstream` chain (corporate actions, adjust, universe, feature blocks, guard last) | 972d89f | 91790e9 |
| R21-2 | Read-only proposal of corporate-action factor changes with ex-date price evidence | 23597ce | ead0771 |
| R20-10 | `data_refresh`: range-chunk step starts after the newest chunk; superseded overlap refusals excused | 398e3c5 | 21962b7 |

Data work done by LEAD through the job runner (all recorded with run ids in the decision log):
- Nine raw sources brought to 2026-11-09 (eight of them now to 2026-11-11). FII/DII forward capture cannot be backfilled; 2026-09-30..2026-11-08 stays missing and flagged.
- Corporate actions refreshed (163 new actions); owner-approved correction applied (113 rows: AASTHA bonus 2026-09-28 factor 0.5 and 112 dividends).
- `adjust_prices`, universe, delivery, pledging, circuits, F&O, flows, breadth rebuilt to 2026-11-09.
- History guard (cutoff 2026-09-29, Postgres indicators skipped by owner decision): PASSES with 1,475 allowed differences, all of owner-approved kinds: 5 symbols with a new action after the cutoff (BLSE, BUILDPRO, CONCORDBIO, MOLDTECH, MOLDTKPAC), 13 empty-column type changes, 114 from the approved correction (113 rows + AASTHA's adjusted prices), 1,343 circuits late fills (band columns on 2026-09-28 and 2026-09-29 only).
- Breadth: rebuilt under the owner's one-time exemption; comparison with the kept old file shows 0 history differences (7,934 dates before 2026-09-30 identical).
- No block listed any orphan symbol file.
- Combined Stage 10 verifier: 4 of 5; flows fails only on the known publication-timing check.

## 3. Open branches

There is NO unreviewed work on any branch. Every task branch is merged. The only branches not merged into main are four superseded tester branches, each replaced by a later `-r2` branch that IS merged; they are kept, not deleted:

| Branch | Head | Status |
|---|---|---|
| tester/R20-1 | c2ee1b1 | superseded by tester/R20-1-r2 (merged) |
| tester/R20-5 | 847e3a3 | superseded by tester/R20-5-r2 (merged) |
| tester/R20-6 | bd5a998 | superseded by tester/R20-6-r2 (merged) |
| tester/R21-2 | b4c9f03 | superseded by tester/R21-2-r2 (merged) |

Uncommitted in the main checkout, left for the owner as he decided: `scripts/jobs.json` (job entries added since 88dd5a5: history guard, corporate-actions refresh and correction, adjust_prices, universe and feature rebuilds, data_refresh, the one-run breadth guard entry) and `docs/agent-team/STAGE_QUEUE.md` (row 20 status, new row 21). The shell guard stops LEAD from staging the allowlist.

Worktrees: `LYNX_V1-devA` (row21/reparse-proposal, clean), `LYNX_V1-devB` (row20/refresh-announcements-range, clean), `LYNX_V1-test`, `-sec`, `-verify` (detached at 398e3c5, clean).

## 4. Blockers

- OWNER item 0, circuits guard failure: RESOLVED. The owner chose A (2026-11-12); the narrow guard extension (R20-7) is merged and the guard passes with the 1,343 late fills listed as approved. Breadth (A2) is also done with 0 history differences. Nothing is blocked on item 0 any more.
- OWNER item 1, split-parser bug: OPEN, waiting for the owner. The parser is fixed (R21-1) but no stored data has changed. The proposed list is `logs/row21/proposed_changes.md` and `.json` (sha256 544381d44a1689dc630cb79fd0e74650f26a65e27fe025849f1d4c30095109ef): 341 changes in 331 symbols: 337 splits gain a factor (164 at 0.1, 105 at 0.2, 62 at 0.5, 6 others), 4 non-equity bonuses lose theirs (ZEEL 2014-03-03, DRREDDY 2011-03-17, NTPC 2015-03-20, COROMANDEL 2012-07-13). Price check: 297 consistent, 1 inconsistent (JINDALSTEL 2008-01-21, ratio 0.1465 against 0.2), 43 without a usable price. Until this is applied, adjusted prices for those symbols carry fake split-day drops, and M1, M1b, M2, M3 ran on them.
- `data_refresh` has not completed a run. Its first run failed at the announcements step; the fix (R20-10) is merged; the rerun is not done. Adjusted prices, universe and features end 2026-11-09 while eight raw sources end 2026-11-11.
- Machine: 16 GB laptop. Free RAM fell under the 3 GB gate several times and to 0.5 GB once; commits (the hook runs the full invariant suite) and jobs waited. The machine also slept for hours several times.

## 5. The owner's pending decisions

A. Row 21 changes list: approve all 341 (LEAD's recommendation), or the 297 price-consistent lines only, or name exclusions.
B. FII flows end 2026-09-25 because the NSDL monthly files are not in the row-20 source list: add `data.ingest_nsdl_fpi` to `data_refresh`? LEAD recommends yes, as its own reviewed step.
C. Paper-log backfill (six weeks: 2026-10-01, 10-09, 10-16, 10-23, 10-30, 11-08): the owner's order is "not before row 21 is merged and re-adjusted"; it then needs an explicit go, because its rows are permanent.
D. Commit `scripts/jobs.json` and `docs/agent-team/STAGE_QUEUE.md` (the owner said he would do this himself).
E. Standing choices that hold until changed: no rebases (task branches merge main); a tab with an empty queue stops cycling; LEAD stops its loop when waiting only on the owner; Docker/Postgres stays down and the indicators table is out of scope.

## 6. First task for each role on restart

- LEAD: read `docs/team/board/OWNER.md` and the end of `logs/agent_team_decisions.md`. Check free RAM (2 GB gate) and that no job runs; rerun the `data_refresh` job (about 1 hour; `--start 2026-11-09`; the announcements step should now start at 2026-11-10) and confirm the guard passes at its end. Then, on the owner's answer to A, write briefs R21-3 onward: (1) apply step that changes ONLY the parser-derived fields of approved rows; (2) guard extension admitting exactly the approved rows and their symbols' adjusted columns; (3) re-adjust and rebuild; (4) M3 report re-run on corrected data with `docs/vanta/M3_PREREG.md` unchanged, labelled "data-corrected re-run", original report kept; (5) count of M1-M3 trades and candidate weeks that touched an affected symbol-date.
- DEV-A: R21-3 (the apply step), once LEAD has written the brief after the owner's approval. Until then nothing.
- DEV-B: a small follow-up to R20-10 from the reviews: in `scripts/run_data_refresh.py` ignore, with a WARNING, a chunk file whose end date is today or later (a stray future-dated file would make the announcements step report "already current" forever), and log a WARNING when `--start` falls in a hole before the newest chunk. Tests first. LEAD writes the brief.
- TESTER, SECURITY, VERIFIER: nothing is waiting. Take the oldest READY dev report whose head SHA has no report of yours, as in `roles/`.
- Everyone: every commit runs the full invariant suite in the pre-commit hook (2-4 minutes); check for a running pytest and 3 GB free RAM first; never bypass it. Only LEAD runs jobs or writes under `data/`.

## 7. Known limits worth remembering

- The history guard trusts the files it is given (`--expected-changes`, `--expected-fills`) and its `--exempt-dataset` flag. LEAD passes only the two committed files under `docs/agent-team/corrections/`; the breadth exemption was used for exactly one run (20261112_112406) and its job entry must not be run again.
- The guard snapshot was taken with `--skip-db`: the Postgres indicators table is neither guarded nor rebuilt.
- `data_refresh` checkpoint: a same-day rerun skips completed downstream steps; use `--restart-downstream` if the ingesters added files in between. Its `--start` argument lives in the job entry and must be moved forward before each weekly run.
- The job runner once recorded a finished guard run as "died"; a rerun gave exit 0. A background wait can be stopped by the terminal under memory pressure without affecting the detached job; read the result with `run_job.py status`.
