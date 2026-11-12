# TESTER handover (2026-11-12)

Loop stopped (cron job f18b4ad2 cancelled). Worktree C:\Users\chris\PycharmProjects\LYNX_V1-test is on a detached head (398e3c5), tracked tree clean (only the untracked data junction and .env). Nothing uncommitted: all my work is on tester/* branches.

## Done: every READY dev head has a test report in reports\ for its current head
| Task | Head tested | Result | Tester branch (last commit) |
|---|---|---|---|
| R20-1 | e781c8f | APPROVED | tester/R20-1-r2 b7dea27 |
| R20-2 | 8cbe2db | APPROVED | tester/R20-2 53b099d |
| R20-3 | 41cb950 | APPROVED | tester/R20-3 41961be |
| R20-4 | f6deaac | APPROVED | tester/R20-4 393a17a |
| R20-5 | 162bccd | APPROVED | tester/R20-5-r2 7436ade |
| R20-6 | abaeab5 | APPROVED | tester/R20-6-r2 07d907d |
| R20-7 | c40d643 | APPROVED | tester/R20-7 252bf2a |
| R20-8 | d80327c | APPROVED | tester/R20-8 b8289f6 |
| R20-9 | 972d89f | APPROVED | tester/R20-9 024c5fc |
| R20-10 | 398e3c5 | APPROVED | tester/R20-10 f472274 |
| R21-1 | 2c9edf7 | APPROVED | tester/R21-1 d714963 |
| R21-2 | 23597ce | APPROVED | tester/R21-2-r2 04bef73 |

Each tester branch adds only its tests/test_<ID>_tester.py. LEAD merges them with the tasks (tester/R20-1-r2, R20-5-r2, R20-6-r2 and R21-2-r2 supersede earlier tester/<ID> branches).

## In progress
Nothing.

## Open notes for LEAD (all LOW, none blocking)
- R20-10: a hole in the announcements chunks before the newest chunk end is silently skipped when --start falls inside it (briefed behaviour); a WARNING would make it visible.
- R20-9: checkpoint skipping is by step name only (briefed).
- R20-7: unchanged real value on one listed date beside a fill on the other fails closed.
- R21-1: equity bonus whose subject mentions "preference" gets no factor (fails safe).

## Exact next step (when restarted)
Re-run the loop: read BOARD.md and reports\*-dev.md; take the oldest READY dev report whose head SHA has no reports\<ID>-test.md for that SHA (my tester/roles\tester.md). `git switch --detach <SHA>` in LYNX_V1-test (delete any untracked copy of a tester test file first if the switch complains), run the task tests and tests/invariants (needs free RAM >= 3 GB), add edge tests on tester/<ID>, prove them with mutants, write reports\<ID>-test.md. Merge-only heads: confirm 0 diff lines in the task's files, rerun tests + my tester tests + invariants, rewrite the report.

Branch: detached at 398e3c5 (worktree LYNX_V1-test). Last tester commit: f472274 on tester/R20-10.
