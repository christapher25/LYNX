# BOARD (LEAD edits this file; everyone reads it)

TEAM STOPPED 2026-11-13 on the owner's WRAP-UP AND STOP. Handover: docs\team\HANDOVER.md in the repo (and board\handover\ per role).
main: 21962b7 + the wrap-up commit (see docs\team\HANDOVER.md for the final hash). Nothing is pushed.

| ID | Lane | Title | Status | Branch | Head SHA | Merged at |
|---|---|---|---|---|---|---|
| R20-1 | DEV-A | Corporate-actions refresh mode, tests, audit script | merged | row20/corp-actions-refresh | e781c8f | cafdf68 |
| R20-2 | DEV-A | History guard: empty-column tolerance, expected changes; refresh keeps schema; NSE rate limiter | merged | row20/guard-and-schema | 8cbe2db | 5b403b7 |
| R20-3 | DEV-B | Owner-approved correction script (AASTHA bonus + 112 dividends); applied by LEAD | merged | row20/precutoff-correction | 41cb950 | 606d73d |
| R20-4 | DEV-B | Unexplained-moves report | merged | row20/unexplained-moves-report | f6deaac | 07cf8cc |
| R20-5 | DEV-B | run_data_refresh follow-ups | merged | row20/refresh-wrapper-followups | 162bccd | a10686b |
| R20-6 | DEV-B | Paper-log backfill wrapper (NOT run) | merged | row20/paper-log-backfill | abaeab5 | dd403e5 |
| R20-7 | DEV-A | History guard: late fills + one-time dataset exemption | merged | row20/guard-late-fills | c40d643 | 3d02c5b |
| R20-8 | DEV-B | Breadth comparison report | merged | row20/breadth-compare | d80327c | 28448dd |
| R20-9 | DEV-B | data_refresh --downstream chain | merged | row20/data-refresh-chain | 972d89f | 91790e9 |
| R20-10 | DEV-B | data_refresh: announcements range start / overlap refusal | merged | row20/refresh-announcements-range | 398e3c5 | 21962b7 |
| R21-1 | DEV-B | Parser fix (splits without a factor; non-equity bonus); no stored data changed | merged | row21/parser-fix | 2c9edf7 | 28448dd |
| R21-2 | DEV-A | Proposed-changes list (read-only); official list run by LEAD | merged | row21/reparse-proposal | 23597ce | ead0771 |
| R21-3.. | - | Apply step, guard extension, re-adjust, M3 data-corrected re-run, affected-trade count | BLOCKED-OWNER (approval of the 341-line list) | - | - | - |

No unreviewed work exists on any branch. Unmerged branches are only four SUPERSEDED tester branches (tester/R20-1, tester/R20-5, tester/R20-6, tester/R21-2), each replaced by its -r2 branch, which is merged. Nothing was deleted.
No job is running. All loops are stopped.
