# Handover: DEV-B (2026-11-13)

Loop stopped (cron job 1c4291aa deleted). Worktree C:\Users\chris\PycharmProjects\LYNX_V1-devB is clean (tracked files), on branch row20/refresh-announcements-range at 398e3c5. Nothing uncommitted, nothing in progress.

## Done (every DEV-B task has a dev report in LYNX-team\reports\<ID>-dev.md)
| Task | State at handover | Branch | Last commit |
|---|---|---|---|
| R20-3 precutoff correction script | merged 606d73d | row20/precutoff-correction | 41cb950 |
| R20-4 unexplained-moves report | merged 07cf8cc | row20/unexplained-moves-report | f6deaac |
| R20-5 run_data_refresh follow-ups | merged a10686b | row20/refresh-wrapper-followups | 162bccd |
| R20-6 paper-log backfill wrapper | merged dd403e5 | row20/paper-log-backfill | abaeab5 |
| R20-8 breadth compare (STOP rule) | merged 28448dd | row20/breadth-compare | d80327c |
| R21-1 parser fix | merged 28448dd | row21/parser-fix | 2c9edf7 |
| R20-9 data_refresh --downstream chain | merged 91790e9 | row20/data-refresh-chain | 972d89f |
| R20-10 announcements range/overlap fix | READY and fully approved (TESTER, SECURITY, VERIFIER all at 398e3c5); waiting for LEAD to merge tester/R20-10 (f472274) with it. Board row may still say "assigned" | row20/refresh-announcements-range | 398e3c5 |

## In progress
Nothing.

## Exact next step
- For DEV-B: none. No brief is assigned and nothing is in changes.
- For LEAD: merge row20/refresh-announcements-range (398e3c5) together with tester/R20-10 (f472274) into main, update the BOARD row for R20-10, then rerun the data_refresh job (same --start 2026-11-09, --downstream) so the announcements step starts at 2026-11-10.
- Optional, only if LEAD asks: R20-10 Round 1 for the verifier's finding 1 (a stray chunk file whose end date is in the future would make the announcements step log "already current" every day and silently skip it): in chunk_files() of scripts/run_data_refresh.py ignore chunks ending after yesterday with a WARNING, test first. A new head would need all three reviews again.

## Notes for whoever continues
- Pre-commit hook runs the full invariant suite (about 80-200 s): check free RAM (>= 3 GB) and that no other pytest is running before committing. Never rebase: bring a branch up to date with git merge main.
- Real data was never written by DEV-B; no job was run. The real runs (R20-3 correction, R20-6 backfill, R20-8 compare, data_refresh) are LEAD's.
- Open observations recorded in the reports: R20-9 runtime is about 50-55 min downstream (corporate-actions refresh about 31 min under the R20-2 rate limiter); the checkpoint skips completed downstream steps on a same-day rerun (use --restart-downstream if the ingesters added files).
