# SECURITY handover (2026-11-12)

Role: SECURITY (read-only checker). Worktree: C:\Users\chris\PycharmProjects\LYNX_V1-sec (detached checkouts only).
Branch: none. SECURITY never edits tracked files, so there is nothing to commit. The worktree is detached at 398e3c5db21427f08f3671306434439b7af0fcc1 (the R20-10 head). `git status` shows only untracked files (.env, data/ junction content).
Loop: cron job cb482983 (every 20 min) was cancelled with CronDelete. No other loop or background job is running.

## Done (every report is in LYNX-team\reports\<ID>-sec.md, all CLEAR, each for the head SHA below)
| Task | Head SHA reviewed |
|---|---|
| R20-1 | e781c8f |
| R20-2 | 8cbe2db |
| R20-3 | 41cb950 |
| R20-4 | f6deaac |
| R20-5 | 162bccd |
| R20-6 | abaeab5 |
| R20-7 | c40d643 |
| R20-8 | d80327c |
| R20-9 | 972d89f |
| R20-10 | 398e3c5 |
| R21-1 | 2c9edf7 |
| R21-2 | 23597ce |

## In progress
Nothing. No half-finished review.

## Exact next step
Run the cycle in roles\security.md: list `reports\*-dev.md`, take the oldest READY report whose head SHA has no `<ID>-sec.md` for that SHA, `git switch --detach <SHA>` in LYNX_V1-sec, read `git diff main...<SHA>` (for a re-review, `git diff <old reviewed SHA> <new SHA>`; a head that is only a merge of main needs only a check that the task's own files are unchanged), and write `reports\<ID>-sec.md`. If none: "idle".

## Open points for LEAD (all LOW or operational, none block a CLEAR)
- NSE rate: fixed in R20-2 (process-wide 0.6 s limiter). Still no backoff on a 429 (fails safe).
- Guard files (`--expected-changes`, `--expected-fills`, `--exempt-dataset`) are trusted as given. Always pass the owner-approved JSONs in docs\agent-team\corrections\. R20-9's downstream chain passes no `--exempt-dataset`; confirm how the one-time breadth exemption applies before the real run.
- Real runs that rewrite shared data\ (R20-3 apply script, R20-9 `--downstream`, R20-6 backfill) are LEAD-only jobs in the main checkout via scripts\run_job.py. For R20-6, confirm 068593b is an ancestor of the recorded code_head.
- Several scripts take an unrestricted `--out` / `--out-dir` (R20-8, R21-2, R20-4); optional: refuse paths inside data\.
