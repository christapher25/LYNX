# VERIFIER handover (2026-11-13)

Role: VERIFIER (read-only). Worktree: C:\Users\chris\PycharmProjects\LYNX_V1-verify, detached checkouts only. Branch: none; nothing committed (my role never edits tracked files; my only writes are reports\<ID>-verify.md in the team folder). Last commit checked out: 398e3c5 (R20-10). The 20-minute loop (cron job ea7e7c6c) is deleted.

## Done (every report is for the SHA named)
- R20-1 e781c8f APPROVED, R20-2 8cbe2db APPROVED, R20-3 41cb950 APPROVED, R20-4 f6deaac APPROVED
- R20-5 162bccd APPROVED (after CHANGES REQUIRED at c91b956: range key hid an earlier failure)
- R20-6 abaeab5 APPROVED, R20-7 c40d643 APPROVED, R20-8 d80327c APPROVED (earlier d: 1f4e645 fix for NaT dates / empty old file)
- R20-9 972d89f APPROVED (main af02a30 is identical in scripts and tests), R20-10 398e3c5 APPROVED
- R21-1 2c9edf7 APPROVED, R21-2 23597ce APPROVED
Reports: LYNX-team\reports\<ID>-verify.md. Method used: dev-report head SHA vs the SHA in <ID>-verify.md; a merge-of-main-only head gets a short re-issue after checking the diff of the reviewed files is empty.

## In progress
Nothing. No READY dev report without a verify report for its current head at the time of stopping.

## Open points for LEAD / owner (from my reports, not blocking)
- R21-2: JINDALSTEL 2008-01-21 split (0.2, observed 0.1465, not same_day) is the one line that stays inconsistent; owner to look by hand.
- R20-10 MINOR: a chunk file with a FUTURE end date would make the announcements step report "already current" forever; fix = ignore chunks ending today or later (not yet done).
- R20-9: the corporate-actions refresh now takes about 31 min (rate limiter), the whole downstream chain roughly 55-70 min.
- R20-7: --exempt-dataset is a plain flag; pass it only for the one breadth run it was granted for.

## Exact next step
Restart the loop only when the owner says so: `/loop 20m You are VERIFIER. Follow C:\Users\chris\PycharmProjects\LYNX-team\roles\verifier.md and run one cycle.` The next cycle: list dev reports whose "Head SHA" differs from the one in reports\<ID>-verify.md, take the oldest READY one, `git switch --detach <SHA>` in LYNX_V1-verify, review per roles\verifier.md.
Lesson to keep: for gate scripts, try empty and missing-key inputs (R20-8 NaT date).
