# DEV-A handover (2026-11-12)

Role: DEV-A. Worktree: C:\Users\chris\PycharmProjects\LYNX_V1-devA. Loop stopped (cron job 642fb25d deleted).

## Branch and last commit
- Branch checked out now: row21/reparse-proposal
- Last commit: 23597ce "Merge main (91790e9) into row21/reparse-proposal" (merge of main on top of 1882509). Tree clean: nothing uncommitted, nothing staged, no WIP commit needed.
- BOARD (15:40 and 16:15) records R21-2 as merged into main at ead0771.

## Done (all merged per BOARD.md)
- R20-1 corporate-actions refresh: Round 1 fix (reworded duplicates inside one fetch, FETCH FAILED via log.warning), merged cafdf68.
- R20-2 history guard: (a) empty-column dtype tolerance, (c) --expected-changes, (b) refresh keeps each file's stored Arrow schema, (d) process-wide NSE rate limiter. Merged 5b403b7.
- R20-7 history guard: --expected-fills (owner-approved circuits late fills; real dry check 1,343 ALLOWED, 0 failures) and --exempt-dataset (one-time features_* exemption). Merged 3d02c5b.
- R21-2 proposed corporate-action changes list (read-only script plus tests; Round 1: bad closes and bad factors are no_price, combined same-day factor and check_combined). Real run: 341 proposed = 337 gain + 4 lose; LEAD ran the official list. Merged ead0771.
- All dev reports are in LYNX-team\reports (R20-1-dev.md, R20-2-dev.md, R20-7-dev.md, R21-2-dev.md, plus NOTE-DEV-A-20261110-2100.md).

## In progress
Nothing. BOARD.md (updated 2026-11-12 16:15) assigns DEV-A no task; the only open task is R20-10, which belongs to DEV-B.

## Exact next step
Read BOARD.md and tasks\ for a new DEV-A assignment. If there is none, stay idle. Before starting anything new: `git switch -c <branch from the brief> main` from a clean tree (main is now at or after ead0771).

## Things to know for the next session
- No git rebase (it is on the ask-the-owner list and was denied): bring a branch up to date with `git merge main` into your own branch.
- Every commit runs the full invariant suite in the pre-commit hook (about 2.5-3 minutes). Check free RAM >= 3 GB and that no other pytest/invariants run is in progress first; a hook run was once killed for low memory and had to be retried.
- Never `git checkout -- <file>` to revert a mutation while the code under test is uncommitted: it also discards the uncommitted work. Commit first, or mutate a backed-up copy and restore the copy.
- Big multi-line python or heredoc edits in the shell were sometimes rejected by the tool wrapper; writing the script to the scratchpad folder with the Write tool and running it worked every time.
- The worktree's data folder is a junction to the main checkout's data folder (shared and real): never write under it from here.
