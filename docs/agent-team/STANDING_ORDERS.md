# Standing orders for the agent-team lead (v2)

These orders let the lead run the build without waiting for the user: develop, test, execute long jobs, verify, commit, and move to the next stage. They sit under CLAUDE.md: if anything here conflicts with CLAUDE.md's forbidden list, CLAUDE.md wins.

## 1. Work from the stage queue

- Work through `docs/agent-team/STAGE_QUEUE.md` top to bottom. A row can start when everything in its "Depends on" column is accepted.
- Run independent rows in parallel (at most 3 teammates at once), never two rows that edit the same files.
- A row marked **STOP** in the queue: do the preparatory work, write what the user must decide to `logs/agent_team_decisions.md` as `NEEDS USER`, and do not start rows that depend on it. Keep independent rows moving.
- After a row is accepted, tick it in the queue (`[x]`, the date and the commit hash) and continue. Do not wait for the user between rows.

## 2. Decide, don't ask

When a teammate asks a question or something is unclear, decide it yourself, in this order of authority:

1. CLAUDE.md (forbidden list and build rules).
2. `docs/LYNX-v6-Final-Architecture-and-Setup-Documentation.txt`, `docs/vanta/VANTA_SPEC.md`, `docs/vanta/INTERFACES.md`.
3. Existing code and tests in the repository (match what is already there).
4. If still open: the more conservative choice (stricter test, smaller scope, no new dependency, no data deleted).

Write every decision to `logs/agent_team_decisions.md` (append, never rewrite): time, who asked, the question, the decision, and the source that decided it.

## 3. Stop and ask the user only for these

- Anything the CLAUDE.md forbidden list rules out, or a change to CLAUDE.md itself.
- Pushing, force operations, `git reset --hard`, `git clean`, rebasing, rewriting history, deleting data files or raw downloads.
- A long job that is not in `scripts/jobs.json` and does not meet the rules for adding it (section 5).
- Adding a new third-party dependency to requirements.
- A spec question where the possible readings would change results (not just style): a label rule, a cost rule, an exclusion rule, a threshold.
- A row marked STOP in the stage queue (milestones such as M1 GO/NO-GO).
- A teammate has failed the same task twice and its replacement has also failed.

For these, write the question to `logs/agent_team_decisions.md` marked `NEEDS USER` (with your recommendation and why), park only the affected work, and keep everything else moving.

## 4. Errors

- When a teammate reports an error, fails on an API error, or goes idle with its work unfinished, send it a follow-up by name at once: the exact error, the likely cause, and the next concrete step.
- Same teammate, same error, twice: stop it and spawn a replacement with the full context (task, owned files, what was tried, the error).
- A red invariant suite blocks every acceptance until it is green again. Find which change caused it and send that teammate the output.
- A failed job (`run_job.py wait` exits 1): read the error tail it prints, decide whether it is a code bug (send it to the owning teammate), a data problem (log it, NEEDS USER if it needs a data decision) or transient (start it again once, with `--resume` if the job supports it).

## 5. Executing long jobs

Teammates cannot run long jobs; you run them, always through the job runner:

1. `python scripts/run_job.py start <name>` (returns immediately; the job runs detached and survives the session).
2. `python scripts/run_job.py wait <name>` as a **background** command. You are notified when it exits: exit 0 = succeeded, 1 = failed / timeout / died. Keep coordinating teammates while it runs.
3. When it exits, run the stage's verifier and the invariant suite, then accept or send the failure to the owner.

One job at a time unless the allowlist marks both as `parallel_ok`. Never start a long job with Start-Process, `&`, nohup or a plain command.

You may add a job to `scripts/jobs.json` (with the Edit tool, logged as `JOB ADDED` in the decision log) only if all of these hold: it is resumable from a checkpoint; it writes only under `data/processed/`, a feature folder or `logs/`; it deletes nothing; it is expected to finish within 8 hours (`max_hours` ≤ 8). Anything else is NEEDS USER.

## 6. Accepting work and committing

A task is accepted only when all of these hold:

1. `python -m pytest tests/invariants -q` passes (run it yourself; never rely on the teammate's report).
2. The stage's passing check from the queue passes (its verifier or job).
3. The reviewer teammate reports no issues on the diff.

Then commit locally, one commit per accepted task, with a message naming the stage, the task and the check result. The pre-commit hook runs the invariant suite again; never bypass it (`--no-verify` is blocked). Never push. Record the commit hash in the queue and the decision log.

## 7. Usage limits and context

- Keep at most 3 teammates alive. Shut a teammate down as soon as its work is accepted; spawn a fresh one for the next row.
- If a usage limit stops work, wait for the reset (Claude Code continues the lead automatically); then message each parked teammate by name to resume, or respawn it from the queue if it is gone.
- Give teammates focused spawn prompts: the row, the owned files, the passing check, the sources to read. Not the whole history.

## 8. Finishing a run

Do not end while any startable row is pending or in progress; wait for teammates and jobs (their messages and job exits wake you). When nothing startable is left (everything accepted, parked or STOP):

1. Write `logs/agent_team_report.md`: what each teammate delivered (files, tests, commits), job runs and their results, the final invariant-suite result, every `NEEDS USER` item with your recommendation, and what the next run should start with.
2. Shut the teammates down.
