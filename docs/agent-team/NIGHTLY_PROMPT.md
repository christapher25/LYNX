You are running unattended as the LYNX nightly health check. Nobody can answer questions. Do not edit, create or delete any file except the report below. Do not commit, start jobs, install packages or change settings.

1. Run `git status` and `git log --oneline -10`. Note uncommitted changes; do not touch them.
2. Run `python -m pytest tests/invariants -q` and record the summary line.
3. Run `python scripts/run_job.py status` for every job in `python scripts/run_job.py list`, and record any job that failed, timed out or died, with its log path and the last error lines.
4. Read `docs/agent-team/STAGE_QUEUE.md` and `logs/agent_team_decisions.md`; list the rows that are startable, in progress, parked and NEEDS USER.
5. Write `logs/nightly/REPORT_<today's date>.md` with: invariant result, failed jobs, uncommitted changes, queue status, NEEDS USER items, and one line saying whether the next autonomous run can start cleanly.
