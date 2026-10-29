# Kickoff prompt for an autonomous run

Paste everything below the line into the lead session after starting it (Step 8 of the setup guide).

---

You are the lead (orchestrator) of the LYNX agent team. Run autonomously under docs/agent-team/STANDING_ORDERS.md and CLAUDE.md; both are binding. You coordinate, run jobs through scripts/run_job.py, verify and commit. You do not implement stage work yourself.

1. Check the real state first: read CLAUDE.md, docs/agent-team/STANDING_ORDERS.md, docs/agent-team/STAGE_QUEUE.md and logs/agent_team_decisions.md (I record my decisions on STOP rows there); run `git status`, `python -m pytest tests/invariants -q` and `python scripts/run_job.py list`. Confirm you have the task tools (TaskCreate) and that agent teams are on. Tick any queue row that the repository or the decision log shows is already done, with the evidence, and log it.
2. Create the shared task list from every startable queue row, split into subtasks of about 5 per row, with dependencies.
3. Spawn teammates for the startable rows, at most 3 at a time, named after their role and row (for example `data-s11`, `quant-13a`, `tests`, `reviewer`), using the agent types lynx-data-engineer, lynx-quant-engineer, lynx-test-writer and lynx-reviewer. Give each a focused spawn prompt: the row, the files it owns, the passing check, the sources to read.
4. Run long jobs yourself with `python scripts/run_job.py start <name>`, then `python scripts/run_job.py wait <name>` as a background command.
5. Accept work only per the standing orders (invariants run by you, the passing check, a clean review), commit locally, tick the row, and move to the next startable row without waiting for me.
6. Stop for me only on the standing orders' list; log those as NEEDS USER with your recommendation and keep independent rows moving.
7. When nothing startable is left, write logs/agent_team_report.md and shut the teammates down.
