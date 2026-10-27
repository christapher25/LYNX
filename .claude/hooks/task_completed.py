"""TaskCompleted hook: a task can only be marked complete while the invariant suite passes.

Exit code 2 blocks the completion. Claude Code passes this script's stderr back
to the agent that tried to close the task, so the failure arrives as a
follow-up message telling it exactly what broke.
"""
from __future__ import annotations

import sys

from _common import read_hook_input, run_invariants


def main() -> int:
    event = read_hook_input()
    who = event.get("teammate_name") or "this agent"
    task = event.get("task_subject") or event.get("task_id") or "this task"

    passed, tail = run_invariants()
    if passed:
        return 0

    sys.stderr.write(
        f"Task '{task}' was NOT marked complete: the invariant suite fails.\n"
        f"{who}, fix the failure below if it is in files you own, re-run "
        f"`python -m pytest tests/invariants -q`, then mark the task complete again.\n"
        f"If the failure is in files you do not own, do not edit them: message the "
        f"lead with the failing test name and this output, and wait.\n\n"
        f"--- pytest output (last lines) ---\n{tail}\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
