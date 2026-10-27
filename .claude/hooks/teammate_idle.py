"""TeammateIdle hook: a teammate that stops while the invariant suite is red is sent back to work.

Teammates sometimes stop after hitting an error instead of recovering. When a
teammate is about to go idle, this hook runs the invariant suite:

* passing  -> exit 0, the teammate goes idle and the lead is notified as usual;
* failing  -> exit 2 with the failure on stderr, so the teammate receives it as
              a follow-up and keeps working;
* failing for the MAX_RETRIES-th time in a row -> exit 0 and let it go idle, so
  the lead (orchestrator) takes over instead of the teammate looping forever.
"""
from __future__ import annotations

import sys

from _common import bump_counter, read_hook_input, reset_counter, run_invariants

MAX_RETRIES = 3


def main() -> int:
    event = read_hook_input()
    who = event.get("teammate_name") or event.get("session_id") or "teammate"
    key = f"idle-{who}"

    passed, tail = run_invariants()
    if passed:
        reset_counter(key)
        return 0

    attempt = bump_counter(key)
    if attempt >= MAX_RETRIES:
        reset_counter(key)
        # Let it go idle: the lead gets the idle notification and must follow up.
        print(
            f"{who} stopped with the invariant suite still failing after "
            f"{MAX_RETRIES} attempts; handing over to the lead."
        )
        return 0

    sys.stderr.write(
        f"Do not stop yet ({attempt}/{MAX_RETRIES - 1} automatic follow-ups): "
        f"the invariant suite is failing.\n"
        f"1. If the failure comes from your own change, fix it and re-run "
        f"`python -m pytest tests/invariants -q`.\n"
        f"2. If it comes from files another teammate owns, do not edit them: "
        f"message the lead with the failing test and this output, then stop.\n"
        f"3. If you hit an error you cannot resolve (missing data, a dependency "
        f"not built yet), message the lead with the exact error text and what "
        f"you tried, then stop.\n\n"
        f"--- pytest output (last lines) ---\n{tail}\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
