"""Shared helpers for the LYNX agent-team hooks.

Both hooks run the invariant suite (tests/invariants/) with the same Python
interpreter that launched them, so the project's virtual environment is used
when Claude Code was started from an activated venv in the PyCharm terminal.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
INVARIANT_DIR = PROJECT_DIR / "tests" / "invariants"
STATE_DIR = PROJECT_DIR / ".claude" / "hooks" / ".state"
TAIL_LINES = 40


def read_hook_input() -> dict:
    """Read the JSON event that Claude Code sends on stdin; never crash on bad input."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (ValueError, OSError):
        return {}


def run_invariants() -> tuple[bool, str]:
    """Run the invariant suite. Returns (passed, tail of output).

    Zero collected tests counts as a failure, matching the CI rule that a green
    run with no tests is not safety.
    """
    if not INVARIANT_DIR.is_dir():
        return False, f"Invariant test folder not found: {INVARIANT_DIR}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(INVARIANT_DIR), "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])
    # pytest exit codes: 0 = passed, 5 = no tests collected, others = failures/errors
    if proc.returncode == 5:
        return False, "No invariant tests were collected.\n" + tail
    return proc.returncode == 0, tail


def bump_counter(key: str) -> int:
    """Increment and return a per-key retry counter stored under .claude/hooks/.state/."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80] or "unknown"
    path = STATE_DIR / f"{safe}.count"
    try:
        count = int(path.read_text().strip()) + 1
    except (OSError, ValueError):
        count = 1
    path.write_text(str(count))
    return count


def reset_counter(key: str) -> None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80] or "unknown"
    try:
        (STATE_DIR / f"{safe}.count").unlink()
    except OSError:
        pass
