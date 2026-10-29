"""PreToolUse hook for Bash and PowerShell: blocks commands that would get around the safety gates.

Blocked (exit code 2; the reason goes back to the agent as feedback):
  * skipping git hooks:            git commit --no-verify / -n, git push --no-verify
  * switching the hooks off:       git config core.hooksPath ..., git -c core.hooksPath=...
  * editing the job allowlist or the gates from the shell (use the Edit tool, where
    the change is visible in the diff and the decision log): writes to scripts/jobs.json,
    .githooks/, .claude/settings.json

Everything else is allowed (exit 0). The deny rules in .claude/settings.json still
apply on top of this (push, reset --hard, clean, rm -rf ...).
"""
from __future__ import annotations

import json
import re
import sys

RULES = [
    (re.compile(r"\bgit\b[^\n;|&]*\bcommit\b[^\n;|&]*(--no-verify|\s-n(\s|$))"),
     "Skipping the pre-commit hook is not allowed. Make the invariant suite pass, then commit normally."),
    (re.compile(r"\bgit\b[^\n;|&]*--no-verify"),
     "--no-verify is not allowed: the git hooks are the project's safety gates."),
    (re.compile(r"core\.hooksPath", re.IGNORECASE),
     "Changing core.hooksPath is not allowed: it would switch off the pre-commit gate."),
    (re.compile(r"(scripts[\\/]+jobs\.json|\.githooks[\\/]|\.claude[\\/]+settings\.json)", re.IGNORECASE),
     "Do not touch the job allowlist, git hooks or Claude settings from the shell. Propose the change "
     "with the Edit tool so it shows in the diff, and log it in logs/agent_team_decisions.md."),
]

# Read-only commands that may mention protected paths (viewing is fine).
READ_ONLY = re.compile(r"^\s*(cat|type|Get-Content|gc|less|more|head|tail|git\s+(diff|log|show|status))\b",
                       re.IGNORECASE)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0  # never block on malformed input
    if event.get("tool_name") not in ("Bash", "PowerShell"):
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    for pattern, reason in RULES:
        if pattern.search(command):
            if pattern is RULES[3][0] and READ_ONLY.match(command):
                continue
            sys.stderr.write(f"Blocked by .claude/hooks/guard_shell.py: {reason}\nCommand: {command[:300]}\n")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
