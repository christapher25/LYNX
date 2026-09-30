"""
scripts/verify_common.py

Shared reporting for the Stage 10 block verifiers. Every check is a function that returns a list of problems. It is run on
the real output (`report`: must be empty) and on an in-memory copy with one bug injected (`expect_caught`: must be
non-empty). An injected bug that escapes fails the verifier exactly like a real defect; an unmeasured item is `pending`
and is never printed as a pass.
"""

from __future__ import annotations

from loguru import logger


class Reporter:
    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []
        self.pending: list[str] = []

    def report(self, label: str, problems: list[str], detail: str = "") -> None:
        ok = not problems
        (logger.success if ok else logger.error)(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
        if not ok:
            self.failures.append(label)
            for line in problems[:5]:
                logger.error(f"        {line}")

    def expect_caught(self, bug: str, problems: list[str]) -> None:
        caught = bool(problems)
        (logger.success if caught else logger.error)(
            f"{'PASS' if caught else 'FAIL'}  injected bug caught: {bug}"
            + (f"  ({len(problems)} problem(s); e.g. {problems[0]})" if caught else "  <- THE CHECK DID NOT FIRE"))
        if not caught:
            self.failures.append(f"injected bug escaped: {bug}")

    def mark_pending(self, label: str, why: str) -> None:
        self.pending.append(label)
        logger.warning(f"PENDING  {label} ({why})")

    def finish(self) -> int:
        if self.failures:
            logger.error(f"{self.name} checkpoint FAILED: {self.failures}")
            return 1
        logger.success(f"{self.name} checkpoint PASSED" + (f" (PENDING, not verified: {self.pending})" if self.pending else ""))
        return 0
