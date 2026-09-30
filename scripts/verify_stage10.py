"""
Stage 10 (India block) combined checkpoint.

Runs all five sub-block verifiers this checkpoint covers -- delivery, F&O, flows, pledging, circuits -- each of which
independently validates its own Parquet output against its own raw archive (see each module's docstring for what it
checks). This script exists so one command answers "is Stage 10 done" the way scripts/verify_stage9_checkpoint.py
answers it for Stage 9; every sub-block remains independently runnable as `python -m scripts.verify_stage10_<name>`.

Breadth (features/blocks/breadth.py) has its own verifier, scripts/verify_stage10_breadth.py, already run and passing
(2026-09-24); it is not re-run here because it is not one of the five sub-blocks this checkpoint was asked to cover.

Usage: python -m scripts.verify_stage10
"""

from __future__ import annotations

import sys
import time

from loguru import logger

from scripts import (
    verify_stage10_circuits,
    verify_stage10_delivery,
    verify_stage10_flows,
    verify_stage10_fno,
    verify_stage10_pledging,
)

SUBBLOCKS = [
    ("delivery", verify_stage10_delivery),
    ("F&O", verify_stage10_fno),
    ("flows", verify_stage10_flows),
    ("pledging", verify_stage10_pledging),
    ("circuits", verify_stage10_circuits),
]


def main() -> int:
    results: dict[str, int] = {}
    elapsed: dict[str, float] = {}
    for name, module in SUBBLOCKS:
        logger.info(f"===== Stage 10 / {name} =====")
        started = time.monotonic()
        results[name] = module.main()
        elapsed[name] = time.monotonic() - started

    logger.info("===== Stage 10 combined checkpoint summary =====")
    for name, code in results.items():
        (logger.success if code == 0 else logger.error)(f"{'PASS' if code == 0 else 'FAIL'}  {name}  ({elapsed[name]:.0f}s)")

    failed = [name for name, code in results.items() if code != 0]
    if failed:
        logger.error(f"Stage 10 checkpoint FAILED: {failed}")
        return 1
    logger.success("Stage 10 checkpoint PASSED: delivery, F&O, flows, pledging, circuits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
