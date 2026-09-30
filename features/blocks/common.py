"""
features/blocks/common.py

Small helpers shared by the market-level Stage 10 blocks (flows, breadth, and the symbol blocks' calendars).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Collection, Sequence

import pandas as pd
from loguru import logger


def validated_sessions(sessions: Sequence[date]) -> pd.DatetimeIndex:
    """Trading-session dates as a normalised DatetimeIndex; sorted ascending with no duplicates or it raises."""
    index = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions]).normalize()
    if any(b <= a for a, b in zip(index[:-1], index[1:])):
        raise ValueError("sessions must be sorted ascending with no duplicates")
    return index


def prune_orphaned_symbol_files(directory: Path, symbols: Collection[str]) -> list[str]:
    """
    Delete a per-symbol feature Parquet for any symbol no longer in the current universe. A symbol can
    legitimately fall out of every year's top-700 turnover ranking when Stage 5 is rebuilt with corrected
    point-in-time data (found 2026-09-30: a Stage 5 rebuild that added weekend sessions and fixed the
    2020-07-13 parsing gap upstream dropped RATNAVEER everywhere). A --no-resume rebuild only overwrites
    files for symbols in the CURRENT universe -- it never deletes one for a symbol that fell out -- so a
    stale file lingers with nothing marking it stale, and the checkpoint's "one file per universe symbol,
    no extras" check correctly flags it forever. Mirrors signals.indicators.prune_orphaned_symbols, same
    root cause, a different storage layer (a directory of Parquet files here, Postgres there).
    """
    directory = Path(directory)
    present = {p.stem for p in directory.glob("*.parquet")}
    orphaned = sorted(present - set(symbols))
    for symbol in orphaned:
        (directory / f"{symbol}.parquet").unlink()
    if orphaned:
        logger.warning(f"{directory}: pruned {len(orphaned)} symbol file(s) no longer in the universe: {orphaned[:10]}")
    return orphaned
