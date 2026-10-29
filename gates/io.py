"""
gates/io.py

Stage 11 -- loads the Stage 10 circuits and F&O feature Parquets into the one frame `evaluate_gates` reads. I/O only; no gate logic.

A (symbol, date) present in one block but absent from the other gets NaN values and the missing flag set True -- never a default.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from config import CIRCUIT_FEATURES_DIR, FNO_FEATURES_DIR

CIRCUIT_COLUMNS = ["hit_upper_circuit", "hit_lower_circuit", "is_missing_circuit_hits", "price_band_pct"]
FNO_COLUMNS = ["fno_in_ban", "is_missing_ban"]


def _read(directory, symbol: str, columns: list[str], start, end) -> pd.DataFrame:
    filters = []
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end)))
    return pd.read_parquet(directory / f"{symbol}.parquet", columns=["symbol", "date", *columns], filters=filters or None)


def _unique(frame: pd.DataFrame, block: str, symbol: str) -> pd.DataFrame:
    duplicated = frame.duplicated(["symbol", "date"], keep=False)
    if duplicated.any():
        raise ValueError(f"{block} block for {symbol} has {int(duplicated.sum())} rows sharing a (symbol, date); a merge would multiply rows")
    return frame


def load_gate_inputs(symbols: Iterable[str], start=None, end=None, dates: Iterable | None = None) -> pd.DataFrame:
    """Rows (symbol, date, hit_*, fno_in_ban, price_band_pct, flags) for EVERY requested symbol on EVERY session of the grid.

    The grid is `dates` when given (exactly those sessions, intersected with [start, end]); otherwise the union of dates found in the
    requested symbols' files within [start, end]. A (symbol, date) on the grid that neither block holds -- including a symbol with no
    file in either block -- comes back with NaN values and both missing flags True, so the gate says UNAVAILABLE, never a missing row
    and never PASS. Duplicate (symbol, date) rows within a block raise ValueError.
    """
    symbols = list(dict.fromkeys(symbols))
    frames = []
    for symbol in symbols:
        circuits = _read(CIRCUIT_FEATURES_DIR, symbol, CIRCUIT_COLUMNS, start, end) if (CIRCUIT_FEATURES_DIR / f"{symbol}.parquet").exists() else None
        fno = _read(FNO_FEATURES_DIR, symbol, FNO_COLUMNS, start, end) if (FNO_FEATURES_DIR / f"{symbol}.parquet").exists() else None
        if circuits is not None:
            circuits = _unique(circuits, "circuits", symbol)
        if fno is not None:
            fno = _unique(fno, "fno", symbol)
        if circuits is None and fno is None:
            continue
        frame = (circuits if fno is None else fno if circuits is None else circuits.merge(fno, on=["symbol", "date"], how="outer"))
        frame["symbol"] = symbol
        frames.append(frame)
    columns = ["symbol", "date", *CIRCUIT_COLUMNS, *FNO_COLUMNS]
    found = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    found["date"] = pd.to_datetime(found["date"])
    if dates is not None:
        grid = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique()
        if start is not None:
            grid = grid[grid >= pd.Timestamp(start)]
        if end is not None:
            grid = grid[grid <= pd.Timestamp(end)]
        found = found[found["date"].isin(grid)]
    else:
        grid = pd.DatetimeIndex(found["date"].unique())
    full = pd.MultiIndex.from_product([symbols, grid], names=["symbol", "date"]).to_frame(index=False)
    out = full.merge(found, on=["symbol", "date"], how="left")
    for column in CIRCUIT_COLUMNS + FNO_COLUMNS:
        if column not in out.columns:
            out[column] = float("nan")
    for flag in ("is_missing_circuit_hits", "is_missing_ban"):
        out[flag] = out[flag].astype("boolean").fillna(True).astype(bool)     # a row one block lacks is missing, not clear
    return out.sort_values(["symbol", "date"], ignore_index=True)
