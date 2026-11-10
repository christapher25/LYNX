"""History guard: prove a refresh did not rewrite history.

    python -m scripts.history_guard snapshot [--cutoff 2026-09-29] [--skip-db]
    python -m scripts.history_guard verify   [--only NAME ...]

`snapshot` records, per dataset and per symbol (or file), a digest of every column over the rows dated on or before
the cutoff, plus the row count and a digest of the date column. `verify` recomputes them and exits 1 listing every
difference. Rows dated after the cutoff are never compared, so new data is always fine.

Datasets (snapshot file logs/history_guard/<name>.json): corporate_actions (one file per symbol, by ex_date),
adjusted (Stage 4), universe (Stage 5, one file per year <= the cutoff year), features_<block> (one entry per Parquet),
indicators (Postgres, rows with date <= cutoff, read-only SELECTs).

The single exemption: in `adjusted`, the backward-adjusted columns (ADJ_* and ADJUSTMENT_APPLIED, as produced by
data/adjust_prices.py) of a symbol that has a corporate action with ex_date after the cutoff AND a usable adjustment_factor
(only those change adjusted prices). Such differences are reported as ALLOWED. Anything else fails: unadjusted columns of
that symbol, any other symbol, added or removed rows, universe, features, indicators.

Universe status columns (LAST_DATE, IS_DELISTED, TERMINAL_VALUE) are compared only for symbols already delisted at
snapshot time (LAST_DATE before the cutoff); for still-active symbols they legitimately move with the data end.

Read-only on data: the only thing written is the snapshot JSON under the output directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CUTOFF = "2026-09-29"
INDEX_NAME = "_index"
NEW_ACTION_LABEL = "(symbol has a new corporate action; derived from adjusted columns?)"
DATE_COLUMNS = ("DATE", "date", "ex_date")
UNIVERSE_KEY_COLUMNS = ["SYMBOL", "RANK", "AVG_TURNOVER", "FIRST_DATE"]
UNIVERSE_STATUS_COLUMNS = ["LAST_DATE", "IS_DELISTED", "TERMINAL_VALUE"]


def adjusted_columns() -> frozenset[str]:
    """The backward-adjusted columns, taken from data/adjust_prices.py (its single source of truth)."""
    from data.adjust_prices import PRICE_COLS
    return frozenset({f"ADJ_{c}" for c in PRICE_COLS} | {"ADJ_CLOSE", "ADJ_VOLUME", "ADJUSTMENT_APPLIED"})


@dataclass
class Sources:
    actions_dir: Path
    adjusted_dir: Path
    universe_dir: Path
    features_dir: Path
    out_dir: Path
    # cutoff -> iterable of (symbol, DataFrame[date, <indicator columns>]) with rows dated <= cutoff. None: no indicators dataset.
    indicator_reader: Callable[[str], Iterable[tuple[str, pd.DataFrame]]] | None = None


def default_sources(skip_db: bool = False) -> Sources:
    import config
    return Sources(
        actions_dir=config.CORPORATE_ACTIONS_DIR,
        adjusted_dir=config.ADJUSTED_PRICES_DIR,
        universe_dir=config.UNIVERSE_DIR,
        features_dir=config.FEATURE_BLOCKS_DIR,
        out_dir=config.LOG_DIR / "history_guard",
        indicator_reader=None if skip_db else postgres_indicator_reader,
    )


def postgres_indicator_reader(cutoff: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Read-only: one SELECT per symbol, rows with date <= cutoff, ordered by date."""
    from config import get_db_connection
    from signals.indicators import INDICATOR_COLUMNS, INDICATORS_TABLE
    conn = get_db_connection()
    try:
        conn.set_session(readonly=True, autocommit=True)
        cols = ["date", *INDICATOR_COLUMNS]
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT symbol FROM {INDICATORS_TABLE} WHERE date <= %s ORDER BY symbol", (cutoff,))
            symbols = [r[0] for r in cur.fetchall()]
        for symbol in symbols:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {', '.join(cols)} FROM {INDICATORS_TABLE} WHERE symbol = %s AND date <= %s ORDER BY date",
                            (symbol, cutoff))
                frame = pd.DataFrame(cur.fetchall(), columns=cols)
            frame["date"] = pd.to_datetime(frame["date"])
            yield symbol, frame
    finally:
        conn.close()


# ----------------------------------------------------------------------------------------------- digests
def _digest_series(series: pd.Series) -> str:
    try:
        hashed = pd.util.hash_pandas_object(series, index=False).to_numpy()
    except TypeError:  # unhashable objects in an object column: fall back to their text
        hashed = pd.util.hash_pandas_object(series.astype(str), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:32]


def _date_column(frame: pd.DataFrame) -> str | None:
    return next((c for c in DATE_COLUMNS if c in frame.columns), None)


def _digest_frame(frame: pd.DataFrame, sort_col: str, columns: list[str] | None = None, track_all_null: bool = False) -> dict:
    frame = frame.sort_values(sort_col, kind="stable").reset_index(drop=True)
    cols = list(frame.columns) if columns is None else columns
    unit = {
        "rows": int(len(frame)),
        "columns": {c: _digest_series(frame[c]) for c in cols},
        "keys": _digest_series(frame[sort_col]),
    }
    if track_all_null:  # not part of any digest: only names the columns that held no value in the history rows
        unit["all_null"] = sorted(c for c in cols if not frame[c].notna().any())
    return unit


def _all_null_digests(rows: int) -> set[str]:
    """Digests of an all-null column of `rows` rows in every representation a Parquet round trip can produce."""
    return {
        _digest_series(pd.Series([None] * rows, dtype=object)),
        _digest_series(pd.Series([float("nan")] * rows, dtype=object)),
        _digest_series(pd.Series([float("nan")] * rows, dtype="float64")),
        _digest_series(pd.Series([pd.NaT] * rows, dtype="datetime64[ns]")),
        _digest_series(pd.Series([pd.NaT] * rows, dtype="datetime64[us]")),
    }


def _history(frame: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    """Rows dated <= cutoff (rows with no date are kept, never silently dropped)."""
    col = _date_column(frame)
    if col is None:
        return frame, None
    d = pd.to_datetime(frame[col])
    return frame[(d <= cutoff) | d.isna()], col


def _file_unit(path: Path, cutoff: pd.Timestamp, track_all_null: bool = False) -> dict | None:
    frame, col = _history(pd.read_parquet(path), cutoff)
    if col is None:
        raise ValueError(f"{path}: no date column among {DATE_COLUMNS}; cannot apply a cutoff")
    return _digest_frame(frame, col, track_all_null=track_all_null) if len(frame) else None


def _dir_units(directory: Path, cutoff: pd.Timestamp, track_all_null: bool = False) -> dict[str, dict]:
    units = {}
    for path in sorted(directory.glob("*.parquet")):
        unit = _file_unit(path, cutoff, track_all_null)
        if unit is not None:
            units[path.stem] = unit
    return units


def _universe_units(directory: Path, cutoff: pd.Timestamp, status_symbols: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """One unit per year file <= cutoff year. Key columns always; status columns only for the delisted-at-snapshot symbols."""
    units = {}
    for path in sorted(directory.glob("*.parquet")):
        if not path.stem.isdigit() or int(path.stem) > cutoff.year:
            continue
        frame = pd.read_parquet(path)
        if status_symbols is None:
            inactive = sorted(frame.loc[pd.to_datetime(frame["LAST_DATE"]) < cutoff, "SYMBOL"])
        else:
            inactive = status_symbols.get(path.stem, [])
        unit = _digest_frame(frame[UNIVERSE_KEY_COLUMNS], "SYMBOL")
        sub = frame[frame["SYMBOL"].isin(inactive)][["SYMBOL", *UNIVERSE_STATUS_COLUMNS]]
        unit["status"] = _digest_frame(sub, "SYMBOL") if len(sub) else None
        unit["status_symbols"] = list(inactive)
        units[path.stem] = unit
    return units


def _indicator_units(reader, cutoff_text: str, cutoff: pd.Timestamp) -> dict[str, dict]:
    units = {}
    for symbol, frame in reader(cutoff_text):
        frame, col = _history(frame, cutoff)
        if len(frame):
            units[symbol] = _digest_frame(frame, "date")
    return units


def dataset_names(src: Sources, include_indicators: bool = True) -> list[str]:
    names = []
    if src.actions_dir.is_dir():
        names.append("corporate_actions")
    if src.adjusted_dir.is_dir():
        names.append("adjusted")
    if src.universe_dir.is_dir():
        names.append("universe")
    if src.features_dir.is_dir():
        names += [f"features_{p.name}" for p in sorted(src.features_dir.iterdir()) if p.is_dir()]
    if include_indicators and src.indicator_reader is not None:
        names.append("indicators")
    return names


def compute_units(name: str, src: Sources, cutoff_text: str, status_symbols: dict[str, list[str]] | None = None) -> dict[str, dict]:
    cutoff = pd.Timestamp(cutoff_text)
    if name == "corporate_actions":
        return _dir_units(src.actions_dir, cutoff, track_all_null=True)
    if name == "adjusted":
        return _dir_units(src.adjusted_dir, cutoff)
    if name == "universe":
        return _universe_units(src.universe_dir, cutoff, status_symbols)
    if name.startswith("features_"):
        return _dir_units(src.features_dir / name[len("features_"):], cutoff)
    if name == "indicators":
        if src.indicator_reader is None:
            raise RuntimeError("indicators snapshot exists but no indicator reader is available (--skip-db?)")
        return _indicator_units(src.indicator_reader, cutoff_text, cutoff)
    raise ValueError(f"unknown dataset {name!r}")


# ----------------------------------------------------------------------------------------------- snapshot
def snapshot(cutoff: str, src: Sources, force: bool = False) -> list[str]:
    """Write one JSON per dataset plus _index.json. Refuses (FileExistsError, nothing written) if any of them exists, unless
    force: a second snapshot after a refresh would silently re-baseline the very history it is meant to protect."""
    pd.Timestamp(cutoff)  # validates
    names = dataset_names(src)
    wanted = [src.out_dir / f"{n}.json" for n in names] + [src.out_dir / f"{INDEX_NAME}.json"]
    existing = [p.name for p in wanted if p.exists()]
    if existing and not force:
        raise FileExistsError(f"snapshot already exists in {src.out_dir} ({existing[:6]}); refusing to re-baseline. "
                              "Pass --force only if you mean to accept the current history as the new baseline.")
    src.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        units = compute_units(name, src, cutoff)
        payload = {"dataset": name, "cutoff": cutoff, "units": units}
        (src.out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        logger.info("snapshot {}: {:,} unit(s), {:,} row(s) <= {}", name, len(units), sum(u["rows"] for u in units.values()), cutoff)
        written.append(name)
    index = {"cutoff": cutoff, "created": datetime.now(timezone.utc).isoformat(), "datasets": written,
             "skipped": [] if src.indicator_reader is not None else ["indicators"]}
    (src.out_dir / f"{INDEX_NAME}.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return written


# ----------------------------------------------------------------------------------------------- verify
@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    kinds: dict[str, int] = field(default_factory=lambda: {"new-action adjusted": 0, "empty-column type": 0, "owner-approved": 0})

    def allow(self, kind: str, line: str) -> None:
        self.allowed.append(line)
        self.kinds[kind] += 1


def new_action_symbols(actions_dir: Path, cutoff: pd.Timestamp) -> set[str]:
    """Symbols with an action dated after the cutoff and a usable adjustment_factor (the only kind that changes adjusted prices)."""
    out = set()
    if not actions_dir.is_dir():
        return out
    for path in sorted(actions_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["ex_date", "adjustment_factor"])
        if ((pd.to_datetime(df["ex_date"]) > cutoff) & df["adjustment_factor"].notna()).any():
            out.add(path.stem)
    return out


def _compare_units(name: str, key: str, old: dict | None, new: dict | None, allowed_adj: set[str], adj_cols: frozenset[str],
                   report: Report) -> None:
    """allowed_adj: symbols with a new factor-bearing action. Only `adjusted` ADJ_* differences for them are allowed; for any
    other failing unit of such a symbol the line is labelled (informational, still a failure)."""
    where = f"{name}:{key}"
    before = len(report.failures)
    try:
        _compare_units_inner(name, key, where, old, new, allowed_adj, adj_cols, report)
    finally:
        if key in allowed_adj:
            for i in range(before, len(report.failures)):
                report.failures[i] += f" {NEW_ACTION_LABEL}"


def _empty_column_type_changes(old: dict, new: dict, changed: list[str]) -> list[str]:
    """corporate_actions only. A changed column is excusable iff the snapshot digest is that of an all-null column of the snapshot's
    row count (in any of the representations) AND the column holds no value in the history rows now (new unit's all_null)."""
    shapes = _all_null_digests(old["rows"])
    now_empty = set(new.get("all_null", []))
    return [c for c in changed if old["columns"][c] in shapes and c in now_empty]


def _compare_units_inner(name, key, where, old, new, allowed_adj, adj_cols, report) -> None:
    if old is None:
        report.failures.append(f"{where}: unit has rows <= cutoff now but did not at snapshot ({new['rows']} rows added)")
        return
    if new is None:
        report.failures.append(f"{where}: unit missing or has no rows <= cutoff now ({old['rows']} rows at snapshot)")
        return
    if old["rows"] != new["rows"]:
        report.failures.append(f"{where}: row count {old['rows']} -> {new['rows']} (rows <= cutoff added or removed)")
        return
    if old["keys"] != new["keys"]:
        report.failures.append(f"{where}: date column differs (same row count, different dates)")
        return
    if set(old["columns"]) != set(new["columns"]):
        report.failures.append(f"{where}: column set changed {sorted(set(old['columns']) ^ set(new['columns']))}")
        return
    changed = sorted(c for c in old["columns"] if old["columns"][c] != new["columns"][c])
    if name == "corporate_actions" and changed:
        empty = _empty_column_type_changes(old, new, changed)
        if empty:
            report.allow("empty-column type", f"{where}: empty column changed stored type {empty}")
            changed = [c for c in changed if c not in empty]
    if "status" in old or "status" in new:
        if old.get("status") != new.get("status"):
            changed.append("<delisted-status columns>")
    if not changed:
        return
    if name == "adjusted" and key in allowed_adj and all(c in adj_cols for c in changed):
        report.allow("new-action adjusted", f"{where}: adjusted column(s) {changed} changed; symbol has a new corporate action after the cutoff")
    else:
        report.failures.append(f"{where}: column(s) changed {changed}")


def verify(src: Sources, only: list[str] | None = None, skip_db: bool = False) -> Report:
    """Recompute and compare. Fails if the index is missing, if any indexed dataset has no snapshot file or cannot be
    enumerated now, or if a dataset exists now that the index does not list. indicators may only be absent when the index says
    it was skipped AND skip_db is set."""
    report = Report()
    if skip_db:
        src = dataclasses.replace(src, indicator_reader=None)
    index_path = src.out_dir / f"{INDEX_NAME}.json"
    if not index_path.exists():
        report.failures.append(f"no snapshot index at {index_path}; run `snapshot` first")
        return report
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed, skipped = set(index["datasets"]), set(index.get("skipped", []))
    now_names = set(dataset_names(src))
    names = sorted(indexed | now_names)
    if only:
        names = [n for n in names if n in only]
    for n in names:
        if n in now_names and n not in indexed:
            hint = " (snapshot skipped indicators; verify with --skip-db)" if n == "indicators" and n in skipped else ""
            report.failures.append(f"{n}: exists now but is not in the snapshot index{hint}")
        elif n in indexed and n not in now_names:
            hint = " (verify was run with --skip-db but the snapshot includes indicators)" if n == "indicators" else ""
            report.failures.append(f"{n}: in the snapshot index but cannot be enumerated now (missing directory or reader){hint}")
        elif n in indexed and not (src.out_dir / f"{n}.json").exists():
            report.failures.append(f"{n}: snapshot file {n}.json is missing")
    adj_cols = adjusted_columns()
    new_actions = new_action_symbols(src.actions_dir, pd.Timestamp(index["cutoff"]))
    for name in names:
        path = src.out_dir / f"{name}.json"
        if name not in indexed or name not in now_names or not path.exists():
            continue
        snap = json.loads(path.read_text(encoding="utf-8"))
        cutoff = snap["cutoff"]
        status_symbols = {k: u.get("status_symbols", []) for k, u in snap["units"].items()} if name == "universe" else None
        try:
            now = compute_units(name, src, cutoff, status_symbols)
        except Exception as exc:  # a dataset that cannot be recomputed is a failure, never a skip
            report.failures.append(f"{name}: cannot recompute ({type(exc).__name__}: {exc})")
            continue
        for key in sorted(set(snap["units"]) | set(now)):
            _compare_units(name, key, snap["units"].get(key), now.get(key), new_actions, adj_cols, report)
        logger.info("verified {} against snapshot (cutoff {})", name, cutoff)
    return report


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None, sources: Sources | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot and verify history (rows dated <= cutoff).")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    s.add_argument("--skip-db", action="store_true", help="do not snapshot the Postgres indicators table (recorded in the index)")
    s.add_argument("--force", action="store_true", help="overwrite an existing snapshot (re-baselines history; use deliberately)")
    v = sub.add_parser("verify")
    v.add_argument("--only", nargs="*", help="dataset names to verify (default: every indexed dataset)")
    v.add_argument("--skip-db", action="store_true", help="valid only if the snapshot was also made with --skip-db")
    args = parser.parse_args(argv)

    src = sources if sources is not None else default_sources(skip_db=getattr(args, "skip_db", False))
    if args.command == "snapshot":
        try:
            snapshot(args.cutoff, src, force=args.force)
        except FileExistsError as exc:
            logger.error("{}", exc)
            return 1
        return 0

    report = verify(src, args.only, skip_db=args.skip_db)
    for line in report.allowed:
        logger.warning("ALLOWED {}", line)
    for line in report.failures:
        logger.error("FAIL {}", line)
    logger.info("ALLOWED counts: new-action adjusted={} empty-column type={} owner-approved={}",
                report.kinds["new-action adjusted"], report.kinds["empty-column type"], report.kinds["owner-approved"])
    if report.failures:
        logger.error("history guard FAILED: {} difference(s), {} allowed", len(report.failures), len(report.allowed))
        return 1
    logger.success("history guard passed: history unchanged ({} allowed)", len(report.allowed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
