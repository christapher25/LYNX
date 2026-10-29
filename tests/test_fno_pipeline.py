"""
tests/test_fno_pipeline.py

run_fno_pipeline end to end on a tiny synthetic archive: the write path, scoped-run == the same rows of a full run, checkpoint /
resume, and the orphan prune (a stale file for a symbol outside the universe is removed on a full run, never on a scoped one).
Every I/O dependency is redirected into tmp_path; nothing touches the real archive or FNO_FEATURES_DIR.
"""

import json

import numpy as np
import pandas as pd
import pytest

from features.blocks import fno

DAYS = pd.bdate_range("2024-01-01", periods=70)
UNIVERSE = ["AAA", "BBB", "CCC"]                 # CCC is never in F&O; AAA is banned on a few days
NO_BAN_LIST = {DAYS[10], DAYS[40]}               # sessions with no ban file at all
NO_FNO_FILE = {DAYS[25]}


def _write_archive(root):
    bhav, ban = root / "bhav", root / "ban"
    bhav.mkdir(), ban.mkdir()
    rng = np.random.default_rng(3)
    for i, day in enumerate(DAYS):
        if day not in NO_FNO_FILE:
            rows = []
            for symbol in ("AAA", "BBB"):
                base = 1000 + 10 * i + (5 if symbol == "BBB" else 0)
                for expiry, share in ((day + pd.Timedelta(days=5), 0.7), (day + pd.Timedelta(days=35), 0.3)):
                    rows.append((symbol, "FUTSTK", expiry, np.nan, np.nan, day, base * share + rng.integers(0, 9)))
                rows.append((symbol, "OPTSTK", day + pd.Timedelta(days=5), 100.0, "CE", day, 40.0 + i))
                rows.append((symbol, "OPTSTK", day + pd.Timedelta(days=5), 100.0, "PE", day, 30.0 + i))
            pd.DataFrame(rows, columns=["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE", "DATE", "OPEN_INT"]) \
                .to_parquet(bhav / f"{day:%Y-%m-%d}.parquet", index=False)
        if day not in NO_BAN_LIST:
            names = ["AAA"] if 20 <= i < 30 else [None]                       # None = a NIL list
            pd.DataFrame({"DATE": day, "SYMBOL": pd.array(names, dtype="string")}).to_parquet(ban / f"{day:%Y-%m-%d}.parquet", index=False)
    return bhav, ban


@pytest.fixture
def world(tmp_path, monkeypatch):
    bhav, ban = _write_archive(tmp_path)
    real_summaries, real_bans = fno.load_fno_summaries, fno.load_ban_lists
    monkeypatch.setattr(fno, "load_fno_summaries", lambda start=None, end=None: real_summaries(start, end, directory=bhav))
    monkeypatch.setattr(fno, "load_ban_lists", lambda start=None, end=None: real_bans(start, end, directory=ban))
    monkeypatch.setattr(fno, "list_universe_symbols", lambda: list(UNIVERSE))
    monkeypatch.setattr(fno, "load_price_panel", lambda symbol: pd.DataFrame({"date": DAYS, "volume_adj_factor": 1.0}))
    monkeypatch.setattr(fno, "FNO_FEATURES_DIR", tmp_path / "out")
    monkeypatch.setattr(fno, "FNO_FEATURES_CHECKPOINT", tmp_path / "checkpoint.json")
    return tmp_path


def _read(directory, symbol):
    return pd.read_parquet(directory / f"{symbol}.parquet")


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return all((a[c].to_numpy() == b[c].to_numpy()).all() if a[c].dtype in (bool, object) or c == "symbol" or c == "date"
               else np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True) for c in a.columns)


class TestWritePath:
    def test_one_file_per_symbol_row_aligned_with_the_spine(self, world):
        summary = fno.run_fno_pipeline()
        out = world / "out"
        assert sorted(p.stem for p in out.glob("*.parquet")) == UNIVERSE
        assert summary["symbols_written"] == 3 and summary["rows"] == 3 * len(DAYS)
        for symbol in UNIVERSE:
            frame = _read(out, symbol)
            assert list(frame.columns) == fno.OUTPUT_COLUMNS and list(frame["date"]) == list(DAYS) and (frame["symbol"] == symbol).all()

    def test_values_reflect_the_archive(self, world):
        fno.run_fno_pipeline()
        aaa, ccc = _read(world / "out", "AAA"), _read(world / "out", "CCC")
        gap = aaa.set_index("date").loc[DAYS[25]]
        assert gap["fno_day_unavailable"] and gap["is_missing_fno"] and np.isnan(gap["fut_oi"])      # no file: NaN + flagged
        assert ccc["is_missing_fno"].all() and not ccc["fno_day_unavailable"].iloc[:25].any()          # not in F&O: flagged, file exists
        no_list = aaa.set_index("date").loc[DAYS[10]]
        assert no_list["is_missing_ban"] and np.isnan(no_list["fno_in_ban"])                           # no list: unavailable, not none
        assert aaa["fno_in_ban"].iloc[20:30].eq(1.0).all()
        assert ccc.set_index("date").loc[DAYS[5], "fno_in_ban"] == 0.0                                  # a cash-only name on a real list
        assert not ccc.set_index("date").loc[DAYS[5], "is_missing_ban"]


class TestScopedEqualsFull:
    START = DAYS[50]

    def test_scoped_rows_equal_the_same_rows_of_a_full_run(self, world):
        fno.run_fno_pipeline()
        fno.run_fno_pipeline(start=self.START, output_dir=world / "scoped")
        for symbol in UNIVERSE:
            full = _read(world / "out", symbol)
            full = full[full["date"] >= self.START].reset_index(drop=True)
            assert len(full) == 20 and _equal(_read(world / "scoped", symbol), full)

    def test_a_scoped_run_that_drops_its_warmup_differs(self, world, monkeypatch):
        """The broken variant: with no warm-up history the trailing windows are wrong, and this test sees it."""
        fno.run_fno_pipeline()
        monkeypatch.setattr(fno, "WARMUP_DAYS", 0)
        fno.run_fno_pipeline(start=self.START, output_dir=world / "scoped")
        full = _read(world / "out", "AAA")
        full = full[full["date"] >= self.START].reset_index(drop=True)
        assert not _equal(_read(world / "scoped", "AAA"), full)

    def test_scoped_run_respects_end_and_symbols(self, world):
        fno.run_fno_pipeline(start=DAYS[50], end=DAYS[55], symbols=["AAA", "NOTINUNIVERSE"], output_dir=world / "scoped")
        assert [p.stem for p in (world / "scoped").glob("*.parquet")] == ["AAA"]
        assert list(_read(world / "scoped", "AAA")["date"]) == list(DAYS[50:56])

    def test_scoped_run_never_touches_the_checkpoint_or_real_output(self, world):
        fno.run_fno_pipeline(start=DAYS[50], output_dir=world / "scoped")
        assert not (world / "checkpoint.json").exists() and not (world / "out").exists()


class TestCheckpointResume:
    def test_checkpoint_lists_every_symbol(self, world):
        fno.run_fno_pipeline()
        assert json.loads((world / "checkpoint.json").read_text())["completed_symbols"] == UNIVERSE

    def test_resume_writes_only_what_is_missing(self, world):
        fno.run_fno_pipeline()
        (world / "out" / "BBB.parquet").unlink()
        (world / "checkpoint.json").write_text(json.dumps({"completed_symbols": ["AAA", "CCC"]}))
        before = (world / "out" / "AAA.parquet").stat().st_mtime_ns
        summary = fno.run_fno_pipeline(resume=True)
        assert summary["symbols_written"] == 1 and (world / "out" / "BBB.parquet").exists()
        assert (world / "out" / "AAA.parquet").stat().st_mtime_ns == before

    def test_no_resume_rewrites_everything(self, world):
        fno.run_fno_pipeline()
        assert fno.run_fno_pipeline(resume=False)["symbols_written"] == 3

    def test_a_completed_resume_has_nothing_left_to_do(self, world):
        fno.run_fno_pipeline()
        assert fno.run_fno_pipeline(resume=True)["symbols_written"] == 0


class TestOrphanPrune:
    def _stale(self, world):
        out = world / "out"
        out.mkdir(exist_ok=True)
        pd.DataFrame({"symbol": ["GONE"]}).to_parquet(out / "GONE.parquet")
        return out / "GONE.parquet"

    def test_full_run_removes_a_symbol_that_left_the_universe(self, world):
        stale = self._stale(world)
        fno.run_fno_pipeline()
        assert not stale.exists()

    def test_the_prune_test_can_fail(self, world, monkeypatch):
        """Broken variant: with the prune disabled the stale file survives, so the test above is sensitive to it."""
        stale = self._stale(world)
        monkeypatch.setattr(fno, "prune_orphaned_symbol_files", lambda *a, **k: [])
        fno.run_fno_pipeline()
        assert stale.exists()

    def test_scoped_run_never_prunes(self, world):
        stale = self._stale(world)
        fno.run_fno_pipeline(start=DAYS[50], symbols=["AAA"], output_dir=world / "out")
        assert stale.exists()


class TestVerifierStaleCarryInjection:
    """The checkpoint verifier's injected stale-carry-forward bug must be able to fire: it must target gaps that follow an
    observed OI row (leading pre-listing gaps have nothing to carry and made the injection a no-op on real history)."""

    @staticmethod
    def _frame():
        from scripts import verify_stage10_fno_features as v
        n = 12
        frame = pd.DataFrame({c: np.nan for c in v.OUTPUT_COLUMNS}, index=range(n))
        frame["symbol"] = "AAA"
        frame["date"] = pd.bdate_range("2024-01-01", periods=n)
        missing = [True, True, True, False, False, False, True, False, True, False, False, False]   # 3 leading gaps
        frame["is_missing_fno"] = missing
        frame["fno_day_unavailable"] = [False] * 8 + [True] + [False] * 3
        frame["is_missing_ban"] = True
        for c in v.OI_FEATURES:
            frame[c] = np.where(frame["is_missing_fno"], np.nan, np.arange(n) + 1.0)
        return v, frame

    @staticmethod
    def _want(v, frame, dates):
        """What an independent recompute reports for a clean frame: exactly the stored row (the gaps are NaN + flagged)."""
        rows = frame.set_index("date")
        return {("AAA", d): {c: rows.loc[d, c] for c in v.VALUE_COLUMNS} for d in dates}

    def test_gap_dates_exclude_leading_gaps(self):
        v, frame = self._frame()
        dates = v.stale_gap_dates(frame)
        assert set(dates) == {frame["date"][6], frame["date"][8]}

    def test_injected_ffill_is_caught_and_clean_passes(self):
        v, frame = self._frame()
        want = self._want(v, frame, v.stale_gap_dates(frame))
        assert v.compare_violations({"AAA": frame}, want) == 0
        stale = frame.copy()
        stale[v.OI_FEATURES] = stale[v.OI_FEATURES].ffill()
        assert v.compare_violations({"AAA": stale}, want) > 0

    def test_old_selection_would_have_missed_it(self):
        """Broken variant: choosing the first missing rows (the old rule) picks only leading gaps, where ffill is a no-op."""
        v, frame = self._frame()
        old = list(frame.loc[frame["is_missing_fno"], "date"])[:3]
        want = self._want(v, frame, old)
        stale = frame.copy()
        stale[v.OI_FEATURES] = stale[v.OI_FEATURES].ffill()
        assert v.compare_violations({"AAA": stale}, want) == 0
