"""
tests/test_features_common.py

Tests for features/blocks/common.py's shared helpers.
"""

from datetime import date

import pandas as pd
import pytest

from features.blocks.common import prune_orphaned_symbol_files, validated_sessions


class TestValidatedSessions:
    def test_sorted_ascending_no_duplicates_passes(self):
        out = validated_sessions([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
        assert list(out) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]

    def test_a_duplicate_raises(self):
        with pytest.raises(ValueError, match="sorted ascending"):
            validated_sessions([date(2024, 1, 1), date(2024, 1, 1)])

    def test_out_of_order_raises(self):
        with pytest.raises(ValueError, match="sorted ascending"):
            validated_sessions([date(2024, 1, 2), date(2024, 1, 1)])


class TestPruneOrphanedSymbolFiles:
    """
    A symbol that legitimately falls out of the universe (e.g. a Stage 5 rebuild with corrected
    point-in-time data shifts trailing-turnover rankings enough to drop it from every year's top-700)
    must not leave its last-computed feature file sitting on disk forever with no signal it is stale.
    Found 2026-09-30: RATNAVEER's delivery/pledging/circuits Parquet files all survived a --no-resume
    rebuild untouched, because each pipeline only overwrites files for symbols in the CURRENT universe --
    it never deletes one for a symbol that fell out. The checkpoint's own "one file per universe symbol,
    no extras" check caught it in all three blocks at once. Mirrors
    signals.indicators.prune_orphaned_symbols, same day, same root cause, a different storage layer.
    """

    def test_a_symbol_no_longer_in_the_universe_is_deleted(self, tmp_path):
        for sym in ("RELIANCE", "TCS", "RATNAVEER"):
            (tmp_path / f"{sym}.parquet").write_bytes(b"")
        pruned = prune_orphaned_symbol_files(tmp_path, ["RELIANCE", "TCS"])
        assert pruned == ["RATNAVEER"]
        assert not (tmp_path / "RATNAVEER.parquet").exists()

    def test_current_universe_symbols_are_never_deleted(self, tmp_path):
        for sym in ("RELIANCE", "TCS"):
            (tmp_path / f"{sym}.parquet").write_bytes(b"")
        prune_orphaned_symbol_files(tmp_path, ["RELIANCE", "TCS"])
        assert (tmp_path / "RELIANCE.parquet").exists()
        assert (tmp_path / "TCS.parquet").exists()

    def test_nothing_orphaned_is_a_no_op(self, tmp_path):
        (tmp_path / "RELIANCE.parquet").write_bytes(b"")
        pruned = prune_orphaned_symbol_files(tmp_path, ["RELIANCE", "TCS", "INFY"])
        assert pruned == []
        assert (tmp_path / "RELIANCE.parquet").exists()

    def test_multiple_orphans_are_all_deleted_and_returned_sorted(self, tmp_path):
        for sym in ("ZULU", "RELIANCE", "ALPHA"):
            (tmp_path / f"{sym}.parquet").write_bytes(b"")
        pruned = prune_orphaned_symbol_files(tmp_path, ["RELIANCE"])
        assert pruned == ["ALPHA", "ZULU"]
        assert (tmp_path / "RELIANCE.parquet").exists()

    def test_non_parquet_files_are_ignored(self, tmp_path):
        (tmp_path / "RELIANCE.parquet").write_bytes(b"")
        (tmp_path / "README.md").write_text("not a symbol file")
        pruned = prune_orphaned_symbol_files(tmp_path, ["RELIANCE"])
        assert pruned == []
        assert (tmp_path / "README.md").exists()


class TestListOrphansOnly:
    """--list-orphans-only: report orphans, delete nothing. The default (deleting) behaviour is covered above."""

    def test_list_only_keeps_the_orphan_file_and_reports_it(self, tmp_path):
        for sym in ("RELIANCE", "RATNAVEER"):
            (tmp_path / f"{sym}.parquet").write_bytes(b"x")
        orphans = prune_orphaned_symbol_files(tmp_path, ["RELIANCE"], list_only=True)
        assert orphans == ["RATNAVEER"]
        assert (tmp_path / "RATNAVEER.parquet").read_bytes() == b"x"
        assert (tmp_path / "RELIANCE.parquet").exists()

    def test_list_only_logs_the_orphans(self, tmp_path):
        from loguru import logger
        (tmp_path / "RATNAVEER.parquet").write_bytes(b"")
        msgs = []
        sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
        try:
            prune_orphaned_symbol_files(tmp_path, [], list_only=True)
        finally:
            logger.remove(sink)
        assert any("RATNAVEER" in m for m in msgs)

    def test_default_still_deletes(self, tmp_path):
        (tmp_path / "RATNAVEER.parquet").write_bytes(b"")
        assert prune_orphaned_symbol_files(tmp_path, []) == ["RATNAVEER"]
        assert not (tmp_path / "RATNAVEER.parquet").exists()


import inspect as _inspect
import subprocess as _subprocess
import sys as _sys

import pytest as _pytest

_BLOCKS = ["delivery", "pledging", "circuits", "fno"]


class TestPipelinesExposeTheFlag:
    @_pytest.mark.parametrize("block", _BLOCKS)
    def test_runner_has_list_orphans_only_defaulting_false(self, block):
        import importlib
        mod = importlib.import_module(f"features.blocks.{block}")
        fn = getattr(mod, f"run_{block}_pipeline")
        param = _inspect.signature(fn).parameters["list_orphans_only"]
        assert param.default is False

    @_pytest.mark.parametrize("block", _BLOCKS)
    def test_cli_accepts_the_exact_flag_and_forwards_list_only(self, block):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "features" / "blocks" / f"{block}.py").read_text(encoding="utf-8")
        assert '"--list-orphans-only"' in src
        assert "list_only=list_orphans_only" in src

    @_pytest.mark.parametrize("block", _BLOCKS)
    def test_help_lists_the_flag(self, block):
        out = _subprocess.run([_sys.executable, "-m", f"features.blocks.{block}", "--help"],
                              capture_output=True, text=True, timeout=120)
        assert out.returncode == 0 and "--list-orphans-only" in out.stdout


class TestFlagHelpTextIsExplicit:
    @_pytest.mark.parametrize("module", [f"features.blocks.{b}" for b in _BLOCKS] + ["signals.indicators"])
    def test_help_says_orphans_are_listed_not_deleted_and_processing_continues(self, module):
        out = _subprocess.run([_sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0
        assert "orphans are listed, not deleted; normal processing continues" in " ".join(out.stdout.split())
