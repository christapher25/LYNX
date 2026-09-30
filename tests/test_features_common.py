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
