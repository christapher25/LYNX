"""
tests/test_ingest_indices.py

Tests for data/ingest_indices.py's CLI. Before this, main() took no arguments at all (always a full backfill),
so there was no way to request just the weekend special sessions (Budget-day Saturdays, Muhurat) that
data/raw/DATA_NOTES.md §10 confirms exist in the F&O archive on 30 dates -- whether the index archive also
serves them was unprobed. Wires --weekends to the same days="weekends" mechanism every other Stage 10 ingester
uses (data/ingest_common.trading_day_candidates), not a new one.
"""

import sys
from datetime import date

from data.ingest_indices import FIRST_AVAILABLE


class TestWeekendsCLI:
    def test_weekends_flag_selects_the_weekend_day_selector(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_indices as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_indices.py", "--weekends", "--days", "5"])
        mod.main()
        assert captured["days"] == "weekends"

    def test_default_is_weekdays(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_indices as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_indices.py"])
        mod.main()
        assert captured["days"] == "weekdays"
        assert captured["start"] == FIRST_AVAILABLE

    def test_days_flag_still_selects_a_recent_window(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_indices as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_indices.py", "--days", "10"])
        mod.main()
        assert (captured["end"] - captured["start"]).days == 10

    def test_start_before_first_available_is_clamped(self, monkeypatch):
        captured = {}

        def fake_run_ingestion(**kwargs):
            captured.update(kwargs)

        import data.ingest_indices as mod
        monkeypatch.setattr(mod, "run_ingestion", fake_run_ingestion)
        monkeypatch.setattr(sys, "argv", ["ingest_indices.py", "--start", "2000-01-01"])
        mod.main()
        assert captured["start"] == FIRST_AVAILABLE
