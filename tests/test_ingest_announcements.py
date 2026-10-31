"""
tests/test_ingest_announcements.py

Parsing, chunking, checkpoint/resume and failure handling for the NSE corporate-announcements ingester. Network is never
touched: fetching is injected.
"""

import json
from datetime import date

import pandas as pd
import pytest

from data import ingest_announcements as ia
from data.entity_resolution import SymbolMaster

MASTER = SymbolMaster.from_frame(pd.DataFrame({
    "SYMBOL": ["TCS", "INFY"], "ISIN": ["INE467B01029", "INE009A01021"],
    "FIRST_DATE": pd.to_datetime(["2004-08-25", "1995-01-01"]), "LAST_DATE": pd.to_datetime(["2026-09-29"] * 2),
}))


def raw(symbol="TCS", an_dt="02-Jan-2020 18:30:15", diss="02-Jan-2020 18:30:20", **over):
    base = {
        "symbol": symbol, "sm_name": "Tata Consultancy Services Limited", "sm_isin": "INE467B01029",
        "an_dt": an_dt, "exchdisstime": diss, "desc": "Outcome of Board Meeting",
        "attchmntText": "TCS informed the Exchange about results", "attchmntFile": "https://x/y.pdf",
        "seq_id": "1", "hasXbrl": True, "smIndustry": "IT", "sort_date": "2020-01-02 18:30:15",
        "dt": "02012020183015", "difference": "00:00:05", "bflag": None, "csvName": None, "old_new": None,
        "orgid": None, "fileSize": "1 MB", "attFileSize": "1 MB",
    }
    base.update(over)
    return base


class TestChunks:
    def test_month_aligned_and_clamped(self):
        got = ia.month_chunks(date(2020, 1, 15), date(2020, 3, 10))
        assert got == [(date(2020, 1, 15), date(2020, 1, 31)), (date(2020, 2, 1), date(2020, 2, 29)),
                       (date(2020, 3, 1), date(2020, 3, 10))]

    def test_single_day(self):
        assert ia.month_chunks(date(2020, 1, 2), date(2020, 1, 2)) == [(date(2020, 1, 2), date(2020, 1, 2))]

    def test_year_boundary(self):
        assert ia.month_chunks(date(2019, 12, 30), date(2020, 1, 2)) == [
            (date(2019, 12, 30), date(2019, 12, 31)), (date(2020, 1, 1), date(2020, 1, 2))]


class TestTimestamps:
    def test_later_of_the_two_stamps_is_publication_time(self):
        ts, prec = ia.parse_published_at("02-Jan-2020 18:30:15", "02-Jan-2020 18:30:20")
        assert prec == "second" and ts.strftime("%H:%M:%S") == "18:30:20" and str(ts.tzinfo) != "None"

    def test_dash_dissemination_time_falls_back_to_an_dt(self):
        ts, prec = ia.parse_published_at("02-Jan-2020 17:01:00", "-")
        assert prec == "second" and ts.strftime("%H:%M") == "17:01"

    def test_timestamp_is_ist(self):
        ts, _ = ia.parse_published_at("02-Jan-2020 18:30:15", None)
        assert ts.utcoffset().total_seconds() == 5.5 * 3600

    def test_date_only_stamp_is_pushed_to_end_of_day(self):
        ts, prec = ia.parse_published_at("06-Jan-2003 00:00:00", "-")
        assert prec == "date" and ts.strftime("%H:%M:%S") == "23:59:59"

    def test_unparseable_stamp_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError):
            ia.parse_published_at("garbage", "-")
        with pytest.raises(ValueError):
            ia.parse_published_at(None, None)


class TestParse:
    def test_columns_and_nan_for_dashes(self):
        df = ia.parse_announcements([raw(attchmntFile="-", fileSize="-")], date(2020, 1, 2), date(2020, 1, 2))
        r = df.iloc[0]
        assert pd.isna(r["ATTACHMENT_URL"]) and r["PUBLISHED_DATE"] == pd.Timestamp("2020-01-02")
        assert r["SYMBOL_RAW"] == "TCS" and r["ISIN"] == "INE467B01029" and r["CATEGORY"] == "Outcome of Board Meeting"

    def test_record_outside_requested_range_fails_the_chunk(self):
        with pytest.raises(ValueError, match="outside"):
            ia.parse_announcements([raw(an_dt="03-Jan-2020 09:00:00", diss="-")], date(2020, 1, 2), date(2020, 1, 2))

    def test_exact_duplicates_dropped_distinct_kept(self):
        df = ia.parse_announcements([raw(), raw(), raw(seq_id="2")], date(2020, 1, 2), date(2020, 1, 2))
        assert len(df) == 2

    def test_non_list_payload_rejected(self):
        with pytest.raises(ValueError):
            ia.parse_announcements({"error": "blocked"}, date(2020, 1, 2), date(2020, 1, 2))

    def test_empty_payload_gives_empty_frame_with_schema(self):
        df = ia.parse_announcements([], date(2020, 1, 2), date(2020, 1, 2))
        assert df.empty and "PUBLISHED_AT" in df.columns


class TestBuildChunk:
    def test_resolved_and_unresolved_are_both_kept(self):
        payload = [raw(), raw(symbol="-", sm_isin="-", sm_name="Mystery Ltd", seq_id="9")]
        df = ia.build_chunk_frame(payload, date(2020, 1, 2), date(2020, 1, 2), MASTER)
        assert len(df) == 2 and df["TICKER"].notna().sum() == 1
        assert df.loc[df["TICKER"].isna(), "RESOLUTION_METHOD"].iloc[0] == "unresolved"
        assert ia.announcement_invariant_violations(df, MASTER) == {}


class TestRunResume:
    def _payload_for(self, start, end):
        d = start.strftime("%d-%b-%Y")
        return [raw(an_dt=f"{d} 10:00:00", diss="-", seq_id=str(start))]

    def test_writes_manifest_and_resumes_without_refetching(self, tmp_path):
        calls = []

        def fetch(s, e):
            calls.append((s, e))
            return self._payload_for(s, e), 123

        kw = dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, fetch_fn=fetch, delay=0)
        res = ia.run(date(2020, 1, 2), date(2020, 2, 3), **kw)
        assert res["ok"] == 2 and len(calls) == 2
        assert sorted(p.name for p in tmp_path.glob("announcements_*.parquet")) == [
            "announcements_2020-01-02_2020-01-31.parquet", "announcements_2020-02-01_2020-02-03.parquet"]
        recs = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
        assert [r["status"] for r in recs] == ["ok", "ok"] and all(r["rows"] == 1 for r in recs)

        res2 = ia.run(date(2020, 1, 2), date(2020, 2, 3), **kw)
        assert res2["skipped"] == 2 and len(calls) == 2            # nothing refetched

    def test_failure_is_recorded_and_retried_next_run_only_for_that_chunk(self, tmp_path):
        state = {"fail": True}

        def fetch(s, e):
            if s.month == 2 and state["fail"]:
                raise ConnectionError("boom")
            return self._payload_for(s, e), 1

        kw = dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, fetch_fn=fetch, delay=0)
        res = ia.run(date(2020, 1, 2), date(2020, 2, 3), **kw)
        assert res["ok"] == 1 and res["failed"] == 1
        state["fail"] = False
        res2 = ia.run(date(2020, 1, 2), date(2020, 2, 3), **kw)
        assert res2["ok"] == 1 and res2["skipped"] == 1

    def test_empty_chunk_is_not_written_and_is_retried(self, tmp_path):
        kw = dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, fetch_fn=lambda s, e: ([], 2), delay=0)
        res = ia.run(date(2020, 1, 2), date(2020, 1, 5), **kw)
        assert res["empty"] == 1 and not list(tmp_path.glob("announcements_*.parquet"))

    def test_out_of_range_record_fails_chunk_and_writes_nothing(self, tmp_path):
        kw = dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, delay=0,
                  fetch_fn=lambda s, e: ([raw(an_dt="09-Mar-2021 10:00:00", diss="-")], 1))
        res = ia.run(date(2020, 1, 2), date(2020, 1, 5), **kw)
        assert res["failed"] == 1 and not list(tmp_path.glob("announcements_*.parquet"))

    def test_rewriting_a_grown_month_removes_the_superseded_partial_file(self, tmp_path):
        kw = dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, delay=0,
                  fetch_fn=lambda s, e: (self._payload_for(s, e), 1))
        ia.run(date(2020, 3, 1), date(2020, 3, 10), **kw)
        ia.run(date(2020, 3, 1), date(2020, 3, 20), **kw)
        assert [p.name for p in tmp_path.glob("announcements_*.parquet")] == ["announcements_2020-03-01_2020-03-20.parquet"]


class TestHasXbrl:
    @pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("true", True), ("false", False),
                                                ("False", False), ("TRUE", True)])
    def test_explicit_values(self, value, expected):
        df = ia.parse_announcements([raw(hasXbrl=value)], date(2020, 1, 2), date(2020, 1, 2))
        assert df["HAS_XBRL"].iloc[0] == expected

    @pytest.mark.parametrize("value", [None, "-", ""])
    def test_missing_is_na_not_false(self, value):
        df = ia.parse_announcements([raw(hasXbrl=value)], date(2020, 1, 2), date(2020, 1, 2))
        assert pd.isna(df["HAS_XBRL"].iloc[0])

    def test_unrecognised_string_is_an_error(self):
        with pytest.raises(ValueError):
            ia.parse_announcements([raw(hasXbrl="maybe")], date(2020, 1, 2), date(2020, 1, 2))


class TestOverlap:
    def _kw(self, tmp_path, n=1, calls=None):
        def fetch(s, e):
            if calls is not None:
                calls.append((s, e))
            return [raw(an_dt=f"{s.strftime('%d-%b-%Y')} 10:00:00", diss="-", seq_id=f"{s}-{i}") for i in range(n)], 1
        return dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, fetch_fn=fetch, delay=0)

    def test_week_inside_existing_month_is_skipped_not_duplicated(self, tmp_path):
        calls = []
        ia.run(date(2020, 3, 1), date(2020, 3, 31), **self._kw(tmp_path, calls=calls))
        res = ia.run(date(2020, 3, 9), date(2020, 3, 15), **self._kw(tmp_path, calls=calls))
        assert res["skipped"] == 1 and len(calls) == 1
        assert len(list(tmp_path.glob("announcements_*.parquet"))) == 1

    def test_month_after_week_supersedes_the_week(self, tmp_path):
        ia.run(date(2020, 3, 9), date(2020, 3, 15), **self._kw(tmp_path))
        res = ia.run(date(2020, 3, 1), date(2020, 3, 31), **self._kw(tmp_path))
        assert res["ok"] == 1
        assert [p.name for p in tmp_path.glob("announcements_*.parquet")] == ["announcements_2020-03-01_2020-03-31.parquet"]

    def test_partial_overlap_fails_loudly_and_writes_nothing(self, tmp_path):
        ia.run(date(2020, 3, 1), date(2020, 3, 10), **self._kw(tmp_path))
        res = ia.run(date(2020, 3, 5), date(2020, 3, 20), **self._kw(tmp_path))
        assert res["failed"] == 1 and res["ok"] == 0
        assert [p.name for p in tmp_path.glob("announcements_*.parquet")] == ["announcements_2020-03-01_2020-03-10.parquet"]


class TestNeverReplaceWithFewerRows:
    def test_smaller_replacement_aborts_and_keeps_existing_file(self, tmp_path):
        def kw(n):
            return dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, delay=0,
                        fetch_fn=lambda s, e: ([raw(an_dt=f"{s.strftime('%d-%b-%Y')} 10:00:00", diss="-", seq_id=str(i))
                                                for i in range(n)], 1))
        ia.run(date(2020, 3, 9), date(2020, 3, 15), **kw(3))
        res = ia.run(date(2020, 3, 1), date(2020, 3, 31), **kw(1))
        assert res["failed"] == 1 and res["ok"] == 0
        files = [p.name for p in tmp_path.glob("announcements_*.parquet")]
        assert files == ["announcements_2020-03-09_2020-03-15.parquet"]
        assert len(pd.read_parquet(tmp_path / files[0])) == 3


class TestFetchGuards:
    def test_html_or_error_body_is_rejected(self):
        class Resp:
            status_code = 200
            content = b"<html>Access Denied</html>"
            def json(self): raise ValueError("no json")
        with pytest.raises(ValueError):
            ia.decode_payload(Resp())

    def test_json_list_accepted(self):
        class Resp:
            status_code = 200
            content = b"[]"
            def json(self): return []
        assert ia.decode_payload(Resp()) == []


class TestCrashRecovery:
    def _kw(self, tmp_path, n):
        return dict(out_dir=tmp_path, manifest_path=tmp_path / "m.jsonl", master=MASTER, delay=0,
                    fetch_fn=lambda s, e: ([raw(an_dt=f"{s.strftime('%d-%b-%Y')} 10:00:00", diss="-", seq_id=str(i))
                                            for i in range(n)], 1))

    def _frame(self, n):
        return pd.DataFrame({"x": range(n)})

    def test_covered_run_removes_leftover_inside_files(self, tmp_path):
        # state after a crash between the atomic write of the month and the unlink of the week
        self._frame(5).to_parquet(tmp_path / "announcements_2020-03-01_2020-03-31.parquet")
        self._frame(3).to_parquet(tmp_path / "announcements_2020-03-09_2020-03-15.parquet")
        res = ia.run(date(2020, 3, 1), date(2020, 3, 31), **self._kw(tmp_path, 1))
        assert res["skipped"] == 1
        assert [p.name for p in tmp_path.glob("announcements_*.parquet")] == ["announcements_2020-03-01_2020-03-31.parquet"]

    def test_leftover_kept_when_covering_file_has_fewer_rows(self, tmp_path):
        self._frame(2).to_parquet(tmp_path / "announcements_2020-03-01_2020-03-31.parquet")
        self._frame(3).to_parquet(tmp_path / "announcements_2020-03-09_2020-03-15.parquet")
        ia.run(date(2020, 3, 1), date(2020, 3, 31), **self._kw(tmp_path, 1))
        assert len(list(tmp_path.glob("announcements_*.parquet"))) == 2

    def test_fewer_rows_guard_compares_against_the_sum_of_superseded_files(self, tmp_path):
        ia.run(date(2020, 3, 2), date(2020, 3, 8), **self._kw(tmp_path, 3))
        ia.run(date(2020, 3, 9), date(2020, 3, 15), **self._kw(tmp_path, 3))
        res = ia.run(date(2020, 3, 1), date(2020, 3, 31), **self._kw(tmp_path, 4))     # 4 < 3 + 3, though 4 > 3
        assert res["failed"] == 1 and res["ok"] == 0
        assert len(list(tmp_path.glob("announcements_*.parquet"))) == 2
