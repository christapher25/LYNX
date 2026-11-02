"""
tests/test_isin_bridge.py

Row 16: an announcement whose exchange symbol tag does not resolve (renamed company: the NSE API stamps the CURRENT symbol on
old records) is resolved through its ISIN to the symbol that was VALID ON THE ANNOUNCEMENT DATE. Never the current symbol,
never a guess. Each mutant below is a deliberately broken bridge that these tests must reject.
"""

from datetime import date

import pandas as pd
import pytest

from data import isin_bridge
from data.entity_resolution import SymbolMaster, resolve

# OLDNAME -> NEWNAME keeps ISIN A (rename 2015-06-01). TWINA/TWINB overlap on ISIN B (ambiguous). LATE lists 2020-01-01.
FRAME = pd.DataFrame({
    "SYMBOL": ["OLDNAME", "NEWNAME", "TWINA", "TWINB", "LATE", "STABLE"],
    "ISIN": ["INE111A01011", "INE111A01011", "INE222B01012", "INE222B01012", "INE333C01013", "INE444D01014"],
    "FIRST_DATE": pd.to_datetime(["2005-01-03", "2015-06-01", "2010-01-04", "2010-01-04", "2020-01-01", "2000-01-03"]),
    "LAST_DATE": pd.to_datetime(["2015-05-29", "2026-09-29", "2026-09-29", "2026-09-29", "2026-09-29", "2026-09-29"]),
})


@pytest.fixture
def master():
    return SymbolMaster.from_frame(FRAME)


def rec(symbol, isin, day, name="x"):
    return {"SYMBOL_RAW": symbol, "ISIN": isin, "COMPANY_NAME": name, "PUBLISHED_DATE": pd.Timestamp(day)}


def run(master, **kw):
    return resolve(pd.DataFrame([rec(**kw)]), master, {}, use_isin_bridge=True).iloc[0]


class TestPointInTime:
    def test_old_record_maps_to_symbol_valid_then_not_current_symbol(self, master):
        r = run(master, symbol="NEWNAME", isin="INE111A01011", day="2010-03-01")      # API stamped the current name
        assert r["TICKER"] == "OLDNAME" and r["RESOLUTION_METHOD"] == "isin_bridge" and pd.isna(r["UNRESOLVED_REASON"])

    def test_record_after_rename_maps_to_new_symbol(self, master):
        r = run(master, symbol="OLDNAME", isin="INE111A01011", day="2020-03-01")       # old tag on a post-rename record
        assert r["TICKER"] == "NEWNAME" and r["RESOLUTION_METHOD"] == "isin_bridge"

    def test_symbol_that_traded_then_keeps_method_symbol(self, master):
        r = run(master, symbol="NEWNAME", isin="INE111A01011", day="2020-03-01")
        assert r["TICKER"] == "NEWNAME" and r["RESOLUTION_METHOD"] == "symbol"           # bridge never overrides a good tag

    def test_symbol_not_in_master_is_bridged_by_isin(self, master):
        r = run(master, symbol="GHOSTSYM", isin="INE444D01014", day="2020-01-02")
        assert r["TICKER"] == "STABLE" and r["RESOLUTION_METHOD"] == "isin_bridge"


class TestNeverGuess:
    def test_ambiguous_isin_on_a_date_stays_unresolved(self, master):
        r = run(master, symbol="WRONG", isin="INE222B01012", day="2020-01-02")
        assert pd.isna(r["TICKER"]) and r["RESOLUTION_METHOD"] == "unresolved"
        assert r["UNRESOLVED_REASON"] == "bridge_isin_ambiguous"

    def test_record_before_first_listing_stays_unresolved(self, master):
        r = run(master, symbol="LATE", isin="INE333C01013", day="2019-06-01")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "bridge_before_first_listing"

    def test_record_after_last_session_stays_unresolved(self, master):
        r = run(master, symbol="OLDNAME", isin="INE444D01014", day="2030-01-01")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "bridge_after_last_session"

    def test_gap_between_windows_stays_unresolved(self, master):
        r = run(master, symbol="GHOSTSYM", isin="INE111A01011", day="2015-05-31")        # between the two windows
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "bridge_gap_between_windows"

    def test_isin_never_seen_is_unresolved_with_reason(self, master):
        r = run(master, symbol="GHOSTSYM", isin="INE999Z01019", day="2020-01-02")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "bridge_isin_not_in_master"

    def test_no_isin_keeps_original_reason(self, master):
        r = run(master, symbol="GHOSTSYM", isin=None, day="2020-01-02")
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_not_in_master"

    def test_symbol_isin_conflict_is_not_overridden(self, master):
        r = run(master, symbol="STABLE", isin="INE333C01013", day="2020-06-01")         # contradictory identifiers
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_isin_conflict"

    def test_unknown_date_cannot_be_bridged(self, master):
        frame = pd.DataFrame([{"SYMBOL_RAW": "GHOSTSYM", "ISIN": "INE444D01014", "COMPANY_NAME": "x", "PUBLISHED_DATE": pd.NaT}])
        r = resolve(frame, master, {}, use_isin_bridge=True).iloc[0]
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "bridge_no_date"


class TestOffByDefault:
    def test_legacy_resolve_unchanged_without_flag(self, master):
        r = resolve(pd.DataFrame([rec("GHOSTSYM", "INE444D01014", "2020-01-02")]), master, {}).iloc[0]
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_not_in_master"


# ---- mutants: broken bridges the contract tests above must reject -----------------------------------------------------

def _current_symbol_bridge(isin, day, master):
    """Mutant: picks the symbol with the LATEST window (the current symbol), ignoring the date."""
    cands = master.symbols_for_isin(isin)
    return (max(cands, key=lambda c: c[2])[0], None) if cands else (None, "bridge_isin_not_in_master")


def _guessing_bridge(isin, day, master):
    """Mutant: on ambiguity, picks the first candidate."""
    cands = master.symbols_for_isin(isin)
    return (cands[0][0], None) if cands else (None, "bridge_isin_not_in_master")


def _ignore_listing_bridge(isin, day, master):
    """Mutant: ignores the date entirely when only one symbol ever carried the ISIN."""
    cands = {s for s, _, _ in master.symbols_for_isin(isin)}
    return (next(iter(cands)), None) if len(cands) == 1 else (None, "bridge_isin_ambiguous")


class TestMutantsAreRejected:
    CASES = [
        ("current_symbol", _current_symbol_bridge, dict(symbol="NEWNAME", isin="INE111A01011", day="2010-03-01"), "OLDNAME"),
        ("guess_on_ambiguity", _guessing_bridge, dict(symbol="WRONG", isin="INE222B01012", day="2020-01-02"), None),
        ("ignore_first_listing", _ignore_listing_bridge, dict(symbol="LATE", isin="INE333C01013", day="2019-06-01"), None),
    ]

    @pytest.mark.parametrize("label,mutant,kw,expected", CASES, ids=[c[0] for c in CASES])
    def test_mutant_fails_the_contract(self, master, monkeypatch, label, mutant, kw, expected):
        good = run(master, **kw)
        assert (None if pd.isna(good["TICKER"]) else good["TICKER"]) == expected          # real bridge satisfies it
        monkeypatch.setattr(isin_bridge, "bridge_isin", mutant)
        bad = run(master, **kw)
        assert (None if pd.isna(bad["TICKER"]) else bad["TICKER"]) != expected             # mutant violates it


class TestBridgeFunction:
    def test_returns_symbol_and_none(self, master):
        assert isin_bridge.bridge_isin("INE111A01011", date(2010, 3, 1), master) == ("OLDNAME", None)

    def test_boundaries_are_inclusive(self, master):
        assert isin_bridge.bridge_isin("INE111A01011", date(2015, 5, 29), master)[0] == "OLDNAME"
        assert isin_bridge.bridge_isin("INE111A01011", date(2015, 6, 1), master)[0] == "NEWNAME"


class TestReResolveMode:
    def _frame(self):
        base = {"TICKER": None, "RESOLUTION_METHOD": "unresolved"}
        rows = [
            dict(SYMBOL_RAW="NEWNAME", ISIN="INE111A01011", PUBLISHED_DATE=pd.Timestamp("2010-03-01"), UNRESOLVED_REASON="symbol_outside_traded_window", **base),
            dict(SYMBOL_RAW="X", ISIN="INE222B01012", PUBLISHED_DATE=pd.Timestamp("2020-01-02"), UNRESOLVED_REASON="symbol_not_in_master", **base),
            dict(SYMBOL_RAW="S", ISIN=None, PUBLISHED_DATE=pd.Timestamp("2020-01-02"), UNRESOLVED_REASON="symbol_not_in_master", **base),
            dict(SYMBOL_RAW="TWINA", ISIN="INE222B01012", PUBLISHED_DATE=pd.Timestamp("2020-01-02"), UNRESOLVED_REASON="symbol_isin_conflict", **base),
            dict(SYMBOL_RAW="STABLE", ISIN="INE444D01014", PUBLISHED_DATE=pd.Timestamp("2020-01-02"), TICKER="STABLE",
                 RESOLUTION_METHOD="symbol", UNRESOLVED_REASON=None),
        ]
        return pd.DataFrame(rows)

    def test_only_eligible_rows_change(self, master):
        df = self._frame()
        new, bridged, changed = isin_bridge.re_resolve_frame(df, master)
        assert (bridged, changed) == (1, 1)
        assert new.loc[0, "TICKER"] == "OLDNAME" and new.loc[0, "RESOLUTION_METHOD"] == "isin_bridge"
        assert new.loc[1, "UNRESOLVED_REASON"] == "bridge_isin_ambiguous" and pd.isna(new.loc[1, "TICKER"])
        pd.testing.assert_frame_equal(new.iloc[2:], df.iloc[2:])                       # conflict / no-ISIN / resolved untouched

    def test_resumable_and_idempotent(self, master, tmp_path):
        self._frame().to_parquet(tmp_path / "announcements_2010-03-01_2010-03-31.parquet")
        first = isin_bridge.re_resolve_dir(tmp_path, master)
        assert first["bridged"] == 1 and first["rewritten"] == 1
        again = isin_bridge.re_resolve_dir(tmp_path, master)                           # checkpointed: skipped entirely
        assert again["skipped"] == 1 and again["bridged"] == 0
        forced = isin_bridge.re_resolve_dir(tmp_path, master, resume=False)            # no checkpoint: still a no-op on data
        assert forced["bridged"] == 0 and forced["rewritten"] == 0


# ---- owner rule (2026-11-02): the override IS the purpose ---------------------------------------------------------------

class TestOverrideRule:
    """Tag symbol in master but OUTSIDE its traded window + ISIN resolves uniquely on the date -> bridge. Tag symbol traded
    on/near the date but contradicts the ISIN -> symbol_isin_conflict, never bridged."""

    def test_tag_outside_window_with_unique_isin_is_bridged(self, master):
        r = run(master, symbol="OLDNAME", isin="INE111A01011", day="2020-03-01")      # OLDNAME last traded 2015-05-29, >365d before
        assert r["TICKER"] == "NEWNAME" and r["RESOLUTION_METHOD"] == "isin_bridge"

    def test_tag_that_traded_near_the_date_but_contradicts_isin_is_a_conflict(self, master):
        r = run(master, symbol="STABLE", isin="INE333C01013", day="2020-06-01")        # STABLE traded then; ISIN belongs to LATE
        assert pd.isna(r["TICKER"]) and r["UNRESOLVED_REASON"] == "symbol_isin_conflict"


# ---- verifier PIT check: exactly ONE distinct symbol contains the date ------------------------------------------------

def _pit_any_mutant(ticker, isin, day, master):
    return any(s == ticker and lo <= day <= hi for s, lo, hi in master.symbols_for_isin(isin))


class TestPitWindowCheck:
    def test_unique_containing_symbol_passes(self, master):
        assert isin_bridge.pit_window_ok("OLDNAME", "INE111A01011", date(2010, 3, 1), master)

    def test_wrong_symbol_fails(self, master):
        assert not isin_bridge.pit_window_ok("NEWNAME", "INE111A01011", date(2010, 3, 1), master)

    def test_ambiguity_fails_and_the_lenient_mutant_misses_it(self, master):
        assert not isin_bridge.pit_window_ok("TWINA", "INE222B01012", date(2020, 1, 2), master)    # two symbols contain the date
        assert _pit_any_mutant("TWINA", "INE222B01012", date(2020, 1, 2), master)                  # mutant would have passed it

    def test_date_outside_window_fails(self, master):
        assert not isin_bridge.pit_window_ok("OLDNAME", "INE111A01011", date(2015, 5, 30), master)


# ---- more mutants: boundary off-by-one, bridge overriding an already-good tag -------------------------------------------

def _exclusive_boundary_bridge(isin, day, master):
    inside = {s for s, lo, hi in master.symbols_for_isin(isin) if lo < day < hi}
    return (next(iter(inside)), None) if len(inside) == 1 else (None, "bridge_gap_between_windows")


def _late_boundary_bridge(isin, day, master):
    inside = {s for s, lo, hi in master.symbols_for_isin(isin) if lo <= day <= hi + pd.Timedelta(days=1).to_pytimedelta()}
    return (next(iter(inside)), None) if len(inside) == 1 else (None, "bridge_gap_between_windows")


class TestBoundaryAndOverrideMutants:
    def test_exclusive_window_mutant_rejected(self, master, monkeypatch):
        kw = dict(symbol="GHOSTSYM", isin="INE111A01011", day="2015-05-29")             # last day of OLDNAME's window
        assert run(master, **kw)["TICKER"] == "OLDNAME"
        monkeypatch.setattr(isin_bridge, "bridge_isin", _exclusive_boundary_bridge)
        assert pd.isna(run(master, **kw)["TICKER"])

    def test_window_extended_by_one_day_mutant_rejected(self, master, monkeypatch):
        kw = dict(symbol="GHOSTSYM", isin="INE111A01011", day="2015-05-30")             # one day after the window: real bridge refuses
        assert pd.isna(run(master, **kw)["TICKER"])
        monkeypatch.setattr(isin_bridge, "bridge_isin", _late_boundary_bridge)
        assert run(master, **kw)["TICKER"] == "OLDNAME"

    def test_bridge_overriding_a_good_tag_mutant_rejected(self, master, monkeypatch):
        # OLDNAME traded 2015-05-29; on 2015-06-15 the tag is within the post-delisting slack, so it is a good tag.
        kw = dict(symbol="OLDNAME", isin="INE111A01011", day="2015-06-15")
        good = run(master, **kw)
        assert good["TICKER"] == "OLDNAME" and good["RESOLUTION_METHOD"] == "symbol"
        monkeypatch.setattr(SymbolMaster, "traded_near", lambda self, sym, day: False)  # mutant: treat every tag as failed
        bad = run(master, **kw)
        assert bad["TICKER"] == "NEWNAME" and bad["RESOLUTION_METHOD"] == "isin_bridge"  # the override the real code must not make


# ---- checkpoint fingerprint, parquet schema stability -------------------------------------------------------------------

class TestCheckpointFingerprint:
    def _write(self, tmp_path):
        TestReResolveMode()._frame().to_parquet(tmp_path / "announcements_2010-03-01_2010-03-31.parquet")

    def test_master_change_invalidates_checkpoint(self, master, tmp_path):
        self._write(tmp_path)
        assert isin_bridge.re_resolve_dir(tmp_path, master)["bridged"] == 1
        assert isin_bridge.re_resolve_dir(tmp_path, master)["skipped"] == 1
        grown = SymbolMaster.from_frame(pd.concat([FRAME, pd.DataFrame({
            "SYMBOL": ["NEWCO"], "ISIN": ["INE555E01015"], "FIRST_DATE": pd.to_datetime(["2020-01-01"]),
            "LAST_DATE": pd.to_datetime(["2026-10-30"])})], ignore_index=True))
        assert grown.fingerprint() != master.fingerprint()
        again = isin_bridge.re_resolve_dir(tmp_path, grown)
        assert again["skipped"] == 0                                                   # checkpoint ignored, file reprocessed

    def test_fingerprint_covers_isin_window_count(self):
        extra = pd.concat([FRAME, FRAME.iloc[[0]].assign(ISIN="INE666F01016")], ignore_index=True)
        a, b = SymbolMaster.from_frame(FRAME).fingerprint(), SymbolMaster.from_frame(extra).fingerprint()
        assert a["symbols"] == b["symbols"] and a["max_last_date"] == b["max_last_date"] and a != b


class TestParquetStability:
    def test_schema_and_untouched_rows_survive_rewrite(self, master, tmp_path):
        df = TestReResolveMode()._frame()
        df["EXTRA_NUM"] = [1.5, 2.5, 3.5, 4.5, 5.5]
        df["EXTRA_STR"] = pd.array(["a", "b", "c", "d", "e"], dtype="string")
        f = tmp_path / "announcements_2010-03-01_2010-03-31.parquet"
        df.to_parquet(f, index=False)
        before = pd.read_parquet(f)
        assert isin_bridge.re_resolve_dir(tmp_path, master)["rewritten"] == 1
        after = pd.read_parquet(f)
        assert list(after.columns) == list(before.columns)
        assert after.dtypes.to_dict() == before.dtypes.to_dict()
        pd.testing.assert_frame_equal(after.iloc[2:].reset_index(drop=True), before.iloc[2:].reset_index(drop=True))
        # only TICKER / RESOLUTION_METHOD / UNRESOLVED_REASON may differ on the eligible rows
        same = [c for c in before.columns if c not in ("TICKER", "RESOLUTION_METHOD", "UNRESOLVED_REASON")]
        pd.testing.assert_frame_equal(after[same], before[same])

    def test_file_with_nothing_eligible_is_byte_identical(self, master, tmp_path):
        df = TestReResolveMode()._frame().iloc[2:]
        f = tmp_path / "announcements_2010-03-01_2010-03-31.parquet"
        df.to_parquet(f, index=False)
        raw = f.read_bytes()
        isin_bridge.re_resolve_dir(tmp_path, master)
        assert f.read_bytes() == raw
