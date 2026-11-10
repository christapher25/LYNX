"""R20-3: tests for scripts/apply_corp_action_correction.py. No network: the ingester's fetch is faked."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import ingest_corporate_actions as ing
from scripts import apply_corp_action_correction as cor

ROOT = Path(__file__).resolve().parent.parent
COMMITTED = ROOT / "docs" / "agent-team" / "corrections" / "2026-11-10_precutoff_corporate_actions.json"


def rec(subject, ex, isin="INE000A01010"):
    return {"subject": subject, "exDate": ex, "recDate": ex, "bcStartDate": "-", "bcEndDate": "-", "isin": isin,
            "series": "EQ", "faceVal": "10"}


DIV = rec("Interim Dividend - Rs 5 Per Share", "15-Sep-2026")
BONUS = rec("Bonus 1:1", "28-Sep-2026")
OLD = [rec("Interim Dividend - Rs 6 Per Share", "29-Apr-2026")]


class Fake:
    def __init__(self, records=None):
        self.records = records or {}
        self.calls = []

    def __call__(self, session, symbol, year):
        self.calls.append((symbol, year))
        v = self.records.get((symbol, year), [])
        if isinstance(v, Exception):
            raise v
        return v


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = tmp_path / "corporate_actions"
    out.mkdir()
    monkeypatch.setattr(ing, "OUTPUT_DIR", out)
    monkeypatch.setattr(ing, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ing, "get_session", lambda: object())
    monkeypatch.setattr(cor, "SUMMARY_DIR", tmp_path / "corrections")
    fake = Fake()
    monkeypatch.setattr(ing, "fetch_symbol_year", fake)
    return type("Env", (), {"out": out, "fake": fake, "tmp": tmp_path})


def seed(env, symbol, records=None):
    fake = Fake({(symbol, 2026): records if records is not None else OLD})
    saved = ing.fetch_symbol_year
    ing.fetch_symbol_year = fake
    try:
        frame = ing.ingest_symbol(object(), symbol, 2026, 2026)
    finally:
        ing.fetch_symbol_year = saved
    ing.write_parquet_atomic(frame, env.out / f"{symbol}.parquet")
    return pd.read_parquet(env.out / f"{symbol}.parquet")


def approved(env, entries, cutoff="2026-09-29", name="approved.json"):
    p = env.tmp / name
    p.write_text(json.dumps({"approved_by": "t", "source_log": "t", "history_cutoff": cutoff, "actions": entries}),
                 encoding="utf-8")
    return p


def entry(symbol, ex, typ, factor=None):
    return {"symbol": symbol, "ex_date": ex, "action_type": typ, "adjustment_factor": factor}


DIV_E = entry("AAA", "2026-09-15", "dividend")
BONUS_E = entry("AAA", "2026-09-28", "bonus", 0.5)


def run(env, entries, *extra, **kw):
    return cor.main(["--approved", str(approved(env, entries, **kw)), *extra])


def _nulls_as_none(df):
    return df.astype(object).where(df.notna(), None)


def cmp_old(old, new):
    """Existing rows unchanged value by value (a null is a null whether pandas shows None or NaN)."""
    new_old = new[new["raw_subject"].isin(old["raw_subject"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(_nulls_as_none(old.reset_index(drop=True)), _nulls_as_none(new_old),
                                  check_dtype=False)


# T1 -------------------------------------------------------------------
def test_t1_appends_only_approved_rows_and_keeps_old_rows(env):
    old = seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [DIV, BONUS]}
    assert run(env, [DIV_E, BONUS_E]) == 0
    new = pd.read_parquet(env.out / "AAA.parquet")
    assert len(new) == len(old) + 2
    cmp_old(old, new)
    assert list(new["ex_date"]) == sorted(new["ex_date"])
    added = new[new["ex_date"] >= pd.Timestamp("2026-09-01")]
    assert added["adjustment_factor"].dropna().tolist() == [0.5]
    assert new.loc[new["ex_date"] < pd.Timestamp("2026-09-01"), "adjustment_factor"].isna().all()


# T2 -------------------------------------------------------------------
def test_t2_unapproved_fetched_rows_are_never_added(env):
    seed(env, "AAA")
    extra_before = rec("Final Dividend - Rs 2 Per Share", "10-Aug-2026")
    extra_after = rec("Final Dividend - Rs 3 Per Share", "05-Oct-2026")
    env.fake.records = {("AAA", 2026): OLD + [DIV, extra_before, extra_after]}
    assert run(env, [DIV_E]) == 0
    new = pd.read_parquet(env.out / "AAA.parquet")
    assert sorted(new["raw_subject"]) == sorted([OLD[0]["subject"], DIV["subject"]])


# T3 -------------------------------------------------------------------
def test_t3_rerun_is_a_noop(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [DIV, BONUS]}
    assert run(env, [DIV_E, BONUS_E]) == 0
    p = env.out / "AAA.parquet"
    before, mtime = p.read_bytes(), p.stat().st_mtime_ns
    assert run(env, [DIV_E, BONUS_E]) == 0
    assert p.read_bytes() == before and p.stat().st_mtime_ns == mtime
    summaries = sorted((env.tmp / "corrections").glob("*.json"))
    last = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert last["present"] == 2 and last["added"] == 0 and last["failed"] == 0


# T4 -------------------------------------------------------------------
def test_t4_entry_not_fetched_fails_and_file_untouched(env):
    seed(env, "AAA")
    p = env.out / "AAA.parquet"
    before = p.read_bytes()
    env.fake.records = {("AAA", 2026): OLD}            # approved dividend not returned
    assert run(env, [DIV_E]) == 1
    assert p.read_bytes() == before


def test_t4_two_fetched_rows_match_one_entry_fails(env):
    seed(env, "AAA")
    p = env.out / "AAA.parquet"
    before = p.read_bytes()
    twin = rec("Interim Dividend - Rs 7 Per Share", "15-Sep-2026")    # same date, type, null factor
    env.fake.records = {("AAA", 2026): OLD + [DIV, twin]}
    assert run(env, [DIV_E]) == 1
    assert p.read_bytes() == before


# T5 -------------------------------------------------------------------
def test_t5_fetch_error_fails_that_symbol_others_continue(env):
    seed(env, "AAA")
    seed(env, "BBB")
    pa_, pb = env.out / "AAA.parquet", env.out / "BBB.parquet"
    before = pa_.read_bytes()
    env.fake.records = {("AAA", 2026): ValueError("HTTP 403"), ("BBB", 2026): OLD + [DIV]}
    assert run(env, [DIV_E, entry("BBB", "2026-09-15", "dividend")]) == 1
    assert pa_.read_bytes() == before
    assert len(pd.read_parquet(pb)) == 2


def test_t5_write_error_fails_symbol(env, monkeypatch):
    seed(env, "AAA")
    p = env.out / "AAA.parquet"
    before = p.read_bytes()
    env.fake.records = {("AAA", 2026): OLD + [DIV]}
    monkeypatch.setattr(cor, "_write_table_atomic", lambda t, path: (_ for _ in ()).throw(OSError("disk full")))
    assert run(env, [DIV_E]) == 1
    assert p.read_bytes() == before


# T6 -------------------------------------------------------------------
def test_t6_symbol_without_file_fails_and_no_file_created(env):
    env.fake.records = {("NOFILE", 2026): [DIV]}
    assert run(env, [entry("NOFILE", "2026-09-15", "dividend")]) == 1
    assert not (env.out / "NOFILE.parquet").exists()
    assert env.fake.calls == []


# T7 -------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    [{"symbol": "AAA", "ex_date": "2026-09-15", "action_type": "dividend"}],                # missing factor key
    [entry("AAA", "15-09-2026", "dividend")],                                               # bad date
    [entry("AAA", "2026-09-15", "dividend"), entry("AAA", "2026-09-15", "dividend")],       # duplicate
    [entry("AAA", "2026-09-30", "dividend")],                                               # after cutoff
    [entry("AAA", "2026-09-15", "dividend", "x")],                                          # factor not a number
    [],                                                                                     # empty list
])
def test_t7_invalid_list_exits_2_without_fetch(env, bad):
    seed(env, "AAA")
    assert run(env, bad) == 2
    assert env.fake.calls == []


def test_t7_not_json_exits_2(env):
    p = env.tmp / "bad.json"
    p.write_text("{nope", encoding="utf-8")
    assert cor.main(["--approved", str(p)]) == 2
    assert env.fake.calls == []


# T8 -------------------------------------------------------------------
def test_t8_dry_run_writes_nothing(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [DIV]}
    snap = {f: f.read_bytes() for f in env.tmp.rglob("*") if f.is_file()}
    appr = approved(env, [DIV_E])
    snap[appr] = appr.read_bytes()
    assert cor.main(["--approved", str(appr), "--dry-run"]) == 0
    after = {f: f.read_bytes() for f in env.tmp.rglob("*") if f.is_file()}
    assert after == snap


# T9 -------------------------------------------------------------------
def test_t9_typed_column_keeps_type_and_null_column_gains_type(env):
    seed(env, "AAA")
    path = env.out / "AAA.parquet"
    stored = pq.read_schema(path)
    assert stored.field("adjustment_factor").type == pa.null()
    assert stored.field("dividend_amount").type == pa.float64()
    env.fake.records = {("AAA", 2026): OLD + [BONUS]}                 # bonus: factor 0.5, no dividend_amount
    assert run(env, [BONUS_E]) == 0
    t = pq.read_table(path)
    assert t.schema.field("adjustment_factor").type == pa.float64()
    assert t.schema.field("dividend_amount").type == pa.float64()      # kept although the new row is null there
    assert t.schema.field("ex_date").type == stored.field("ex_date").type
    assert t.schema.field("symbol").type == stored.field("symbol").type
    d = t.to_pandas()
    assert d["adjustment_factor"].notna().sum() == 1
    assert d.loc[d["adjustment_factor"].notna(), "ex_date"].iloc[0] == pd.Timestamp("2026-09-28")


def test_t9_null_column_stays_null_when_added_rows_have_no_value(env):
    seed(env, "AAA")
    env.fake.records = {("AAA", 2026): OLD + [DIV]}
    assert run(env, [DIV_E]) == 0
    assert pq.read_schema(env.out / "AAA.parquet").field("adjustment_factor").type == pa.null()


# T10 ------------------------------------------------------------------
def test_t10_committed_approved_list():
    d = json.loads(COMMITTED.read_text(encoding="utf-8"))
    acts = d["actions"]
    assert len(acts) == 113
    assert sum(a["action_type"] == "dividend" for a in acts) == 112
    bonus = [a for a in acts if a["action_type"] == "bonus"]
    assert len(bonus) == 1
    assert (bonus[0]["symbol"], bonus[0]["ex_date"], bonus[0]["adjustment_factor"]) == ("AASTHA", "2026-09-28", 0.5)
    assert all("2026-07-07" <= a["ex_date"] <= "2026-09-28" for a in acts)
    assert d["history_cutoff"] == "2026-09-29"


def test_no_network_or_literal_data_path_in_source():
    src = (ROOT / "scripts" / "apply_corp_action_correction.py").read_text(encoding="utf-8")
    assert "requests." not in src and "print(" not in src
    assert "raw/corporate_actions" not in src and "raw\\corporate_actions" not in src


# Round 1: the stored pandas metadata block is kept, so a corrected file reads back with the dtypes of an untouched one ------
@pytest.mark.parametrize("records,entries", [([DIV], [DIV_E]), ([DIV, BONUS], [DIV_E, BONUS_E]), ([BONUS], [BONUS_E])])
def test_round1_pandas_metadata_and_dtypes_are_kept(env, records, entries):
    seed(env, "AAA")
    path = env.out / "AAA.parquet"
    stored = pq.read_schema(path)
    assert b"pandas" in stored.metadata
    before = pd.read_parquet(path).dtypes
    env.fake.records = {("AAA", 2026): OLD + records}
    assert run(env, entries) == 0
    assert b"pandas" in pq.read_schema(path).metadata
    after_schema = pq.read_schema(path)
    after = pd.read_parquet(path).dtypes
    gained = {f.name for f in stored if f.type == pa.null() and after_schema.field(f.name).type != pa.null()}
    for col in before.index:
        if col not in gained:
            assert after[col] == before[col], f"{col}: {before[col]} -> {after[col]}"
