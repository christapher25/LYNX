"""Tests for scripts/history_guard.py: snapshot/verify of history (rows dated <= cutoff) across the derived outputs.

Fixtures are tiny hand-built Parquet trees under tmp_path; the Postgres indicators reader is injected.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import history_guard as hg

CUTOFF = "2026-09-29"
DATES = pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"])
LATER = pd.to_datetime(["2026-09-30", "2026-10-01"])
ADJ_COLS = ("ADJ_OPEN", "ADJ_HIGH", "ADJ_LOW", "ADJ_CLOSE", "ADJ_LAST", "ADJ_PREV_CLOSE")


def _prices(symbol, dates, base=100.0):
    n = len(dates)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame({
        "SYMBOL": symbol, "SERIES": "EQ", "DATE": dates,
        "OPEN": close - 1, "HIGH": close + 1, "LOW": close - 2, "CLOSE": close, "LAST": close, "PREV_CLOSE": close - 0.5,
        "VOLUME": np.arange(n) * 10 + 1000,
        "ADJ_CLOSE": close, "ADJ_OPEN": close - 1, "ADJ_HIGH": close + 1, "ADJ_LOW": close - 2, "ADJ_LAST": close,
        "ADJ_PREV_CLOSE": close - 0.5, "ADJ_VOLUME": (np.arange(n) * 10 + 1000).astype(float),
        "ADJUSTMENT_APPLIED": False,
    })


def _action(symbol, ex_date, factor):
    return pd.DataFrame({"symbol": [symbol], "ex_date": pd.to_datetime([ex_date]), "action_type": ["split"],
                         "raw_subject": ["x"], "adjustment_factor": [factor]})


def _feature(symbol, dates):
    return pd.DataFrame({"symbol": symbol, "date": dates, "delivery_pct": np.linspace(10, 20, len(dates)),
                         "is_missing_delivery": False})


def _indicators(dates):
    return pd.DataFrame({"date": dates, "rsi_14": np.linspace(30, 60, len(dates)), "atr_14": np.linspace(1, 2, len(dates))})


class World:
    def __init__(self, root: Path):
        self.root = root
        self.actions, self.adjusted = root / "actions", root / "adjusted"
        self.universe, self.features, self.out = root / "universe", root / "features", root / "guard"
        for d in (self.actions, self.adjusted, self.universe, self.features / "delivery", self.features / "flows"):
            d.mkdir(parents=True)
        self.ind = {}
        all_dates = DATES.append(LATER)
        for sym, base in (("AAA", 100.0), ("BBB", 200.0)):
            _prices(sym, all_dates, base).to_parquet(self.adjusted / f"{sym}.parquet")
            _feature(sym, all_dates).to_parquet(self.features / "delivery" / f"{sym}.parquet")
            _action(sym, "2020-01-01", np.nan).to_parquet(self.actions / f"{sym}.parquet")
            self.ind[sym] = _indicators(all_dates)
        pd.DataFrame({"date": all_dates, "fii": np.arange(len(all_dates), dtype=float)}).to_parquet(self.features / "flows" / "fii.parquet")
        pd.DataFrame({"SYMBOL": ["AAA", "BBB", "OLD"], "RANK": [1, 2, 3], "AVG_TURNOVER": [3.0, 2.0, 1.0],
                      "FIRST_DATE": pd.to_datetime(["2000-01-01"] * 3),
                      "LAST_DATE": pd.to_datetime(["2026-09-29", "2026-09-29", "2024-01-01"]),
                      "IS_DELISTED": [False, False, True], "TERMINAL_VALUE": [np.nan, np.nan, 5.0]}).to_parquet(self.universe / "2026.parquet")

    def sources(self):
        return hg.Sources(actions_dir=self.actions, adjusted_dir=self.adjusted, universe_dir=self.universe,
                          features_dir=self.features, out_dir=self.out,
                          indicator_reader=lambda cutoff: ((s, d[d["date"] <= pd.Timestamp(cutoff)]) for s, d in sorted(self.ind.items())))

    def snapshot(self):
        hg.snapshot(CUTOFF, self.sources())

    def verify(self):
        return hg.verify(self.sources())

    def edit(self, path: Path, fn):
        df = pd.read_parquet(path)
        fn(df)
        df.to_parquet(path)

    def set_value(self, path: Path, row, col, value):
        self.edit(path, lambda d: d.loc.__setitem__((row, col), value))

    def new_split(self, symbol, factor=0.5, apply_to_prices=True):
        """A post-cutoff split: the action file keeps its old row and gains a new one; prices adjust backward."""
        pd.concat([_action(symbol, "2020-01-01", np.nan), _action(symbol, "2026-10-01", factor)],
                  ignore_index=True).to_parquet(self.actions / f"{symbol}.parquet")
        if not apply_to_prices:
            return

        def adj(d):
            m = d["DATE"] < pd.Timestamp("2026-10-01")
            for c in ADJ_COLS:
                d.loc[m, c] = d.loc[m, c] * factor
            d.loc[m, "ADJ_VOLUME"] = d.loc[m, "ADJ_VOLUME"] / factor
            d.loc[m, "ADJUSTMENT_APPLIED"] = True
        self.edit(self.adjusted / f"{symbol}.parquet", adj)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    w.snapshot()
    return w


def failing_text(report):
    return " | ".join(str(f) for f in report.failures)


def test_clean_verify_passes(world):
    r = world.verify()
    assert r.failures == [] and r.allowed == []


def test_snapshot_writes_one_file_per_dataset_under_out_dir_only(tmp_path):
    w = World(tmp_path)
    before = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.parquet")}
    w.snapshot()
    after = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.parquet")}
    assert before == after
    names = {p.stem for p in w.out.glob("*.json")}
    assert {"corporate_actions", "adjusted", "universe", "features_delivery", "features_flows", "indicators"} <= names
    assert {p.parent for p in tmp_path.rglob("*.json")} == {w.out}


def test_new_rows_after_cutoff_are_fine(world):
    p = world.adjusted / "AAA.parquet"
    pd.concat([pd.read_parquet(p), _prices("AAA", pd.to_datetime(["2026-10-05"]))], ignore_index=True).to_parquet(p)
    world.ind["AAA"] = pd.concat([world.ind["AAA"], _indicators(pd.to_datetime(["2026-10-05"]))], ignore_index=True)
    assert world.verify().failures == []


def test_changed_value_after_cutoff_is_ignored(world):
    world.edit(world.adjusted / "AAA.parquet", lambda d: d.__setitem__("CLOSE", np.where(d["DATE"] > CUTOFF, 1.0, d["CLOSE"])))
    assert world.verify().failures == []


def test_new_symbol_with_only_post_cutoff_rows_is_fine(world):
    _prices("NEW", LATER).to_parquet(world.adjusted / "NEW.parquet")
    assert world.verify().failures == []


# ---- mutants: each must fail ----------------------------------------------------------------------
def test_mutant_unadjusted_value_changed(world):
    world.set_value(world.adjusted / "AAA.parquet", 1, "CLOSE", 999.0)
    r = world.verify()
    assert r.failures and "AAA" in failing_text(r) and "CLOSE" in failing_text(r)


def test_mutant_adjusted_value_changed_without_new_action(world):
    world.set_value(world.adjusted / "AAA.parquet", 1, "ADJ_CLOSE", 999.0)
    r = world.verify()
    assert r.failures and "ADJ_CLOSE" in failing_text(r) and r.allowed == []


def test_mutant_row_deleted_at_or_before_cutoff(world):
    p = world.adjusted / "AAA.parquet"
    pd.read_parquet(p).drop(index=1).reset_index(drop=True).to_parquet(p)
    assert world.verify().failures


def test_mutant_row_deleted_even_with_new_action(world):
    world.new_split("AAA")
    p = world.adjusted / "AAA.parquet"
    pd.read_parquet(p).drop(index=1).reset_index(drop=True).to_parquet(p)
    assert world.verify().failures


def test_mutant_symbol_file_removed(world):
    (world.adjusted / "BBB.parquet").unlink()
    assert world.verify().failures


def test_mutant_new_row_dated_before_cutoff(world):
    p = world.adjusted / "AAA.parquet"
    pd.concat([pd.read_parquet(p), _prices("AAA", pd.to_datetime(["2026-09-26"]))], ignore_index=True).to_parquet(p)
    assert world.verify().failures


def test_mutant_corporate_action_history_changed(world):
    _action("AAA", "2020-01-01", 0.5).to_parquet(world.actions / "AAA.parquet")
    r = world.verify()
    assert r.failures and "corporate_actions" in failing_text(r)


def test_mutant_late_action_dated_before_cutoff_added(world):
    pd.concat([_action("AAA", "2020-01-01", np.nan), _action("AAA", "2026-09-01", 0.5)]).to_parquet(world.actions / "AAA.parquet")
    assert world.verify().failures


def test_mutant_feature_changed(world):
    world.set_value(world.features / "delivery" / "AAA.parquet", 0, "delivery_pct", 99.0)
    r = world.verify()
    assert r.failures and "features_delivery" in failing_text(r)


def test_mutant_market_level_feature_changed(world):
    world.set_value(world.features / "flows" / "fii.parquet", 0, "fii", 99.0)
    assert world.verify().failures


def test_mutant_universe_changed(world):
    world.set_value(world.universe / "2026.parquet", 0, "RANK", 9)
    r = world.verify()
    assert r.failures and "universe" in failing_text(r)


def test_universe_active_row_extending_last_date_is_not_a_failure(world):
    world.set_value(world.universe / "2026.parquet", 0, "LAST_DATE", pd.Timestamp("2026-11-09"))
    assert world.verify().failures == []


def test_mutant_universe_delisted_status_changed(world):
    world.set_value(world.universe / "2026.parquet", 2, "TERMINAL_VALUE", 6.0)
    assert world.verify().failures


def test_mutant_indicator_value_changed(world):
    world.ind["AAA"].loc[1, "rsi_14"] = 1.0
    r = world.verify()
    assert r.failures and "indicators" in failing_text(r)


def test_mutant_indicator_row_deleted(world):
    world.ind["AAA"] = world.ind["AAA"].drop(index=1).reset_index(drop=True)
    assert world.verify().failures


def test_mutant_indicator_null_becomes_zero(tmp_path):
    w = World(tmp_path)
    w.ind["BBB"].loc[0, "rsi_14"] = np.nan
    w.snapshot()
    w.ind["BBB"].loc[0, "rsi_14"] = 0.0
    assert w.verify().failures


# ---- the allowed case -----------------------------------------------------------------------------
def test_allowed_adjusted_change_for_symbol_with_new_action(world):
    world.new_split("AAA")
    r = world.verify()
    assert r.failures == []
    assert len(r.allowed) == 1 and "AAA" in str(r.allowed[0]) and "ADJ_CLOSE" in str(r.allowed[0])


def test_allowed_case_still_fails_if_unadjusted_column_also_changed(world):
    world.new_split("AAA")
    world.set_value(world.adjusted / "AAA.parquet", 0, "CLOSE", 1.0)
    assert world.verify().failures


def test_allowed_case_does_not_cover_other_symbols(world):
    world.new_split("AAA", apply_to_prices=False)
    world.set_value(world.adjusted / "BBB.parquet", 1, "ADJ_CLOSE", 1.0)
    r = world.verify()
    assert r.failures and "BBB" in failing_text(r)


def test_new_action_without_a_factor_does_not_excuse_adjusted_change(world):
    pd.concat([_action("AAA", "2020-01-01", np.nan), _action("AAA", "2026-10-01", np.nan)], ignore_index=True) \
        .to_parquet(world.actions / "AAA.parquet")
    world.set_value(world.adjusted / "AAA.parquet", 1, "ADJ_CLOSE", 1.0)
    assert world.verify().failures


def test_new_action_does_not_excuse_feature_changes(world):
    world.new_split("AAA")
    world.set_value(world.features / "delivery" / "AAA.parquet", 0, "delivery_pct", 99.0)
    assert world.verify().failures


def test_new_action_does_not_excuse_indicator_changes(world):
    world.new_split("AAA")
    world.ind["AAA"].loc[1, "rsi_14"] = 1.0
    assert world.verify().failures


# ---- CLI / hygiene --------------------------------------------------------------------------------
def test_main_exit_codes(world):
    assert hg.main(["verify"], sources=world.sources()) == 0
    world.set_value(world.adjusted / "AAA.parquet", 1, "CLOSE", 999.0)
    assert hg.main(["verify"], sources=world.sources()) == 1


def test_main_snapshot_default_cutoff(tmp_path):
    w = World(tmp_path)
    assert hg.main(["snapshot"], sources=w.sources()) == 0
    assert json.loads((w.out / "adjusted.json").read_text())["cutoff"] == "2026-09-29"


def test_verify_without_snapshot_fails(tmp_path):
    w = World(tmp_path)
    assert hg.main(["verify"], sources=w.sources()) == 1


SRC = Path(hg.__file__).read_text(encoding="utf-8")


def test_module_is_read_only_on_data():
    assert not re.search(r"os\.remove|os\.unlink|\.unlink\(|rmtree|\.rmdir\(|\bDELETE\b|\bUPDATE\b|\bINSERT\b|\bDROP\b|\bTRUNCATE\b", SRC)
    assert not re.search(r"^\s*print\(", SRC, re.M)
    assert "loguru" in SRC


# ---- review fixes ------------------------------------------------------------------------------------
import dataclasses  # noqa: E402


def test_second_snapshot_refuses_to_overwrite_without_force(world):
    before = {p.name: p.read_bytes() for p in world.out.glob("*.json") if p.name != "_index.json"}
    world.set_value(world.adjusted / "AAA.parquet", 1, "CLOSE", 999.0)  # history changed after the baseline
    with pytest.raises(FileExistsError):
        hg.snapshot(CUTOFF, world.sources())
    assert {p.name: p.read_bytes() for p in world.out.glob("*.json") if p.name != "_index.json"} == before
    assert world.verify().failures  # the baseline still catches the change


def test_cli_second_snapshot_exits_1_and_force_rebaselines(world):
    world.set_value(world.adjusted / "AAA.parquet", 1, "CLOSE", 999.0)
    assert hg.main(["snapshot"], sources=world.sources()) == 1
    assert hg.main(["verify"], sources=world.sources()) == 1
    assert hg.main(["snapshot", "--force"], sources=world.sources()) == 0
    assert hg.main(["verify"], sources=world.sources()) == 0


def test_snapshot_refusal_writes_nothing_even_for_a_dataset_that_did_not_exist(world):
    (world.out / "adjusted.json").unlink()  # only one file present; the others exist -> still refuse, nothing rewritten
    mtime = (world.out / "universe.json").stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        hg.snapshot(CUTOFF, world.sources())
    assert not (world.out / "adjusted.json").exists()
    assert (world.out / "universe.json").stat().st_mtime_ns == mtime


def test_index_is_written_with_names_cutoff_and_created_time(world):
    idx = json.loads((world.out / "_index.json").read_text())
    assert idx["cutoff"] == CUTOFF and idx["created"]
    assert {"corporate_actions", "adjusted", "universe", "features_delivery", "features_flows", "indicators"} <= set(idx["datasets"])
    assert idx.get("skipped", []) == []


def test_mutant_deleted_snapshot_json_fails_verify(world):
    (world.out / "adjusted.json").unlink()
    r = world.verify()
    assert r.failures and "adjusted" in failing_text(r)


def test_mutant_missing_index_fails_verify(world):
    (world.out / "_index.json").unlink()
    assert world.verify().failures


def test_mutant_dataset_directory_vanished_fails_verify(world):
    import shutil
    shutil.rmtree(world.features / "flows")
    assert world.verify().failures


def test_dataset_present_now_but_not_in_index_fails(world):
    (world.features / "newblock").mkdir()
    pd.DataFrame({"date": DATES, "x": 1.0}).to_parquet(world.features / "newblock" / "f.parquet")
    r = world.verify()
    assert r.failures and "features_newblock" in failing_text(r)


def test_indicators_missing_because_skip_db_is_a_failure(world):
    r = hg.verify(world.sources(), skip_db=True)
    assert r.failures and "indicators" in failing_text(r)


def test_skipped_snapshot_verified_with_reader_fails_and_with_skip_db_passes(tmp_path):
    w = World(tmp_path)
    no_db = dataclasses.replace(w.sources(), indicator_reader=None)
    hg.snapshot(CUTOFF, no_db)
    assert json.loads((w.out / "_index.json").read_text())["skipped"] == ["indicators"]
    assert "indicators" in failing_text(hg.verify(w.sources()))                 # reader available, index says skipped
    assert hg.verify(no_db, skip_db=True).failures == []                        # consistent skip
    assert hg.verify(w.sources(), skip_db=True).failures == []                  # --skip-db wins even if a reader exists
    assert hg.main(["verify", "--skip-db"], sources=w.sources()) == 0
    assert hg.main(["verify"], sources=w.sources()) == 1


def test_label_on_failures_for_symbols_with_a_new_action_still_fails(world):
    world.new_split("AAA")
    world.set_value(world.features / "delivery" / "AAA.parquet", 0, "delivery_pct", 99.0)
    world.set_value(world.features / "delivery" / "BBB.parquet", 0, "delivery_pct", 99.0)
    r = world.verify()
    label = "symbol has a new corporate action; derived from adjusted columns?"
    aaa = [f for f in r.failures if "features_delivery:AAA" in f]
    bbb = [f for f in r.failures if "features_delivery:BBB" in f]
    assert aaa and label in aaa[0]
    assert bbb and label not in bbb[0]


def test_label_does_not_apply_without_a_factor(world):
    pd.concat([_action("AAA", "2020-01-01", np.nan), _action("AAA", "2026-10-01", np.nan)], ignore_index=True) \
        .to_parquet(world.actions / "AAA.parquet")
    world.set_value(world.features / "delivery" / "AAA.parquet", 0, "delivery_pct", 99.0)
    r = world.verify()
    assert r.failures and "new corporate action" not in failing_text(r)


# ------------------------------------------------------------------ R20-2 (a): empty-column dtype change, corporate_actions only
def _ca_frame(symbol, rows, ratio=None, rec=None):
    """rows: list of ex_date strings. ratio / rec: per-row values or None (all-None object column, like the real files)."""
    n = len(rows)
    ratio = [None] * n if ratio is None else ratio
    rec = [None] * n if rec is None else rec
    return pd.DataFrame({"symbol": [symbol] * n, "ex_date": pd.to_datetime(rows), "action_type": ["dividend"] * n,
                         "raw_subject": ["x"] * n, "adjustment_factor": pd.array([None] * n, dtype=object),
                         "ratio": pd.Series(ratio, dtype=object), "record_date": pd.Series(rec, dtype=object)})


def _ca_world(tmp_path, extra_all_none=True):
    w = World(tmp_path)
    for sym in ("AAA", "BBB"):
        _ca_frame(sym, ["2020-01-01", "2026-09-20"]).to_parquet(w.actions / f"{sym}.parquet")
    return w


def _append_valued_row(path, ex_date="2026-10-05"):
    df = pd.read_parquet(path)
    row = _ca_frame("AAA", [ex_date], ratio=[2.0], rec=[pd.Timestamp("2026-10-04")])
    pd.concat([df.astype(object), row.astype(object)], ignore_index=True).to_parquet(path)


def test_a_fixture_really_changes_stored_type(tmp_path):
    w = _ca_world(tmp_path)
    path = w.actions / "AAA.parquet"
    assert pd.read_parquet(path)["ratio"].dtype == object
    _append_valued_row(path)
    df = pd.read_parquet(path)
    assert df["ratio"].dtype == np.float64 and str(df["record_date"].dtype).startswith("datetime64")


def test_a_T1_empty_column_changing_type_is_allowed(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _append_valued_row(w.actions / "AAA.parquet")
    r = w.verify()
    assert r.failures == [], failing_text(r)
    text = " | ".join(r.allowed)
    assert "empty column changed stored type" in text and "ratio" in text and "record_date" in text


def test_a_T1_old_format_snapshot_without_all_null_still_verifies(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    p = w.out / "corporate_actions.json"
    snap = json.loads(p.read_text())
    for u in snap["units"].values():
        u.pop("all_null", None)
    p.write_text(json.dumps(snap))
    assert w.verify().failures == []
    _append_valued_row(w.actions / "AAA.parquet")
    assert w.verify().failures == []


def test_a_new_snapshot_records_all_null_for_corporate_actions_only(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    ca = json.loads((w.out / "corporate_actions.json").read_text())
    assert ca["units"]["AAA"]["all_null"] == ["adjustment_factor", "ratio", "record_date"]
    for name in ("adjusted", "features_delivery", "universe", "indicators"):
        for u in json.loads((w.out / f"{name}.json").read_text())["units"].values():
            assert "all_null" not in u


def _retype_all_none_column(path, col):
    df = pd.read_parquet(path)
    df[col] = np.nan  # all-NaN float64: same shape of change, no value
    df.to_parquet(path)


def test_a_T2_same_input_in_adjusted_and_features_still_fails(tmp_path):
    w = World(tmp_path)
    for p in (w.adjusted / "AAA.parquet", w.features / "delivery" / "AAA.parquet"):
        df = pd.read_parquet(p)
        df["xtra"] = pd.Series([None] * len(df), dtype=object)
        df.to_parquet(p)
    w.snapshot()
    for p in (w.adjusted / "AAA.parquet", w.features / "delivery" / "AAA.parquet"):
        _retype_all_none_column(p, "xtra")
    text = failing_text(w.verify())
    assert "adjusted:AAA" in text and "features_delivery:AAA" in text


def test_a_T3_column_with_a_value_at_snapshot_that_is_now_null_fails(tmp_path):
    w = _ca_world(tmp_path)
    _ca_frame("AAA", ["2020-01-01", "2026-09-20"], ratio=[1.0, None]).to_parquet(w.actions / "AAA.parquet")
    w.snapshot()
    _ca_frame("AAA", ["2020-01-01", "2026-09-20"]).to_parquet(w.actions / "AAA.parquet")
    r = w.verify()
    assert "corporate_actions:AAA" in failing_text(r) and "ratio" in failing_text(r)


def test_a_T3_empty_column_that_now_holds_a_history_value_fails(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _ca_frame("AAA", ["2020-01-01", "2026-09-20"], ratio=[None, 3.0]).to_parquet(w.actions / "AAA.parquet")
    r = w.verify()
    assert "corporate_actions:AAA" in failing_text(r) and "ratio" in failing_text(r)


def test_a_T4_type_change_plus_removed_history_row_fails(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _ca_frame("AAA", ["2020-01-01", "2026-10-05"], ratio=[None, 2.0]).to_parquet(w.actions / "AAA.parquet")  # history row dropped
    assert "row count" in failing_text(w.verify())


def test_a_T4_type_change_plus_changed_ex_date_fails(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    df = pd.concat([_ca_frame("AAA", ["2020-01-01", "2026-09-21"]).astype(object),
                    _ca_frame("AAA", ["2026-10-05"], ratio=[2.0]).astype(object)], ignore_index=True)
    df.to_parquet(w.actions / "AAA.parquet")
    assert "date column differs" in failing_text(w.verify())


# ------------------------------------------------------------------ R20-2 (c): owner-approved expected changes
def _entry(symbol, ex_date, action_type, factor):
    return {"symbol": symbol, "ex_date": ex_date, "action_type": action_type, "adjustment_factor": factor}


BONUS = _entry("AAA", "2026-09-28", "bonus", 0.5)
DIV = _entry("BBB", "2026-09-23", "dividend", None)


def _write_expected(w, actions, cutoff=CUTOFF, **extra):
    path = w.root / "expected.json"
    path.write_text(json.dumps({"history_cutoff": cutoff, "actions": actions, **extra}))
    return path


def _add_rows(path, *entries, **overrides):
    df = pd.read_parquet(path).astype(object)
    rows = pd.DataFrame({
        "symbol": [e["symbol"] for e in entries], "ex_date": pd.to_datetime([e["ex_date"] for e in entries]),
        "action_type": [e["action_type"] for e in entries], "raw_subject": ["new"] * len(entries),
        "adjustment_factor": pd.Series([e["adjustment_factor"] for e in entries], dtype=object),
        "ratio": pd.Series([None] * len(entries), dtype=object), "record_date": pd.Series([None] * len(entries), dtype=object),
    }).astype(object)
    pd.concat([df, rows], ignore_index=True).to_parquet(path)


def _verify_expected(w, actions, **kw):
    return hg.verify(w.sources(), expected=hg.load_expected(_write_expected(w, actions, **kw)))


def _approved_world(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _add_rows(w.actions / "AAA.parquet", BONUS)
    _add_rows(w.actions / "BBB.parquet", DIV)
    return w


def _adj_change(w, symbol):
    w.edit(w.adjusted / f"{symbol}.parquet", lambda d: d.__setitem__("ADJ_CLOSE", d["ADJ_CLOSE"] * 0.5))


def test_c_T5_approved_bonus_and_dividend_pass_and_are_listed(tmp_path):
    w = _approved_world(tmp_path)
    _adj_change(w, "AAA")
    r = _verify_expected(w, [BONUS, DIV])
    assert r.failures == [], failing_text(r)
    added = [a for a in r.allowed if "owner-approved action(s) added" in a]
    assert len(added) == 2 and any("bonus" in a for a in added) and any("dividend" in a for a in added)
    assert any("adjusted:AAA" in a and "owner-approved correction" in a for a in r.allowed)
    assert r.kinds["owner-approved"] == 3


def test_c_T5_dividend_only_symbol_gets_no_adjusted_exemption(tmp_path):
    w = _approved_world(tmp_path)
    _adj_change(w, "BBB")
    r = _verify_expected(w, [BONUS, DIV])
    assert "adjusted:BBB" in failing_text(r)


def test_c_without_the_option_the_added_rows_fail(tmp_path):
    w = _approved_world(tmp_path)
    assert "corporate_actions:AAA" in failing_text(w.verify())


def test_c_T6_approved_row_plus_altered_history_row_fails(tmp_path):
    w = _approved_world(tmp_path)
    w.edit(w.actions / "AAA.parquet", lambda d: d.__setitem__("raw_subject", ["x", "CHANGED", "new"]))
    assert "corporate_actions:AAA" in failing_text(_verify_expected(w, [BONUS, DIV]))


def test_c_T7_expected_entry_not_present_fails(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    assert "expected action not present" in failing_text(_verify_expected(w, [BONUS]))


def test_c_T7_entry_matching_two_rows_is_ambiguous(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _add_rows(w.actions / "AAA.parquet", BONUS, BONUS)
    assert "ambiguous" in failing_text(_verify_expected(w, [BONUS]))


def test_c_T7_unapproved_extra_history_row_beside_an_approved_one_fails(tmp_path):
    w = _approved_world(tmp_path)
    _add_rows(w.actions / "AAA.parquet", _entry("AAA", "2026-09-27", "split", 0.25))
    r = _verify_expected(w, [BONUS, DIV])
    assert "corporate_actions:AAA" in failing_text(r)


def test_c_T7_wrong_cutoff_fails(tmp_path):
    w = _approved_world(tmp_path)
    r = _verify_expected(w, [BONUS, DIV], cutoff="2026-09-28")
    assert "history_cutoff" in failing_text(r)


def test_c_T7_expected_entry_for_symbol_without_a_unit_fails(tmp_path):
    w = _approved_world(tmp_path)
    r = _verify_expected(w, [BONUS, DIV, _entry("ZZZ", "2026-09-01", "dividend", None)])
    assert "ZZZ" in failing_text(r)


def test_c_T7_malformed_json_exits_1_and_bad_entries_are_rejected(tmp_path):
    w = _approved_world(tmp_path)
    bad = w.root / "bad.json"
    bad.write_text("{not json")
    assert hg.main(["verify", "--expected-changes", str(bad)], sources=w.sources()) == 1
    for payload in ({"actions": []}, {"history_cutoff": CUTOFF, "actions": [{"symbol": "AAA"}]},
                    {"history_cutoff": CUTOFF, "actions": "x"}):
        bad.write_text(json.dumps(payload))
        assert hg.main(["verify", "--expected-changes", str(bad)], sources=w.sources()) == 1


def test_c_cli_accepts_approved_changes_and_ignores_extra_keys(tmp_path):
    w = _approved_world(tmp_path)
    _adj_change(w, "AAA")
    path = _write_expected(w, [BONUS, DIV], note="extra keys are ignored")
    assert hg.main(["verify", "--expected-changes", str(path)], sources=w.sources()) == 0


def test_c_float_factor_is_compared_after_rounding(tmp_path):
    w = _ca_world(tmp_path)
    w.snapshot()
    _add_rows(w.actions / "AAA.parquet", _entry("AAA", "2026-09-28", "bonus", 0.1 + 0.2))
    r = _verify_expected(w, [_entry("AAA", "2026-09-28", "bonus", 0.3)])
    assert r.failures == [], failing_text(r)
