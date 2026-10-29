"""Unit tests for gates/override_gates.py: the explicit truth table, the unavailable policy, and loud failure on corrupt input."""

import numpy as np
import pandas as pd
import pytest

from gates.override_gates import BLOCK, DISABLED, PASS, UNAVAILABLE, GateConfig, classify, evaluate_gates

NAN = np.nan


def _row(up=0.0, lo=0.0, ban=0.0, hits_missing=False, ban_missing=False):
    return {"hit_upper_circuit": up, "hit_lower_circuit": lo, "fno_in_ban": ban,
            "is_missing_circuit_hits": hits_missing, "is_missing_ban": ban_missing}


def _eval(rows, **config):
    return evaluate_gates(pd.DataFrame(rows), GateConfig(**config))


class TestClassify:
    def test_truth_table(self):
        value = pd.Series([0.0, 1.0, NAN, 0.0, 1.0, NAN], name="v")
        flag = pd.Series([False, False, False, True, True, True])
        assert list(classify(value, flag)) == [PASS, BLOCK, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE]

    def test_nan_flag_is_unavailable_not_pass(self):
        out = classify(pd.Series([0.0, 1.0], name="v"), pd.Series([NAN, None], dtype=object))
        assert list(out) == [UNAVAILABLE, UNAVAILABLE]

    def test_corrupt_value_raises(self):
        with pytest.raises(ValueError, match="outside"):
            classify(pd.Series([0.5], name="v"), pd.Series([False]))


class TestDefaultGates:
    def test_default_enables_upper_circuit_and_ban_only(self):
        out = _eval([_row(lo=1.0)])
        assert out.loc[0, "gate_lower_circuit"] == DISABLED
        assert out.loc[0, "gate_status"] == PASS and out.loc[0, "gate_allowed"]

    def test_upper_hit_blocks(self):
        out = _eval([_row(up=1.0)])
        assert out.loc[0, "gate_status"] == BLOCK and not out.loc[0, "gate_allowed"]
        assert out.loc[0, "gate_reasons"] == "upper_circuit=BLOCK"

    def test_ban_blocks(self):
        assert _eval([_row(ban=1.0)]).loc[0, "gate_reasons"] == "fno_ban=BLOCK"

    def test_clear_row_passes_with_empty_reasons(self):
        out = _eval([_row()])
        assert out.loc[0, "gate_status"] == PASS and out.loc[0, "gate_allowed"] and out.loc[0, "gate_reasons"] == ""

    def test_block_beats_unavailable_and_both_are_listed(self):
        out = _eval([_row(up=1.0, ban=NAN, ban_missing=True)])
        assert out.loc[0, "gate_status"] == BLOCK
        assert out.loc[0, "gate_reasons"] == "upper_circuit=BLOCK;fno_ban=UNAVAILABLE"

    def test_missing_ban_list_is_unavailable_not_nobody_banned(self):
        out = _eval([_row(ban=NAN, ban_missing=True)])
        assert out.loc[0, "gate_status"] == UNAVAILABLE and not out.loc[0, "gate_allowed"]


class TestUnavailablePolicy:
    def test_allow_policy_allows_but_status_still_says_unavailable(self):
        out = _eval([_row(ban=NAN, ban_missing=True)], unavailable_policy="allow")
        assert out.loc[0, "gate_status"] == UNAVAILABLE and out.loc[0, "gate_allowed"]

    def test_allow_policy_never_allows_a_block(self):
        assert not _eval([_row(ban=1.0)], unavailable_policy="allow").loc[0, "gate_allowed"]

    def test_disabled_gate_unavailable_input_is_ignored(self):
        out = _eval([_row(ban=NAN, ban_missing=True)], enabled=("upper_circuit",))
        assert out.loc[0, "gate_status"] == PASS


class TestLoudFailure:
    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            evaluate_gates(pd.DataFrame([_row()]).drop(columns="is_missing_ban"))

    def test_disabled_gate_column_not_required(self):
        evaluate_gates(pd.DataFrame([_row()]).drop(columns="hit_lower_circuit"), GateConfig(enabled=("upper_circuit", "fno_ban")))

    @pytest.mark.parametrize("kwargs", [{"enabled": ()}, {"enabled": ("bogus",)}, {"unavailable_policy": "maybe"}])
    def test_bad_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            GateConfig(**kwargs)

    def test_index_is_preserved(self):
        df = pd.DataFrame([_row(), _row(up=1.0)], index=[10, 7])
        assert list(evaluate_gates(df).index) == [10, 7]


# --- gates/io.py: every requested (symbol, date) gets a row; duplicates raise ----------------------------------------------

from gates import io as gate_io  # noqa: E402

DAYS = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])


@pytest.fixture
def blocks(tmp_path, monkeypatch):
    circuits, fno = tmp_path / "circuits", tmp_path / "fno"
    circuits.mkdir()
    fno.mkdir()
    monkeypatch.setattr(gate_io, "CIRCUIT_FEATURES_DIR", circuits)
    monkeypatch.setattr(gate_io, "FNO_FEATURES_DIR", fno)

    def write(symbol, days=DAYS, block=("circuits", "fno"), dup=False):
        if "circuits" in block:
            c = pd.DataFrame({"symbol": symbol, "date": days, "hit_upper_circuit": 0.0, "hit_lower_circuit": 0.0,
                              "is_missing_circuit_hits": False, "price_band_pct": 20.0})
            pd.concat([c, c.iloc[:1]] if dup else [c]).to_parquet(circuits / f"{symbol}.parquet")
        if "fno" in block:
            pd.DataFrame({"symbol": symbol, "date": days, "fno_in_ban": 0.0, "is_missing_ban": False}).to_parquet(fno / f"{symbol}.parquet")
    return write


def _skipping_loader(symbols, start=None, end=None, dates=None):
    """MUTANT: the old behaviour -- a symbol in neither block is silently dropped."""
    out = gate_io.load_gate_inputs([s for s in symbols if (gate_io.CIRCUIT_FEATURES_DIR / f"{s}.parquet").exists()
                                    or (gate_io.FNO_FEATURES_DIR / f"{s}.parquet").exists()], start, end, dates)
    return out


def _gate_ok_for_absent(loader) -> bool:
    frame = loader(["AAA", "GHOST"], dates=DAYS)
    ghost = frame[frame["symbol"] == "GHOST"]
    if len(ghost) != len(DAYS):
        return False
    verdict = evaluate_gates(ghost.reset_index(drop=True), GateConfig(enabled=("upper_circuit", "lower_circuit", "fno_ban")))
    return bool((verdict["gate_status"] == UNAVAILABLE).all() and not verdict["gate_allowed"].any())


class TestLoaderCompleteness:
    def test_absent_symbol_gets_unavailable_rows(self, blocks):
        blocks("AAA")
        assert _gate_ok_for_absent(gate_io.load_gate_inputs)

    def test_mutant_skipping_loader_is_caught(self, blocks):
        blocks("AAA")
        assert not _gate_ok_for_absent(_skipping_loader)

    def test_absent_date_in_one_symbol_is_unavailable(self, blocks):
        blocks("AAA", days=DAYS[:2])
        blocks("BBB")
        out = gate_io.load_gate_inputs(["AAA", "BBB"])
        row = out[(out["symbol"] == "AAA") & (out["date"] == DAYS[2])]
        assert len(row) == 1 and row["is_missing_circuit_hits"].all() and row["is_missing_ban"].all()
        assert len(out) == 6

    def test_one_block_only_flags_the_other(self, blocks):
        blocks("AAA", block=("circuits",))
        out = gate_io.load_gate_inputs(["AAA"])
        assert out["is_missing_ban"].all() and not out["is_missing_circuit_hits"].any() and out["fno_in_ban"].isna().all()

    def test_explicit_dates_honoured_exactly(self, blocks):
        blocks("AAA")
        out = gate_io.load_gate_inputs(["AAA", "GHOST"], dates=[DAYS[0], DAYS[2]])
        assert len(out) == 4 and set(out["date"]) == {DAYS[0], DAYS[2]}


def _merge_without_uniqueness_check(circuits, fno):
    return circuits.merge(fno, on=["symbol", "date"], how="outer")


class TestLoaderUniqueness:
    def test_duplicate_raises(self, blocks):
        blocks("AAA", dup=True)
        with pytest.raises(ValueError, match="sharing a"):
            gate_io.load_gate_inputs(["AAA"])

    def test_mutant_merge_without_check_multiplies_rows(self, blocks):
        blocks("AAA", dup=True)
        c = pd.read_parquet(gate_io.CIRCUIT_FEATURES_DIR / "AAA.parquet")
        f = pd.read_parquet(gate_io.FNO_FEATURES_DIR / "AAA.parquet")
        assert len(_merge_without_uniqueness_check(c, f)) > len(f)       # the harm the check prevents
