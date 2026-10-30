"""Stage 11 additions for M1 (user decisions 15 and 16, 2026-10-30): lower-circuit gate on for buy signals, a liquidity-cliff gate
(outside the top 200 by trailing turnover, point-in-time), and the explicit list of gates NOT APPLIED. Mutants run on source copies."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gates.liquidity as liq
import gates.override_gates as og
from gates.liquidity import MIN_HISTORY, TRAILING_SESSIONS, trailing_turnover_rank
from gates.override_gates import (BLOCK, BUY_SIGNAL_GATES, LIQUIDITY_TOP_N, NOT_APPLIED_GATES, PASS, UNAVAILABLE, GateConfig,
                                  evaluate_gates)


def _frame(rank, rank_missing=False, up=0.0, lo=0.0, ban=0.0):
    return pd.DataFrame([{"hit_upper_circuit": up, "hit_lower_circuit": lo, "fno_in_ban": ban, "is_missing_circuit_hits": False,
                          "is_missing_ban": False, "liquidity_rank": rank, "is_missing_liquidity_rank": rank_missing}])


M1 = GateConfig(enabled=BUY_SIGNAL_GATES)


# ---- the gate -------------------------------------------------------------------------------------------------------

def test_buy_signal_set_is_circuits_ban_and_liquidity():
    assert set(BUY_SIGNAL_GATES) == {"upper_circuit", "lower_circuit", "fno_ban", "liquidity_cliff"}
    assert LIQUIDITY_TOP_N == 200


@pytest.mark.parametrize("rank,expected", [(1, PASS), (199, PASS), (200, PASS), (201, BLOCK), (700, BLOCK), (2300, BLOCK)])
def test_cliff_is_strictly_outside_the_top_200(rank, expected):
    assert evaluate_gates(_frame(float(rank)), M1).loc[0, "gate_liquidity_cliff"] == expected


def test_unknown_rank_is_unavailable_never_pass():
    for rank, flag in ((np.nan, True), (np.nan, False), (50.0, True)):
        out = evaluate_gates(_frame(rank, flag), M1)
        assert out.loc[0, "gate_liquidity_cliff"] == UNAVAILABLE and not out.loc[0, "gate_allowed"]


def test_lower_circuit_blocks_a_buy_under_the_buy_signal_config():
    out = evaluate_gates(_frame(10.0, lo=1.0), M1)
    assert out.loc[0, "gate_status"] == BLOCK and out.loc[0, "gate_reasons"] == "lower_circuit=BLOCK"


def test_a_clean_liquid_row_passes():
    out = evaluate_gates(_frame(10.0), M1)
    assert out.loc[0, "gate_status"] == PASS and out.loc[0, "gate_allowed"]


def test_threshold_is_configurable_and_validated():
    assert evaluate_gates(_frame(120.0), GateConfig(enabled=BUY_SIGNAL_GATES, liquidity_top_n=100)).loc[0, "gate_liquidity_cliff"] == BLOCK
    with pytest.raises(ValueError):
        GateConfig(enabled=BUY_SIGNAL_GATES, liquidity_top_n=0)


def test_invalid_rank_raises():
    for bad in (0.0, -3.0, 2.5):
        with pytest.raises(ValueError, match="liquidity_rank"):
            evaluate_gates(_frame(bad), M1)


def test_missing_rank_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        evaluate_gates(_frame(10.0).drop(columns=["liquidity_rank"]), M1)


def test_gates_without_liquidity_do_not_need_its_columns():
    out = evaluate_gates(_frame(10.0).drop(columns=["liquidity_rank", "is_missing_liquidity_rank"]), GateConfig())
    assert out.loc[0, "gate_liquidity_cliff"] == og.DISABLED


def test_not_applied_gates_are_listed_with_reasons():
    assert {"corporate_action_frequency", "pledging"} <= set(NOT_APPLIED_GATES)
    assert all(isinstance(v, str) and v for v in NOT_APPLIED_GATES.values())
    assert not set(NOT_APPLIED_GATES) & set(BUY_SIGNAL_GATES)


def _mutant(module, old, new):
    src = Path(module.__file__).read_text(encoding="utf-8")
    assert old in src, f"mutation anchor moved: {old!r}"
    m = types.ModuleType("mutant")
    m.__file__ = module.__file__
    sys.modules["mutant"] = m                                                           # dataclasses look their module up here
    exec(compile(src.replace(old, new), "mutant", "exec"), m.__dict__)
    return m


def test_mutant_cliff_at_or_above_200_is_caught():
    m = _mutant(og, "numeric > top_n", "numeric >= top_n")
    assert m.evaluate_gates(_frame(200.0), M1).loc[0, "gate_liquidity_cliff"] == BLOCK      # the mutant is wrong at the boundary...
    assert evaluate_gates(_frame(200.0), M1).loc[0, "gate_liquidity_cliff"] == PASS          # ...the real gate is not


# ---- the point-in-time rank -----------------------------------------------------------------------------------------

def _panel(n_days=TRAILING_SESSIONS + 10, symbols=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for k, s in enumerate(symbols):
        for d in dates:
            rows.append((s, d, 100.0, 1000.0 * (len(symbols) - k) * (1 + 0.01 * rng.random())))   # A most liquid, then B, then C
    return pd.DataFrame(rows, columns=["SYMBOL", "DATE", "ADJ_CLOSE", "ADJ_VOLUME"])


def test_rank_orders_by_trailing_turnover():
    p = _panel()
    r = trailing_turnover_rank(p)
    last = r[r["date"] == r["date"].max()].set_index("symbol")
    assert list(last.loc[["A", "B", "C"], "liquidity_rank"]) == [1.0, 2.0, 3.0]
    assert not last["is_missing_liquidity_rank"].any()


def test_rank_is_nan_and_flagged_before_enough_history():
    r = trailing_turnover_rank(_panel())
    early = r[r["date"] == sorted(r["date"].unique())[MIN_HISTORY - 1]]       # only MIN_HISTORY-1 prior sessions exist
    assert early["liquidity_rank"].isna().all() and early["is_missing_liquidity_rank"].all()
    ok = r[r["date"] == sorted(r["date"].unique())[MIN_HISTORY]]
    assert ok["liquidity_rank"].notna().all()


def test_rank_at_t_ignores_session_t_itself():
    p = _panel()
    t = sorted(p["DATE"].unique())[-1]
    base = trailing_turnover_rank(p)
    trap = p.copy()
    trap.loc[(trap["SYMBOL"] == "C") & (trap["DATE"] == t), "ADJ_VOLUME"] = 1e12      # C explodes ON t: unknowable before t
    after = trailing_turnover_rank(trap)
    pick = lambda f: f[f["date"] == t].set_index("symbol")["liquidity_rank"]          # noqa: E731
    assert (pick(base) == pick(after)).all()
    nxt = trap.copy()
    nxt.loc[len(nxt)] = ["C", pd.Timestamp(t) + pd.offsets.BDay(1), 100.0, 1e12]
    assert trailing_turnover_rank(nxt).query("date > @t").set_index("symbol").loc["C", "liquidity_rank"].item() < 3.0   # known one day later


def test_truncation_never_changes_earlier_ranks():
    p = _panel(n_days=TRAILING_SESSIONS + 60, symbols=("A", "B", "C", "D"), seed=3)
    full = trailing_turnover_rank(p).set_index(["symbol", "date"])["liquidity_rank"]
    cut = pd.Timestamp(sorted(p["DATE"].unique())[TRAILING_SESSIONS + 30])
    part = trailing_turnover_rank(p[p["DATE"] <= cut]).set_index(["symbol", "date"])["liquidity_rank"]
    pd.testing.assert_series_equal(full.loc[part.index], part)


def test_mutant_including_session_t_is_caught():
    m = _mutant(liq, ".shift(1)", ".shift(0)")
    p = _panel()
    t = sorted(p["DATE"].unique())[-1]
    p.loc[(p["SYMBOL"] == "C") & (p["DATE"] == t), "ADJ_VOLUME"] = 1e12
    got = lambda mod: (lambda r: r[r["date"] == t].set_index("symbol")["liquidity_rank"])(mod.trailing_turnover_rank(p))  # noqa: E731
    assert got(m)["C"] == 1.0 and got(liq)["C"] == 3.0


def test_nan_turnover_is_missing_not_zero():
    p = _panel()
    p.loc[p["SYMBOL"] == "B", "ADJ_VOLUME"] = np.nan
    r = trailing_turnover_rank(p)
    last = r[r["date"] == r["date"].max()].set_index("symbol")
    assert np.isnan(last.loc["B", "liquidity_rank"]) and last.loc["B", "is_missing_liquidity_rank"]
    assert list(last.loc[["A", "C"], "liquidity_rank"]) == [1.0, 2.0]                   # B did not take a rank, and was not ranked last as 0


def test_a_name_that_stopped_trading_stops_taking_a_rank_slot():
    p = _panel(symbols=("Z", "A", "B"))
    last = sorted(p["DATE"].unique())[-1]
    p = p[~((p["SYMBOL"] == "Z") & (p["DATE"] > sorted(p["DATE"].unique())[TRAILING_SESSIONS - 40]))]      # Z, the most liquid, dies
    r = trailing_turnover_rank(p)
    end = r[r["date"] == last].set_index("symbol")
    assert list(end.loc[["A", "B"], "liquidity_rank"]) == [1.0, 2.0]
    assert "Z" not in end.index or np.isnan(end.loc["Z", "liquidity_rank"])
    soon = r[r["date"] == sorted(p["DATE"].unique())[TRAILING_SESSIONS - 38]].set_index("symbol")             # just after: still ranked
    assert soon.loc["A", "liquidity_rank"] == 2.0


def test_start_argument_limits_output_but_not_history():
    p = _panel()
    start = pd.Timestamp(sorted(p["DATE"].unique())[-3])
    r = trailing_turnover_rank(p, start=start)
    assert r["date"].min() == start and r["liquidity_rank"].notna().all()


# ---- loader merge ---------------------------------------------------------------------------------------------------

def test_loader_rank_merge_missing_row_is_flagged(monkeypatch, tmp_path):
    import gates.io as gio
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cdir, fdir = tmp_path / "c", tmp_path / "f"
    cdir.mkdir(), fdir.mkdir()
    pd.DataFrame({"symbol": "A", "date": dates, "hit_upper_circuit": 0.0, "hit_lower_circuit": 0.0, "is_missing_circuit_hits": False,
                  "price_band_pct": 20.0}).to_parquet(cdir / "A.parquet")
    pd.DataFrame({"symbol": "A", "date": dates, "fno_in_ban": 0.0, "is_missing_ban": False}).to_parquet(fdir / "A.parquet")
    monkeypatch.setattr(gio, "CIRCUIT_FEATURES_DIR", cdir)
    monkeypatch.setattr(gio, "FNO_FEATURES_DIR", fdir)
    ranks = pd.DataFrame({"symbol": ["A"], "date": dates[:1], "liquidity_rank": [5.0], "is_missing_liquidity_rank": [False]})
    out = gio.load_gate_inputs(["A"], liquidity_ranks=ranks).set_index("date")
    assert out.loc[dates[0], "liquidity_rank"] == 5.0 and not out.loc[dates[0], "is_missing_liquidity_rank"]
    assert np.isnan(out.loc[dates[1], "liquidity_rank"]) and out.loc[dates[1], "is_missing_liquidity_rank"]
    bare = gio.load_gate_inputs(["A"])                                                  # no rank supplied: unknown, never a default
    assert bare["liquidity_rank"].isna().all() and bare["is_missing_liquidity_rank"].all()
