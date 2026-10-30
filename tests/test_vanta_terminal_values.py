"""Stage 13A: Stage 5 delisted terminal values wired into the label simulator (user decision 8, 2026-10-30).

An open trade at the end of a delisted name's history exits at the Stage 5 terminal value (the last traded adjusted close) instead of
NO_DATA. A name with no terminal value stays NO_DATA. Mutants are applied to a source copy of vanta/labels.py, never to the repo.
"""

import math
import types
from pathlib import Path

import pandas as pd
import pytest

import vanta.labels as labels
from vanta.labels import load_terminal_values, simulate_trade, terminal_kwargs

SYM = "GONECO"
ZERO = lambda *a: (0.0, 0.0, 0.0)  # noqa: E731
FLAT = (100, 101, 99, 100)
LAST = pd.Timestamp("2024-01-04")


def _bars(ohlc, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=len(ohlc))
    cols = {}
    for pre in ("raw_", ""):
        for k, name in enumerate(("open", "high", "low", "close")):
            cols[pre + name] = [r[k] for r in ohlc]
    return pd.DataFrame(cols, index=idx)


def _run(module=labels, **kw):
    b = _bars([FLAT, FLAT, FLAT, (100, 101, 99, 100.5)])               # history ends 2024-01-04, h=5 runs past it
    hits = pd.DataFrame(columns=["date", "symbol", "side"])
    return module.simulate_trade(SYM, b, b.index[0], 5, 1.0, 2.0, 2.0, hits, set(b.index), cost_fn=ZERO, **kw)


# ---- simulate_trade -------------------------------------------------------------------------------------------------

def test_terminal_value_is_the_exit_price():
    r = _run(delisted_after=LAST, terminal_value=100.5)
    assert r["fill_status"] == "FILLED" and r["exit_reason"] == "DELISTED"
    assert r["exit_price"] == 100.5 and r["exit_date"] == LAST and r["gross_return"] == pytest.approx(0.005)


def test_no_terminal_value_stays_no_data():
    for kw in ({}, {"delisted_after": None, "terminal_value": None}):
        r = _run(**kw)
        assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"]) and math.isnan(r["exit_price"])


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -3.0])
def test_unusable_terminal_value_stays_no_data(bad):
    r = _run(delisted_after=LAST, terminal_value=bad)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])


def test_terminal_value_disagreeing_with_last_adjusted_close_is_not_used():
    # a terminal value in different price units than the bars would produce a fantasy return: unpriced, never guessed
    r = _run(delisted_after=LAST, terminal_value=50.25)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])


def test_active_name_is_not_exited_by_a_terminal_value():
    # delisted_after earlier than the last bar: the name traded after that date, so it is not the end of its history
    r = _run(delisted_after=pd.Timestamp("2024-01-02"), terminal_value=100.5)
    assert r["fill_status"] == "NO_DATA"


def test_bars_cut_before_the_delisting_date_get_no_delisted_exit():
    # the bars end 2024-01-04 but the name's last traded date is later: the history is truncated, not ended
    r = _run(delisted_after=pd.Timestamp("2024-01-10"), terminal_value=100.5)
    assert r["fill_status"] == "NO_DATA" and math.isnan(r["y_net"])


def test_mutant_restoring_le_on_the_ended_test_is_caught():
    src = Path(labels.__file__).read_text(encoding="utf-8")
    needle = "idx[last_j] == pd.Timestamp(delisted_after)"
    assert needle in src, "mutation anchor moved: update this test"
    mutant = types.ModuleType("labels_mutant_le")
    mutant.__file__ = labels.__file__
    exec(compile(src.replace(needle, "idx[last_j] <= pd.Timestamp(delisted_after)"), "labels_mutant_le", "exec"), mutant.__dict__)
    assert _run(mutant, delisted_after=pd.Timestamp("2024-01-10"), terminal_value=100.5)["fill_status"] == "FILLED"   # the bug
    assert _run(delisted_after=pd.Timestamp("2024-01-10"), terminal_value=100.5)["fill_status"] == "NO_DATA"


def test_a_real_stop_still_wins_over_the_terminal_value():
    b = _bars([FLAT, FLAT, (99, 99, 90, 91)])
    hits = pd.DataFrame(columns=["date", "symbol", "side"])
    r = simulate_trade(SYM, b, b.index[0], 5, 1.0, 2.0, 2.0, hits, set(b.index), cost_fn=ZERO,
                       delisted_after=b.index[-1], terminal_value=91.0)
    assert r["exit_reason"] == "STOP"


def test_mutant_ignoring_terminal_value_is_caught():
    src = Path(labels.__file__).read_text(encoding="utf-8")
    needle = "exit_px, reason = last_j, delist_px, \"DELISTED\""
    assert needle in src, "mutation anchor moved: update this test"
    mutant = types.ModuleType("labels_mutant")
    mutant.__file__ = labels.__file__
    exec(compile(src.replace(needle, "exit_px, reason = last_j, ac[last_j] * 1.5, \"DELISTED\""), "labels_mutant", "exec"), mutant.__dict__)
    assert _run(mutant, delisted_after=LAST, terminal_value=100.5)["exit_price"] != 100.5     # the real test above would fail on it
    assert _run(delisted_after=LAST, terminal_value=100.5)["exit_price"] == 100.5


# ---- load_terminal_values / terminal_kwargs ---------------------------------------------------------------------------

def _write_adjusted(directory, symbol, last_date, last_close):
    pd.DataFrame({"SYMBOL": symbol, "DATE": pd.to_datetime(["2000-01-03", last_date]), "ADJ_CLOSE": [10.0, last_close]}
                 ).to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def dirs(tmp_path):
    uni, adj = tmp_path / "universe", tmp_path / "adjusted"
    uni.mkdir(), adj.mkdir()
    rows = lambda *r: pd.DataFrame(r, columns=["SYMBOL", "RANK", "AVG_TURNOVER", "FIRST_DATE", "LAST_DATE", "IS_DELISTED", "TERMINAL_VALUE"])  # noqa: E731
    t = pd.Timestamp
    # 2023 snapshot: MIDYR died on 2023-09-01 (not yet "delisted" at the 2023 cutoff), OLD died in 2022, LIVE trades on
    rows(("MIDYR", 1, 9.0, t("2000-01-03"), t("2023-09-01"), False, None),
         ("OLD", 2, 8.0, t("2000-01-03"), t("2022-05-02"), True, 7.5),
         ("LIVE", 3, 7.0, t("2000-01-03"), t("2026-09-29"), False, None)).to_parquet(uni / "2023.parquet")
    # 2024 snapshot (the newest, cutoff 2024-01-01): MIDYR is now delisted by Stage 5's own rule, but fell outside the top list
    rows(("OLD", 1, 8.0, t("2000-01-03"), t("2022-05-02"), True, 7.5),
         ("LIVE", 2, 7.0, t("2000-01-03"), t("2026-09-29"), False, None),
         ("NOVAL", 3, 6.0, t("2000-01-03"), t("2022-01-03"), True, None),
         ("NOFILE", 4, 5.0, t("2000-01-03"), t("2022-01-03"), True, 4.0)).to_parquet(uni / "2024.parquet")
    _write_adjusted(adj, "MIDYR", "2023-09-01", 12.5)
    _write_adjusted(adj, "OLD", "2022-05-02", 7.5)
    _write_adjusted(adj, "LIVE", "2026-09-29", 55.0)
    _write_adjusted(adj, "NOVAL", "2022-01-03", float("nan"))
    return uni, adj


def test_flagged_symbol_uses_the_stage5_terminal_value(dirs):
    tv = load_terminal_values(*dirs)
    assert tv["OLD"] == (pd.Timestamp("2022-05-02"), 7.5)


def test_symbol_that_ended_before_the_newest_cutoff_gets_the_same_rule_value_from_stage4(dirs):
    tv = load_terminal_values(*dirs)
    assert tv["MIDYR"] == (pd.Timestamp("2023-09-01"), 12.5)          # Stage 5 never stored it (not in the newest top list)


def test_live_symbol_has_no_terminal_value(dirs):
    assert "LIVE" not in load_terminal_values(*dirs)


def test_no_usable_terminal_value_means_absent(dirs):
    tv = load_terminal_values(*dirs)
    assert "NOVAL" not in tv          # delisted, but its last adjusted close is NaN
    assert "NOFILE" not in tv         # flagged with 4.0 but no Stage 4 file to confirm the last date: not trusted blindly


def test_terminal_kwargs_round_trip(dirs):
    tv = load_terminal_values(*dirs)
    assert terminal_kwargs("OLD", tv) == {"delisted_after": pd.Timestamp("2022-05-02"), "terminal_value": 7.5}
    assert terminal_kwargs("LIVE", tv) == {}
    assert terminal_kwargs("NOVAL", tv) == {}
