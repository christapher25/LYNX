"""
tests/invariants/test_pitstore_invariants.py

Stage 12 invariant: a read as of date t returns only what was knowable at t. Three silent-failure modes, each proved with
planted data and with a deliberately broken reader that the test must catch:

  1. LEAK      -- a row/value whose availability is after t is never returned (a reader that admits rows <= t+1 session, or that
                  treats a next-session-published block's row date as its availability, fails).
  2. TRUNCATION -- the read at t is identical when everything after t is deleted from the data and the calendar.
  3. MISSING   -- a NaN-flagged row stays NaN + flagged: no forward fill into it, no zero fill, an absent row is not carried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitstore import store
from pitstore.store import BlockSpec, select_as_of

SESSIONS = list(pd.bdate_range("2024-03-04", periods=12).date)        # Mon 4 Mar .. Tue 19 Mar
D = lambda i: pd.Timestamp(SESSIONS[i])                                # noqa: E731


def _spec_symbol(source_col=None):
    return BlockSpec(name="sym", scope="symbol", value_columns=("v",), flag_columns=("is_missing_sym",), source_column=source_col)


def _spec_market():
    # mirrors flows: row date = placement session, source_date = the reporting day, which must be strictly before the row date
    return BlockSpec(name="mkt", scope="market", value_columns=("net",), flag_columns=("is_missing_mkt",), source_column="source_date")


def _sym_frame(symbol="AAA", rows=range(12), missing=()):
    n = len(list(rows))
    frame = pd.DataFrame({"symbol": symbol, "date": [D(i) for i in rows], "v": [100.0 + i for i in rows]})
    frame["is_missing_sym"] = False
    for i in missing:
        frame.loc[frame["date"] == D(i), ["v", "is_missing_sym"]] = [np.nan, True]
    return frame


def _mkt_frame(rows=range(12)):
    frame = pd.DataFrame({"date": [D(i) for i in rows], "net": [10.0 * i for i in rows]})
    frame["source_date"] = [D(i - 1) if i >= 1 else pd.NaT for i in rows]    # figure of the previous session, public the next
    frame["is_missing_mkt"] = False
    return frame


def _read(frames, specs, as_of, **kw):
    kw.setdefault("before_nightly_run", False)
    return select_as_of(frames, specs, ["AAA"], pd.Timestamp(as_of), SESSIONS, **kw)


def _one(out, column):
    return out[column].iloc[-1]


# ---------------------------------------------------------------- 1. no leak

def test_future_rows_are_never_returned():
    frames = {"sym": _sym_frame()}
    out = _read(frames, {"sym": _spec_symbol()}, SESSIONS[5], lookback_sessions=12)
    assert out["date"].max() == D(5)
    assert (out["date"] <= D(5)).all()
    assert out["v"].tolist() == [100.0 + i for i in range(6)]


def test_planted_future_value_is_excluded_even_when_huge():
    frame = _sym_frame()
    frame.loc[frame["date"] == D(6), "v"] = 1e9                    # the tell-tale future row
    out = _read({"sym": frame}, {"sym": _spec_symbol()}, SESSIONS[5], lookback_sessions=12)
    assert 1e9 not in out["v"].tolist()


def test_before_nightly_run_excludes_the_as_of_session_itself():
    out = _read({"sym": _sym_frame()}, {"sym": _spec_symbol()}, SESSIONS[5], before_nightly_run=True)
    assert out["date"].iloc[-1] == D(4)


def test_misdated_row_is_not_available_before_its_source_is_public():
    """A flows-style row dated D(5) whose figure describes D(5) itself (published D(6)) must NOT be readable as of D(5)."""
    frame = _mkt_frame()
    frame.loc[frame["date"] == D(5), "source_date"] = D(5)         # planted: figure of the row's own day
    frame.loc[frame["date"] == D(5), "net"] = 777.0
    out = _read({"mkt": frame}, {"mkt": _spec_market()}, SESSIONS[5])
    row = out.iloc[-1]
    assert row["date"] == D(5)
    assert np.isnan(row["net"]) and row["mkt_row_absent"] and row["is_missing_mkt"]
    # and it IS readable once D(6) has run
    later = _read({"mkt": frame}, {"mkt": _spec_market()}, SESSIONS[6], lookback_sessions=2)
    assert later.loc[later["date"] == D(5), "net"].iloc[0] == 777.0


def _mutant_plus_one_session(monkeypatch):
    original = store.knowable_cutoff
    monkeypatch.setattr(store, "knowable_cutoff", lambda as_of, before: original(as_of, before) + pd.Timedelta(days=1))


def _mutant_date_as_availability(monkeypatch):
    monkeypatch.setattr(store, "compute_availability", lambda spec, frame, sessions: pd.to_datetime(frame["date"]))


def test_mutant_reader_admitting_t_plus_one_is_caught(monkeypatch):
    _mutant_plus_one_session(monkeypatch)
    out = _read({"sym": _sym_frame()}, {"sym": _spec_symbol()}, SESSIONS[5])
    with pytest.raises(AssertionError):
        assert out["date"].max() <= D(5)                           # the invariant, as the real tests state it


def test_mutant_row_date_as_availability_is_caught(monkeypatch):
    frame = _mkt_frame()
    frame.loc[frame["date"] == D(5), "source_date"] = D(5)
    frame.loc[frame["date"] == D(5), "net"] = 777.0
    _mutant_date_as_availability(monkeypatch)
    out = _read({"mkt": frame}, {"mkt": _spec_market()}, SESSIONS[5])
    with pytest.raises(AssertionError):
        assert np.isnan(out.iloc[-1]["net"])                       # the leak the real reader prevents


# ---------------------------------------------------------------- 2. truncation invariance

@pytest.mark.parametrize("t", [2, 5, 8, 11])
def test_read_is_identical_when_data_after_t_is_deleted(t):
    frames = {"sym": _sym_frame(missing=(3,)), "mkt": _mkt_frame()}
    specs = {"sym": _spec_symbol(), "mkt": _spec_market()}
    full = select_as_of(frames, specs, ["AAA"], D(t), SESSIONS, lookback_sessions=4, before_nightly_run=False)
    cut = {k: v[v["date"] <= D(t)] for k, v in frames.items()}
    truncated = select_as_of(cut, specs, ["AAA"], D(t), SESSIONS[: t + 1], lookback_sessions=4, before_nightly_run=False)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), truncated.reset_index(drop=True))


def test_truncation_check_would_catch_a_reader_peeking_forward(monkeypatch):
    """Mutant: a reader whose cutoff runs one session ahead. Its full-data read differs from the truncated-data read."""
    frames = {"sym": _sym_frame()}
    specs = {"sym": _spec_symbol()}
    _mutant_plus_one_session(monkeypatch)
    t = 5
    full = select_as_of(frames, specs, ["AAA"], D(t), SESSIONS, lookback_sessions=3, before_nightly_run=False)
    truncated = select_as_of({"sym": frames["sym"][frames["sym"]["date"] <= D(t)]}, specs, ["AAA"], D(t), SESSIONS[: t + 1],
                             lookback_sessions=3, before_nightly_run=False)
    assert not full.reset_index(drop=True).equals(truncated.reset_index(drop=True))


# ---------------------------------------------------------------- 3. missing stays missing

def test_nan_flagged_row_is_neither_forward_filled_nor_zeroed():
    out = _read({"sym": _sym_frame(missing=(4,))}, {"sym": _spec_symbol()}, SESSIONS[5], lookback_sessions=3)
    row = out.loc[out["date"] == D(4)].iloc[0]
    assert np.isnan(row["v"]) and bool(row["is_missing_sym"]) and not bool(row["sym_row_absent"])
    assert row["v"] != 0
    assert out.loc[out["date"] == D(5), "v"].iloc[0] == 105.0     # the next present row is untouched


def test_absent_row_is_not_carried_forward_from_an_older_row():
    frame = _sym_frame(rows=[0, 1, 2, 3])                          # nothing for sessions 4 and 5
    out = _read({"sym": frame}, {"sym": _spec_symbol()}, SESSIONS[5])
    row = out.iloc[-1]
    assert row["date"] == D(5)
    assert np.isnan(row["v"]) and bool(row["is_missing_sym"]) and bool(row["sym_row_absent"])


def test_symbol_without_any_data_is_all_missing_not_dropped():
    out = select_as_of({"sym": _sym_frame("AAA")}, {"sym": _spec_symbol()}, ["AAA", "ZZZ"], D(5), SESSIONS, before_nightly_run=False)
    z = out[out["symbol"] == "ZZZ"].iloc[0]
    assert np.isnan(z["v"]) and bool(z["is_missing_sym"]) and bool(z["sym_row_absent"])


def test_duplicate_symbol_date_rows_raise():
    frame = pd.concat([_sym_frame(), _sym_frame().iloc[[3]]])
    with pytest.raises(ValueError, match="duplicate"):
        _read({"sym": frame}, {"sym": _spec_symbol()}, SESSIONS[5])


def test_mutant_zero_fill_is_caught(monkeypatch):
    real = store._mask_rows

    def zero_filling(frame, spec, mask):
        out = real(frame, spec, mask)
        out[list(spec.value_columns)] = out[list(spec.value_columns)].fillna(0.0)
        return out
    monkeypatch.setattr(store, "_mask_rows", zero_filling)
    out = _read({"sym": _sym_frame(rows=[0, 1, 2, 3])}, {"sym": _spec_symbol()}, SESSIONS[5])
    with pytest.raises(AssertionError):
        assert np.isnan(out.iloc[-1]["v"])


# ---------------------------------------------------------------- real files (skip where absent)

def test_real_block_files_truncation_invariance():
    from config import DELIVERY_FEATURES_DIR, FLOW_FEATURES_DIR
    if not (DELIVERY_FEATURES_DIR / "RELIANCE.parquet").exists() or not (FLOW_FEATURES_DIR / "fii_flows.parquet").exists():
        pytest.skip("Stage 10 feature files not present")
    from pitstore.store import BLOCK_SPECS, read_as_of, truncated_loader, default_loader
    from data.trading_calendar import trading_sessions
    sessions = trading_sessions()
    blocks = list(BLOCK_SPECS)
    for t in [pd.Timestamp("2022-08-08"), pd.Timestamp("2024-02-19"), pd.Timestamp("2026-01-15")]:
        for before in (False, True):
            a = read_as_of(["RELIANCE", "TCS"], t, blocks, sessions, before_nightly_run=before, lookback_sessions=5)
            b = read_as_of(["RELIANCE", "TCS"], t, blocks, [x for x in sessions if pd.Timestamp(x) <= t], before_nightly_run=before,
                           lookback_sessions=5, loader=truncated_loader(default_loader, t))
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
            assert (a["date"] <= (t - pd.Timedelta(days=1) if before else t)).all()


def test_before_nightly_run_is_a_required_argument():
    with pytest.raises(TypeError):
        select_as_of({}, {}, ["AAA"], D(5), SESSIONS)
    with pytest.raises(TypeError):
        store.read_as_of(["AAA"], D(5), [], SESSIONS)


# ---------------------------------------------------------------- measured-late publications: lag of one session

_LATE_BLOCKS = {"delivery": "is_missing_delivery", "fno_oi": "is_missing_fno"}


def _late_frame(name):
    spec = store.BLOCK_SPECS[name]
    frame = pd.DataFrame({"symbol": "AAA", "date": [D(i) for i in range(12)]})
    for c in spec.value_columns:
        frame[c] = [1000.0 + i for i in range(12)]
    for c in spec.flag_columns:
        frame[c] = False
    return spec, frame


@pytest.mark.parametrize("name", list(_LATE_BLOCKS))
def test_late_published_value_dated_s_is_not_returned_at_s_but_is_at_next_session(name):
    spec, frame = _late_frame(name)
    first = spec.value_columns[0]
    at_s = select_as_of({name: frame}, {name: spec}, ["AAA"], D(5), SESSIONS, before_nightly_run=False, lookback_sessions=2)
    row_s = at_s[at_s["date"] == D(5)].iloc[0]
    assert np.isnan(row_s[first]) and row_s[f"{name}_row_absent"] and row_s[_LATE_BLOCKS[name]]
    assert at_s[at_s["date"] == D(4)][first].iloc[0] == 1004.0              # the previous session's figure is readable
    at_next = select_as_of({name: frame}, {name: spec}, ["AAA"], D(6), SESSIONS, before_nightly_run=False, lookback_sessions=2)
    assert at_next[at_next["date"] == D(5)][first].iloc[0] == 1005.0 and not at_next[at_next["date"] == D(5)][f"{name}_row_absent"].iloc[0]


def test_ban_and_circuit_columns_keep_row_date_availability():
    for name in ("fno_ban", "circuits"):
        spec = store.BLOCK_SPECS[name]
        frame = pd.DataFrame({"symbol": "AAA", "date": [D(i) for i in range(12)]})
        for c in spec.value_columns:
            frame[c] = 1.0
        for c in spec.flag_columns:
            frame[c] = False
        out = select_as_of({name: frame}, {name: spec}, ["AAA"], D(5), SESSIONS, before_nightly_run=False)
        assert not out[f"{name}_row_absent"].iloc[0] and out[spec.value_columns[0]].iloc[0] == 1.0


def test_lag_without_a_documented_reason_is_refused():
    with pytest.raises(ValueError):
        BlockSpec("x", "symbol", ("v",), ("f",), None, None, 1, "")


@pytest.mark.parametrize("name", list(_LATE_BLOCKS))
def test_mutant_old_row_date_availability_fails_the_lag_test(monkeypatch, name):
    monkeypatch.setattr(store, "compute_availability", lambda spec, frame, sessions: pd.to_datetime(frame["date"]))
    spec, frame = _late_frame(name)
    at_s = select_as_of({name: frame}, {name: spec}, ["AAA"], D(5), SESSIONS, before_nightly_run=False)
    with pytest.raises(AssertionError):
        assert at_s[f"{name}_row_absent"].iloc[0]                           # what the real reader guarantees


def test_mutant_forward_fill_of_an_absent_row_is_caught(monkeypatch):
    real = store._mask_rows

    def forward_filling(frame, spec, mask):
        out = real(frame, spec, mask)
        filled = out.groupby("symbol")[list(spec.value_columns)].ffill() if "symbol" in out else out[list(spec.value_columns)].ffill()
        out[list(spec.value_columns)] = filled
        return out
    monkeypatch.setattr(store, "_mask_rows", forward_filling)
    # the planted frame's last two sessions are absent after masking, but an older row exists to carry forward
    out = _read({"sym": _sym_frame(rows=[0, 1, 2, 3])}, {"sym": _spec_symbol()}, SESSIONS[5], lookback_sessions=3)
    with pytest.raises(AssertionError):
        assert np.isnan(out.iloc[-1]["v"])
