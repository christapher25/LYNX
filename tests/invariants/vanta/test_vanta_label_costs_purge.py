"""
tests/invariants/vanta/test_vanta_label_costs_purge.py

VANTA Stage 13A invariants N6 and N11 (docs/vanta/VANTA_SPEC.md).

  N6   Labels use the cost model of the trade date   the same trade priced before and after a known rate-change date gets a
                                                      different cost (costs/model.py introduces STT on 2004-10-01: 0.1% on each
                                                      delivery leg, so the round trip costs at least 0.2% more afterwards).
  N11  Purging                                        no training label with label_end_date >= test_start - embargo survives.

Same pattern as the other VANTA invariant tests: real module (skipped until it exists), a correct toy, deliberately broken toys.
For a trade straddling a rate change the spec ('trade-date rates') does not say which date's rates apply; that ambiguity is
DELIBERATELY not asserted. N6 only uses trades wholly before / wholly after the change.

Final N11 signature assumed: purge_training_rows(labels_df, test_start, embargo, sessions); embargo in SESSIONS, `sessions` a
sorted DatetimeIndex trading calendar; cutoff = the session `embargo` positions before test_start.
"""

import pandas as pd
import pytest

from vanta_toys_labels import make_bars, make_hits, purge_training_rows, toy

SYM, ATR, KS, KT, H = "TESTCO", 2.0, 1.0, 2.0, 5
STT_START = pd.Timestamp("2004-10-01")


# ---------------------------------------------------------------- N6
def _quiet_trade(sim, start):
    """A time-exit trade with no stop/target touch: entry 100, exit close 100.5, identical price path whatever the date."""
    bar = (100, 101, 99.5, 100.5)
    bars = make_bars([(99, 101, 98, 100)] + [bar] * 6, start=start)
    hits, cov = make_hits(bars, SYM)
    return sim(SYM, bars, bars.index[0], H, KS, KT, ATR, hits, cov)   # cost_fn=None: the default, trade-date cost model


def check_n6(sim):
    """N6: output changes across a known STT rate-change date (trade dates, not the label-build date, pick the rates)."""
    before = _quiet_trade(sim, "2004-09-20")     # entry 2004-09-21 .. exit 2004-09-27, all before 2004-10-01
    after = _quiet_trade(sim, "2004-10-04")      # entry 2004-10-05 .. exit 2004-10-11, all after
    assert before["exit_reason"] == after["exit_reason"] == "TIME"
    assert before["gross_return"] == pytest.approx(after["gross_return"])      # identical price path
    assert before["exit_date"] < STT_START <= after["entry_date"]

    legs = lambda r: r["cost_entry"] + r["cost_exit"]                           # noqa: E731
    assert legs(after) > legs(before)
    assert legs(after) - legs(before) >= 0.002 * 0.99                           # STT 0.1% buy + 0.1% sell appears
    assert after["y_net"] < before["y_net"]
    assert before["y_net"] - after["y_net"] >= 0.002 * 0.99
    # y_net is gross minus ALL stored cost legs
    for r in (before, after):
        assert r["y_net"] == pytest.approx(r["gross_return"] - r["cost_entry"] - r["cost_exit"] - r["cost_impact"])


def test_n6_real():
    sim = pytest.importorskip("vanta.labels").simulate_trade
    check_n6(sim)


def test_n6_reference_toy_passes():
    check_n6(toy())


@pytest.mark.parametrize("bug", ["frozen_cost_date"])
def test_n6_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n6(toy(bug))


# ---------------------------------------------------------------- N11
# Session calendar with a weekend and a holiday (2019-12-25) near the test start, so a calendar-day embargo is a different cutoff.
SESSIONS = pd.bdate_range("2019-05-01", "2020-03-31").drop(pd.Timestamp("2019-12-25"))
TEST_START = pd.Timestamp("2020-01-02")
EMBARGO = 5                                          # SESSIONS (v6: 5, 20 or 60 sessions), not calendar days
CUTOFF = SESSIONS[SESSIONS.get_loc(TEST_START) - EMBARGO]
assert CUTOFF == pd.Timestamp("2019-12-26")         # 5 sessions back: 01-01, 12-31, 12-30, 12-27, 12-26 (calendar-day cutoff: 12-28)


def _labels():
    d = pd.Timestamp
    # (name, decision_date, label_end_date)
    rows = [
        ("old", d("2019-06-03"), d("2019-06-10")),
        ("just_before", d("2019-12-16"), d("2019-12-24")),                 # kept: strictly before the cutoff
        ("on_boundary", d("2019-12-18"), CUTOFF),                          # dropped: >= cutoff
        ("calendar_gap", d("2019-12-19"), d("2019-12-27")),                # dropped; a calendar-day embargo (cutoff 12-28) keeps it
        ("in_embargo", d("2019-12-20"), d("2019-12-31")),                  # dropped
        ("straddles_test", d("2019-12-30"), d("2020-01-10")),              # dropped (label ends inside the test fold)
        ("long_label_early_decision", d("2019-11-01"), d("2019-12-30")),   # dropped: decision early, label_end late
    ]
    return pd.DataFrame(rows, columns=["name", "decision_date", "label_end_date"])


def check_n11(purge):
    """N11: purging drops every training row with label_end_date >= test_start - embargo SESSIONS; keeps the rest."""
    df = _labels()
    kept = purge(df, TEST_START, EMBARGO, SESSIONS)
    assert list(kept["name"]) == ["old", "just_before"]
    assert (kept["label_end_date"] < CUTOFF).all()
    # a bigger embargo only ever removes more rows
    kept_more = purge(df, TEST_START, 20, SESSIONS)
    assert set(kept_more["name"]) <= set(kept["name"])
    assert list(kept_more["name"]) == ["old"]
    # the input is not mutated
    assert len(df) == 7


def test_n11_real():
    purge = pytest.importorskip("vanta.walkforward").purge_training_rows
    check_n11(purge)


def test_n11_reference_toy_passes():
    check_n11(purge_training_rows)


@pytest.mark.parametrize("bug", ["strict_inequality", "no_embargo", "uses_decision_date", "calendar_days"])
def test_n11_broken_toy_fails(bug):
    broken = lambda df, ts, emb, sess: purge_training_rows(df, ts, emb, sess, _bugs=frozenset({bug}))   # noqa: E731
    with pytest.raises(AssertionError):
        check_n11(broken)
