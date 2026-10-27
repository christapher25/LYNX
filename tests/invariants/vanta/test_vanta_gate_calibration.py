"""
tests/invariants/vanta/test_vanta_gate_calibration.py

VANTA Stage 13C invariants N5, N8, N9 (docs/vanta/VANTA_SPEC.md): what is allowed into the conformal calibration set.

  N5  The calibration set never holds an unmatured outcome    every calibration row has matured_at <= T; an injected row with
                                                                matured_at = T+1 is caught; adding outcomes after T never changes
                                                                p-values at T.
  N8  Calibration covers all scored candidates                 the calibration row count for a week equals candidates scored,
                                                                not names published.
  N9  Calibration and test share model version, candidate      a mismatched row (model_version, candidate_filter_version, or
      filter and OOS status                                    is_oos False) is rejected.

World: 4 prior weekly decision dates x 30 scored candidates (score 0..29, losers are scores < 20), T = 2024-06-03, every prior
week matured by T. p-values are checked against an independent plain-loop restatement of the spec formula (oracle_p).

"Rejected" is tested both ways (open Q3 to the lead): select_calibration must EXCLUDE offending rows, and validate_calibration
must RAISE ValueError on a set containing one.
"""

import numpy as np
import pandas as pd
import pytest

from vanta_toys_gate import FILTER, MODEL, cal_rows, calibration_world, oracle_p, toy_api

T = pd.Timestamp("2024-06-03")
PI_T = (1.0, 0.0, 0.0)
KW = dict(model_version=MODEL, candidate_filter_version=FILTER)
CANDS = pd.DataFrame({"symbol": list("ABCDEF"), "score": [-1.0, 5.5, 19.5, 25.5, 29.5, 50.0]})


def _gate(api, cal):
    return api.run_gate(CANDS, cal, T, PI_T, tau_weeks=26, q=0.3, cap=10, ess_floor=0, **KW)


def _real_api():
    return pytest.importorskip("vanta.gate")


# ---------------------------------------------------------------- N5
def _late_rows(decision_date, matured_at):
    """A loser with an extreme score: it would move every p-value if it ever leaked into the calibration set."""
    return cal_rows(decision_date, ["LATE"], [100.0], [-0.5], [matured_at])


def check_n5(api):
    """N5: every calibration row has matured_at <= T; matured_at = T+1 is caught; later outcomes never change p at T."""
    cal = calibration_world()
    base = api.select_calibration(cal, T, **KW)
    assert len(base) == 120 and (base["matured_at"] <= T).all()
    api.validate_calibration(base, T, **KW)                                    # a clean set passes

    d = T - pd.Timedelta(weeks=1)
    injected = pd.concat([cal, _late_rows(d, T + pd.Timedelta(days=1))], ignore_index=True)
    sel = api.select_calibration(injected, T, **KW)
    assert len(sel) == 120 and "LATE" not in set(sel["symbol"])
    with pytest.raises(ValueError):
        api.validate_calibration(injected, T, **KW)

    # a row whose outcome is not yet known (matured_at / y_net null) is not calibration data either
    pending = cal_rows(d, ["PEND"], [100.0], [np.nan], [pd.NaT])
    sel = api.select_calibration(pd.concat([cal, pending], ignore_index=True), T, **KW)
    assert len(sel) == 120

    # the boundary: matured_at == T is usable at T (the rule is <=)
    edge = _late_rows(d, T)
    sel = api.select_calibration(pd.concat([cal, edge], ignore_index=True), T, **KW)
    assert len(sel) == 121 and "LATE" in set(sel["symbol"])

    # adding outcomes after T (even losers with extreme scores, and changing their y) never changes p-values at T
    p0 = _gate(api, cal)["p_value"].to_numpy()
    np.testing.assert_allclose(p0, oracle_p(base, T, PI_T, 26, CANDS["score"]))
    future = pd.concat([cal, _late_rows(d, T + pd.Timedelta(days=1)), _late_rows(d, T + pd.Timedelta(days=30))], ignore_index=True)
    np.testing.assert_allclose(_gate(api, future)["p_value"].to_numpy(), p0)
    future.loc[future["symbol"] == "LATE", "y_net"] = 9.0
    np.testing.assert_allclose(_gate(api, future)["p_value"].to_numpy(), p0)


def test_n5_real():
    check_n5(_real_api())


def test_n5_reference_toy_passes():
    check_n5(toy_api())


@pytest.mark.parametrize("bug", ["no_maturity_filter", "maturity_off_by_one", "strict_maturity"])
def test_n5_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n5(toy_api(bug))


# ---------------------------------------------------------------- N8
def check_n8(api):
    """N8: calibration row count for a week equals candidates scored (30), not names published (3)."""
    cal = calibration_world()
    assert int(cal.groupby("decision_date")["published"].sum().iloc[0]) == 3          # the trap: only 3 per week were published
    sel = api.select_calibration(cal, T, **KW)
    counts = sel.groupby("decision_date").size()
    assert len(counts) == 4 and (counts == 30).all()
    log = _gate(api, cal)
    assert len(log) == len(CANDS)                                                    # the log covers every candidate
    assert (log["n_calibration"] == 120).all()
    np.testing.assert_allclose(log["p_value"].to_numpy(), oracle_p(sel, T, PI_T, 26, CANDS["score"]))
    # published-only calibration would see no losers at all (published = top scores = winners): p would be 1/(1+n), never large
    assert log["p_value"].max() > 0.5


def test_n8_real():
    check_n8(_real_api())


def test_n8_reference_toy_passes():
    check_n8(toy_api())


@pytest.mark.parametrize("bug", ["published_only"])
def test_n8_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n8(toy_api(bug))


# ---------------------------------------------------------------- N9
def check_n9(api):
    """N9: a calibration row with another model_version, another candidate_filter_version, or is_oos False is rejected."""
    cal = calibration_world()
    d = T - pd.Timedelta(weeks=1)
    m = d + pd.Timedelta(days=6)
    bad = {
        "wrong_model": cal_rows(d, ["BAD"], [100.0], [-0.5], [m], model="vanta-ranker@f06:2017-12-31:deadbee:000000"),
        "wrong_filter": cal_rows(d, ["BAD"], [100.0], [-0.5], [m], filt="cfilter@f07:2018-12-31:a1b2c3d:other0"),
        "not_oos": cal_rows(d, ["BAD"], [100.0], [-0.5], [m], oos=False),
    }
    base_p = _gate(api, cal)["p_value"].to_numpy()
    for name, row in bad.items():
        mixed = pd.concat([cal, row], ignore_index=True)
        sel = api.select_calibration(mixed, T, **KW)
        assert len(sel) == 120 and "BAD" not in set(sel["symbol"]), name
        with pytest.raises(ValueError):
            api.validate_calibration(mixed, T, **KW)
        log = _gate(api, mixed)
        assert (log["n_calibration"] == 120).all(), name
        np.testing.assert_allclose(log["p_value"].to_numpy(), base_p, err_msg=name)
    api.validate_calibration(cal, T, **KW)                                           # all-matching passes


def test_n9_real():
    check_n9(_real_api())


def test_n9_reference_toy_passes():
    check_n9(toy_api())


@pytest.mark.parametrize("bug", ["no_version_filter", "no_filter_version_check", "no_oos_filter"])
def test_n9_broken_toy_fails(bug):
    with pytest.raises(AssertionError):
        check_n9(toy_api(bug))
