"""Unit tests for vanta.gate / vanta.gains edge cases (the invariants live in tests/invariants/vanta)."""

import numpy as np
import pandas as pd
import pytest

from vanta import gains, gate


def test_pvalues_match_bruteforce_with_ties_and_zero_loss():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, 80).astype(float)             # many ties
    y = rng.integers(-1, 2, 80).astype(float)            # includes exact zeros
    w = rng.uniform(0.1, 1, 80)
    t = np.array([-1.0, 0.0, 2.0, 3.0, 5.0, 9.0])
    ref = [(1 + w[(s >= x) & (y <= 0)].sum()) / (1 + w.sum()) for x in t]
    np.testing.assert_allclose(gate.conformal_pvalues(s, y, w, t), ref)


def test_bh_empty_and_all_pass():
    assert gate.bh_select([], 0.3).shape == (0,)
    assert gate.bh_select([0.01, 0.02], 0.3).all()


def test_gate_tie_break_higher_score_and_order_preserved():
    d = pd.Timestamp("2024-05-27")
    n = 50
    cal = pd.DataFrame({"decision_date": d, "symbol": range(n), "horizon": 5, "score": np.linspace(0, 1, n),
                        "model_version": "m", "fold_id": "f", "is_oos": True, "regime_posterior": [[1, 0, 0]] * n,
                        "hmm_version": "h", "y_net": -0.01, "matured_at": d + pd.Timedelta(days=6),
                        "candidate_filter_version": "c"})
    c = pd.DataFrame({"symbol": list("abcd"), "score": [5.0, 7.0, 6.0, 8.0]})
    out = gate.run_gate(c, cal, pd.Timestamp("2024-06-03"), (1, 0, 0), model_version="m", candidate_filter_version="c",
                        cap=2, ess_floor=0)
    assert list(out["symbol"]) == list("abcd")
    assert list(out["published"]) == [False, True, False, True]
    assert out["cap_binding"].all()


def test_gains_nan_ignored_and_misaligned_mask_rejected():
    df = pd.DataFrame({"y_net": [0.1, np.nan, 0.3, 0.2], "atr_pct": 0.02})
    g = gains.build_fold_gains(df, [True, True, True, False], 5)
    assert np.isnan(g.iloc[1]) and g.notna().sum() == 3
    with pytest.raises(ValueError):
        gains.build_fold_gains(df, [True, False], 5)
