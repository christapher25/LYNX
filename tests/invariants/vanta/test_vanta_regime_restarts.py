"""
tests/invariants/vanta/test_vanta_regime_restarts.py

User decision 12 (logs/agent_team_decisions.md): the regime HMM is fitted with 5 seeded EM restarts and the
restart with the highest TRAINING log likelihood is kept. Deterministic; training rows only; inference unchanged
(forward recursion). Restarts must not leak anything beyond train_end.
"""

import numpy as np
import pandas as pd

from vanta import regime as R

from test_vanta_regime import _series


def _same(a, b):
    return all(np.array_equal(getattr(a, f), getattr(b, f))
               for f in ("startprob", "transmat", "means", "variances", "mu", "sigma"))


def test_default_is_five_restarts():
    df = _series()
    p = R.fit_hmm(df, train_end=df.index[199])
    assert len(p.restart_logliks) == 5


def test_fit_is_deterministic_given_seed():
    df = _series()
    end = df.index[199]
    assert _same(R.fit_hmm(df, end), R.fit_hmm(df, end))


def test_kept_restart_has_highest_training_likelihood():
    df = _series()
    p = R.fit_hmm(df, train_end=df.index[199])
    assert p.train_loglik == max(p.restart_logliks)
    one = R.fit_hmm(df, train_end=df.index[199], n_restarts=1)
    assert p.train_loglik >= one.train_loglik - 1e-9
    assert one.restart_logliks[0] == p.restart_logliks[0]  # restart 0 is the same tercile init


def test_seed_changes_the_extra_restarts_only():
    df = _series()
    end = df.index[199]
    a = R.fit_hmm(df, end, seed=1)
    b = R.fit_hmm(df, end, seed=2)
    assert a.restart_logliks[0] == b.restart_logliks[0]
    assert a.restart_logliks[1:] != b.restart_logliks[1:]


def test_restarts_use_training_rows_only():
    df = _series()
    end = df.index[199]
    tampered = df.copy()
    tampered.loc[tampered.index > end, :] = 99.0
    assert _same(R.fit_hmm(df, end), R.fit_hmm(tampered, end))


def test_restart_loglik_is_the_final_parameters_likelihood():
    df = _series()
    p = R.fit_hmm(df, train_end=df.index[199], n_restarts=1)
    tr = df.loc[df.index <= df.index[199], list(R.FEATURES)]
    Z = R._standardise(p, tr.to_numpy(float))
    # parameters are stored permuted by VIX order; likelihood is permutation-invariant
    logb = R._log_emissions(Z, p.means, p.variances)
    ll = R._e_step(p.startprob, p.transmat, logb)[2]
    assert abs(ll - p.train_loglik) < 1e-6 * abs(ll)


def test_a_later_restart_can_win_over_the_tercile_start():
    """On this seeded series restart 0 (tercile init) reaches a worse local optimum; the kept fit must be the better restart, so a
    fit that always keeps restart 0 fails here."""
    df = _series(n=300, seed=3)
    end = df.index[199]
    p = R.fit_hmm(df, end)
    assert p.restart_logliks[0] < max(p.restart_logliks) - 1.0
    assert p.train_loglik == max(p.restart_logliks) > p.restart_logliks[0]
    one = R.fit_hmm(df, end, n_restarts=1)
    assert not _same(p, one)                      # the single-start fit is a different (worse) optimum
    assert p.train_loglik > one.train_loglik + 1.0
