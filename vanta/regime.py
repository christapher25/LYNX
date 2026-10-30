"""
VANTA Stage 13B: filtered regime posterior (3-state Gaussian HMM).

v6 section 8.3 / SPEC N7. The only regime quantity this module returns at
inference is the FILTERED state probability P(s_t | x_1..x_t) from an explicit
forward recursion. No Viterbi, no smoothed (forward-backward) posterior, no
hmmlearn predict_proba, no learned gating network. The posterior is used
downstream as soft weights.

Fitting (`fit_hmm`) runs Baum-Welch EM on the training window only. EM needs
forward-backward by construction; those smoothed quantities exist only inside
the fit and never leave it. Everything fitted (standardisation, winsorisation
bounds, HMM parameters) comes from rows <= train_end.

Missing data: a NaN observation is never filled. With diagonal covariance each
feature is conditionally independent given the state, so a missing feature is
dropped from the emission term (marginalised); a row with every feature missing
propagates through the transition step only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_STATES = 3
FEATURES = ("log_vix", "fii_net_20d_cr", "universe_pct_above_200dma", "usdinr_chg_5d")
# Sessions after the first India VIX observation before the posterior is reported
# (the 20-session FII sum and 5-session USDINR change need to be formed).
WARMUP_SESSIONS = 20
USDINR_CHANGE_SESSIONS = 5
USDINR_MAX_STALE_DAYS = 5  # calendar days; a rate older than this is missing, not carried
_VAR_FLOOR = 1e-3  # on standardised features
_WINSOR = (1.0, 99.0)


@dataclass(frozen=True)
class RegimeHMM:
    """Fitted parameters. Features are standardised then clipped with train-only stats."""

    startprob: np.ndarray  # (K,)
    transmat: np.ndarray  # (K, K), rows sum to 1
    means: np.ndarray  # (K, D) standardised space
    variances: np.ndarray  # (K, D) standardised space
    mu: np.ndarray  # (D,)
    sigma: np.ndarray  # (D,)
    lo: np.ndarray  # (D,) clip bounds, standardised space
    hi: np.ndarray  # (D,)
    feature_names: tuple
    train_end: pd.Timestamp | None
    n_train: int
    train_loglik: float = float("nan")  # training-window log likelihood of the kept restart
    restart_logliks: tuple = ()  # one per restart, in restart order (restart 0 = tercile init)


# ---------------------------------------------------------------- emissions

def _standardise(params: RegimeHMM, X: np.ndarray) -> np.ndarray:
    Z = (X - params.mu) / params.sigma
    return np.clip(Z, params.lo, params.hi)  # NaN stays NaN


def _log_emissions(Z: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(T, K) log p(x_t | s=k) over OBSERVED dims only; all-NaN row gives 0 (no information)."""
    obs = ~np.isnan(Z)
    Z0 = np.where(obs, Z, 0.0)  # masked out below, never enters the sum
    diff2 = (Z0[:, None, :] - means[None, :, :]) ** 2
    ll = -0.5 * (np.log(2 * np.pi * variances)[None, :, :] + diff2 / variances[None, :, :])
    return np.where(obs[:, None, :], ll, 0.0).sum(axis=2)


# ----------------------------------------------------------------- inference

def filter_posterior(params: RegimeHMM, X) -> np.ndarray:
    """Filtered posterior P(s_t | x_1..x_t) for every row, by forward recursion.

    alpha_1 ∝ startprob * b_1;   alpha_t ∝ (alpha_{t-1} @ A) * b_t, normalised each step.
    Row t depends on rows <= t only. Returns (T, K); rows sum to 1.
    """
    X = _as_array(params, X)
    T = len(X)
    out = np.empty((T, params.startprob.size))
    if T == 0:
        return out
    logb = _log_emissions(_standardise(params, X), params.means, params.variances)
    prior = params.startprob
    for t in range(T):
        if t > 0:
            prior = out[t - 1] @ params.transmat
        a = np.log(np.maximum(prior, 1e-300)) + logb[t]
        a = np.exp(a - a.max())
        out[t] = a / a.sum()
    return out


def filtered_posterior_frame(params: RegimeHMM, features: pd.DataFrame,
                             warmup: int = WARMUP_SESSIONS) -> pd.DataFrame:
    """Filtered posterior indexed like `features`, columns p0..p2 (0 = calmest by VIX).

    Rows before the filter's first observed feature, and the first `warmup` sessions
    after the first observed India VIX, are NaN (the filter still runs through them).
    """
    post = filter_posterior(params, features)
    df = pd.DataFrame(post, index=features.index, columns=[f"p{k}" for k in range(post.shape[1])])
    start = first_available(features, warmup)
    if start is None:
        df[:] = np.nan
    else:
        df.loc[df.index < start] = np.nan
    return df


def first_available(features: pd.DataFrame, warmup: int = WARMUP_SESSIONS):
    """Date of the first reported posterior: `warmup` sessions after the first observed VIX."""
    vix = features[FEATURES[0]].notna().to_numpy()
    if not vix.any():
        return None
    i = int(np.argmax(vix)) + warmup
    return features.index[i] if i < len(features) else None


def _as_array(params: RegimeHMM, X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X[list(params.feature_names)].to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


# ------------------------------------------------------------------- fitting

def fold_train_end(sessions: pd.DatetimeIndex, test_start, embargo_sessions: int) -> pd.Timestamp:
    """Last date a fold's HMM may be fitted on: `embargo_sessions` full sessions before test_start.

    Embargo is counted in trading sessions (config.EMBARGO_DAYS) and must be at least the label horizon.
    """
    if embargo_sessions < 1:
        raise ValueError("embargo must be at least 1 session")
    pos = int(sessions.searchsorted(pd.Timestamp(test_start), side="left"))
    last = pos - 1 - embargo_sessions
    if last < 0:
        raise ValueError("no training sessions before test_start minus embargo")
    return sessions[last]


N_RESTARTS = 5
RESTART_SEED = 20240611


def fit_hmm(features: pd.DataFrame, train_end, n_states: int = N_STATES,
            n_iter: int = 200, tol: float = 1e-6, n_restarts: int = N_RESTARTS,
            seed: int = RESTART_SEED) -> RegimeHMM:
    """Fit on rows with index <= train_end ONLY. Deterministic for a given seed.

    EM runs `n_restarts` times: restart 0 from the VIX-tercile init, the rest from seeded
    random soft assignments (seed + restart index). The restart with the highest TRAINING
    log likelihood is kept. States are ordered by mean of the first feature (log VIX):
    state 0 calmest, K-1 most stressed.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    train_end = pd.Timestamp(train_end)
    tr = features.loc[features.index <= train_end, list(FEATURES)]
    tr = tr[tr.notna().any(axis=1)]
    X = tr.to_numpy(dtype=float)
    if len(X) < 10 * n_states:
        raise ValueError(f"too few training rows ({len(X)})")
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Z = (X - mu) / sigma
    lo = np.nanpercentile(Z, _WINSOR[0], axis=0)
    hi = np.nanpercentile(Z, _WINSOR[1], axis=0)
    Z = np.clip(Z, lo, hi)

    # restart 0: deterministic terciles of the first feature (rows lacking it go to the middle state)
    key = np.where(np.isnan(Z[:, 0]), np.nanmedian(Z[:, 0]), Z[:, 0])
    order = np.argsort(key, kind="stable")
    resp0 = np.zeros((len(Z), n_states))
    for k, idx in enumerate(np.array_split(order, n_states)):
        resp0[idx, k] = 1.0

    fits = []
    for r in range(n_restarts):
        if r == 0:
            resp = resp0
        else:
            resp = np.random.default_rng(seed + r).dirichlet(np.full(n_states, 0.3), size=len(Z))
        fits.append(_run_em(Z, resp, n_states, n_iter, tol))
    lls = tuple(f[4] for f in fits)
    pi, A, means, variances, ll = fits[int(np.argmax(lls))]  # ties -> earliest restart

    perm = np.argsort(means[:, 0], kind="stable")
    return RegimeHMM(
        startprob=pi[perm], transmat=A[np.ix_(perm, perm)], means=means[perm], variances=variances[perm],
        mu=mu, sigma=sigma, lo=lo, hi=hi, feature_names=tuple(FEATURES), train_end=train_end, n_train=len(X),
        train_loglik=ll, restart_logliks=lls,
    )


def _run_em(Z: np.ndarray, resp: np.ndarray, n_states: int, n_iter: int, tol: float):
    """One Baum-Welch run from initial responsibilities; returns (pi, A, means, variances, loglik)."""
    means, variances = _m_step_emission(Z, resp)
    A = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    pi = np.full(n_states, 1.0 / n_states)
    prev = -np.inf
    ll = -np.inf
    for _ in range(n_iter):
        logb = _log_emissions(Z, means, variances)
        gamma, xi_sum, ll = _e_step(pi, A, logb)
        pi = gamma[0] + 1e-12
        pi /= pi.sum()
        A = xi_sum + 1e-12
        A /= A.sum(axis=1, keepdims=True)
        means, variances = _m_step_emission(Z, gamma)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    # likelihood of the final parameters (ll above is for the parameters before the last M step)
    ll = _e_step(pi, A, _log_emissions(Z, means, variances))[2]
    return pi, A, means, variances, ll


def _m_step_emission(Z: np.ndarray, resp: np.ndarray):
    obs = ~np.isnan(Z)
    Z0 = np.where(obs, Z, 0.0)
    w = resp[:, :, None] * obs[:, None, :]  # (T, K, D)
    wsum = w.sum(axis=0) + 1e-12
    means = (w * Z0[:, None, :]).sum(axis=0) / wsum
    var = (w * (Z0[:, None, :] - means[None]) ** 2).sum(axis=0) / wsum
    return means, np.maximum(var, _VAR_FLOOR)


def _e_step(pi: np.ndarray, A: np.ndarray, logb: np.ndarray):
    """Scaled forward-backward on the TRAINING window (fit-time only; never used for inference)."""
    T, K = logb.shape
    m = logb.max(axis=1, keepdims=True)
    b = np.exp(logb - m)
    alpha = np.empty((T, K))
    beta = np.ones((T, K))
    c = np.empty(T)
    a = pi * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        xi = alpha[t][:, None] * A * (b[t + 1] * beta[t + 1])[None, :] / c[t + 1]
        xi_sum += xi
    return gamma, xi_sum, float(np.log(c).sum() + m.sum())


# ------------------------------------------------------------- data loading

def build_regime_features(start="2014-05-14", end=None) -> pd.DataFrame:
    """Real-data regime features indexed by India VIX session dates. Missing stays NaN.

    log_vix                   log of India VIX close (data/raw/indices)
    fii_net_20d_cr            Stage 10 flows block, 20 reported-day FPI equity net
    universe_pct_above_200dma Stage 10 breadth block (point-in-time universe proxy, not Nifty 500)
    usdinr_chg_5d             log(rate_t / rate_{t-5 sessions}); rate = latest at or before t, max 5 days old
    """
    import config

    idx_dir = config.RAW_DATA_DIR / "indices"
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    rows = []
    for f in sorted(idx_dir.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        if d < start_ts or (end_ts is not None and d > end_ts):
            continue
        v = pd.read_parquet(f, columns=["INDEX_NAME", "DATE", "CLOSE"],
                            filters=[("INDEX_NAME", "==", "India VIX")])
        if len(v):
            rows.append((pd.Timestamp(v["DATE"].iloc[0]), float(v["CLOSE"].iloc[0])))
    vix = pd.Series(dict(rows), dtype=float).sort_index()
    vix = vix[vix > 0]
    sessions = vix.index
    out = pd.DataFrame(index=sessions)
    out.index.name = "date"
    out[FEATURES[0]] = np.log(vix)

    fii = pd.read_parquet(config.FLOW_FEATURES_DIR / "fii_flows.parquet").set_index("date")
    out[FEATURES[1]] = fii["fpi_equity_net_20d_cr"].reindex(sessions)

    br = pd.read_parquet(config.BREADTH_FEATURES_DIR / "universe_breadth.parquet").set_index("date")
    out[FEATURES[2]] = br["universe_pct_above_200dma"].reindex(sessions)

    fx = pd.read_parquet(config.RAW_DATA_DIR / "macro" / "usdinr.parquet")
    fx = fx[fx["currency"] == "USD"].drop_duplicates("date", keep="last").set_index("date")["rate"].sort_index()
    asof = fx.reindex(fx.index.union(sessions)).ffill()
    last_seen = pd.Series(fx.index, index=fx.index).reindex(asof.index).ffill()
    stale = (pd.Series(asof.index, index=asof.index) - last_seen).dt.days
    asof = asof.where(stale <= USDINR_MAX_STALE_DAYS).reindex(sessions)
    out[FEATURES[3]] = np.log(asof / asof.shift(USDINR_CHANGE_SESSIONS))
    return out
