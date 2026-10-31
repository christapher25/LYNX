"""
Stage 14 baselines: A0 (momentum 12-1), A1 (SignalForge port: deterministic score + XGBoost win classifier) and B1 (CatBoost on
net labels, the main baseline and the M2 score). Interface: docs/vanta/INTERFACES.md section 7; pre-registration:
docs/vanta/M2_PREREG.md section 3.

Everything here is model-agnostic. A model is anything with `fit(X, y)` and `predict(X)`; a factory is `factory(seed) -> model`.
CatBoost / XGBoost are imported lazily inside `b1_model_factory` / `a1_model_factory`, so the rest of the module (and its tests)
work without them.

Rules this module keeps:
  * X is a float ndarray with NaN for a missing feature. Nothing is filled, imputed or interpolated; a row with a missing feature is
    kept, a row with a missing LABEL is dropped (a missing label is not a loser and not a zero return).
  * A fit sees exactly the rows `wf.run_fold` hands it (already purged against test_start - embargo, N11); no state survives a call.
  * A score at decision date d reads the `features` rows of date d only.
  * B1 score = plain mean of the seed models' predictions, seeds exactly B1_SEEDS.
  * Rank IC is Spearman per decision date, averaged by date; a sparse date is excluded, not counted as 0.
"""

from __future__ import annotations

import types
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import vanta.walkforward as wf
from vanta import m1

MIN_NAMES = wf.MIN_NAMES

B1_SEEDS = (0, 1, 2, 3, 4)          # M2_PREREG section 3: fixed, never tuned

# B1 hyperparameters (M2_PREREG section 3: chosen once on fold 0's inner split only, recorded here BEFORE the first out-of-sample run,
# never changed after OOS results are seen). No seed key: the seed comes only from B1_SEEDS.
# PLACEHOLDER UNTIL THE TUNING JOB HAS RUN: these are the CatBoost defaults the grid is centred on. The lead replaces this dict with
# the output of `python -m scripts.run_baselines --arm tune` (see its report) and commits it before any B1 OOS run.
B1_PARAMS: dict = {
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "verbose": 0,
    "allow_writing_files": False,
}

# Per-horizon B1 parameters (lead decision 2026-10-31: tuned separately for h=5 and h=20 on fold 0's inner split, because the labels differ).
# B1_PARAMS is the template. PLACEHOLDERS (= the template) until the tuning job has run; the lead commits the tuned values before the first
# OOS run. The driver reads this dict only, never the tuning JSON.
B1_PARAMS_BY_HORIZON: dict = {5: dict(B1_PARAMS), 20: dict(B1_PARAMS)}

# The grid the tuning job searches (also fixed here, before any OOS run). Everything not listed stays at B1_PARAMS.
B1_PARAM_GRID: list = [
    {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
    for d in (4, 6) for lr in (0.03, 0.08) for l2 in (3.0, 10.0)
]

# A1 (SignalForge port): XGBoost binary classifier on y_net > 0. Published when P(win) > A1_PUBLISH_ABOVE. Not judged in M2.
A1_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "n_jobs": -1,
}
A1_PUBLISH_ABOVE = 0.5

_SEED_KEYS = {"random_seed", "random_state", "seed"}


# ----------------------------------------------------------------------------------------------------- model factories
def b1_model_factory(**overrides) -> Callable[[int], object]:
    """factory(seed) -> CatBoostRegressor(random_seed=seed, **B1_PARAMS with overrides). Native NaN handling; nothing imputes."""
    bad = _SEED_KEYS & set(overrides)
    if bad:
        raise ValueError(f"the seed comes from B1_SEEDS, not from overrides: {sorted(bad)}")
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError("catboost is not installed: B1 needs the 'catboost' package (adding a dependency needs the user's "
                          "sign-off)") from e
    params = {**B1_PARAMS, **overrides}

    def factory(seed: int):
        return CatBoostRegressor(random_seed=int(seed), **params)
    return factory


class _WinProbability:
    """XGBClassifier whose predict() returns P(y == 1), so the baseline protocol (predict -> score) holds for A1."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return np.asarray(self.model.predict_proba(np.asarray(X, float))[:, 1], float)


def a1_model_factory(**overrides) -> Callable[[int], object]:
    """factory(seed) -> XGBoost binary classifier whose predict() is P(win) in [0, 1]. NaN features are handled natively."""
    bad = _SEED_KEYS & set(overrides)
    if bad:
        raise ValueError(f"the seed comes from the caller, not from overrides: {sorted(bad)}")
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("xgboost is not installed: A1 needs the 'xgboost' package (adding a dependency needs the user's "
                          "sign-off)") from e
    params = {**A1_PARAMS, **overrides}

    def factory(seed: int):
        return _WinProbability(XGBClassifier(random_state=int(seed), **params))
    return factory


# ----------------------------------------------------------------------------------------------------- ensemble
def fit_ensemble(factory: Callable[[int], object], seeds: Sequence[int], X: np.ndarray, y: np.ndarray) -> list:
    """One factory(seed) model per seed, in order, each fitted on the SAME (X, y)."""
    models = []
    for s in seeds:
        m = factory(s)
        m.fit(X, y)
        models.append(m)
    return models


def predict_ensemble(models: Sequence, X_new: np.ndarray) -> np.ndarray:
    """Plain mean over the models' predictions (not the median)."""
    return np.vstack([np.asarray(m.predict(X_new), float) for m in models]).mean(axis=0)


# ----------------------------------------------------------------------------------------------------- fit / score closures
def _check_features(features: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    missing = [c for c in ("decision_date", "symbol", *feature_cols) if c not in features.columns]
    if missing:
        raise KeyError(f"features lack column(s) {missing}")
    if features.duplicated(["decision_date", "symbol"]).any():
        raise ValueError("features hold duplicate (decision_date, symbol) rows; refusing to pick one")


def _target(y_net: np.ndarray, target: str) -> np.ndarray:
    y = np.asarray(y_net, float)
    if target == "net":
        return y
    if target == "binary":                     # y_net > 0 wins; a flat trade is a loser; a missing label stays missing
        return np.where(np.isnan(y), np.nan, (y > 0.0).astype(float))
    raise ValueError(f"target must be 'net' or 'binary', got {target!r}")


def make_fit_fn(features: pd.DataFrame, feature_cols: Sequence[str], factory: Callable[[int], object],
                seeds: Sequence[int] = B1_SEEDS, target: str = "net") -> Callable:
    """fit_fn(rows, fold) for wf.run_fold. Fits on exactly `rows` (the purged training rows) joined to the feature panel; fresh models
    on every call, nothing carried between folds."""
    cols = list(feature_cols)
    seeds = tuple(seeds)
    _check_features(features, cols)
    panel = features[["decision_date", "symbol", *cols]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))

    def fit_fn(rows: pd.DataFrame, fold):
        r = rows[["decision_date", "symbol", "y_net"]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))
        r = r.merge(panel, on=["decision_date", "symbol"], how="left")
        y = _target(r["y_net"].to_numpy(float), target)
        keep = np.isfinite(y)
        X = r[cols].to_numpy(float)
        Xf, yf = X[keep], y[keep]
        keys = r.loc[keep, ["decision_date", "symbol"]].reset_index(drop=True)
        if not len(yf):
            raise ValueError("no training rows with a finite label")
        models = fit_ensemble(factory, seeds, Xf, yf)
        return types.SimpleNamespace(models=models, seeds=seeds, train_keys=keys, feature_cols=cols)

    return fit_fn


def make_score_fn(features: pd.DataFrame, feature_cols: Sequence[str]) -> Callable:
    """score_fn(model, rows, fold, reader) -> one score per row of `rows`, same order. Looks up each row's own (decision_date, symbol)
    in the feature panel: no cross-date normalisation, no neighbouring date. A row with every feature missing (or absent from the panel)
    scores NaN (unavailable), never a model output on an all-NaN vector."""
    cols = list(feature_cols)
    _check_features(features, cols)
    panel = features[["decision_date", "symbol", *cols]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))

    def score_fn(model, rows: pd.DataFrame, fold, reader):
        r = rows[["decision_date", "symbol"]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))
        r = r.merge(panel, on=["decision_date", "symbol"], how="left")
        if len(r) != len(rows):
            raise ValueError("feature lookup changed the number of rows")
        X = r[cols].to_numpy(float)
        if not len(X):
            return np.array([])
        score = predict_ensemble(model.models, X)
        score[np.isnan(X).all(axis=1)] = np.nan
        return score

    return score_fn


# ----------------------------------------------------------------------------------------------------- A0
def momentum_a0(close: pd.Series, dates) -> pd.Series:
    """A0: close[t-21] / close[t-252] - 1 on the symbol's own adjusted bars; identical to vanta.m1.momentum_12_1 (the definition the
    M1/M1b drivers used), including its NaN pattern."""
    return m1.momentum_12_1(close, pd.DatetimeIndex(dates))


# ----------------------------------------------------------------------------------------------------- IC, selection, net per signal
def fold_rank_ic(scored: pd.DataFrame, min_names: int = MIN_NAMES) -> dict:
    """Spearman rank IC per decision date over complete (score, y_net) pairs, then the unweighted mean over dates."""
    per = wf.rank_ic_by_date(scored, min_names=min_names)
    return {"per_date": per, "mean": float(per.mean()) if len(per) else float("nan"), "n_dates": int(len(per))}


def top_n_per_date(scored: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Per decision date the n highest scores; a non-finite score is never selected, ties keep input order, the label is never read."""
    parts = []
    for _, g in scored.groupby("decision_date", sort=True):
        g = g[np.isfinite(g["score"].to_numpy(float))]
        parts.append(g.sort_values("score", ascending=False, kind="mergesort").head(n))
    return pd.concat(parts) if parts else scored.iloc[:0]


def net_per_signal(selected: pd.DataFrame) -> dict:
    """Pooled mean of y_net over FILLED published rows with a finite label. NO_FILL_* and FILLED-with-NaN (unmatured) are counted
    apart and never enter the mean (never as a 0 return). No fill_status column means every row is FILLED."""
    fs = selected["fill_status"] if "fill_status" in selected.columns else pd.Series("FILLED", index=selected.index)
    y = selected["y_net"].astype(float)
    filled = fs == "FILLED"
    ok = filled & y.notna()
    no_data = fs == "NO_DATA"                  # no bar / no ATR at the decision date: reported apart from NO_FILL_* (a different cause)
    return {"mean": float(y[ok].mean()) if ok.any() else float("nan"), "n_published": int(len(selected)),
            "n_filled": int(ok.sum()), "n_no_fill": int((~filled & ~no_data).sum()), "n_no_data": int(no_data.sum()),
            "n_unmatured": int((filled & y.isna()).sum())}


def fold_report(results: Sequence, seeds: Sequence[int] = B1_SEEDS, n: int = 10, min_names: int = MIN_NAMES,
                publish_above: float | None = None) -> list[dict]:
    """One dict per fold, computed on THAT fold's rows only. `publish_above` (A1) keeps only scores strictly above it before the top-n
    cut; rank IC always uses every scored row."""
    out = []
    for r in results:
        sc = r.scored
        ic = fold_rank_ic(sc, min_names)
        pool = sc if publish_above is None else sc[sc["score"] > publish_above]
        net = net_per_signal(top_n_per_date(pool, n))
        f = r.fold
        out.append({"fold_id": f.fold_id, "train_start": f.train_start, "val_start": f.val_start, "test_start": f.test_start,
                    "test_end": f.test_end, "embargo": f.embargo, "seeds": tuple(seeds), "n_train": r.n_train,
                    "rank_ic": ic["mean"], "n_ic_dates": ic["n_dates"], "net_per_signal": net["mean"],
                    "n_published": net["n_published"], "n_filled": net["n_filled"], "n_no_fill": net["n_no_fill"],
                    "n_no_data": net["n_no_data"], "n_unmatured": net["n_unmatured"]})
    return out


# ----------------------------------------------------------------------------------------------------- hyperparameters
def choose_hyperparameters(grid: Sequence[dict], model_for: Callable[[dict, int], object], features: pd.DataFrame,
                           labels: pd.DataFrame, fold, sessions, seed: int = 0, min_names: int = MIN_NAMES) -> dict:
    """The grid entry with the highest mean rank IC, using FOLD 0's inner split only (M2_PREREG section 3).

    Fit rows: decisions in [train_start, val_start) with label_end_date < session(val_start - embargo). Evaluation rows: decisions in
    [val_start, test_start) with label_end_date < session(test_start - embargo). Nothing from the test window, later folds or
    unmatured boundary rows. Rows with a missing label are dropped. Ties go to the first entry in grid order."""
    dec = pd.to_datetime(labels["decision_date"])
    fit_lab = wf.purge_training_rows(labels[(dec >= fold.train_start) & (dec < fold.val_start)], fold.val_start, fold.embargo, sessions)
    ev_lab = wf.purge_training_rows(labels[(dec >= fold.val_start) & (dec < fold.test_start)], fold.test_start, fold.embargo, sessions)
    cols = [c for c in features.columns if c not in ("decision_date", "symbol")]
    panel = features.assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))

    def rows(lab):
        r = lab[["decision_date", "symbol", "y_net"]].assign(decision_date=lambda d: pd.to_datetime(d["decision_date"]))
        r = r.merge(panel, on=["decision_date", "symbol"], how="left")
        return r[np.isfinite(r["y_net"].to_numpy(float))]

    fit_r, ev_r = rows(fit_lab), rows(ev_lab)
    if not len(fit_r) or not len(ev_r):
        raise ValueError(f"fold {fold.fold_id}: empty inner split (fit rows {len(fit_r)}, evaluation rows {len(ev_r)})")
    best, best_ic = None, -np.inf
    for params in grid:
        m = model_for(params, seed)
        m.fit(fit_r[cols].to_numpy(float), fit_r["y_net"].to_numpy(float))
        ic = fold_rank_ic(ev_r.assign(score=np.asarray(m.predict(ev_r[cols].to_numpy(float)), float)), min_names)["mean"]
        if np.isfinite(ic) and ic > best_ic:
            best, best_ic = params, ic
    if best is None:
        raise ValueError("no grid entry produced a finite rank IC on the inner evaluation rows")
    return best


# ----------------------------------------------------------------------------------------------------- evaluation window
def first_scored_and_judged_week(scored: pd.DataFrame, oos_start, window_weeks: int = 52) -> dict:
    """First week with a finite OOS score, and the first JUDGED week (M2_PREREG section 4): the first scored week T >= oos_start for
    which `window_weeks` weeks of scored history exist before T, i.e. first_scored_week <= T - window_weeks weeks. This is the calibration
    window vanta/m1b.py calibration_rows (line 68-74) and vanta/gate.py _usable_mask (line 59-74) use: decision_date >= T - 52 weeks,
    so the window is fully covered by scored history. Same rule for every horizon. NOTE: m1b has no code that drops early weeks for thin
    calibration (it splits at m1.OOS_START, m1.py:33, and the gate falls back to age-only weights / its ESS floor); this is the
    lead's stated reading of that sentence. Either result is NaT when no such week exists."""
    ok = scored[np.isfinite(scored["score"].to_numpy(float))]
    weeks = pd.DatetimeIndex(sorted(ok["decision_date"].unique()))
    if not len(weeks):
        return {"first_scored_week": pd.NaT, "first_judged_week": pd.NaT}
    cand = weeks[(weeks >= pd.Timestamp(oos_start)) & (weeks - pd.Timedelta(weeks=window_weeks) >= weeks[0])]
    return {"first_scored_week": weeks[0], "first_judged_week": cand[0] if len(cand) else pd.NaT}
