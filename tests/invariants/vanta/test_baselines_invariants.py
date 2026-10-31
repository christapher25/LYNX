"""
Stage 14 baselines (A0 momentum 12-1, A1 SignalForge port, B1 CatBoost) invariants, queue row 15c.
Sources: v6 section 11.2 and Stage 14 row of section 15.1, docs/vanta/M2_PREREG.md (sections 3, 4, 7), N11 (VANTA_SPEC.md line 77).
Interface: docs/vanta/INTERFACES.md section 7 ("Stage 14 baselines"; every name INVENTED, the spec has no module paths).

  BL1  rank IC per decision date (Spearman), NaN labels dropped never zero-filled, sparse dates excluded not counted as 0,
       averaged by date within a fold (unweighted, never pooled)
  BL2  top-10 selection by score only (never the label), NaN score never published, stable ties
  BL3  net return per published signal = pooled mean over FILLED signals with a finite label; NO_FILL and unmatured are
       counted separately, never as a 0 return
  BL4  B1 prediction = mean over exactly seeds 0-4 (M2_PREREG section 3); each seed fits the same data once; deterministic
  BL5  A0 = close[t-21] / close[t-252] - 1 on adjusted closes, NaN on short history, equals vanta.m1.momentum_12_1
  BL6  missing FEATURES stay NaN in the fit matrix (row kept, no zero, no mean imputation); missing LABELS drop the row
  BL7  A1 target is the binary net outcome (y_net > 0); NaN stays missing, never a loser
  BL8  scores at date t depend only on features knowable at t (truncation / poison of later rows)
  BL9  a fold's fit sees only the rows run_fold hands it (purged, N11); a poisoned boundary row never reaches the model
  BL10 no model state is carried from one fold into the next (no fit across a fold boundary)
  BL11 the leaky walk-forward configuration is still flagged when a baseline is plugged in
  BL12 per-fold report: rank IC, net per published signal, seeds, fold dates, one row per fold
  BL13 hyperparameters are chosen on the earliest fold's inner data only (never its test window, never later folds, purged)

Each check_* takes an `api` (vanta.baselines or the toy below). The real module skips until it exists; the toy must pass and
each deliberately broken toy must fail. The toy lives in this file (the assigned test folder has no support-file slot).
"""

import types

import numpy as np
import pandas as pd
import pytest

TS = pd.Timestamp
NAN = np.nan
B1_SEEDS = (0, 1, 2, 3, 4)
NO_BUGS = frozenset()


# ======================================================================================================================
# Reference implementation ("toy") with switchable bugs. Not a VANTA module.
# ======================================================================================================================
def _pos(sess, d):
    return int(pd.DatetimeIndex(sess).get_loc(TS(d)))


def _toy(bugs=NO_BUGS, hidden_labels=None):
    bugs = frozenset(bugs)
    carry = {"X": None, "y": None, "keys": None}          # module-level state for the carry_over mutant

    # ---------------------------------------------------------------- rank IC
    def fold_rank_ic(scored, min_names=10):
        per = {}
        for d, g in scored.groupby("decision_date", sort=True):
            if "zero_fill_label" in bugs:
                g = g[["score", "y_net"]].assign(y_net=g["y_net"].fillna(0.0)).dropna()
            else:
                g = g[["score", "y_net"]].dropna()
            if len(g) < min_names:
                if "short_as_zero" in bugs:
                    per[d] = 0.0
                continue
            if "pearson" in bugs:
                r = g["score"].corr(g["y_net"])
            else:
                r = g["score"].rank().corr(g["y_net"].rank())
            if not np.isnan(r):
                per[d] = float(r)
        s = pd.Series(per, dtype=float)
        if "pooled" in bugs:
            allg = scored[["score", "y_net"]].dropna()
            m = float(allg["score"].rank().corr(allg["y_net"].rank())) if len(allg) else NAN
        elif "weighted" in bugs and len(s):
            n = scored.dropna(subset=["score", "y_net"]).groupby("decision_date").size().reindex(s.index)
            m = float((s * n).sum() / n.sum())
        else:
            m = float(s.mean()) if len(s) else NAN
        return {"per_date": s, "mean": m, "n_dates": int(len(s))}

    # ---------------------------------------------------------------- selection and net per signal
    def top_n_per_date(scored, n=10):
        parts = []
        for _, g in scored.groupby("decision_date", sort=True):
            if "nan_score_zero" not in bugs:
                g = g[np.isfinite(g["score"].to_numpy(float))]
            else:
                g = g.assign(score=g["score"].fillna(0.0))
            if "by_label" in bugs:
                g = g.sort_values("y_net", ascending=False, kind="mergesort")
            elif "lowest" in bugs:
                g = g.sort_values("score", ascending=True, kind="mergesort")
            elif "unstable_ties" in bugs:
                g = g.sort_values(["score", "symbol"], ascending=[False, False], kind="mergesort")
            else:
                g = g.sort_values("score", ascending=False, kind="mergesort")
            parts.append(g.head(n))
        return pd.concat(parts) if parts else scored.iloc[:0]

    def net_per_signal(selected):
        fs = selected["fill_status"] if "fill_status" in selected.columns else pd.Series("FILLED", index=selected.index)
        y = selected["y_net"].astype(float)
        filled = fs == "FILLED"
        ok = filled & y.notna()
        n_nofill = int((~filled).sum())
        n_unm = int((filled & y.isna()).sum())
        if "no_fill_zero" in bugs:
            m = float(y.where(filled, 0.0)[filled | ~filled].dropna().mean()) if len(y) else NAN
        elif "unmatured_zero" in bugs:
            m = float(y[filled].fillna(0.0).mean()) if filled.any() else NAN
        elif "weekly_mean" in bugs:
            m = float(y[ok].groupby(selected.loc[ok, "decision_date"]).mean().mean()) if ok.any() else NAN
        else:
            m = float(y[ok].mean()) if ok.any() else NAN
        return {"mean": m, "n_published": int(len(selected)), "n_filled": int(ok.sum()), "n_no_fill": n_nofill,
                "n_unmatured": n_unm}

    # ---------------------------------------------------------------- A0
    def momentum_a0(close, dates, lookback=252, skip=21):
        if "no_skip" in bugs:
            skip = 0
        if "skip20" in bugs:
            skip = 20
        if "lookback251" in bugs:
            lookback = 251
        close = close[close > 0]
        idx = close.index
        pos = idx.get_indexer(dates)
        out = np.full(len(dates), NAN)
        c = close.to_numpy()
        for k, p in enumerate(pos):
            if p < 0:
                continue
            if p < lookback and "zero_short" not in bugs:
                continue
            if p < lookback:
                out[k] = 0.0
                continue
            a = c[min(p + 1, len(c) - 1)] if "uses_future_close" in bugs else c[p - skip]
            v = a / c[p - lookback] - 1.0
            out[k] = 1.0 / (1.0 + v) - 1.0 if "inverted" in bugs else v
        return pd.Series(out, index=dates)

    # ---------------------------------------------------------------- ensemble
    def fit_ensemble(factory, seeds, X, y):
        if "four_seeds" in bugs:
            seeds = tuple(seeds)[:4]
        if "single_seed" in bugs:
            seeds = tuple(seeds)[:1]
        if "shifted_seeds" in bugs:
            seeds = tuple(s + 1 for s in seeds)
        models = []
        for s in seeds:
            m = factory(0 if "seed_ignored" in bugs else s)
            if "subsample" in bugs and s > 0:
                m.fit(X[:-1], y[:-1])
                models.append(m)
                continue
            m.fit(X, y)
            models.append(m)
        return models

    def predict_ensemble(models, X_new):
        p = np.vstack([np.asarray(m.predict(X_new), float) for m in models])
        if "nondeterministic" in bugs:
            p = p + np.random.default_rng().normal(0, 1e-3, p.shape)
        return np.median(p, axis=0) if "median" in bugs else p.mean(axis=0)

    # ---------------------------------------------------------------- fit / score closures
    def _target(y, target):
        y = y.astype(float)
        if target == "binary":
            out = np.where(np.isnan(y), NAN, (y >= 0.0 if "zero_is_win" in bugs else y > 0.0).astype(float))
            if "nan_loser" in bugs:
                out = np.where(np.isnan(y), 0.0, out)
            return out
        return y.to_numpy() if hasattr(y, "to_numpy") else y

    def make_fit_fn(features, feature_cols, factory, seeds=B1_SEEDS, target="net"):
        cols = list(feature_cols)

        def fit_fn(rows, fold):
            if "fit_nothing" in bugs:
                rows = rows.iloc[:0]
            if "global_labels" in bugs and hidden_labels is not None:
                rows = hidden_labels
            r = rows[["decision_date", "symbol", "y_net"]].merge(features[["decision_date", "symbol"] + cols],
                                                                 on=["decision_date", "symbol"], how="left")
            y = np.asarray(_target(r["y_net"], target), float)
            keep = np.isfinite(y)
            if "zero_fill_labels" in bugs:
                y = np.where(keep, y, 0.0)
                keep = np.ones(len(y), bool)
            X = r[cols].to_numpy(float)
            if "zero_fill_features" in bugs:
                X = np.nan_to_num(X, nan=0.0)
            if "impute_mean" in bugs:
                X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            if "drop_nan_feature_rows" in bugs:
                keep = keep & ~np.isnan(X).any(axis=1)
            Xf, yf, keys = X[keep], y[keep], r.loc[keep, ["decision_date", "symbol"]].reset_index(drop=True)
            if "carry_over" in bugs:
                if carry["X"] is not None:
                    Xf, yf = np.vstack([carry["X"], Xf]), np.concatenate([carry["y"], yf])
                    keys = pd.concat([carry["keys"], keys], ignore_index=True)
                carry.update(X=Xf, y=yf, keys=keys)
            models = fit_ensemble(factory, seeds, Xf, yf)
            return types.SimpleNamespace(models=models, seeds=tuple(seeds), train_keys=keys, feature_cols=cols)

        return fit_fn

    def make_score_fn(features, feature_cols):
        cols = list(feature_cols)

        def score_fn(model, rows, fold, reader):
            feats = features
            if "future_norm" in bugs:
                feats = features.assign(**{c: features[c] - features[c].mean() for c in cols})
            if "next_row" in bugs:
                dates = np.sort(features["decision_date"].unique())
                nxt = dict(zip(dates[:-1], dates[1:]))
                feats = features.assign(decision_date=features["decision_date"].map(lambda d: d))
                look = rows.assign(decision_date=rows["decision_date"].map(lambda d: nxt.get(np.datetime64(d), pd.NaT)))
                r = look.merge(feats[["decision_date", "symbol"] + cols], on=["decision_date", "symbol"], how="left")
            else:
                r = rows[["decision_date", "symbol"]].merge(feats[["decision_date", "symbol"] + cols],
                                                            on=["decision_date", "symbol"], how="left")
            X = r[cols].to_numpy(float)
            if "zero_fill_score_features" in bugs:
                X = np.nan_to_num(X, nan=0.0)
            return predict_ensemble(model.models, X)

        return score_fn

    # ---------------------------------------------------------------- report
    def fold_report(results, seeds=B1_SEEDS, n=10, min_names=10):
        out = []
        every = pd.concat([r.scored for r in results], ignore_index=True)
        for r in results:
            sc = r.scored
            ic = fold_rank_ic(every if "global_mean" in bugs else sc, min_names)
            sel = top_n_per_date(sc, n)
            net = net_per_signal(pd.concat([x for x in [top_n_per_date(q.scored, n) for q in results]]) if
                                 "net_pooled_over_folds" in bugs else sel)
            f = r.fold
            out.append({"fold_id": f.fold_id, "train_start": f.test_start if "dates_wrong" in bugs else f.train_start,
                        "val_start": f.val_start, "test_start": f.test_start, "test_end": f.test_end, "embargo": f.embargo,
                        "seeds": () if "seeds_missing" in bugs else tuple(seeds), "n_train": r.n_train,
                        "rank_ic": ic["mean"], "n_ic_dates": ic["n_dates"], "net_per_signal": net["mean"],
                        "n_published": net["n_published"], "n_filled": net["n_filled"], "n_no_fill": net["n_no_fill"],
                        "n_unmatured": net["n_unmatured"]})
        return out

    # ---------------------------------------------------------------- hyperparameter choice
    def choose_hyperparameters(grid, model_for, features, labels, fold, sessions, seed=0, min_names=10):
        sess = pd.DatetimeIndex(sessions)
        cut_fit = sess[_pos(sess, fold.val_start) - fold.embargo]
        cut_eval = sess[_pos(sess, fold.test_start) - fold.embargo]
        dec, end = pd.to_datetime(labels["decision_date"]), pd.to_datetime(labels["label_end_date"])
        fit_m = (dec >= fold.train_start) & (dec < fold.val_start)
        if "no_purge_fit" not in bugs:
            fit_m &= end < cut_fit
        ev_m = (dec >= fold.val_start) & (dec < fold.test_start)
        if "include_test" in bugs:
            ev_m = (dec >= fold.val_start) & (dec <= fold.test_end)
        if "all_folds" in bugs:
            ev_m = dec >= fold.val_start
        if "no_purge_eval" not in bugs:
            ev_m &= end < cut_eval if not ({"include_test", "all_folds"} & bugs) else (end < end.max() + pd.Timedelta(days=1))
        cols = [c for c in features.columns if c not in ("decision_date", "symbol")]
        fit_r = labels[fit_m].merge(features, on=["decision_date", "symbol"], how="left")
        ev_r = labels[ev_m].merge(features, on=["decision_date", "symbol"], how="left")
        best, best_ic = None, -np.inf
        for params in grid:
            m = model_for(params, seed)
            m.fit(fit_r[cols].to_numpy(float), fit_r["y_net"].to_numpy(float))
            ic = fold_rank_ic(ev_r.assign(score=np.asarray(m.predict(ev_r[cols].to_numpy(float)), float)), min_names)["mean"]
            if ic > best_ic:
                best, best_ic = params, ic
        return best

    return types.SimpleNamespace(B1_SEEDS=B1_SEEDS, fold_rank_ic=fold_rank_ic, top_n_per_date=top_n_per_date,
                                 net_per_signal=net_per_signal, momentum_a0=momentum_a0, fit_ensemble=fit_ensemble,
                                 predict_ensemble=predict_ensemble, make_fit_fn=make_fit_fn, make_score_fn=make_score_fn,
                                 fold_report=fold_report, choose_hyperparameters=choose_hyperparameters)


# ======================================================================================================================
# Shared spy models and fixtures
# ======================================================================================================================
class SpyModel:
    """Records what it was fitted on. predict(X) = first feature column + offset(seed**2): seed-dependent on purpose."""

    def __init__(self, seed):
        self.seed, self.X, self.y = seed, None, None

    def fit(self, X, y):
        self.X, self.y = np.array(X, float, copy=True), np.array(y, float, copy=True)
        return self

    def predict(self, X):
        return np.asarray(X, float)[:, 0] + float(self.seed) ** 2


def spy_factory(log=None):
    def factory(seed):
        m = SpyModel(seed)
        if log is not None:
            log.append(m)
        return m
    return factory


class MemoriserModel:
    """1-nearest-neighbour on the first feature (unique per row): exact on rows it was fitted on, noise elsewhere."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        X0 = np.asarray(X, float)[:, 0]
        o = np.argsort(X0)
        self.x, self.y = X0[o], np.asarray(y, float)[o]
        return self

    def predict(self, X):
        if not len(getattr(self, "x", [])):
            return np.zeros(len(X))
        q = np.asarray(X, float)[:, 0]
        i = np.clip(np.searchsorted(self.x, q), 1, len(self.x) - 1) if len(self.x) > 1 else np.zeros(len(q), int)
        j = np.where(np.abs(self.x[i] - q) < np.abs(self.x[np.maximum(i - 1, 0)] - q), i, np.maximum(i - 1, 0))
        return self.y[j]


def _scored(rows):
    return pd.DataFrame(rows, columns=["decision_date", "symbol", "score", "y_net"]).assign(
        decision_date=lambda d: pd.to_datetime(d["decision_date"]))


# ======================================================================================================================
# BL1 rank IC
# ======================================================================================================================
def check_rank_ic_nan_labels_dropped(api):
    """BL1: scores 1..5, labels [.1,.2,NaN,.4,.3]; the NaN pair is DROPPED: Spearman over the remaining 4 = 0.8. Zero-filling
    the NaN label would give 0.6; a Pearson would give a different number."""
    d = "2020-01-03"
    sc = _scored([(d, "A", 1.0, 0.1), (d, "B", 2.0, 0.2), (d, "C", 3.0, NAN), (d, "D", 4.0, 0.4), (d, "E", 5.0, 0.3)])
    r = api.fold_rank_ic(sc, min_names=3)
    assert r["mean"] == pytest.approx(0.8)
    assert r["n_dates"] == 1


def check_rank_ic_mean_of_dates_not_pooled_not_weighted(api):
    """BL1: IC is computed PER decision date then averaged by date, unweighted: date A (3 names, IC +1) and date B (5 names, IC -1)
    average to exactly 0. Pooling all rows or weighting by names gives a non-zero number."""
    a, b = "2020-01-03", "2020-01-10"
    rows = [(a, f"S{i}", float(i), float(i)) for i in (1, 2, 3)] + [(b, f"S{i}", float(i), float(6 - i)) for i in (1, 2, 3, 4, 5)]
    r = api.fold_rank_ic(_scored(rows), min_names=3)
    assert r["n_dates"] == 2
    assert list(np.round(np.asarray(r["per_date"], float), 12)) == [1.0, -1.0]
    assert r["mean"] == pytest.approx(0.0, abs=1e-12)


def check_rank_ic_sparse_date_excluded_not_zero(api):
    """BL1: a date left with fewer than min_names complete pairs has NO IC (excluded); it is not averaged in as 0. Date A IC=+1
    (3 pairs), date B has 5 names but only 2 finite labels: the fold mean is 1.0 over 1 date (a zero would give 0.5)."""
    a, b = "2020-01-03", "2020-01-10"
    rows = [(a, f"S{i}", float(i), float(i)) for i in (1, 2, 3)]
    rows += [(b, "S1", 1.0, 0.5), (b, "S2", 2.0, 0.6), (b, "S3", 3.0, NAN), (b, "S4", 4.0, NAN), (b, "S5", 5.0, NAN)]
    r = api.fold_rank_ic(_scored(rows), min_names=3)
    assert r["n_dates"] == 1 and r["mean"] == pytest.approx(1.0)


# ======================================================================================================================
# BL2 selection and BL3 net per signal
# ======================================================================================================================
def _week_frame():
    d = TS("2020-01-03")
    return pd.DataFrame({"decision_date": d, "symbol": list("ABCDEFG"),
                         "score": [0.9, 0.1, NAN, 0.9, 0.5, 0.7, 0.3],
                         "y_net": [-0.30, 0.20, 0.90, 0.01, 0.05, 0.02, 0.50]})


def check_top_n_by_score_only(api):
    """BL2: top 3 by score descending: A and D tie at 0.9 (stable: input order A before D), then F (0.7). The NaN score (C, which
    has the best label) is never selected; the label plays no part (B's and G's big labels do not help them)."""
    sel = api.top_n_per_date(_week_frame(), n=3)
    assert list(sel["symbol"]) == ["A", "D", "F"]
    neg = _week_frame().iloc[:4].assign(score=[-0.5, NAN, -0.2, -0.9])      # a NaN score must not outrank real negatives
    assert list(api.top_n_per_date(neg, n=2)["symbol"]) == ["C", "A"]


def check_top_n_per_date_each_date(api):
    """BL2: n applies per decision date: two dates with 3 names each and n=2 give 4 rows, the 2 best of each date."""
    f = pd.concat([_week_frame().iloc[:3].assign(score=[3.0, 2.0, 1.0]),
                   _week_frame().iloc[:3].assign(decision_date=TS("2020-01-10"), score=[1.0, 2.0, 3.0])])
    sel = api.top_n_per_date(f, n=2)
    assert [(d.date().isoformat(), s) for d, s in zip(sel["decision_date"], sel["symbol"])] == [
        ("2020-01-03", "A"), ("2020-01-03", "B"), ("2020-01-10", "C"), ("2020-01-10", "B")]


def check_net_per_signal_known_answer(api):
    """BL3: published rows y_net [0.10, -0.02, NaN(NO_FILL), NaN(FILLED, unmatured)]: mean over the 2 FILLED finite = 0.04;
    n_filled 2, n_no_fill 1, n_unmatured 1. A NO_FILL or unmatured counted as a 0 return would give 0.0266 / 0.02."""
    d = TS("2020-01-03")
    sel = pd.DataFrame({"decision_date": d, "symbol": list("ABCD"), "y_net": [0.10, -0.02, NAN, NAN],
                        "fill_status": ["FILLED", "FILLED", "NO_FILL_UPPER_LOCK", "FILLED"]})
    r = api.net_per_signal(sel)
    assert r["mean"] == pytest.approx(0.04)
    assert (r["n_published"], r["n_filled"], r["n_no_fill"], r["n_unmatured"]) == (4, 2, 1, 1)


def check_net_per_signal_pooled_per_signal(api):
    """BL3: 'per published signal' is a POOLED mean over signals, not a mean of weekly means: week 1 has one signal at +0.10, week
    2 has three at 0.0 each: pooled 0.025, mean of weekly means 0.05. No fill_status column means every row is FILLED."""
    sel = pd.DataFrame({"decision_date": [TS("2020-01-03")] + [TS("2020-01-10")] * 3, "symbol": list("ABCD"),
                        "y_net": [0.10, 0.0, 0.0, 0.0]})
    r = api.net_per_signal(sel)
    assert r["mean"] == pytest.approx(0.025) and r["n_filled"] == 4 and r["n_no_fill"] == 0


# ======================================================================================================================
# BL4 seeds
# ======================================================================================================================
def check_seed_list_is_zero_to_four(api):
    """BL4 (M2_PREREG section 3): the B1 seed list is exactly (0, 1, 2, 3, 4), fixed."""
    assert tuple(api.B1_SEEDS) == (0, 1, 2, 3, 4)


def check_b1_prediction_is_mean_over_seeds_0_to_4(api):
    """BL4: with model_s(x) = x + s**2, seeds 0..4 give x + mean(0,1,4,9,16) = x + 6. Four seeds give 3.5, seeds 1..5 give 11, one
    seed 0, the median 4, so each of those mutants fails. The factory is called once per seed, in order, with the seed itself."""
    log = []
    X = np.array([[1.0], [2.0]])
    models = api.fit_ensemble(spy_factory(log), api.B1_SEEDS, X, np.array([0.1, 0.2]))
    assert [m.seed for m in log] == [0, 1, 2, 3, 4] and len(models) == 5
    assert api.predict_ensemble(models, X) == pytest.approx([7.0, 8.0])


def check_each_seed_sees_identical_training_data_and_is_deterministic(api):
    """BL4: every seed's model is fitted on the SAME (X, y) (seeds differ only in the model's own randomness) and the ensemble is
    deterministic: fitting twice gives identical predictions."""
    X = np.array([[1.0, NAN], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0.1, 0.2, 0.3])
    la, lb = [], []
    pa = api.predict_ensemble(api.fit_ensemble(spy_factory(la), api.B1_SEEDS, X, y), X)
    pb = api.predict_ensemble(api.fit_ensemble(spy_factory(lb), api.B1_SEEDS, X, y), X)
    np.testing.assert_array_equal(pa, pb)
    for m in la:
        np.testing.assert_array_equal(m.X, X)
        np.testing.assert_array_equal(m.y, y)


# ======================================================================================================================
# BL5 A0
# ======================================================================================================================
def _closes(n=300):
    sess = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(50.0 + np.arange(n) + 3.0 * np.sin(np.arange(n)), index=sess)


def check_a0_known_answer(api):
    """BL5: A0 at position p = close[p-21] / close[p-252] - 1, by hand from the series (the 12-1 definition: skip the latest
    month). No-skip (close[p-0]), skip 20, lookback 251 and an inverted ratio all give other numbers."""
    c = _closes()
    p = [260, 299]
    got = api.momentum_a0(c, c.index[p])
    exp = [c.iloc[q - 21] / c.iloc[q - 252] - 1.0 for q in p]
    assert got.to_numpy() == pytest.approx(exp, rel=1e-12)


def check_a0_short_history_is_nan_not_zero(api):
    """BL5 + no-zero-fill: with fewer than 252 prior bars the score is NaN (missing), never 0."""
    c = _closes()
    got = api.momentum_a0(c, c.index[[100, 251]])
    assert got.isna().all()


def check_a0_truncation_causal(api):
    """BL5 + BL8: A0 at date t is unchanged when every bar after t is deleted (it reads only the past)."""
    c = _closes()
    t = c.index[[270, 280]]
    full = api.momentum_a0(c, t)
    cut = api.momentum_a0(c.iloc[:281], t)
    np.testing.assert_array_equal(full.to_numpy(), cut.to_numpy())
    one = api.momentum_a0(c.iloc[:271], t[:1])
    np.testing.assert_array_equal(full.to_numpy()[:1], one.to_numpy())


def check_a0_equals_stage7_driver_definition(api):
    """BL5: A0 reproduces vanta.m1.momentum_12_1 (the definition the M1 driver used) on a noisy series, including the NaN
    pattern."""
    m1 = pytest.importorskip("vanta.m1")
    rng = np.random.default_rng(3)
    sess = pd.bdate_range("2018-01-01", periods=420)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 420))), index=sess)
    dates = sess[[10, 252, 300, 419]]
    np.testing.assert_array_equal(api.momentum_a0(c, dates).to_numpy(), m1.momentum_12_1(c, dates).to_numpy())


# ======================================================================================================================
# BL6 / BL7 fit matrix
# ======================================================================================================================
def _fit_fixture():
    d = pd.to_datetime(["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10", "2020-01-17", "2020-01-17"])
    sym = ["A", "B", "A", "B", "A", "B"]
    feats = pd.DataFrame({"decision_date": d, "symbol": sym, "f0": [1.0, 2.0, NAN, 4.0, 5.0, 6.0],
                          "f1": [10.0, NAN, 30.0, 40.0, 50.0, 60.0]})
    rows = pd.DataFrame({"decision_date": d, "symbol": sym, "y_net": [0.10, -0.05, 0.02, NAN, 0.0, 0.30],
                         "label_end_date": d + pd.Timedelta(days=7)})
    return feats, rows


def check_missing_features_stay_nan_in_fit(api):
    """BL6: the fit matrix keeps the row whose features are missing and keeps NaN as NaN: no 0, no mean imputation, row not
    dropped. Rows 0,1,2,4,5 have a finite label (row 3's label is NaN and is dropped); NaNs at (1,f1) and (2,f0)."""
    feats, rows = _fit_fixture()
    log = []
    api.make_fit_fn(feats, ["f0", "f1"], spy_factory(log), seeds=(0,))(rows, None)
    X, y = log[0].X, log[0].y
    exp = np.array([[1.0, 10.0], [2.0, NAN], [NAN, 30.0], [5.0, 50.0], [6.0, 60.0]])
    np.testing.assert_array_equal(X, exp)
    np.testing.assert_array_equal(y, [0.10, -0.05, 0.02, 0.0, 0.30])


def check_missing_labels_dropped_never_zero(api):
    """BL6 + BL1: a training row whose net label is missing is DROPPED from the fit, never filled with 0: y holds 5 values and
    the only 0.0 in it is the genuine 0.0 label (row 4)."""
    feats, rows = _fit_fixture()
    log = []
    api.make_fit_fn(feats, ["f0", "f1"], spy_factory(log), seeds=(0,))(rows, None)
    y = log[0].y
    assert len(y) == 5 and int((y == 0.0).sum()) == 1


def check_fit_returns_train_keys_and_seeds(api):
    """BL9/BL12 support: the fitted object exposes `.seeds`, `.models` (one per seed) and `.train_keys` (decision_date, symbol of
    the rows actually fitted: the NaN-label row is absent)."""
    feats, rows = _fit_fixture()
    m = api.make_fit_fn(feats, ["f0", "f1"], spy_factory(), seeds=B1_SEEDS)(rows, None)
    assert tuple(m.seeds) == B1_SEEDS and len(m.models) == 5
    assert len(m.train_keys) == 5
    assert not ((m.train_keys["decision_date"] == TS("2020-01-10")) & (m.train_keys["symbol"] == "B")).any()


def check_a1_binary_target(api):
    """BL7: the A1 target is y_net > 0: 0.10 -> 1, -0.05 -> 0, 0.02 -> 1, 0.0 -> 0 (a flat trade is a loser, prereg 'loser =
    y_net <= 0'), 0.30 -> 1; the NaN label stays missing and the row is dropped (a NaN is not a loser)."""
    feats, rows = _fit_fixture()
    log = []
    api.make_fit_fn(feats, ["f0", "f1"], spy_factory(log), seeds=(0,), target="binary")(rows, None)
    np.testing.assert_array_equal(log[0].y, [1.0, 0.0, 1.0, 0.0, 1.0])


# ======================================================================================================================
# BL8 causality of scores
# ======================================================================================================================
def _feature_panel():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-06", periods=4, freq="5B")
    rows = [(d, f"S{i}", rng.normal(), rng.normal()) for d in dates for i in range(6)]
    return pd.DataFrame(rows, columns=["decision_date", "symbol", "f0", "f1"]), dates


def check_scores_depend_only_on_features_up_to_t(api):
    """BL8: scores for decision date d2 are identical whether the feature store holds rows through d4, only through d2, or has
    its LATER rows replaced by huge shuffled values. A cross-sectional/global normalisation over the whole frame, or a lookup of
    the next date's features, changes the answer."""
    feats, dates = _feature_panel()
    d2 = dates[1]
    model = types.SimpleNamespace(models=[SpyModel(0)])
    rows = feats[feats["decision_date"] == d2][["decision_date", "symbol"]].reset_index(drop=True)

    def score(frame):
        return np.asarray(api.make_score_fn(frame, ["f0", "f1"])(model, rows, None, None), float)

    full = score(feats)
    trunc = score(feats[feats["decision_date"] <= d2])
    poisoned = feats.copy()
    later = poisoned["decision_date"] > d2
    poisoned.loc[later, ["f0", "f1"]] = np.random.default_rng(9).normal(size=(int(later.sum()), 2)) * 1e6
    np.testing.assert_array_equal(full, trunc)
    np.testing.assert_array_equal(full, score(poisoned))
    np.testing.assert_array_equal(full, feats[feats["decision_date"] == d2]["f0"].to_numpy())   # SpyModel(0): f0 + 0


def check_missing_score_features_stay_nan(api):
    """BL6/BL8: a scored row whose feature is missing is handed to the model as NaN (never 0)."""
    feats, dates = _feature_panel()
    feats.loc[0, "f0"] = NAN
    seen = {}

    class Probe:
        def predict(self, X):
            seen["X"] = np.array(X, float)
            return np.zeros(len(X))

    rows = feats[feats["decision_date"] == dates[0]][["decision_date", "symbol"]].reset_index(drop=True)
    api.make_score_fn(feats, ["f0", "f1"])(types.SimpleNamespace(models=[Probe()]), rows, None, None)
    assert np.isnan(seen["X"][0, 0]) and not np.isnan(seen["X"][1:, 0]).any()


# ======================================================================================================================
# BL9 / BL10 / BL11: plugged into the real walk-forward harness
# ======================================================================================================================
H = 20


def _wf():
    return pytest.importorskip("vanta.walkforward")


def _sess():
    return pd.bdate_range("2008-01-01", "2019-12-31")


def _labels(n_sym=12, step=5, seed=0):
    rng = np.random.default_rng(seed)
    sess = _sess()
    dec = sess[::step][:-(H // step + 2)]
    rows = [(f"S{s:02d}", d, H, sess[int(sess.get_loc(d)) + H], rng.normal(0, 0.05)) for s in range(n_sym) for d in dec]
    return pd.DataFrame(rows, columns=["symbol", "decision_date", "horizon", "label_end_date", "y_net"])


def _unique_features(labels, seed=1):
    rng = np.random.default_rng(seed)
    f0 = rng.permutation(len(labels)).astype(float) + 0.5          # unique per row
    return labels[["decision_date", "symbol"]].assign(f0=f0, f1=rng.normal(size=len(labels)))


_LAB = None


def _lab():
    global _LAB
    if _LAB is None:
        _LAB = _labels()
    return _LAB


def _folds():
    return _wf().make_folds(_sess(), H, val_years=0, include_val_in_fit=True)


def check_fit_sees_only_purged_rows(api):
    """BL9 (N11): through wf.run_fold, planted sentinel rows (label ends ON the cutoff session; decision/label after test_start)
    never reach the model's fit matrix, while a row ending one session before the cutoff does. The sentinel is identified by its
    unique first-feature value."""
    wf, sess = _wf(), _sess()
    lab = _lab()
    f = _folds()[0]
    cutoff = sess[int(sess.get_loc(f.test_start)) - f.embargo]
    plant = pd.DataFrame([("PLANT_EQ", cutoff - pd.Timedelta(days=40), H, cutoff, 9e9),
                          ("PLANT_AFTER", f.test_start - pd.Timedelta(days=3), H, f.test_start + pd.Timedelta(days=20), 9e9),
                          ("PLANT_OK", cutoff - pd.Timedelta(days=60), H, sess[int(sess.get_loc(cutoff)) - 1], 7e9)],
                         columns=lab.columns)
    labels = pd.concat([lab, plant], ignore_index=True)
    feats = _unique_features(labels)
    for s, v in (("PLANT_EQ", -111.0), ("PLANT_AFTER", -222.0), ("PLANT_OK", -333.0)):
        feats.loc[feats["symbol"] == s, "f0"] = v
    log = []
    fit = api.make_fit_fn(feats, ["f0", "f1"], spy_factory(log), seeds=(0,))
    res = wf.run_fold(f, labels, sess, fit, api.make_score_fn(feats, ["f0", "f1"]))
    x0 = log[0].X[:, 0]
    assert -111.0 not in x0 and -222.0 not in x0
    assert -333.0 in x0
    assert len(log[0].X) == res.n_train


def check_no_model_state_carried_across_folds(api):
    """BL10 (no fit across a fold boundary): in a 3-fold run each fold's fitted object is its own, and its training keys lie
    inside [train_start, test_start) with label_end_date before test_start - embargo. A baseline that accumulates rows from
    earlier folds (warm start / shared buffer) is caught by train_keys that start before the fold's rolling train_start."""
    wf, sess = _wf(), _sess()
    lab = _lab()
    feats = _unique_features(lab)
    caught = []
    inner = api.make_fit_fn(feats, ["f0", "f1"], spy_factory(), seeds=(0,))

    def fit(rows, fold):
        m = inner(rows, fold)
        caught.append((fold, m))
        return m

    wf.run_walkforward(_folds()[:3], lab, sess, fit, api.make_score_fn(feats, ["f0", "f1"]))
    assert len(caught) == 3
    assert len({id(m) for _, m in caught}) == 3 and len({id(m.models[0]) for _, m in caught}) == 3
    ends = lab.set_index(["decision_date", "symbol"])["label_end_date"]
    for fold, m in caught:
        k = m.train_keys
        assert len(k) > 0
        assert (k["decision_date"] >= fold.train_start).all() and (k["decision_date"] < fold.test_start).all()
        cutoff = sess[int(sess.get_loc(fold.test_start)) - fold.embargo]
        e = ends.reindex(pd.MultiIndex.from_frame(k[["decision_date", "symbol"]])).to_numpy()
        assert (e < np.datetime64(cutoff)).all()


def check_leaky_configuration_flagged_with_baseline(api, detect=None):
    """BL11: a baseline (1-NN memoriser as the 'model', so it can exploit leaked rows) run with fit_rows_policy='include_test' is
    flagged by wf.detect_leak (IC above the 0.10 ceiling AND the N11 audit); the same baseline under the safe 'purged' policy is
    not flagged. A detector that never flags, or a baseline whose fit ignores its rows, fails."""
    wf, sess = _wf(), _sess()
    detect = detect or wf.detect_leak
    lab = _lab()
    feats = _unique_features(lab)
    fit = api.make_fit_fn(feats, ["f0", "f1"], lambda s: MemoriserModel(s), seeds=(0,))
    score = api.make_score_fn(feats, ["f0", "f1"])
    folds = _folds()[:3]
    clean = wf.run_walkforward(folds, lab, sess, fit, score)
    leaky = wf.run_walkforward(folds, lab, sess, fit, score, fit_rows_policy="include_test")
    assert not detect(clean, lab, horizon=H).flagged, "clean baseline run flagged"
    rep = detect(leaky, lab, horizon=H)
    assert rep.flagged, "leaky baseline run not flagged"
    assert any("rank IC" in r for r in rep.reasons) and any("N11" in r for r in rep.reasons), rep.reasons


# ======================================================================================================================
# BL12 per-fold report
# ======================================================================================================================
def _fold_obj(i, y):
    return types.SimpleNamespace(fold_id=i, train_start=TS(f"{y-4}-01-02"), val_start=TS(f"{y}-01-02"),
                                 test_start=TS(f"{y}-01-02"), test_end=TS(f"{y}-12-30"), embargo=20)


def _fold_result(i, y, dates, ics):
    rows = []
    for d, sgn in zip(dates, ics):
        for k in range(1, 5):
            rows.append((TS(d), f"S{k}", float(k), float(k) * sgn * 0.01))
    sc = _scored(rows).assign(fill_status="FILLED")
    return types.SimpleNamespace(fold=_fold_obj(i, y), n_train=100 + i, scored=sc)


def check_fold_report_per_fold_values(api):
    """BL12: one report row per fold. Fold 0 (one date, IC +1) and fold 1 (three dates, IC -1) report rank_ic +1 and -1 (an IC
    pooled across folds would be the same number twice), n_ic_dates 1 and 3; top-2 net per signal is the mean of the two best
    scores' labels (0.04+0.03 over 2 signals = 0.035 for fold 0; -0.035 for fold 1); the seed list and the fold's own
    train/test dates are reported."""
    r0 = _fold_result(0, 2013, ["2013-01-04"], [+1])
    r1 = _fold_result(1, 2014, ["2014-01-03", "2014-01-10", "2014-01-17"], [-1, -1, -1])
    rep = api.fold_report([r0, r1], seeds=B1_SEEDS, n=2, min_names=3)
    assert len(rep) == 2
    assert [r["fold_id"] for r in rep] == [0, 1]
    assert rep[0]["rank_ic"] == pytest.approx(1.0) and rep[1]["rank_ic"] == pytest.approx(-1.0)
    assert (rep[0]["n_ic_dates"], rep[1]["n_ic_dates"]) == (1, 3)
    assert rep[0]["net_per_signal"] == pytest.approx(0.035) and rep[1]["net_per_signal"] == pytest.approx(-0.035)
    assert (rep[0]["n_published"], rep[1]["n_published"]) == (2, 6)
    for r, res in zip(rep, (r0, r1)):
        assert tuple(r["seeds"]) == B1_SEEDS
        assert r["train_start"] == res.fold.train_start and r["test_start"] == res.fold.test_start
        assert r["test_end"] == res.fold.test_end and r["n_train"] == res.n_train
        assert {"n_filled", "n_no_fill", "n_unmatured"} <= set(r)


# ======================================================================================================================
# BL13 hyperparameters from the earliest fold's inner data only
# ======================================================================================================================
class SignModel:
    """predict = sign * first feature; records the labels it was fitted on. Its 'quality' is decided only by the evaluation rows."""

    def __init__(self, sign):
        self.sign = sign

    def fit(self, X, y):
        type(self).fit_y = np.array(y, float)
        return self

    def predict(self, X):
        return self.sign * np.asarray(X, float)[:, 0]


def check_hyperparameters_use_only_earliest_inner_data(api):
    """BL13 (M2_PREREG section 3): the choice between sign=-1 and sign=+1 is made on fold 0's validation year only. The true
    relation is y = +0.1*f0 + noise there. Poison: every label that matures at or after test_start - embargo (the boundary rows
    of the validation year, fold 0's test window, all later folds) is exactly -f0, so ANY use of them flips the choice to -1;
    train rows whose label ends after val_start - embargo carry a 9e9 sentinel that must never reach the fit."""
    wf, sess = _wf(), _sess()
    lab = _labels(n_sym=20, step=1)
    f = wf.make_folds(sess, H)[0]
    rng = np.random.default_rng(4)
    feats = lab[["decision_date", "symbol"]].assign(f0=rng.normal(size=len(lab)))
    f0 = feats["f0"].to_numpy()
    cut_fit = sess[int(sess.get_loc(f.val_start)) - f.embargo]
    cut_eval = sess[int(sess.get_loc(f.test_start)) - f.embargo]
    y = 0.1 * f0 + rng.normal(size=len(lab))
    dec, end = lab["decision_date"], lab["label_end_date"]
    y = np.where(dec >= f.val_start, np.where(end >= cut_eval, -f0, y), np.where(end >= cut_fit, 9e9, y))
    labels = lab.assign(y_net=y)
    grid = [{"sign": -1.0}, {"sign": 1.0}]            # wrong answer first: a tie or an empty evaluation cannot pass by luck
    SignModel.fit_y = np.array([])
    best = api.choose_hyperparameters(grid, lambda p, seed: SignModel(p["sign"]), feats, labels, f, sess, min_names=10)
    assert best == {"sign": 1.0}
    assert len(SignModel.fit_y) > 0 and np.abs(SignModel.fit_y).max() < 1e9


# ======================================================================================================================
# registry: name -> (check, mutants that must fail it)
# ======================================================================================================================
CHECKS = {
    "rank_ic_nan_labels_dropped": (check_rank_ic_nan_labels_dropped, ["zero_fill_label", "pearson"]),
    "rank_ic_mean_of_dates": (check_rank_ic_mean_of_dates_not_pooled_not_weighted, ["pooled", "weighted"]),
    "rank_ic_sparse_date_excluded": (check_rank_ic_sparse_date_excluded_not_zero, ["short_as_zero", "zero_fill_label"]),
    "top_n_by_score_only": (check_top_n_by_score_only, ["by_label", "lowest", "nan_score_zero", "unstable_ties"]),
    "top_n_per_date": (check_top_n_per_date_each_date, ["lowest"]),
    "net_known_answer": (check_net_per_signal_known_answer, ["no_fill_zero", "unmatured_zero"]),
    "net_pooled": (check_net_per_signal_pooled_per_signal, ["weekly_mean"]),
    "seed_list": (check_seed_list_is_zero_to_four, []),
    "b1_mean_over_seeds": (check_b1_prediction_is_mean_over_seeds_0_to_4,
                           ["four_seeds", "single_seed", "shifted_seeds", "median", "seed_ignored"]),
    "seeds_same_data_deterministic": (check_each_seed_sees_identical_training_data_and_is_deterministic,
                                      ["subsample", "nondeterministic"]),
    "a0_known_answer": (check_a0_known_answer, ["no_skip", "skip20", "lookback251", "inverted", "uses_future_close"]),
    "a0_short_history_nan": (check_a0_short_history_is_nan_not_zero, ["zero_short"]),
    "a0_truncation": (check_a0_truncation_causal, ["uses_future_close"]),
    "a0_equals_m1": (check_a0_equals_stage7_driver_definition, ["no_skip", "skip20", "lookback251", "zero_short"]),
    "features_stay_nan": (check_missing_features_stay_nan_in_fit,
                          ["zero_fill_features", "impute_mean", "drop_nan_feature_rows"]),
    "labels_dropped_not_zero": (check_missing_labels_dropped_never_zero, ["zero_fill_labels"]),
    "fit_train_keys": (check_fit_returns_train_keys_and_seeds, ["zero_fill_labels", "single_seed"]),
    "a1_binary_target": (check_a1_binary_target, ["zero_is_win", "nan_loser"]),
    "scores_causal": (check_scores_depend_only_on_features_up_to_t, ["future_norm", "next_row"]),
    "score_features_nan": (check_missing_score_features_stay_nan, ["zero_fill_score_features"]),
    "fit_only_purged_rows": (check_fit_sees_only_purged_rows, ["global_labels"]),
    "no_carry_across_folds": (check_no_model_state_carried_across_folds, ["carry_over"]),
    "leaky_flagged": (check_leaky_configuration_flagged_with_baseline, ["fit_nothing"]),
    "fold_report": (check_fold_report_per_fold_values, ["global_mean", "seeds_missing", "dates_wrong", "net_pooled_over_folds"]),
    "hyperparameters_inner_only": (check_hyperparameters_use_only_earliest_inner_data,
                                   ["all_folds", "include_test", "no_purge_eval", "no_purge_fit"]),
}
MUTANTS = [(n, b) for n, (_, bugs) in CHECKS.items() for b in bugs]


@pytest.mark.parametrize("name", list(CHECKS))
def test_baselines_real(name):
    api = pytest.importorskip("vanta.baselines")
    CHECKS[name][0](api)


@pytest.mark.parametrize("name", list(CHECKS))
def test_baselines_toy_passes(name):
    CHECKS[name][0](_toy())


@pytest.mark.parametrize("name,bug", MUTANTS)
def test_baselines_mutant_fails(name, bug):
    toy = _toy({bug}, hidden_labels=_lab() if bug == "global_labels" else None)
    with pytest.raises(AssertionError):
        CHECKS[name][0](toy)


def test_leaky_detector_that_never_flags_is_caught():
    """BL11 mutant at the detector: a detector reporting everything clean fails the leaky-run assertion."""
    wf = _wf()

    def never_flags(result, labels_all, **kw):
        return wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, 0)

    with pytest.raises(AssertionError):
        check_leaky_configuration_flagged_with_baseline(_toy(), detect=never_flags)


def test_real_baselines_leaky_detector_never_flags_is_caught():
    api = pytest.importorskip("vanta.baselines")
    wf = _wf()

    def never_flags(result, labels_all, **kw):
        return wf.LeakReport(False, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, 0)

    with pytest.raises(AssertionError):
        check_leaky_configuration_flagged_with_baseline(api, detect=never_flags)
