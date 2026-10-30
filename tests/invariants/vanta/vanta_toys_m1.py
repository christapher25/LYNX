"""
Reference ("toy") implementation of the vanta.m1 pure functions, with switchable deliberate bugs.

Not a VANTA module. The tests run every check against `toy_api()` (must pass) and against `toy_api(<bug>)` (must fail), so a
test that cannot fail is caught. The real vanta.m1 is checked by the same functions via importorskip.
"""

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

NO_BUGS = frozenset()

Q = 0.30
CAP = 10
REFERENCE_NOTIONAL = 100_000
HORIZONS = (5, 20)
K_S_GRID = (1.0, 1.5, 2.0)
K_T_GRID = (1.5, 2.0, 3.0)

MODEL = "m@f01:2012-12-31:abc:123"
FILTER = "cf@v1"


def _pub(log, bugs):
    p = log["published"].astype(bool)
    return log if "fdr_all_candidates" in bugs else log[p]


def realised_fdr(log, bugs=NO_BUGS):
    pub = _pub(log, bugs)
    y = pub["y_net"].astype(float)
    is_filled = pub["fill_status"] == "FILLED"
    ok = is_filled & y.notna()
    loser = (y < 0) if "strict_loser" in bugs else (y <= 0)
    loser = loser & ok
    if "no_fill_as_loser" in bugs:
        loser = loser | ~is_filled
        ok = ok | ~is_filled
    if "nan_as_loser" in bugs:
        loser = loser | (is_filled & y.isna())
        ok = ok | (is_filled & y.isna())
    n_filled, n_losers = int(ok.sum()), int(loser.sum())
    n_no_fill = 0 if "no_fill_hidden" in bugs else int((~is_filled).sum())
    n_unmatured = 0 if "unmatured_hidden" in bugs else int((is_filled & y.isna()).sum())
    fdr = n_losers / n_filled if n_filled else float("nan")
    return dict(fdr=fdr, n_published=int(len(pub)), n_filled=n_filled, n_no_fill=n_no_fill, n_losers=n_losers,
                n_unmatured=n_unmatured)


def fdr_by_year(log, bugs=NO_BUGS):
    if "pooled_years" in bugs:
        f = realised_fdr(log, bugs)["fdr"]
        return {int(y): f for y in log["decision_date"].dt.year.unique()}
    out = {}
    for y, g in log.groupby(log["decision_date"].dt.year):
        s = realised_fdr(g, bugs)
        if s["n_filled"]:
            out[int(y)] = s["fdr"]
    return out


def publication_rate(log, decision_dates=None, bugs=NO_BUGS):
    if "per_row_rate" in bugs:
        return float(log["published"].astype(bool).mean())
    weeks = pd.DatetimeIndex(decision_dates) if decision_dates is not None and "ignore_given_weeks" not in bugs \
        else pd.DatetimeIndex(log["decision_date"].unique())
    pub = log[log["published"].astype(bool)]["decision_date"].unique()
    return len(pd.DatetimeIndex(pub).intersection(weeks)) / len(weeks)


def newey_west_se(x, lags, bugs=NO_BUGS):
    x = np.asarray(x, float)
    n = len(x)
    d = x - x.mean()
    var = float(d @ d) / n
    if "no_lags" not in bugs:
        for l in range(1, lags + 1):
            w = 1.0 if "flat_weights" in bugs else 1 - l / (lags + 1)
            var += 2 * w * float(d[l:] @ d[:-l]) / n
    var /= n
    return float(np.sqrt(var)) if var >= 0 else float("nan")


def _week_groups(y, week_ids):
    y = np.asarray(y, float)
    w = np.asarray(week_ids)
    keys = list(dict.fromkeys(w.tolist()))
    return keys, {k: y[w == k] for k in keys}


def weekly_block_bootstrap(y, week_ids, n_boot, seed, bugs=NO_BUGS):
    rng = np.random.default_rng(seed)
    if "iid_signals" in bugs:
        y = np.asarray(y, float)
        return np.array([rng.choice(y, len(y)).mean() for _ in range(n_boot)])
    keys, g = _week_groups(y, week_ids)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        out.append(np.concatenate([g[keys[i]] for i in pick]).mean())
    return np.array(out)


def bootstrap_ci(draws, level=0.95, bugs=NO_BUGS):
    a = (1 - level) / 2 * 100
    if "one_sided" in bugs:
        return float(np.percentile(draws, 100 - level * 100)), float(np.max(draws))
    return float(np.percentile(draws, a)), float(np.percentile(draws, 100 - a))


def weekly_block_bootstrap_diff(y_a, week_a, y_b, week_b, n_boot, seed, bugs=NO_BUGS):
    rng = np.random.default_rng(seed)
    ka, ga = _week_groups(y_a, week_a)
    kb, gb = _week_groups(y_b, week_b)
    union = list(dict.fromkeys(ka + kb))
    out = []
    for _ in range(n_boot):
        pa = rng.integers(0, len(union), len(union))
        pb = rng.integers(0, len(union), len(union)) if "independent_arms" in bugs else pa
        ma = np.concatenate([ga.get(union[i], np.array([])) for i in pa]).mean()
        mb = np.concatenate([gb.get(union[i], np.array([])) for i in pb]).mean()
        out.append(ma - mb)
    return np.array(out)


def choose_k(trades, start="2010-01-01", end="2012-12-31", bugs=NO_BUGS):
    t = trades
    d = t["decision_date"]
    if "uses_all_data" in bugs:
        m = pd.Series(True, index=t.index)
    elif "exclusive_end" in bugs:
        m = (d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))
    else:
        m = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    t = t[m].copy()
    if "nan_as_zero" in bugs:
        t["y_net"] = t["y_net"].fillna(0.0)
    g = t.groupby(["k_s", "k_t"])["y_net"].mean().dropna()
    if "last_tie" in bugs:
        return tuple(g[g == g.max()].index[-1])
    return tuple(g.idxmax())


def _metrics(log, bugs):
    s = realised_fdr(log, bugs)
    return dict(fdr=s["fdr"], n_filled=s["n_filled"], n_no_fill=s["n_no_fill"],
                publication_rate=publication_rate(log, None, bugs), fdr_by_year=fdr_by_year(log, bugs))


def go_report(log, oos_start=pd.Timestamp("2013-01-01"), bugs=NO_BUGS):
    if "oos_includes_insample" in bugs:
        return dict(in_sample=_metrics(log[log["decision_date"] < oos_start], bugs), oos=_metrics(log, bugs))
    return dict(in_sample=_metrics(log[log["decision_date"] < oos_start], bugs),
                oos=_metrics(log[log["decision_date"] >= oos_start], bugs))


def v4_gate_reconstruction(cands, cal, T, *, model_version, candidate_filter_version, q=Q, cap=CAP, bugs=NO_BUGS):
    T = pd.Timestamp(T)
    keep = (cal["model_version"] == model_version) & (cal["candidate_filter_version"] == candidate_filter_version) \
        & cal["is_oos"] & (cal["decision_date"] >= T - pd.Timedelta(weeks=52))
    if "uses_unmatured" not in bugs:
        keep &= cal["matured_at"].notna() & (cal["matured_at"] <= T)
    c = cal[keep & cal["y_net"].notna()]
    slope, icpt = np.polyfit(c["score"].to_numpy(float), c["y_net"].to_numpy(float), 1)
    fit_c = slope * c["score"].to_numpy(float) + icpt
    short = fit_c - c["y_net"].to_numpy(float)
    qq = q if "wrong_quantile" in bugs else 1 - q
    cut = np.quantile(short, qq)
    fitted = slope * cands["score"].to_numpy(float) + icpt
    bound = fitted + cut if "wrong_quantile" in bugs else fitted - cut
    if "bound_is_fitted" in bugs:
        bound = fitted
    ok = (bound >= 0) if "nonstrict" in bugs else (bound > 0)
    out = pd.DataFrame({"symbol": cands["symbol"].to_numpy(), "score": cands["score"].to_numpy(float),
                        "fitted": fitted, "bound": bound, "published": ok}).reset_index(drop=True)
    if "no_cap" not in bugs:
        top = out[out["published"]].sort_values("score", ascending=False).index[:cap]
        out["published"] = out.index.isin(top)
    return out


def impact_floor_per_side(turnover_rank, bugs=NO_BUGS):
    if turnover_rank is None or not np.isfinite(turnover_rank):
        return 0.0 if "nan_as_zero" in bugs else float("nan")
    if turnover_rank < 1:
        raise ValueError("rank must be >= 1")
    if turnover_rank <= (201 if "off_by_one" in bugs else 200):
        return 0.0005
    if turnover_rank <= 700:
        return 0.0020
    return 0.0020 if "default_tail" in bugs else float("nan")


def turnover_ranks(trailing_turnover, bugs=NO_BUGS):
    t = pd.Series(trailing_turnover, dtype=float)
    if "nan_as_zero" in bugs:
        t = t.fillna(0.0)
    return t.rank(ascending=("ascending" in bugs), method="min", na_option="keep")


def fdr_report_text(stats, q, bugs=NO_BUGS):
    f = stats["fdr"]
    if "guarantee" in bugs:
        return f"FDR is guaranteed at most {q:.2f}; realised {f:.3f}. NO_FILL published: {stats['n_no_fill']}."
    s = f"Empirical realised FDR {f:.3f} on {stats['n_filled']} filled trades, against a target q of {q:.2f}, which is not a guarantee."
    if "no_nofill" not in bugs:
        s += f" Published signals that were NO_FILL: {stats['n_no_fill']}."
    return s


FUNCS = ["realised_fdr", "fdr_by_year", "publication_rate", "newey_west_se", "weekly_block_bootstrap", "bootstrap_ci",
         "weekly_block_bootstrap_diff", "choose_k", "go_report", "v4_gate_reconstruction", "impact_floor_per_side",
         "turnover_ranks", "fdr_report_text"]


def toy_api(*bugs):
    b = frozenset(bugs)
    ns = {n: (lambda f: (lambda *a, **k: f(*a, bugs=b, **k)))(globals()[n]) for n in FUNCS}
    return SimpleNamespace(Q=Q, CAP=CAP, REFERENCE_NOTIONAL=REFERENCE_NOTIONAL, HORIZONS=HORIZONS, K_S_GRID=K_S_GRID,
                           K_T_GRID=K_T_GRID, **ns)


def strip_negated_guarantee(text):
    return re.sub(r"\b(not a|no) guarantee\b", "", text, flags=re.I)
