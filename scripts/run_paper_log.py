"""
Forward paper log driver (weekly, resumable). Library and contract: vanta/paper_log.py, tests/test_paper_log.py. Output ONLY under
data/processed/paper_log/ (decisions.jsonl, outcomes.jsonl, unfilled.jsonl, runs.jsonl, models/, cache/). Nothing is deleted or rewritten.

A weekly decision date T is the last session of an ISO week. Run this job once the week is over (Sunday, or Monday before the open) and after
the data pipeline has refreshed every input through T; it REFUSES, with the reasons, when any input is stale or when the data already contain
sessions after T (the entry price would be public; --allow-late records such a run flagged as late).

Stages (--stage, default all, in this order):
  outcomes  append the outcomes of earlier decisions that have matured (hold-to-horizon labels, impact floor), and the never-filled ones.
  models    fit the live B1 (hold labels, tuned vanta.m3 params, 5 seeds) for the quarter of T if its model does not exist yet. Models are
            written once under models/ and never refit or overwritten. Refit schedule: quarterly, at the first session on/after Jan 1,
            Apr 1, Jul 1, Oct 1; training = decision dates in [R - 4 years, R) whose label ended before the session `embargo` sessions
            before R (N11). Hyperparameters are the frozen vanta.m3 values; never retuned.
  decide    score T's candidates with the live models, run the three gate arms and the ungated top 10, append the full candidate list.

Usage:
    python -m scripts.run_paper_log                       # all stages for the latest complete decision week
    python -m scripts.run_paper_log --check               # only report whether the inputs are current for the latest decision week
    python -m scripts.run_paper_log --stage outcomes
    python -m scripts.run_paper_log --decision-date 2026-11-06 [--allow-late]

q = 0.30 is an engineering target; realised FDR is an empirical number, never a guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from loguru import logger

import config
from data.trading_calendar import trading_sessions
from pitstore.store import BLOCK_SPECS
from scripts import run_baselines, run_m1, run_m1b, run_m2
from vanta import baselines as bl
from vanta import m1, m1b, m2, m3
from vanta import paper_log as P
from vanta import regime as R
from vanta import walkforward as wf
from vanta.labels import load_terminal_values

LOG_DIR = config.PROCESSED_DATA_DIR / "paper_log"
CACHE_DIR = LOG_DIR / "cache"
MODEL_DIR = LOG_DIR / "models"
CLIFF = m1.LIQUIDITY_CLIFF_RANK
NO_LEVEL = float("inf")
MIN_SYMBOL_COVERAGE = 0.90          # share of candidate symbols whose block file must reach the date the as-of read needs
MIN_CANDIDATE_FRACTION = 0.5        # candidates at T must be at least this fraction of the median of the last logged weeks
MIN_CANDIDATES_ABSOLUTE = 50


def _logger() -> None:
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.LOG_DIR / "run_paper_log.log", rotation="20 MB")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=["all", "outcomes", "models", "decide"], default="all")
    ap.add_argument("--decision-date", default=None, help="a weekly decision date (default: the latest complete one)")
    ap.add_argument("--allow-late", action="store_true", help="record a decision although later sessions are already in the data (flagged late)")
    ap.add_argument("--check", action="store_true", help="report input currency for the decision week and write nothing")
    return ap


# ---------------------------------------------------------------------------------------------------------------- audit trail
def _runs_line(event: str, **kw) -> None:
    rec = dict(event=event, computed_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw)
    P._append_bytes(LOG_DIR / P.RUNS, [P._line(rec)])


def _date_block_sha(decision_date) -> str:
    d = P._iso(decision_date)
    lines = [ln for ln in P._read_lines(LOG_DIR / P.DECISIONS) if json.loads(ln)["decision_date"] == d]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------- inputs
def latest_equity_session() -> pd.Timestamp | None:
    days = []
    for f in Path(config.BHAVCOPY_DIR).glob("*.parquet"):
        try:
            days.append(pd.Timestamp(f.stem))
        except ValueError:
            continue
    return max(days) if days else None


def _last_dates(directory: Path, symbols) -> dict:
    out = {}
    for s in symbols:
        f = directory / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["date"])["date"]
            out[s] = pd.Timestamp(d.max()) if len(d) else None
    return out


def input_problems(T, sessions, cands: pd.DataFrame) -> list[str]:
    """Every reason the INPUTS are not current for a decision at T. A block whose data end before the date the as-of read needs would be read
    as 'absent' and turned into NaN features without any error, so each block's end date is checked explicitly."""
    probs = []
    T = pd.Timestamp(T)
    sess = pd.DatetimeIndex(sessions)
    pos = int(sess.get_loc(T))
    syms = sorted(cands["symbol"].unique())
    for b in run_baselines.FEATURE_BLOCKS:
        spec = BLOCK_SPECS[b]
        need = sess[max(pos - spec.lag_sessions, 0)]
        if spec.scope == "market":
            d = pd.read_parquet(spec.path, columns=["date"])["date"]
            last = pd.Timestamp(d.max()) if len(d) else None
            if last is None or last < need:
                probs.append(f"block {b}: data end {None if last is None else last.date()}, need {need.date()}")
        else:
            lasts = _last_dates(Path(spec.path), syms)
            ok = sum(1 for v in lasts.values() if v is not None and v >= need)
            if not syms or ok / len(syms) < MIN_SYMBOL_COVERAGE:
                probs.append(f"block {b}: only {ok}/{len(syms)} candidate symbols have data through {need.date()}")
    ok = 0
    for s in syms:
        f = Path(config.ADJUSTED_PRICES_DIR) / f"{s}.parquet"
        if f.exists():
            px = pd.read_parquet(f, columns=["DATE"])["DATE"]
            ok += int(len(px) > 0 and pd.Timestamp(px.max()) >= T)
    if not syms or ok / len(syms) < MIN_SYMBOL_COVERAGE:
        probs.append(f"adjusted prices: only {ok}/{len(syms)} candidate symbols have a bar on {T.date()}")
    if not (Path(config.RAW_DATA_DIR) / "indices" / f"{T.date()}.parquet").exists():
        probs.append(f"index file for {T.date()} is missing (India VIX for the regime posterior)")
    if not (Path(config.UNIVERSE_DIR) / f"{T.year}.parquet").exists():
        probs.append(f"universe snapshot for {T.year} is missing")
    n = len(cands)
    prev = P.read_decisions(LOG_DIR)
    if not prev.empty:
        per = prev[prev["decision_date"] < T].groupby("decision_date").size() / max(1, prev["horizon"].nunique() if "horizon" in prev else 1)
        floor = max(MIN_CANDIDATES_ABSOLUTE, MIN_CANDIDATE_FRACTION * float(per.tail(8).median())) if len(per) else MIN_CANDIDATES_ABSOLUTE
    else:
        floor = MIN_CANDIDATES_ABSOLUTE
    if n < floor:
        probs.append(f"only {n} candidates at {T.date()} (expected at least {floor:.0f}): gate inputs (ban list, price bands) are probably stale")
    return probs


def candidates_for_dates(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Gate-passing candidates (run_m1b.build_candidates, unchanged) for explicit weekly dates, cached under paper_log/cache/ by a key over
    the data and code. Everything run_m1b writes while building (liquidity ranks) also lands under paper_log/cache/m1b, never in m1b/."""
    run_m1b.M1B_DIR = CACHE_DIR / "m1b"
    run_m1b.M1B_DIR.mkdir(parents=True, exist_ok=True)
    run_m1b.FORCE = False
    key = run_m1b._candidate_key(dates, CLIFF)
    f = CACHE_DIR / f"candidates_{dates.min().date()}_{dates.max().date()}_{key}.parquet"
    got = run_m1b._read_or_none(f)
    if got is not None:
        got["decision_date"] = pd.to_datetime(got["decision_date"])
        return got
    c = run_m1b.build_candidates(dates, CLIFF)
    run_m1b._atomic_to_parquet(c, f)
    return c


def label_cache(h: int) -> Path:
    """Chunk directory per horizon AND per hold-label source hash, as run_m3.hold_label_dir: an edit to the label code gives a new directory."""
    return CACHE_DIR / "labels_hold" / f"h{h}_{m3.hold_label_source_fp()}"


def hold_labels(cands: pd.DataFrame, h: int, sessions, directory: Path) -> pd.DataFrame:
    """Hold-to-horizon labels for (symbol, decision_date) pairs, floor impact applied (the M3 label), chunked and cached under `directory`."""
    with run_m2._hold_labels():
        lab = run_m1b.build_labels(cands[["symbol", "decision_date"]], h, NO_LEVEL, NO_LEVEL, directory, sessions, load_terminal_values())
    df = cands.merge(lab, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=h)
    df["fill_status"] = df["fill_status"].fillna("NO_DATA")
    return m1.apply_impact(df, df["liquidity_rank"], "floor")


# ---------------------------------------------------------------------------------------------------------------- outcomes
def run_outcomes(sessions, asof) -> None:
    pend = P.pending_keys(LOG_DIR)
    if pend.empty:
        logger.info("outcomes: nothing pending")
        return
    asof = pd.Timestamp(asof)
    L = len(sessions) - 1
    n_out = n_unf = 0
    for h, g in pend.groupby("horizon"):
        h = int(h)
        pos = sessions.searchsorted(pd.DatetimeIndex(g["decision_date"]).to_numpy(), side="left")
        ready = g[(pos + h + 1) <= L]                      # the horizon-end bar and the next session (maturity) are on disk
        if ready.empty:
            logger.info(f"outcomes h={h}: {len(g)} pending, none old enough yet")
            continue
        cands = ready[["symbol", "decision_date", "liquidity_rank"]].reset_index(drop=True)
        lab = hold_labels(cands, h, sessions, label_cache(h))
        lab["matured_at"] = m1.matured_at(pd.to_datetime(lab["label_end_date"]), sessions)
        keep = ["decision_date", "symbol", "horizon", "y_net", "matured_at", "fill_status", "label_end_date", "entry_date", "exit_reason",
                "gross_return", "impact_rt"]
        lab = lab[keep]
        n_out += P.append_outcomes(LOG_DIR, lab, asof)
        n_unf += P.append_unfilled(LOG_DIR, lab, asof)
    _runs_line("outcomes", asof=P._iso(asof), n_outcomes=n_out, n_unfilled=n_unf)
    logger.info(f"outcomes: appended {n_out} outcomes and {n_unf} unfilled rows (asof {asof.date()})")


# ---------------------------------------------------------------------------------------------------------------- live models
def model_path(R_date, h: int) -> Path:
    return MODEL_DIR / f"h{h}_{pd.Timestamp(R_date).date()}"


def load_model(R_date, h: int):
    d = model_path(R_date, h)
    mf = d / "manifest.json"
    if not mf.exists():
        return None
    man = json.loads(mf.read_text())
    from catboost import CatBoostRegressor
    models = []
    for s in man["seeds"]:
        m = CatBoostRegressor()
        m.load_model(str(d / f"seed{s}.cbm"))
        models.append(m)
    return SimpleNamespace(models=models, seeds=tuple(man["seeds"]), feature_cols=man["feature_cols"], manifest=man)


def ensure_model(R_date, h: int, sessions, cv: dict):
    """The live model for (refit date, horizon): loaded if it exists, else trained once and written. Never overwritten."""
    got = load_model(R_date, h)
    if got is not None:
        return got
    try:
        m3.require_tuned_hold_params(h)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    R_date = pd.Timestamp(R_date)
    t0 = time.time()
    start, end = P.training_window(R_date)
    dates = run_m1.weekly_dates(sessions, start, R_date - pd.Timedelta(days=1))
    cands = candidates_for_dates(dates)
    lab = hold_labels(cands, h, sessions, label_cache(h))
    emb = wf.embargo_for_horizon(h)
    rows = wf.purge_training_rows(lab[["symbol", "decision_date", "label_end_date", "y_net"]], R_date, emb, sessions)
    rows = rows[np.isfinite(rows["y_net"].to_numpy(float))]
    if rows.empty:
        raise SystemExit(f"h={h} refit {R_date.date()}: no matured training rows")
    panel, cols = run_baselines.build_features(cands.merge(rows[["symbol", "decision_date"]], on=["symbol", "decision_date"]), sessions)
    fit = bl.make_fit_fn(panel, cols, bl.b1_model_factory(**m3.hold_params(h)), seeds=bl.B1_SEEDS, target="net")(rows, None)
    final = model_path(R_date, h)
    tmp = final.with_name(final.name + f".partial-{int(time.time())}")
    tmp.mkdir(parents=True, exist_ok=True)
    for s, m in zip(fit.seeds, fit.models):
        m.save_model(str(tmp / f"seed{s}.cbm"))
    keys = fit.train_keys.sort_values(["decision_date", "symbol"])
    man = dict(horizon=h, refit_date=R_date.date().isoformat(), train_window=[start.date().isoformat(), R_date.date().isoformat()],
               embargo_sessions=emb, last_label_end=pd.Timestamp(rows["label_end_date"].max()).date().isoformat(),
               last_train_decision=pd.Timestamp(rows["decision_date"].max()).date().isoformat(), n_train=int(len(keys)),
               params=m3.hold_params(h), seeds=list(fit.seeds), param_hash=P.param_hash(m3.hold_params(h), fit.seeds),
               feature_cols=list(fit.feature_cols), labels="hold-to-horizon net, impact floor",
               train_keys_sha1=hashlib.sha1(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(),
               code_version=cv, trained_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, default=str))     # last: marks a complete model
    tmp.rename(final)                                                              # atomic; refuses (OSError) if `final` exists
    logger.info(f"model h={h} refit {R_date.date()} written: {len(keys):,} rows, {(time.time() - t0) / 60:.1f} min")
    return load_model(R_date, h)


def model_version_string(h: int, man: dict) -> str:
    return (f"b1-hold-live|h{h}|refit {man['refit_date']}|params {man['param_hash']}|seeds {','.join(map(str, man['seeds']))}|"
            f"train rows to {man['last_train_decision']} (labels to {man['last_label_end']})")


# ---------------------------------------------------------------------------------------------------------------- live regime
def _save_hmm(path: Path, p: R.RegimeHMM) -> None:
    np.savez(path, startprob=p.startprob, transmat=p.transmat, means=p.means, variances=p.variances, mu=p.mu, sigma=p.sigma, lo=p.lo,
             hi=p.hi, meta=np.array(json.dumps(dict(feature_names=list(p.feature_names),
                                                    train_end=None if p.train_end is None else str(pd.Timestamp(p.train_end).date()),
                                                    n_train=int(p.n_train), train_loglik=float(p.train_loglik),
                                                    restart_logliks=[float(x) for x in p.restart_logliks]))))


def _load_hmm(path: Path) -> R.RegimeHMM:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return R.RegimeHMM(startprob=z["startprob"], transmat=z["transmat"], means=z["means"], variances=z["variances"], mu=z["mu"],
                       sigma=z["sigma"], lo=z["lo"], hi=z["hi"], feature_names=tuple(meta["feature_names"]),
                       train_end=None if meta["train_end"] is None else pd.Timestamp(meta["train_end"]), n_train=meta["n_train"],
                       train_loglik=meta["train_loglik"], restart_logliks=tuple(meta["restart_logliks"]))


def live_regime(h: int, T) -> tuple:
    """(posterior list of 3, hmm_version) at T: the annual-fold HMM exactly as run_m1.regime_posterior builds it (fold year = T.year, fitted
    on rows up to `h` sessions before Jan 1, 5 restarts), FORWARD-ALGORITHM filtered posterior only, read at T itself. The fitted HMM is
    stored once per (year, horizon) and reused. Missing posterior at T -> ([nan]*3, None); no as-of fill from an earlier day."""
    T = pd.Timestamp(T)
    feats = R.build_regime_features(end=T)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    f = MODEL_DIR / f"hmm_{T.year}_h{h}.npz"
    if f.exists():
        params = _load_hmm(f)
    else:
        te = R.fold_train_end(feats.index, pd.Timestamp(f"{T.year}-01-01"), h)
        params = R.fit_hmm(feats, te)
        _save_hmm(f, params)
    post = R.filtered_posterior_frame(params, feats)
    ver = f"hmm@{T.year}:{pd.Timestamp(params.train_end).date()}:5restarts"
    if T not in post.index or not np.isfinite(post.loc[T].to_numpy(float)).all():
        return [float("nan")] * 3, ver
    return [float(x) for x in post.loc[T].to_numpy(float)], ver


# ---------------------------------------------------------------------------------------------------------------- decide
def decision_rows(T, sessions, cands: pd.DataFrame, models: dict, cv: dict) -> tuple:
    """The full candidate list of T for every horizon with scores, percentiles, regime and publication flags. Returns (frame, model_version)."""
    panel, cols = run_baselines.build_features(cands, sessions)
    rows_q = cands[["decision_date", "symbol"]]
    parts = []
    for h, model in models.items():
        if list(model.feature_cols) != list(cols):
            raise SystemExit(f"h={h}: the live feature set differs from the one the model was trained on; refusing to score")
        score = bl.make_score_fn(panel, cols)(model, rows_q, None, None)
        pct = m1b.percentile_rank(np.asarray(score, float))
        pi, hv = live_regime(h, T)
        status = P.calibration_status(LOG_DIR, h, T)
        cal = P.calibration_frame(LOG_DIR, h, T)
        pub = P.publications(h, pd.DataFrame({"symbol": cands["symbol"].to_numpy(), "score": pct}), cal, T, pi, status)
        base = pd.DataFrame({"symbol": cands["symbol"].to_numpy(), "horizon": h, "score": np.asarray(score, float), "percentile": pct,
                             "mom": cands["mom"].to_numpy(), "liquidity_rank": cands["liquidity_rank"].to_numpy(),
                             "univ_rank": cands["univ_rank"].to_numpy(), "regime_posterior": [pi] * len(cands), "hmm_version": hv,
                             "horizon_model_version": model_version_string(h, model.manifest),
                             "calibration_weeks_of_history": status["weeks_of_history"], "calibration_reason": status["reason"]})
        out = pd.concat([base, pub.drop(columns=["symbol"])], axis=1)
        out["published"] = out[[f"pub_{a}" for a in P.ARMS]].fillna(False).astype(bool).any(axis=1)
        parts.append(out)
    df = pd.concat(parts, ignore_index=True).sort_values(["horizon", "symbol"], kind="stable").reset_index(drop=True)
    version = "b1-hold-live@m3 || " + " || ".join(model_version_string(h, m.manifest) for h, m in models.items())
    return df, version


def run_decide(T, sessions, latest_eq, cv: dict, allow_late: bool, check_only: bool = False) -> int:
    T = pd.Timestamp(T)
    done = P.read_decisions(LOG_DIR)
    if not done.empty and (done["decision_date"] == T).any():
        logger.info(f"decide: {T.date()} is already recorded; nothing to do (NOOP)")
        return 0
    probs = P.freshness_problems(T, sessions, latest_eq, allow_late)
    if probs:
        return _refuse(T, probs, check_only)
    R_date = P.refit_date(T, sessions)
    models = {h: load_model(R_date, h) for h in P.HORIZONS}
    missing = [f"the live model for h={h}, refit {R_date.date()}, does not exist: run --stage models first" for h, m in models.items() if m is None]
    if missing and not check_only:
        return _refuse(T, missing, check_only)
    cands = candidates_for_dates(pd.DatetimeIndex([T]))
    cands = cands[cands["decision_date"] == T].reset_index(drop=True)
    probs = missing + input_problems(T, sessions, cands)
    if probs:
        return _refuse(T, probs, check_only)
    if check_only:
        logger.info(f"check: inputs are current for {T.date()} ({len(cands)} candidates)")
        return 0
    df, version = decision_rows(T, sessions, cands, models, cv)
    late = bool(pd.Timestamp(latest_eq) > T)
    df["late"] = late
    res = P.append_decisions(LOG_DIR, T, df, model_version=version, code_version=P.code_version_string(cv), data_asof=latest_eq)
    _runs_line("decision", decision_date=P._iso(T), result=res, data_asof=P._iso(latest_eq), late=late, code_version=cv,
               n_rows=int(len(df)), decisions_bytes=(LOG_DIR / P.DECISIONS).stat().st_size, block_sha256=_date_block_sha(T),
               gate_status={int(h): sorted(set(g["gate_status"])) for h, g in df.groupby("horizon")})
    logger.info(f"decide: {T.date()} {res}: {len(df)} rows; published (any arm) {int(df['published'].sum())}; "
                + "; ".join(f"h={h} {g['gate_status'].iloc[0]}" for h, g in df.groupby("horizon")))
    return 0


def _refuse(T, problems: list, check_only: bool) -> int:
    for p in problems:
        logger.error(f"REFUSED {pd.Timestamp(T).date()}: {p}")
    if not check_only:
        _runs_line("refused", decision_date=P._iso(T), problems=problems)
    return 3


# ---------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    a = build_parser().parse_args()
    _logger()
    sessions = pd.DatetimeIndex(pd.to_datetime(trading_sessions()))
    latest_eq = latest_equity_session()
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    T = pd.Timestamp(a.decision_date) if a.decision_date else P.latest_complete_decision_date(sessions, now)
    if pd.isna(T):
        logger.error("no complete decision week in the calendar")
        return 3
    if T not in _weekly(sessions):
        logger.error(f"{T.date()} is not a weekly decision date (the last session of an ISO week)")
        return 3
    cv = P.code_version(config.PROJECT_ROOT)
    if cv["dirty"]:
        logger.warning(f"code is dirty ({len(cv['dirty_files'])} changed paths): rows will carry commit {cv['commit'][:12]}+dirty")
    if a.check:
        return run_decide(T, sessions, latest_eq, cv, a.allow_late, check_only=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    if a.stage in ("all", "outcomes"):
        run_outcomes(sessions, latest_eq)
    if a.stage in ("all", "models"):
        if m3.is_placeholder(20) or m3.is_placeholder(60):
            logger.error("vanta.m3 B1_HOLD_PARAMS_BY_HORIZON[20 or 60] is still the placeholder: tune and commit before the paper log trains")
            return 2
        R_date = P.refit_date(T, sessions)
        for h in P.HORIZONS:
            ensure_model(R_date, h, sessions, cv)
    if a.stage in ("all", "decide"):
        rc = run_decide(T, sessions, latest_eq, cv, a.allow_late)
    return rc


def _weekly(sessions):
    from scripts.report_nel_coverage import weekly_decision_dates
    return weekly_decision_dates(sessions)


if __name__ == "__main__":
    sys.exit(main())
