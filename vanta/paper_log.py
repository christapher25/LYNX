"""VANTA forward paper log: the only fresh test (docs/vanta/M3_PREREG.md section 1).

An append-only record of what the system WOULD have published each week, written at decision time, and of what happened later. Contract:
tests/test_paper_log.py (names in docs/vanta/INTERFACES.md section 9a). Files under the log directory (data/processed/paper_log/):

  decisions.jsonl   one JSON object per candidate row (decision date x horizon x symbol), written once per decision date, never edited.
                    Carries the decision date, model version, code version, the score, the weekly percentile and the publication flags.
                    NO outcome field, ever.
  outcomes.jsonl    one row per matured, filled, finite outcome, appended later by a separate call; never edited, never duplicated.
  unfilled.jsonl    rows whose trade never happened (NO_FILL_*, NO_DATA); same rules.
  runs.jsonl        audit trail written by the driver: when each decision was computed, data cut-off, code state.

Rules this module keeps:
  * a decision date already on file is never rewritten: identical content is a NOOP, different content is PaperLogError, and the earlier
    bytes of every file stay a prefix of the file after any later append;
  * input data that ends before the decision date, or whose end date is unknown, is refused (StaleDataError) BEFORE anything is written;
  * an outcome is recorded only when its label has matured by `asof` and is finite (a missing outcome is missing, never 0);
  * calibration for the gate comes only from logged candidates whose outcome has matured by the decision date; too little history is
    reported as INSUFFICIENT (values None / NaN), never fabricated.

q = 0.30 is an engineering target; realised FDR is an empirical number, never a guarantee.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from vanta import m2

# ---------------------------------------------------------------------------------------------------------------- constants
DECISIONS, OUTCOMES, UNFILLED, RUNS = "decisions.jsonl", "outcomes.jsonl", "unfilled.jsonl", "runs.jsonl"
HORIZONS = (20, 60)                          # the judged M3 horizons; h = 5 is not logged
ARMS = m2.ARMS
LIVE_MODEL_KEY = "b1-hold-live@m3"           # the gate's model_version key: constant across refits, as MODEL_VERSION is across folds
FILTER_VERSION = m2.FILTER_VERSION
WINDOW_WEEKS = m2.WINDOW_WEEKS               # calibration window AND the history needed before a gate may publish (the M2/M3 judged rule)
TRAIN_YEARS = 4                              # live model training window (the M3 fold geometry's training length)
CODE_PATHS = ("vanta", "scripts", "features", "gates", "pitstore", "signals", "config.py", "data/*.py")


class PaperLogError(RuntimeError):
    pass


class StaleDataError(PaperLogError):
    pass


# ---------------------------------------------------------------------------------------------------------------- serialisation
def _clean(v):
    """JSON-safe value: NaN / inf -> None (missing, never 0), numpy scalars -> python, Timestamps -> ISO date, arrays / lists recursively."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _iso(d) -> str:
    return pd.Timestamp(d).date().isoformat()


def _line(rec: dict) -> str:
    return json.dumps({k: _clean(v) for k, v in rec.items()}, sort_keys=True, allow_nan=False)


def _read_lines(path: Path) -> list[str]:
    """Complete lines of a log file. A file that does not end with a newline holds a torn last write: refuse (never repair silently)."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise PaperLogError(f"{path.name} ends with a partial line (interrupted write); inspect it by hand, nothing was changed")
    return [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in _read_lines(Path(path))]


def _append_bytes(path: Path, lines: list[str]) -> None:
    """One write of the whole block, flushed to disk. The only way any log file is ever modified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    with open(path, "ab") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())


@contextlib.contextmanager
def _locked(path: Path):
    """Exclusive O_EXCL lock marker beside a log file for the whole check-then-append. A second writer (or a crashed one: the marker is
    left for a human to inspect) gets PaperLogError instead of racing the date-exists check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise PaperLogError(f"{lock.name} exists: another writer is active (or one died); inspect it, nothing was changed") from e
    try:
        yield
    finally:
        os.close(fd)
        os.unlink(lock)


def _key(rec: dict) -> tuple:
    return (rec["decision_date"], rec.get("horizon"), rec["symbol"])


# ---------------------------------------------------------------------------------------------------------------- decisions
def append_decisions(log_dir, decision_date, candidates: pd.DataFrame, *, model_version: str, code_version: str, data_asof) -> str:
    """Record one decision date: EVERY candidate (published or not) with its columns. Returns "APPENDED" or "NOOP" (identical content already
    on file). PaperLogError if the date is on file with different content (nothing written); StaleDataError if `data_asof` (the last date
    the input data covers) is missing or earlier than the decision date (nothing written, no directory created).
    `candidates` needs `symbol` and `published`; a `horizon` column makes (symbol, horizon) the row key."""
    log_dir = Path(log_dir)
    d = pd.Timestamp(decision_date)
    if data_asof is None or pd.isna(data_asof):
        raise StaleDataError("the data cut-off date is unknown: refusing to guess that the inputs reach the decision date")
    if pd.Timestamp(data_asof) < d:
        raise StaleDataError(f"input data ends {_iso(data_asof)}, before the decision date {_iso(d)}")
    for c in ("symbol", "published"):
        if c not in candidates.columns:
            raise PaperLogError(f"candidates lack the required column {c!r}")
    keys = candidates[["symbol", "horizon"]] if "horizon" in candidates.columns else candidates[["symbol"]]
    if keys.duplicated().any():
        raise PaperLogError("duplicate (symbol, horizon) rows in the candidate list")
    rows = []
    for rec in candidates.to_dict("records"):
        r = dict(rec)
        r.update(decision_date=_iso(d), model_version=model_version, code_version=code_version)
        rows.append(_line(r))
    f = log_dir / DECISIONS
    with _locked(f):
        existing = _read_lines(f)
        have = [ln for ln in existing if json.loads(ln)["decision_date"] == _iso(d)]
        if have:
            if have == rows:
                return "NOOP"
            raise PaperLogError(f"decision date {_iso(d)} is already recorded with different content; the log is append-only")
        _append_bytes(f, rows)
    return "APPENDED"


def read_decisions(log_dir) -> pd.DataFrame:
    rows = _read_rows(Path(log_dir) / DECISIONS)
    df = pd.DataFrame(rows)
    if len(df):
        df["decision_date"] = pd.to_datetime(df["decision_date"])
    return df


# ---------------------------------------------------------------------------------------------------------------- outcomes
def _label_records(labels: pd.DataFrame) -> list[dict]:
    recs = []
    for rec in labels.to_dict("records"):
        r = {k: _clean(v) for k, v in rec.items()}
        r["decision_date"] = _iso(rec["decision_date"])
        recs.append(r)
    return recs


def append_outcomes(log_dir, labels: pd.DataFrame, asof) -> int:
    """Record the matured outcomes in `labels` (decision_date, symbol, y_net, matured_at, optional horizon and detail columns): matured_at
    <= asof and y_net finite; a row already on file is skipped; a row with no decision on file is an error. decisions.jsonl is not touched.
    Returns the number of rows written."""
    log_dir = Path(log_dir)
    asof = pd.Timestamp(asof)
    f = log_dir / OUTCOMES
    on_file = {_key(r) for r in _read_rows(f)}
    decisions = {_key(r) for r in _read_rows(log_dir / DECISIONS)}
    out = []
    for r in _label_records(labels):
        y, m = r.get("y_net"), r.get("matured_at")
        if m is None or pd.Timestamp(m) > asof:
            continue
        if y is None or not math.isfinite(float(y)):
            continue
        k = _key(r)
        if k not in decisions:
            raise PaperLogError(f"outcome for {k} has no decision on file")
        if k in on_file:
            continue
        on_file.add(k)
        out.append(_line({**r, "recorded_asof": _iso(asof)}))
    if out:
        _append_bytes(f, out)
    return len(out)


def append_unfilled(log_dir, labels: pd.DataFrame, asof) -> int:
    """Record candidates whose trade never happened (fill_status other than FILLED: NO_FILL_*, NO_DATA) once their decision date is <= asof.
    Same rules as append_outcomes; kept apart so outcomes.jsonl holds only real, finite outcomes."""
    log_dir = Path(log_dir)
    asof = pd.Timestamp(asof)
    f = log_dir / UNFILLED
    on_file = {_key(r) for r in _read_rows(f)}
    decisions = {_key(r) for r in _read_rows(log_dir / DECISIONS)}
    out = []
    for r in _label_records(labels):
        if r.get("fill_status") in (None, "FILLED") or pd.Timestamp(r["decision_date"]) > asof:
            continue
        k = _key(r)
        if k not in decisions:
            raise PaperLogError(f"unfilled row {k} has no decision on file")
        if k in on_file:
            continue
        on_file.add(k)
        out.append(_line({**r, "recorded_asof": _iso(asof)}))
    if out:
        _append_bytes(f, out)
    return len(out)


def read_outcomes(log_dir) -> pd.DataFrame:
    df = pd.DataFrame(_read_rows(Path(log_dir) / OUTCOMES))
    if len(df):
        for c in ("decision_date", "matured_at"):
            df[c] = pd.to_datetime(df[c])
    return df


def pending_keys(log_dir) -> pd.DataFrame:
    """Decision rows (decision_date, horizon, symbol, liquidity_rank when logged) that have neither an outcome nor an unfilled record."""
    log_dir = Path(log_dir)
    dec = read_decisions(log_dir)
    if dec.empty:
        return dec
    done = {_key(r) for r in _read_rows(log_dir / OUTCOMES)} | {_key(r) for r in _read_rows(log_dir / UNFILLED)}
    mask = [(_iso(d), (None if pd.isna(h) else int(h)), s) not in done
            for d, h, s in zip(dec["decision_date"], dec.get("horizon", pd.Series([None] * len(dec))), dec["symbol"])]
    return dec[np.array(mask, bool)].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------- calibration
def _matured_outcomes(log_dir, asof, window_weeks: int, horizon=None) -> pd.DataFrame:
    """Logged candidates joined to their outcomes, keeping only outcomes matured by asof and decisions inside the window."""
    asof = pd.Timestamp(asof)
    dec, out = read_decisions(log_dir), read_outcomes(log_dir)
    if dec.empty or out.empty:
        return pd.DataFrame()
    if horizon is not None and "horizon" in dec.columns:
        dec = dec[dec["horizon"] == horizon]
    on = ["decision_date", "symbol"] + (["horizon"] if "horizon" in dec.columns and "horizon" in out.columns else [])
    j = dec.merge(out[on + ["y_net", "matured_at"]], on=on, how="inner")
    j = j[(j["matured_at"] <= asof) & (j["decision_date"] >= asof - pd.Timedelta(weeks=window_weeks)) & j["y_net"].notna()]
    return j.reset_index(drop=True)


def calibration_summary(log_dir, asof, window_weeks: int = WINDOW_WEEKS, min_rows: int = 30) -> dict:
    """Descriptive summary of the matured logged outcomes inside the window as of `asof` (matured_at <= asof only). Fewer than `min_rows` rows
    is INSUFFICIENT with loser_share / mean_y_net NaN (never 0)."""
    j = _matured_outcomes(log_dir, asof, window_weeks)
    n = int(len(j))
    if n < min_rows:
        return dict(status="INSUFFICIENT", n_matured=n, loser_share=float("nan"), mean_y_net=float("nan"))
    y = j["y_net"].to_numpy(float)
    return dict(status="OK", n_matured=n, loser_share=float((y <= 0).mean()), mean_y_net=float(y.mean()))


def calibration_frame(log_dir, horizon: int, asof) -> pd.DataFrame:
    """Calibration rows for the gate in vanta.gate's format: logged candidates of this horizon whose outcome has matured by `asof`.
    score = the logged weekly percentile; regime_posterior and hmm_version are the ones logged AT that decision (never recomputed)."""
    j = _matured_outcomes(log_dir, asof, window_weeks=10_000, horizon=horizon)
    cols = ["decision_date", "symbol", "horizon", "score", "model_version", "fold_id", "is_oos", "regime_posterior", "hmm_version", "y_net",
            "matured_at", "candidate_filter_version"]
    if j.empty:
        return pd.DataFrame(columns=cols)

    def post(x):
        return [float("nan") if v is None else float(v) for v in x] if isinstance(x, list) and len(x) == 3 else [float("nan")] * 3

    return pd.DataFrame({"decision_date": j["decision_date"], "symbol": j["symbol"], "horizon": horizon, "score": j["percentile"].astype(float),
                         "model_version": LIVE_MODEL_KEY, "fold_id": "live", "is_oos": True,
                         "regime_posterior": [post(x) for x in j["regime_posterior"]], "hmm_version": j.get("hmm_version"),
                         "y_net": j["y_net"].astype(float), "matured_at": j["matured_at"], "candidate_filter_version": FILTER_VERSION})


def calibration_status(log_dir, horizon: int, T, window_weeks: int = WINDOW_WEEKS) -> dict:
    """Whether the gate may publish at T for this horizon. The M2/M3 judged-window rule, applied to the live log: scored history must reach back
    `window_weeks` weeks (the first logged decision date of this horizon <= T - window_weeks weeks) AND at least one matured outcome must be
    usable. Otherwise INSUFFICIENT_CALIBRATION: nothing is published and nothing is fabricated."""
    T = pd.Timestamp(T)
    dec = read_decisions(log_dir)
    if not dec.empty and "horizon" in dec.columns:
        dec = dec[dec["horizon"] == horizon]
    first = None if dec.empty else pd.Timestamp(dec["decision_date"].min())
    usable = calibration_frame(log_dir, horizon, T)
    usable = usable[usable["decision_date"] >= T - pd.Timedelta(weeks=window_weeks)] if len(usable) else usable
    weeks = float("nan") if first is None else (T - first).days / 7.0
    ok = first is not None and first <= T - pd.Timedelta(weeks=window_weeks) and len(usable) > 0
    reason = ("" if ok else "no logged history for this horizon" if first is None else
              f"logged history spans {weeks:.1f} weeks, {window_weeks} needed" if first > T - pd.Timedelta(weeks=window_weeks)
              else "no matured outcome inside the window")
    return dict(status="SUFFICIENT" if ok else "INSUFFICIENT_CALIBRATION", first_logged_week=first, weeks_of_history=weeks,
                required_weeks=window_weeks, n_usable_rows=int(len(usable)), reason=reason)


# ---------------------------------------------------------------------------------------------------------------- the gate arms, live
def publications(horizon: int, cands: pd.DataFrame, cal: pd.DataFrame, T, pi_T, status: dict) -> pd.DataFrame:
    """Publication flags of the three M3 arms and the ungated top 10 for one horizon at T. `cands` has `symbol` and `score` (the weekly
    percentile of the live B1 score, NaN when unscored). With INSUFFICIENT_CALIBRATION the arms are NOT run: pub_* and p_value_* are None
    (not False: they were not evaluated), `gate_status` says why; the ungated comparator needs no calibration and is always recorded.
    Returns one row per candidate in input order."""
    base = cands[["symbol", "score"]].reset_index(drop=True)
    out = pd.DataFrame({"symbol": base["symbol"]})
    ung = m2._top_n(base, m2.CAP)
    out["ungated_top10"] = base["symbol"].isin(set(ung["symbol"])).to_numpy()
    out["gate_status"] = status["status"]
    for arm in ARMS:
        if status["status"] != "SUFFICIENT":
            out[f"pub_{arm}"], out[f"p_value_{arm}"], out[f"abstained_{arm}"] = None, None, None
            continue
        r = m2.gate_arm(arm, base, cal, T, pi_T, model_version=LIVE_MODEL_KEY, candidate_filter_version=FILTER_VERSION)
        out[f"pub_{arm}"] = r["published"].astype(bool).to_numpy()
        out[f"p_value_{arm}"] = r["p_value"].to_numpy(float)
        out[f"abstained_{arm}"] = r["abstained"].astype(bool).to_numpy()
        if arm == "a":
            out["n_calibration"] = int(r["n_calibration"].iloc[0]) if len(r) else 0
            out["effective_n"] = float(r["effective_n"].iloc[0]) if len(r) else float("nan")
            out["age_only_fallback"] = bool(r["age_only_fallback"].iloc[0]) if len(r) else False
            out["state_missing"] = bool(r["state_missing"].iloc[0]) if len(r) else True
    return out


# ---------------------------------------------------------------------------------------------------------------- schedule and freshness
def refit_date(T, sessions) -> pd.Timestamp:
    """The fixed refit schedule: the live B1 models are refit once per calendar quarter, at the first trading session on or after
    Jan 1 / Apr 1 / Jul 1 / Oct 1 that is <= T. A model fitted at R scores every decision date of that quarter and is never refit or
    overwritten. Training: decision dates in [R - 4 years, R) whose label ended before the session `embargo` sessions before R (N11)."""
    T = pd.Timestamp(T)
    sess = pd.DatetimeIndex(sessions)
    q_start = pd.Timestamp(year=T.year, month=3 * ((T.month - 1) // 3) + 1, day=1)
    pos = int(sess.searchsorted(q_start, side="left"))
    if pos >= len(sess) or sess[pos] > T:
        raise PaperLogError(f"no trading session between the start of the quarter {q_start.date()} and {T.date()}")
    return sess[pos]


def training_window(R, years: int = TRAIN_YEARS) -> tuple:
    R = pd.Timestamp(R)
    return R - pd.DateOffset(years=years), R


def latest_complete_decision_date(sessions, today) -> pd.Timestamp:
    """The most recent weekly decision date (last session of an ISO week, as everywhere in M1-M3) whose week is over at `today`. A week in
    progress could still gain a later session, so the current ISO week counts only from its Sunday (the job runs on Sunday, or on Monday
    before the open); special Saturday sessions are therefore already in the calendar. NaT when none."""
    sess = pd.DatetimeIndex(sessions)
    iso = sess.isocalendar()
    key = iso["year"].astype(int).to_numpy() * 100 + iso["week"].astype(int).to_numpy()
    last = pd.Series(sess, index=key).groupby(level=0).max()
    t_iso = pd.Timestamp(today).isocalendar()
    this_week = int(t_iso.year) * 100 + int(t_iso.week)
    done = last[last.index <= this_week] if pd.Timestamp(today).weekday() == 6 else last[last.index < this_week]
    return pd.Timestamp(done.iloc[-1]) if len(done) else pd.NaT


def freshness_problems(T, sessions, latest_equity_session, allow_late: bool = False) -> list[str]:
    """Reasons the decision at T cannot be made now (empty list = fine). Never guesses: a missing date is a problem."""
    probs = []
    T = pd.Timestamp(T)
    sess = pd.DatetimeIndex(sessions)
    if latest_equity_session is None or pd.isna(latest_equity_session):
        return ["the latest equity bhavcopy date is unknown"]
    if T not in sess:
        probs.append(f"{T.date()} is not a trading session in the calendar")
    if pd.Timestamp(latest_equity_session) < T:
        probs.append(f"the equity bhavcopy ends {pd.Timestamp(latest_equity_session).date()}, before the decision date {T.date()}")
    elif pd.Timestamp(latest_equity_session) > T and not allow_late:
        probs.append(f"the data already contains sessions after {T.date()} (latest {pd.Timestamp(latest_equity_session).date()}): the entry "
                     "price is public, so this is not a real-time decision (use --allow-late to record it, flagged late)")
    return probs


# ---------------------------------------------------------------------------------------------------------------- provenance
def param_hash(params: dict, seeds=(0, 1, 2, 3, 4)) -> str:
    return hashlib.sha1(json.dumps({"params": params, "seeds": list(seeds)}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def code_version(repo_root) -> dict:
    """Git commit hash and dirty flag of the code that produced a row. dirty = any modified, staged or untracked file under the code paths.
    If git is unavailable the commit is "unknown" and dirty is True (never assumed clean)."""
    root = str(repo_root)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return dict(commit="unknown", dirty=True, dirty_files=["git unavailable"])
    files = [ln[3:] for ln in st.splitlines() if ln.strip()]
    return dict(commit=commit, dirty=bool(files), dirty_files=files)


def code_version_string(cv: dict) -> str:
    return f"{cv['commit'][:12]}{'+dirty' if cv['dirty'] else ''}"
