"""
Walk-forward harness (Stage 13): fold schedule, per-fold purging (N11), fold-only fitting, rank IC and a leak detector.

Fold schedule (v6 section 10.2): ROLLING calendar-year windows, 4y train / 1y validate / 1y test, rolled by 1y; `window="expanding"`
exists but is not the default (v6: "the window is rolling, not expanding"). Boundaries are derived from the session calendar, never
hard-coded. The embargo is counted in trading SESSIONS and is at least the longest label horizon (v6: "a configuration parameter,
never a constant"); a shorter one is refused unless the caller opts into `allow_unsafe_embargo` (used only to build a deliberately
leaky configuration).

Per fold, the fit sees only `fold_training_rows`: decisions inside the train window with label_end_date strictly before the session
`embargo` positions before test_start (purge_training_rows, VANTA_SPEC.md line 77). Any HMM fit goes through `fit_regime_for_fold`,
which cuts the features at vanta.regime.fold_train_end(sessions, test_start, embargo) BEFORE calling fit_hmm. Store reads go through
`PitReader`, which has no default for before_nightly_run.

IC and the leak detector: rank IC per decision date (Spearman across symbols); IC above 0.10 is "a bug" (v6 section 11.4, the only
plausibility ceiling the spec states). The detector also reports a label-permutation placebo, a future-shifted-label placebo and a
structural purge audit that is independent of the configured embargo. The gate's q is not involved here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from pitstore.store import read_as_of
from vanta.regime import fit_hmm, fold_train_end

IC_CEILING = 0.10         # v6 section 11.4: "An IC above 0.10 is a bug"
PLACEBO_T_FLAG = 3.0      # NOT in the spec: conventional |t| threshold for the permutation placebo (NEEDS USER)
MIN_NAMES = 10            # NOT in the spec: fewest symbols for a session's rank IC to count (NEEDS USER)

_CONFIG_KEY = {5: "swing", 60: "longterm"}   # config.EMBARGO_DAYS has no key for h=20


class LeakageError(RuntimeError):
    """A fold was about to fit on rows that violate the purge/embargo rule."""


# --------------------------------------------------------------------------- embargo and purge

def purge_training_rows(labels_df: pd.DataFrame, test_start, embargo: int, sessions) -> pd.DataFrame:
    """Return the rows of labels_df with label_end_date strictly before the session `embargo` sessions before test_start.

    labels_df is not mutated; kept rows keep their order and columns. `sessions` must be a sorted calendar containing
    test_start and at least `embargo` earlier sessions, otherwise ValueError (a silent shorter embargo would leak)."""
    if isinstance(embargo, bool) or int(embargo) != embargo or embargo < 0:
        raise ValueError(f"embargo must be a non-negative integer number of sessions, got {embargo!r}")
    sess = pd.DatetimeIndex(sessions)
    if not sess.is_monotonic_increasing:
        raise ValueError("sessions must be sorted ascending")
    ts = pd.Timestamp(test_start)
    if ts not in sess:
        raise ValueError(f"test_start {ts.date()} is not in the session calendar")
    pos = int(sess.get_loc(ts)) - int(embargo)
    if pos < 0:
        raise ValueError(f"calendar has fewer than {embargo} sessions before test_start {ts.date()}")
    cutoff = sess[pos]
    return labels_df[pd.to_datetime(labels_df["label_end_date"]) < cutoff]


def embargo_for_horizon(horizon: int, table: dict | None = None) -> int:
    """Embargo in sessions for a label horizon: max(horizon, the configured value for that horizon).

    config.EMBARGO_DAYS is keyed swing (5) and longterm (60); h=20 has no key, so its embargo is the horizon itself."""
    if table is None:
        import config
        table = config.EMBARGO_DAYS
    key = _CONFIG_KEY.get(int(horizon))
    configured = table.get(key) if key else None
    return max(int(horizon), int(configured or 0))


# --------------------------------------------------------------------------- fold schedule

@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    horizons: tuple
    embargo: int
    required_embargo: int      # max(horizons): what the audit holds the fold to, whatever `embargo` was configured
    include_val_in_fit: bool = False

    @property
    def train_stop(self) -> pd.Timestamp:
        """Exclusive end of the decision dates the fit may draw on, before purging."""
        return self.test_start if self.include_val_in_fit else self.val_start


def _first_on_or_after(sess: pd.DatetimeIndex, d) -> pd.Timestamp | None:
    p = int(sess.searchsorted(pd.Timestamp(d), side="left"))
    return sess[p] if p < len(sess) else None


def make_folds(sessions, horizons, embargo: int | None = None, *, train_years: int = 4, val_years: int = 1,
               test_years: int = 1, step_years: int = 1, first_year: int | None = None, window: str = "rolling",
               include_val_in_fit: bool = False, allow_unsafe_embargo: bool = False) -> list[Fold]:
    """Derive the walk-forward folds from the session calendar. Only complete test windows are kept (the calendar must run
    into a later year)."""
    sess = pd.DatetimeIndex(sessions)
    if not sess.is_monotonic_increasing:
        raise ValueError("sessions must be sorted ascending")
    if window not in ("rolling", "expanding"):
        raise ValueError("window must be 'rolling' or 'expanding'")
    hs = tuple(sorted({int(h) for h in ([horizons] if np.isscalar(horizons) else horizons)}))
    if not hs or hs[0] < 1:
        raise ValueError("horizons must be positive integers")
    need = max(hs)                       # multi-horizon training: the longest horizon governs
    if embargo is None:
        embargo = max(embargo_for_horizon(h) for h in hs)
    if isinstance(embargo, bool) or int(embargo) != embargo or embargo < 0:
        raise ValueError(f"embargo must be a non-negative integer number of sessions, got {embargo!r}")
    if embargo < need and not allow_unsafe_embargo:
        raise ValueError(f"embargo {embargo} < longest label horizon {need}: this leaks (v6 10.2)")
    y0 = sess[0].year if first_year is None else int(first_year)
    last_year = sess[-1].year
    folds: list[Fold] = []
    k = 0
    while True:
        ys = y0 + k * step_years
        v_y, t_y, e_y = ys + train_years, ys + train_years + val_years, ys + train_years + val_years + test_years
        if e_y > last_year:              # test window not complete within the calendar
            break
        starts = [_first_on_or_after(sess, pd.Timestamp(year=y, month=1, day=1)) for y in (ys, v_y, t_y)]
        test_end_c = sess[sess < pd.Timestamp(year=e_y, month=1, day=1)]
        if any(s is None for s in starts) or not len(test_end_c):
            break
        train_start = sess[0] if window == "expanding" else starts[0]
        if starts[0] <= sess[0] and sess[0].year > ys:
            k += 1
            continue                     # the calendar does not cover this train window
        folds.append(Fold(len(folds), train_start, starts[1], starts[2], test_end_c[-1], hs, int(embargo), need,
                          include_val_in_fit))
        k += 1
    return folds


# --------------------------------------------------------------------------- training rows and audit

def fold_training_rows(labels: pd.DataFrame, fold: Fold, sessions) -> pd.DataFrame:
    """The rows a fold may fit on: decisions in [train_start, train_stop), purged (N11) against test_start - embargo."""
    d = pd.to_datetime(labels["decision_date"])
    window = labels[(d >= fold.train_start) & (d < fold.train_stop)]
    return purge_training_rows(window, fold.test_start, fold.embargo, sessions)


def audit_training_rows(rows: pd.DataFrame, fold: Fold, sessions) -> int:
    """Independent N11 audit: number of rows that violate the REQUIRED embargo (the longest horizon), whatever the fold was
    configured with. A row violates if its label_end_date is NaT, >= test_start - required_embargo, or its decision is not
    before test_start."""
    if not len(rows):
        return 0
    sess = pd.DatetimeIndex(sessions)
    pos = int(sess.get_loc(fold.test_start)) - fold.required_embargo
    cutoff = sess[max(pos, 0)]
    end = pd.to_datetime(rows["label_end_date"])
    dec = pd.to_datetime(rows["decision_date"])
    bad = end.isna() | (end >= cutoff) | (dec >= fold.test_start)
    return int(bad.sum())


def fit_regime_for_fold(features: pd.DataFrame, fold: Fold, sessions, **fit_kwargs):
    """Fit the regime HMM for one fold: features are cut to [train_start, min(end of train window, fold_train_end)] (the
    validation year is never fitted on) BEFORE fit_hmm sees them, so even a
    fit that ignored its own train_end could not read a row at or after test_start - embargo."""
    sess = pd.DatetimeIndex(sessions)
    window_end = sess[int(sess.searchsorted(fold.train_stop, side="left")) - 1]   # last session of the train window
    train_end = min(window_end, fold_train_end(sess, fold.test_start, fold.embargo))
    idx = features.index
    cut = features.loc[(idx >= fold.train_start) & (idx <= train_end)]
    return fit_hmm(cut, train_end, **fit_kwargs)


# --------------------------------------------------------------------------- point-in-time reads

class PitReader:
    """Feature reads from the Stage 12 store. `before_nightly_run` has NO default: the caller decides, and every read passes it.

    lead_sessions > 0 reads as of a LATER session than the decision date: a deliberate look-ahead, used only to build the
    leaky demonstration configuration (`leaky` is True then)."""

    def __init__(self, sessions, *, before_nightly_run: bool, lead_sessions: int = 0):
        self.sessions = pd.DatetimeIndex(sessions)
        self.before_nightly_run = bool(before_nightly_run)
        self.lead_sessions = int(lead_sessions)

    @property
    def leaky(self) -> bool:
        return self.lead_sessions > 0

    def as_of(self, decision_date) -> pd.Timestamp:
        p = int(self.sessions.get_loc(pd.Timestamp(decision_date))) + self.lead_sessions
        if p >= len(self.sessions):
            raise ValueError("no session at decision_date + lead_sessions")
        return self.sessions[p]

    def read(self, symbols: Sequence[str], decision_date, blocks: Sequence[str], *, lookback_sessions: int = 1, **kw) -> pd.DataFrame:
        return read_as_of(symbols, self.as_of(decision_date), blocks, self.sessions, before_nightly_run=self.before_nightly_run,
                          lookback_sessions=lookback_sessions, **kw)


def drop_absent(frame: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    """Rows where any requested block has no knowable row are UNAVAILABLE, not a value: removed, never filled."""
    mask = pd.Series(False, index=frame.index)
    for b in blocks:
        col = f"{b}_row_absent"
        if col in frame.columns:
            mask |= frame[col].fillna(True).astype(bool)
    return frame[~mask]


# --------------------------------------------------------------------------- IC

def rank_ic_by_date(scored: pd.DataFrame, min_names: int = MIN_NAMES, label: str = "y_net") -> pd.Series:
    """Spearman correlation of score and label across symbols, per decision date. NaN pairs are dropped; a date with fewer than
    `min_names` complete pairs, or a constant side, has no IC (NaN, then excluded from summaries)."""
    out = {}
    for d, g in scored.groupby("decision_date", sort=True):
        g = g[["score", label]].dropna()
        if len(g) < min_names:
            continue
        r = g["score"].rank().corr(g[label].rank())
        if not np.isnan(r):
            out[d] = r
    return pd.Series(out, dtype=float, name="rank_ic")


def ic_summary(ic: pd.Series, horizon: int, step: int = 5) -> dict:
    """Mean, std, IC-IR and a t-stat. Decisions are every `step` sessions but labels span `horizon`, so neighbouring ICs overlap:
    the effective sample is n / max(1, horizon/step) (conservative; no Newey-West)."""
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return dict(n=n, mean=float("nan"), std=float("nan"), ic_ir=float("nan"), t=float("nan"))
    sd = float(ic.std(ddof=1))
    n_eff = n / max(1.0, horizon / step)
    t = float(ic.mean() / (sd / np.sqrt(n_eff))) if sd > 0 else float("nan")
    return dict(n=n, mean=float(ic.mean()), std=sd, ic_ir=float(ic.mean() / sd) if sd > 0 else float("nan"), t=t)


# --------------------------------------------------------------------------- fold runner

FitFn = Callable[[pd.DataFrame, Fold], object]
ScoreFn = Callable[[object, pd.DataFrame, Fold, "PitReader | None"], Sequence[float]]


@dataclass
class FoldResult:
    fold: Fold
    n_train: int
    audit_violations: int
    scored: pd.DataFrame
    fit_rows_policy: str
    regime: object = None


@dataclass
class WalkForwardResult:
    folds: list = field(default_factory=list)

    @property
    def scored(self) -> pd.DataFrame:
        parts = [f.scored.assign(fold_id=f.fold.fold_id) for f in self.folds if len(f.scored)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["decision_date", "symbol", "score", "y_net"])

    @property
    def audit_violations(self) -> int:
        return sum(f.audit_violations for f in self.folds)


def run_fold(fold: Fold, labels: pd.DataFrame, sessions, fit_fn: FitFn, score_fn: ScoreFn, *, reader: PitReader | None = None,
             regime_features: pd.DataFrame | None = None, fit_rows_policy: str = "purged") -> FoldResult:
    """Fit on the fold's purged training rows only, score its test window.

    fit_rows_policy "purged" is the only safe value. "unpurged" (no embargo purge) and "include_test" (all rows through
    test_end) exist to build the deliberately leaky configurations; the audit runs on whatever rows the fit actually gets, so
    they are reported, not hidden. With "purged" a violating row raises LeakageError before the fit is called."""
    sess = pd.DatetimeIndex(sessions)
    d = pd.to_datetime(labels["decision_date"])
    if fit_rows_policy == "purged":
        train = fold_training_rows(labels, fold, sess)
    elif fit_rows_policy == "unpurged":
        train = labels[(d >= fold.train_start) & (d < fold.train_stop)]
    elif fit_rows_policy == "include_test":
        train = labels[(d >= fold.train_start) & (d <= fold.test_end)]
    else:
        raise ValueError(f"unknown fit_rows_policy {fit_rows_policy!r}")
    violations = audit_training_rows(train, fold, sess)
    if fit_rows_policy == "purged" and violations:
        raise LeakageError(f"fold {fold.fold_id}: {violations} training rows violate the required embargo {fold.required_embargo} "
                           f"(configured {fold.embargo}); refusing to fit")
    model = fit_fn(train.copy(), fold)
    regime = fit_regime_for_fold(regime_features, fold, sess) if regime_features is not None else None
    test = labels[(d >= fold.test_start) & (d <= fold.test_end)][["decision_date", "symbol", "y_net"]].reset_index(drop=True)
    scores = np.asarray(score_fn(model, test[["decision_date", "symbol"]].copy(), fold, reader), dtype=float) if len(test) else np.array([])
    if len(scores) != len(test):
        raise ValueError("score_fn must return one score per test row")
    return FoldResult(fold, len(train), violations, test.assign(score=scores), fit_rows_policy, regime)


def run_walkforward(folds: Sequence[Fold], labels: pd.DataFrame, sessions, fit_fn: FitFn, score_fn: ScoreFn, **kw) -> WalkForwardResult:
    return WalkForwardResult([run_fold(f, labels, sessions, fit_fn, score_fn, **kw) for f in folds])


# --------------------------------------------------------------------------- leak detector

@dataclass
class LeakReport:
    flagged: bool
    reasons: list
    mean_ic: float
    t_stat: float
    perm_mean_ic: float
    perm_t: float
    shifted_mean_ic: float
    audit_violations: int
    config_embargo_short: bool
    lead_sessions: int


def shifted_labels(labels_all: pd.DataFrame, n_dates: int) -> pd.DataFrame:
    """For each (symbol, decision_date), the y_net of the same symbol `n_dates` decision dates later: a label entirely beyond
    the real one's window. Returned as columns symbol, decision_date, y_shift.

    The shift is by ROW count per symbol, not by sessions. With FILLED-only labels a symbol's gaps (unfilled decisions) make the
    shift longer than horizon + embargo: conservative (the label is further away), not exact."""
    lab = labels_all[["symbol", "decision_date", "y_net"]].sort_values(["symbol", "decision_date"])
    lab = lab.assign(y_shift=lab.groupby("symbol")["y_net"].shift(-n_dates))
    return lab[["symbol", "decision_date", "y_shift"]]


def detect_leak(result: WalkForwardResult, labels_all: pd.DataFrame, *, horizon: int, step: int = 5, seed: int = 0,
                ic_ceiling: float = IC_CEILING, placebo_t: float = PLACEBO_T_FLAG, reader: PitReader | None = None) -> LeakReport:
    """Flag a run whose IC is implausibly high, whose permutation placebo is non-null, whose future-shifted placebo is
    implausibly high, or whose fits saw rows that violate the required embargo.

    Flags: mean rank IC > ic_ceiling (v6 11.4); |permutation t| > placebo_t; |shifted-label IC| > ic_ceiling; any audited
    training row inside the required embargo (or any fold configured below the longest horizon); a PitReader with lead > 0."""
    scored = result.scored
    reasons: list = []
    ic = rank_ic_by_date(scored)
    s = ic_summary(ic, horizon, step)

    rng = np.random.default_rng(seed)
    perm = scored.copy()
    perm["y_perm"] = perm.groupby("decision_date")["y_net"].transform(lambda v: rng.permutation(v.to_numpy()))
    ps = ic_summary(rank_ic_by_date(perm, label="y_perm"), horizon, step)

    n_dates = int(np.ceil((horizon + max(f.fold.embargo for f in result.folds)) / step)) if result.folds else 1
    sh = scored.merge(shifted_labels(labels_all, n_dates), on=["symbol", "decision_date"], how="left")
    shifted = ic_summary(rank_ic_by_date(sh, label="y_shift"), horizon, step)

    short = any(f.fold.embargo < f.fold.required_embargo for f in result.folds)
    lead = reader.lead_sessions if reader is not None else 0
    if not np.isnan(s["mean"]) and s["mean"] > ic_ceiling:
        reasons.append(f"mean rank IC {s['mean']:.3f} > ceiling {ic_ceiling} (v6 11.4)")
    if not np.isnan(ps["t"]) and abs(ps["t"]) > placebo_t:
        reasons.append(f"permutation placebo non-null (t={ps['t']:.2f})")
    if not np.isnan(shifted["mean"]) and abs(shifted["mean"]) > ic_ceiling:
        reasons.append(f"future-shifted-label placebo IC {shifted['mean']:.3f} beyond {ic_ceiling}")
    if result.audit_violations:
        reasons.append(f"{result.audit_violations} training rows inside the required embargo / not before test_start (N11)")
    if short:
        reasons.append("a fold's embargo is below the longest label horizon")
    if lead > 0:
        reasons.append(f"features read {lead} sessions after the decision date")
    return LeakReport(bool(reasons), reasons, s["mean"], s["t"], ps["mean"], ps["t"], shifted["mean"], result.audit_violations,
                      short, lead)
