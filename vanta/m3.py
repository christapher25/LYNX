"""VANTA Milestone M3 (Stage 13M): the same gate as M2, with B1 retrained on and the trade simulated as HOLD-TO-HORIZON net labels.

Authority: docs/vanta/M3_PREREG.md (committed before any M3 training, tuning or run; not editable). Everything the pre-registration
leaves unchanged is imported from vanta.m2 / vanta.baselines / vanta.m1 and is NOT redefined here, so M3 cannot drift from M2:
q, cap, tau, ESS floor, window, arms a/b/c, the percentile score, the ungated top 10, the five GO criteria, the Newey-West lag and the
block-bootstrap block length (ceil(h/5) weeks).

What M3 adds:
  * B1_HOLD_PARAMS_BY_HORIZON: B1 hyperparameters tuned once per horizon (5, 20, 60) on fold 0's inner split ONLY, on the hold-to-horizon
    labels, with the M2 grid (baselines.B1_PARAM_GRID) and the M2 selection rule (baselines.choose_hyperparameters). PLACEHOLDERS until
    the lead commits the tuned values; a full out-of-sample run is refused while a horizon is still a placeholder.
  * which cells are JUDGED: h = 20 and h = 60 on the floor run. h = 5, and every zero-impact run, are reported and NOT JUDGED.
  * a go_table that accepts h = 60 (vanta.m2.go_table refuses it by design), per-year tables and the report header / footer.

q = 0.30 is an engineering target; realised FDR is an empirical number, never a guarantee. Missing is missing: NaN is never zero.
vanta.m1, vanta.labels and vanta.m2 are NOT edited (the first two are hashed into the M1b label-cache keys).
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd

from vanta import baselines as bl
from vanta import m1, m2

# ---------------------------------------------------------------------------------------------------------------- frozen constants
Q, CAP, TAU_WEEKS, ESS_FLOOR, WINDOW_WEEKS = m2.Q, m2.CAP, m2.TAU_WEEKS, m2.ESS_FLOOR, m2.WINDOW_WEEKS
ARMS = m2.ARMS
B1_SEEDS = bl.B1_SEEDS
IMPACTS = m2.IMPACTS
JUDGED_HORIZONS = (20, 60)
REPORTED_HORIZONS = (5,)
ALL_HORIZONS = (5, 20, 60)
MODEL_VERSION = "b1-hold@m3"
LABEL_VARIANT = "hold"

# Same grid as the M2 tuning (M3_PREREG section 4): the object itself, not a copy that could drift.
B1_HOLD_GRID = bl.B1_PARAM_GRID

# B1 parameters per horizon on HOLD-TO-HORIZON net labels (M3_PREREG section 4: tuned once per horizon on fold 0's inner split only;
# h = 60 is tuned on its own fold 0 and does not reuse h = 20). PLACEHOLDERS (= the CatBoost template) until the tuning jobs have run;
# the lead replaces each entry with the output of `python -m scripts.run_m3 --tune --horizon H` (data/processed/m3/b1hold_tuned_h{H}.json)
# and commits it BEFORE the first out-of-sample run. Frozen after that: never changed once an OOS result has been seen.
B1_HOLD_PARAMS_BY_HORIZON: dict = {
    5: dict(bl.B1_PARAMS),
    20: dict(bl.B1_PARAMS),
    60: dict(bl.B1_PARAMS),
}


def hold_params(horizon: int) -> dict:
    """The B1 hyperparameters of a horizon on the hold-to-horizon labels. Unknown horizon -> ValueError (no fallback to another horizon:
    h = 60 never reuses h = 20)."""
    if horizon not in B1_HOLD_PARAMS_BY_HORIZON:
        raise ValueError(f"no M3 B1 parameters for horizon {horizon!r}; horizons are {ALL_HORIZONS}")
    return B1_HOLD_PARAMS_BY_HORIZON[horizon]


def is_placeholder(horizon: int) -> bool:
    """True while the horizon's parameters still equal the untuned template. No grid entry equals the template (learning rate 0.05 is not
    in the grid), so a tuned entry can never be mistaken for a placeholder."""
    return hold_params(horizon) == bl.B1_PARAMS


def require_tuned_hold_params(horizon: int, mechanics: bool = False) -> None:
    """Refuse a full out-of-sample run while the horizon is still a placeholder (M3_PREREG section 4: recorded before the first OOS run).
    A labelled mechanics check is allowed."""
    if not mechanics and is_placeholder(horizon):
        raise ValueError(f"B1_HOLD_PARAMS_BY_HORIZON[{horizon}] is still the placeholder template: run `python -m scripts.run_m3 --tune "
                         f"--horizon {horizon}`, record the result in vanta/m3.py and commit it before any out-of-sample run "
                         "(or use --mechanics-check)")


def is_judged(impact: str, horizon: int) -> bool:
    """M3_PREREG sections 3, 5, 6: only the FLOOR run at h = 20 and h = 60 is judged. h = 5 and every zero run are reported, NOT JUDGED."""
    if impact not in IMPACTS:
        raise ValueError(f"impact must be one of {IMPACTS}, got {impact!r}")
    return impact == "floor" and horizon in JUDGED_HORIZONS


def hold_label_source_fp() -> str:
    """Hash of the code that defines a hold-to-horizon label (m2.hold_to_horizon_label). Part of the label chunk directory name, so an edit
    to that function gives new chunks and a stale label is never reused. vanta.labels / vanta.m1 are already in every chunk key."""
    return hashlib.sha1(inspect.getsource(m2.hold_to_horizon_label).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------------------------------------------- GO table
def go_table(logs: dict, decision_dates: dict, first_judged_week: dict, invariants_passed: bool, n_boot: int = 300,
             seed: int = 11) -> pd.DataFrame:
    """logs = {impact: {arm: {horizon: (gate_log, ungated_log)}}}; decision_dates = {horizon: DatetimeIndex of scored weeks};
    first_judged_week = {horizon: Timestamp}. One row per impact x arm x horizon, criteria from vanta.m2.go_criteria (M2's five criteria,
    unchanged). `judged` / `go_all` are set only for the floor run at h = 20 and 60 (bool); every other row is reported with its own numbers
    (judged False, go_all None). No multiplicity correction across arms and horizons."""
    rows = []
    for impact, per_arm in logs.items():
        if impact not in IMPACTS:
            raise ValueError("impact must be 'floor' or 'zero'")
        for arm, per_h in per_arm.items():
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm!r}")
            for h, (log, ung) in per_h.items():
                if h not in ALL_HORIZONS:
                    raise ValueError(f"unknown horizon {h!r}")
                c = m2.go_criteria(log, ung, decision_dates[h], h, first_judged_week=first_judged_week[h],
                                   invariants_passed=invariants_passed, n_boot=n_boot, seed=seed)
                judged = is_judged(impact, h)
                ci5 = c["c5"]["ci95"]
                nb = c.get("boot_ci95_block", (float("nan"),) * 2)
                rows.append(dict(impact=impact, arm=arm, horizon=h, judged=judged, c1=c["c1"]["passed"], c2=c["c2"]["passed"],
                                 c3=c["c3"]["passed"], c4=c["c4"]["passed"], c5=c["c5"]["passed"], c2_fdr=c["c2"]["fdr"],
                                 c2_threshold=c["c2"]["threshold"], c3_worst_year_fdr=c["c3"]["worst_year_fdr"],
                                 c4_publication_rate=c["c4"]["publication_rate"], c5_diff=c["c5"]["diff"],
                                 c5_ci_lo=ci5[0], c5_ci_hi=ci5[1], mean_y_net=c["mean_y_net"], ungated_mean_y_net=c["ungated_mean_y_net"],
                                 ungated_fdr=c["ungated_fdr"], nw_se=c.get("nw_se", float("nan")), boot_lo=nb[0], boot_hi=nb[1],
                                 nw_lag_weeks=c["nw_lag_weeks"], n_filled=c["c2"]["n_filled"], n_no_fill=c["c2"]["n_no_fill"],
                                 n_no_data=c["c2"]["n_no_data"], n_judged_weeks=c["n_judged_weeks"],
                                 go_all=c["all_pass"] if judged else None))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- reported, NOT judged
def year_table(log: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year of the decision date: filled published trades with a known outcome, losers (y_net <= 0), realised FDR and net
    return per published signal. NO_FILL / NO_DATA / unmatured rows never enter (not as 0); a year with no filled trade is omitted."""
    ok = log[log["published"].astype(bool) & (log["fill_status"] == "FILLED") & log["y_net"].notna()]
    rows = []
    for y, g in ok.groupby(ok["decision_date"].dt.year):
        y_net = g["y_net"].astype(float)
        n_losers = int((y_net <= 0).sum())
        rows.append(dict(year=int(y), n_filled=int(len(g)), n_losers=n_losers, fdr=n_losers / len(g), net_per_signal=float(y_net.mean())))
    return pd.DataFrame(rows, columns=["year", "n_filled", "n_losers", "fdr", "net_per_signal"])


def publication_rate_by_year(log: pd.DataFrame, decision_dates) -> dict:
    """Share of the weeks of each year (denominator = decision_dates, not the log) with at least one published signal."""
    wk = pd.DatetimeIndex(decision_dates)
    pub = pd.DatetimeIndex(log[log["published"].astype(bool)]["decision_date"].unique())
    return {int(y): float(len(pub[pub.year == y])) / max(1, int((wk.year == y).sum())) for y in sorted(set(wk.year))}


# ---------------------------------------------------------------------------------------------------------------- report text
def report_header() -> str:
    """First paragraph of logs/m3_report.md: M3_PREREG section 1."""
    return ("The 2016-onward judged window was seen in M1, M1b and M2, so M3 is a re-test, not a fresh out-of-sample test. The choice of "
            "hold-to-horizon labels was itself made after the M2 diagnostics (A11) on that same window, which adds a selection effect: a pass "
            "here is weaker evidence than a pass on unseen data. The forward paper log (data/processed/paper_log/, started 2026-11-02) is the "
            "only fresh test.")


def report_footer() -> str:
    return ("Realised FDR is an empirical number, not a guarantee, and q = 0.30 is an engineering target. The GO/NO-GO is the owner's call; "
            "this report does not make it.")
