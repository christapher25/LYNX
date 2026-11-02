"""
Reference ("toy") implementation of the vanta.m3 / scripts.run_m3 contract with switchable deliberate bugs. NOT a VANTA module, no tests.

tests/test_m3_invariants.py runs every check against the real vanta.m3 and scripts.run_m3 (skipped until they exist), against `toy_api()`
(must pass: the check is not over-strict) and against `toy_api(<bug>)` (must fail: the check can fail). The toy mirrors the real driver's
shape (module-level functions that look their collaborators up by name, so a check can monkeypatch them on either object) and is built on
the REAL vanta.m2 / vanta.baselines / vanta.walkforward / scripts.run_m1b primitives.
"""

import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
from vanta import baselines as bl
from vanta import m1, m2
from vanta import walkforward as wf

ARMS = ("a", "b", "c")
ALL_H = (5, 20, 60)


def toy_api(*bugs):
    bugs = frozenset(bugs)
    m3, run = SimpleNamespace(), SimpleNamespace()

    # ================================================================== vanta.m3 (pure)
    m3.Q = 0.30 if "q25" not in bugs else 0.25
    m3.CAP = 10 if "cap5" not in bugs else 5
    m3.TAU_WEEKS = 26 if "tau13" not in bugs else 13
    m3.ESS_FLOOR = 200 if "ess100" not in bugs else 100
    m3.WINDOW_WEEKS = 52 if "win26" not in bugs else 26
    m3.ARMS = ARMS if "four_arms" not in bugs else ARMS + ("d",)
    m3.B1_SEEDS = (0, 1, 2, 3, 4) if "seeds3" not in bugs else (0, 1, 2)
    m3.JUDGED_HORIZONS = (20, 60) if "judge_5" not in bugs else (5, 20, 60)
    m3.REPORTED_HORIZONS = (5,)
    m3.ALL_HORIZONS = ALL_H
    m3.LABEL_VARIANT = "hold" if "label_atr" not in bugs else "atr"
    m3.B1_HOLD_GRID = bl.B1_PARAM_GRID
    m3.B1_HOLD_PARAMS_BY_HORIZON = {5: dict(bl.B1_PARAMS), 20: dict(bl.B1_PARAMS), 60: dict(bl.B1_PARAMS)}

    def hold_params(horizon):
        if horizon not in m3.B1_HOLD_PARAMS_BY_HORIZON:
            if "fallback_20" in bugs:
                return m3.B1_HOLD_PARAMS_BY_HORIZON[20]
            raise ValueError(f"no M3 B1 parameters for horizon {horizon!r}")
        return m3.B1_HOLD_PARAMS_BY_HORIZON[horizon]

    def is_placeholder(horizon):
        return hold_params(horizon) == bl.B1_PARAMS

    def require_tuned_hold_params(horizon, mechanics=False):
        if "no_placeholder_guard" in bugs:
            return
        if not mechanics and is_placeholder(horizon):
            raise ValueError(f"horizon {horizon} is still the placeholder")

    def is_judged(impact, horizon):
        if impact not in ("floor", "zero"):
            raise ValueError(impact)
        if "judge_zero" in bugs:
            return horizon in m3.JUDGED_HORIZONS
        return impact == "floor" and horizon in m3.JUDGED_HORIZONS

    m3.hold_params, m3.is_placeholder, m3.require_tuned_hold_params, m3.is_judged = hold_params, is_placeholder, require_tuned_hold_params, is_judged

    def go_table(logs, decision_dates, first_judged_week, invariants_passed, n_boot=300, seed=11):
        rows = []
        for impact, per_arm in logs.items():
            if impact not in ("floor", "zero"):
                raise ValueError("impact must be 'floor' or 'zero'")
            for arm, per_h in per_arm.items():
                if arm not in ARMS and "accept_any_arm" not in bugs:
                    raise ValueError(f"unknown arm {arm!r}")
                for h, (log, ung) in per_h.items():
                    if h not in ALL_H:
                        raise ValueError(f"unknown horizon {h!r}")
                    hh = 20 if "lag4" in bugs else h              # the lag / block taken from the wrong horizon
                    c = m2.go_criteria(log, ung, decision_dates[h], hh, first_judged_week=first_judged_week[h],
                                       invariants_passed=invariants_passed, n_boot=n_boot, seed=seed)
                    judged = is_judged(impact, h)
                    nb = c.get("boot_ci95_block", (float("nan"),) * 2)
                    rows.append(dict(impact=impact, arm=arm, horizon=h, judged=judged, c1=c["c1"]["passed"], c2=c["c2"]["passed"],
                                     c3=c["c3"]["passed"], c4=c["c4"]["passed"], c5=c["c5"]["passed"], c2_fdr=c["c2"]["fdr"],
                                     c2_threshold=c["c2"]["threshold"], nw_se=c.get("nw_se", float("nan")), boot_lo=nb[0],
                                     boot_hi=nb[1], nw_lag_weeks=c["nw_lag_weeks"], go_all=c["all_pass"] if judged else None))
        return pd.DataFrame(rows)

    m3.go_table = go_table

    # ================================================================== scripts.run_m3 (driver)
    run.m3 = m3
    run.M3_DIR_NAME = "m2" if "writes_m2_dir" in bugs else "m3"

    def out_dir_for(tag):
        return config.PROCESSED_DATA_DIR / (run.M3_DIR_NAME + (f"_{tag}" if tag else ""))

    def hold_label_dir(h):
        base = config.PROCESSED_DATA_DIR / ("m2" if "labels_in_m2" in bugs else "m3") / "labels_hold"
        return base / f"h{h}"

    run.out_dir_for, run.hold_label_dir = out_dir_for, hold_label_dir
    run.REPORT_PATH = config.LOG_DIR / ("m2_report.md" if "report_is_m2" in bugs else "m3_report.md")
    run.NO_LEVEL = math.inf

    # collaborators the checks monkeypatch (names identical to scripts.run_m3's)
    def _unpatched(*a, **k):
        raise RuntimeError("toy collaborator not patched")

    for name in ("trading_sessions", "load_candidates", "feature_panel", "_apply_impact", "load_terminal_values", "_cand_cache_name",
                 "build_atr_labels"):
        setattr(run, name, _unpatched)

    def build_hold_labels(cands, h, sessions, terminal):
        from scripts import run_m1b, run_m2 as rm2                # the real chunked label builder; the check patches build_labels
        if "atr_labels" in bugs:
            labels = run_m1b.build_labels(cands, h, 1.5, 1.5, hold_label_dir(h), sessions, terminal)
        elif "target_only" in bugs:
            labels = run_m1b.build_labels(cands, h, math.inf, 1.5, hold_label_dir(h), sessions, terminal)
        elif "stop_only" in bugs:
            labels = run_m1b.build_labels(cands, h, 1.5, math.inf, hold_label_dir(h), sessions, terminal)
        else:
            with rm2._hold_labels():
                labels = run_m1b.build_labels(cands, h, run.NO_LEVEL, run.NO_LEVEL, hold_label_dir(h), sessions, terminal)
        df = cands.merge(labels, on=["symbol", "decision_date"], how="left", validate="one_to_one").assign(horizon=h)
        df["fill_status"] = df["fill_status"].fillna("NO_DATA")
        return df

    run.build_hold_labels = build_hold_labels

    def _make_folds(sessions, h, df, a):
        first_year = a.first_year or int(df["decision_date"].min().year)
        folds = wf.make_folds(sessions, h, train_years=a.train_years, val_years=a.val_years, first_year=first_year, include_val_in_fit=False)
        if a.max_folds:
            folds = folds[:a.max_folds]
        return folds

    run._make_folds = _make_folds

    def run_tune(h, a):
        out_dir = out_dir_for(a.tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        tuned_f = out_dir / ("b1hold_tuned.json" if "shared_file" in bugs else f"b1hold_tuned_h{h}.json")
        if tuned_f.exists() and not a.force:
            return
        sessions = pd.DatetimeIndex(pd.to_datetime(run.trading_sessions()))
        cands = run.load_candidates(sessions)
        folds = run._make_folds(sessions, h, cands, a)
        f0 = folds[1] if "wrong_fold" in bugs else folds[0]
        d0 = cands if "reads_test_year" in bugs else cands[cands["decision_date"] < f0.test_start].reset_index(drop=True)
        builder = run.build_atr_labels if "atr_labels_tune" in bugs else run.build_hold_labels
        lab = run._apply_impact(builder(d0, h, sessions, run.load_terminal_values()), "floor")
        panel, _ = run.feature_panel(cands if "features_all" in bugs else d0, sessions, run._cand_cache_name(sessions), out_dir, a.force)
        grid = [{"depth": 1}] if "own_grid" in bugs else m3.B1_HOLD_GRID
        best = bl.choose_hyperparameters(grid, lambda p, seed: None, panel, lab, f0, sessions, seed=0)
        tuned_f.write_text(__import__("json").dumps({**bl.B1_PARAMS, **best}, indent=1))

    run.run_tune = run_tune

    def run_score(h, a):
        mech = a.mechanics_check
        try:
            m3.require_tuned_hold_params(h, mech)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if "no_fold_guard" not in bugs and not mech and (a.train_years != 4 or a.val_years != 1 or a.max_folds or a.first_year or a.end):
            raise SystemExit("the pre-registered fold schedule (4 y train / 1 y validation / 1 y test, full window) cannot be overridden "
                             "without --mechanics-check")
        out_dir = out_dir_for(a.tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        run.trading_sessions()                                  # the check's stand-in raises: reaching here means no refusal happened

    run.run_score = run_score
    return SimpleNamespace(m3=m3, run=run, bugs=bugs)


# ====================================================================================================================== report text toy
def toy_report(*bugs):
    """A minimal markdown report with the structure of scripts.run_m3.report (same headings, same table columns), with switchable defects."""
    bugs = frozenset(bugs)

    def table(rows, status):
        head = "| impact | arm | h | c1 inv | FDR | " + ("GO |" if not status else "status |")
        lines = [head, "|---|---|---|---|---|---|"]
        for i, a, h in rows:
            tail = "NOT JUDGED" if status and "unmarked" not in bugs else ("yes" if not status else "n/a")
            lines.append(f"| {i} | {a} | {h} | yes | 0.300 | {tail} |")
        return lines

    def comp(rows):
        lines = ["| impact | arm | h | net/signal | NW SE | block-bootstrap 95% CI |", "|---|---|---|---|---|---|"]
        for i, a, h in rows:
            lag = 4 if "lag4_report" in bugs else {5: 1, 20: 4, 60: 12}[h]
            lines.append(f"| {i} | {a} | {h} | 0.0100 | 0.0050 (lag {lag}) | (0.001, 0.02) |")
        return lines

    go = [("floor", a, h) for a in "abc" for h in (20, 60)]
    if "drop_h60_go" in bugs:
        go = [r for r in go if r[2] != 60]
    if "h5_in_go" in bugs:
        go += [("floor", "a", 5)]
    if "zero_in_go" in bugs:
        go += [("zero", "a", 20)]
    nj = [("zero", a, h) for a in "abc" for h in (20, 60)] + [(i, a, 5) for i in ("floor", "zero") for a in "abc"]
    if "nj_drops_h5" in bugs:
        nj = [r for r in nj if r[2] != 5]

    header = ("The 2016-onward judged window was seen in M1, M1b and M2, so M3 is a re-test, not a fresh out-of-sample test. The choice of "
              "hold-to-horizon labels was itself made after the M2 diagnostics (A11) on that same window, which adds a selection effect: a pass "
              "here is weaker evidence than a pass on unseen data. The forward paper log is the only fresh test.")
    if "header_silent" in bugs:
        header = "Results of the M3 run follow."
    if "header_no_selection" in bugs:
        header = ("The 2016-onward judged window was seen in M1, M1b and M2, so M3 is a re-test, not a fresh test. A pass is weaker "
                  "evidence than a pass on unseen data. The forward paper log is the only fresh test.")
    footer = ("Realised FDR is an empirical number, not a guarantee, and q = 0.30 is an engineering target. The GO/NO-GO is the owner's call; "
              "this report does not make it.")
    if "footer_guarantee" in bugs:
        footer = "Realised FDR is a guarantee. The verdict is GO."

    L = ["# M3 report", ""]
    if "header_buried" not in bugs:
        L += [header, ""]
    L += ["Pre-registration: docs/vanta/M3_PREREG.md. Newey-West lag and bootstrap block = ceil(h/5) weeks.", ""]
    L += ["## GO table: JUDGED cells (floor run; h = 20 and h = 60)", "", *table(go, False), "", "### Net return per published signal (judged cells)", "",
          *comp(go), ""]
    L += ["## NOT JUDGED: h = 5 and the zero-impact runs", "", *table(nj, True), "", "### Net return per published signal (NOT JUDGED cells)", "",
          *comp(nj), ""]
    if "header_buried" in bugs:
        L += [header, ""]
    if "footer_missing" not in bugs:
        L += [footer]
    else:
        L += ["End."]
    return "\n".join(L) + "\n"
