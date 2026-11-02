"""
M3 (Stage 13M, replaces M2 as the gate test) invariants, from docs/vanta/M3_PREREG.md (binding) and N11.
Modules under test: vanta.m3 and scripts.run_m3 (quant-m3's names; this file adapts to them, nothing here is invented except the check names).

  M3-L1   training labels    build_hold_labels yields HOLD-to-horizon labels: no stop, no target, exit = horizon-end close (an ATR k = 1.5
                             label, a stop-only or a target-only label is caught) (PREREG sections 2, 3)
  M3-T1   tuning data        --tune uses fold 0's inner split only: fold 0 of THAT horizon, decision dates before fold 0's test year,
                             hold labels, the M2 grid, outputs under the M3 directory (section 4)
  M3-T2   per-horizon        h = 60 is tuned on its own fold 0 and writes its own file; it never reuses h = 20's result (section 4)
  M3-T3   placeholder        a placeholder parameter set is refused before any output is written (--score); a missing horizon never falls back to
                             another horizon's parameters; a mechanics check is the only bypass (section 4)
  M3-T4   fold schedule      the pre-registered 4 y / 1 y / 1 y schedule over the whole window cannot be overridden on a judged --score run
  M3-O1   outputs            M3 writes under its own directories and report; M2's are never the target (A: nothing overwrites M2)
  M3-R1   GO table (pure)    judged = floor run at h = 20 and 60 only; h = 5 and every zero run are reported (judged False, go_all None)
  M3-R2   report tables      in the rendered report the GO table holds exactly the floor h20/h60 rows, the NOT JUDGED table holds h5 and the
                             zero rows, every one marked NOT JUDGED
  M3-R3   report text        first paragraph states PREREG section 1 (re-test, selection effect, only the paper log is fresh); the report ENDS
                             with the empirical-number footer (no verdict)
  M3-R4   lag and blocks     Newey-West lag and bootstrap block = ceil(h/5) weekly rows (1, 4, 12 at h = 5, 20, 60), in the GO table and the report
  M3-P1   frozen parameters  q 0.30, cap 10, tau 26, ESS 200, 52-week window, arms a/b/c, seeds 0-4 equal M2; the M3 sources override none of them
  M3-P2   CLI                no option changes the frozen parameters; no ATR variant exists; M3 sources never reference K_S / K_T

Each check takes an `api` (the real modules or the toy in tests/test_m3_toy.py). Real ones skip until the modules exist; the toy passes and
every deliberately broken toy fails.
"""

import inspect
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_m2_invariants import A11_BARS, FJW, NO_HITS, _bars, _gate_log, _ungated, _weeks
from test_m3_toy import toy_api, toy_report
from vanta import baselines as bl
from vanta import m1
from vanta import walkforward as wf

TS = pd.Timestamp
COST = lambda *a: (0.001, 0.002, 0.0005)                                                  # noqa: E731


def _real():
    m3 = pytest.importorskip("vanta.m3")
    run = pytest.importorskip("scripts.run_m3")
    return SimpleNamespace(m3=m3, run=run)


def _no_attr_error(ei):
    assert not isinstance(ei.value, (ImportError, NameError, TypeError, AttributeError)), repr(ei.value)


# =============================================================================================================== M3-L1 labels
CAP_BARS = [(99, 101, 98, 100), (100, 101, 99.5, 100),
            (100, 104, 99, 102),                  # high 104 >= target 103 on day 2: an ATR run takes profit at 103
            (102, 103, 98.5, 99), (99, 100, 98, 98.5),
            (98.5, 99, 97.5, 98)]                 # horizon-5 close 98 (the low never reaches the 97 stop)


def check_hold_labels_built_as_hold(api, mp):
    """M3-L1: build_hold_labels drives the shared chunked label builder with hold-to-horizon simulation. Two hand-built symbols through
    the real simulator: A11_BARS (the dip to 90 does not stop it out; exit = horizon-session CLOSE 104: TIME, gross +0.04, y_net 0.04 -
    0.0035, where an ATR k = 1.5 label is STOPPED at 97) and CAP_BARS (the target touch at high 104 does not take profit; exit = close 98:
    gross -0.02, y_net -0.0235, where an ATR label would exit at the target for +0.03). The builder is also given no stop and no target
    distance (infinite k)."""
    from scripts import run_m1b
    seen = []

    def spy(cands, horizon, k_s, k_t, cache_dir, sessions, terminal):
        seen.append((k_s, k_t))
        rows = []
        for sym, ohlc in (("A", A11_BARS), ("B", CAP_BARS)):
            b = _bars(ohlc)
            rows.append(run_m1b.simulate_trade(sym, b, b.index[0], horizon, k_s, k_t, 2.0, *NO_HITS, cost_fn=COST))
        return pd.DataFrame(rows)

    mp.setattr(run_m1b, "build_labels", spy)
    d0 = _bars(A11_BARS).index[0]
    cands = pd.DataFrame({"symbol": ["A", "B"], "decision_date": [d0, d0]})
    df = api.run.build_hold_labels(cands, 5, None, {}).set_index("symbol")
    a, b = df.loc["A"], df.loc["B"]
    assert a["fill_status"] == "FILLED" and a["exit_reason"] == "TIME" and a["exit_price"] == 104
    assert a["gross_return"] == pytest.approx(0.04) and a["y_net"] == pytest.approx(0.04 - 0.0035)
    assert b["exit_reason"] == "TIME" and b["exit_price"] == 98 and b["gross_return"] == pytest.approx(-0.02)
    assert b["y_net"] == pytest.approx(-0.02 - 0.0035)
    assert seen and all(np.isinf(k_s) and np.isinf(k_t) for k_s, k_t in seen), seen
    assert api.m3.LABEL_VARIANT == "hold"


# =============================================================================================================== M3-T tuning
SESS = pd.bdate_range("2008-01-01", "2020-12-31")


def _tune_args(**kw):
    base = dict(tag="t", force=False, first_year=None, train_years=4, val_years=1, max_folds=None, mechanics_check=False, end=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _candidates():
    wk = SESS[SESS.weekday == 4]
    return pd.DataFrame([(d, s) for d in wk for s in ("S0", "S1")], columns=["decision_date", "symbol"])


def _patch_tune_world(api, mp, tmp_path):
    import config
    import vanta.baselines as blmod
    mp.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed", raising=True)
    state = SimpleNamespace(built=[], panels=[], calls=[])
    cands = _candidates()

    def lab(kind):
        def build(c, h, sessions, terminal):
            state.built.append((kind, c.copy()))
            return c.assign(y_net=0.01, label_end_date=c["decision_date"], label_kind=kind, horizon=h)
        return build

    def panel(d, sessions, cand_name, out_dir, force=False):
        state.panels.append(d.copy())
        return d[["decision_date", "symbol"]].assign(f0=1.0), ["f0"]

    def choose(grid, model_for, features, labels, fold, sessions, seed=0, min_names=10):
        state.calls.append(SimpleNamespace(grid=grid, fold=fold, features=features, labels=labels))
        return {"depth": 4 + len(state.calls)}

    run = api.run
    mp.setattr(run, "trading_sessions", lambda: SESS)
    mp.setattr(run, "load_candidates", lambda s: cands)
    mp.setattr(run, "build_hold_labels", lab("hold"))
    if "build_atr_labels" in vars(run) or not hasattr(run, "run_tune") or True:
        mp.setattr(run, "build_atr_labels", lab("atr"), raising=False)
    mp.setattr(run, "_apply_impact", lambda df, impact: df)
    mp.setattr(run, "load_terminal_values", lambda: {})
    mp.setattr(run, "feature_panel", panel)
    mp.setattr(run, "_cand_cache_name", lambda s: "cands.parquet")
    mp.setattr(blmod, "choose_hyperparameters", choose)
    return state


def check_tuning_fold0_inner_split_hold_labels_per_horizon(api, mp, tmp_path):
    """M3-T1/T2 (PREREG section 4): --tune for h = 20 then h = 60 on a synthetic weekly candidate set 2008-2020. Each call hands
    choose_hyperparameters (a) the FOLD 0 of THAT horizon (4 y train / 1 y validation, embargo at least h), (b) the M2 grid, (c) hold labels
    (never the ATR kind), and every label, feature and candidate it built is dated before fold 0's test year (the test window and everything
    later are never read). h = 60 is a separate call writing a separate file with its own result (a h = 20 result is never reused); all
    files land under the M3 output directory, none elsewhere."""
    state = _patch_tune_world(api, mp, tmp_path)
    a = _tune_args()
    for h in (20, 60):
        api.run.run_tune(h, a)
    assert len(state.calls) == 2, "h = 60 must be tuned by its own call (not skipped, not merged with h = 20)"
    for h, call in zip((20, 60), state.calls):
        f0 = wf.make_folds(SESS, h, train_years=4, val_years=1, first_year=2008, include_val_in_fit=False)[0]
        assert call.fold == f0 and call.fold.fold_id == 0 and call.fold.embargo >= h
        assert list(call.grid) == list(bl.B1_PARAM_GRID)
        assert set(call.labels["label_kind"]) == {"hold"}
        for frame in (call.labels, call.features):
            assert frame["decision_date"].max() < f0.test_start and len(frame) > 0
    for kind, c in state.built:
        assert kind == "hold" and c["decision_date"].max() < wf.make_folds(SESS, 20, train_years=4, val_years=1, first_year=2008)[0].test_start
    for p in state.panels:
        assert p["decision_date"].max() < TS("2013-01-01")
    from scripts import run_m2
    out = api.run.out_dir_for("t")
    assert out.resolve() not in [d.resolve() for d in run_m2._dirs("t")] and out.name.startswith("m3")
    files = {p.name: json.loads(p.read_text()) for p in out.glob("b1hold_tuned_h*.json")}
    assert set(files) == {"b1hold_tuned_h20.json", "b1hold_tuned_h60.json"}
    assert files["b1hold_tuned_h20.json"]["depth"] == 5 and files["b1hold_tuned_h60.json"]["depth"] == 6
    made = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert made and all(out in p.parents for p in made), [str(p) for p in made if out not in p.parents]


def _tuned_like():
    return {**bl.B1_PARAMS, "depth": 4, "learning_rate": 0.08}


def check_placeholder_refused_before_any_output(api, mp, tmp_path):
    """M3-T3: hold_params returns each horizon's OWN dict; is_placeholder is true exactly for the untuned template; a placeholder horizon is
    refused (ValueError) by require_tuned_hold_params unless it is a labelled mechanics check; a horizon with no entry raises (never
    another horizon's parameters). run_score for a placeholder horizon exits (SystemExit) BEFORE any directory or file is created."""
    import config
    m3 = api.m3
    mp.setattr(m3, "B1_HOLD_PARAMS_BY_HORIZON", {5: _tuned_like(), 20: {**_tuned_like(), "depth": 6}, 60: dict(bl.B1_PARAMS)}, raising=True)
    assert m3.hold_params(20)["depth"] == 6 and m3.hold_params(5)["depth"] == 4
    assert not m3.is_placeholder(20) and m3.is_placeholder(60)
    m3.require_tuned_hold_params(20)
    m3.require_tuned_hold_params(60, True)                                  # the labelled mechanics check is the only bypass
    with pytest.raises(ValueError):
        m3.require_tuned_hold_params(60)
    mp.setattr(m3, "B1_HOLD_PARAMS_BY_HORIZON", {5: _tuned_like(), 20: _tuned_like()}, raising=True)
    with pytest.raises(ValueError):
        m3.hold_params(60)                                                  # no fallback to horizon 20's parameters
    mp.setattr(m3, "B1_HOLD_PARAMS_BY_HORIZON", {5: _tuned_like(), 20: _tuned_like(), 60: dict(bl.B1_PARAMS)}, raising=True)
    mp.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed", raising=True)
    mp.setattr(api.run, "trading_sessions", lambda: (_ for _ in ()).throw(RuntimeError("reached the data: no refusal happened")))
    with pytest.raises(SystemExit):
        api.run.run_score(60, _tune_args())
    assert not list(tmp_path.rglob("*")), list(tmp_path.rglob("*"))


def check_judged_score_refuses_fold_overrides(api, mp, tmp_path):
    """M3-T4 (PREREG section 4: folds 4 y train / 1 y validation / 1 y test, whole window): with tuned parameters, a non-mechanics --score
    run that overrides the train years, validation years, first year, fold count or end date is refused (SystemExit) before it reads any
    data. Only a labelled mechanics check may use a different schedule."""
    import config
    m3 = api.m3
    mp.setattr(m3, "B1_HOLD_PARAMS_BY_HORIZON", {5: _tuned_like(), 20: _tuned_like(), 60: _tuned_like()}, raising=True)
    mp.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed", raising=True)

    def reached():
        raise RuntimeError("reached the data: the override was not refused")

    mp.setattr(api.run, "trading_sessions", reached)
    for override in (dict(train_years=2), dict(val_years=0), dict(first_year=2012), dict(max_folds=1), dict(end="2012-12-31")):
        with pytest.raises(SystemExit):
            api.run.run_score(20, _tune_args(**override))
    assert not list(tmp_path.rglob("*")), list(tmp_path.rglob("*"))


# =============================================================================================================== M3-O1 outputs
def check_outputs_disjoint_from_m2(api):
    """M3-O1: the M3 output directory (for every tag), its label directory and its report are not, do not lie inside and do not contain the
    corresponding M2 directory/report. Re-training B1 on hold labels must never overwrite M2's scores, logs or labels."""
    from scripts import run_m2
    pairs = [(api.run.out_dir_for(t), d) for t in ("", "x") for d in run_m2._dirs(t)]
    pairs += [(api.run.hold_label_dir(h), run_m2.hold_label_dir(h)) for h in (5, 20, 60)]
    pairs += [(api.run.hold_label_dir(h), d) for h in (5, 20, 60) for d in (*run_m2._dirs(""), run_m2.hold_label_dir(h).parent)]
    for mine, theirs in pairs:
        mine, theirs = mine.resolve(), theirs.resolve()
        assert mine != theirs and theirs not in mine.parents and mine not in theirs.parents, (mine, theirs)
    assert api.run.REPORT_PATH.resolve() != run_m2.REPORT_PATH.resolve()
    assert "m3" in api.run.out_dir_for("").name and "m3" in api.run.REPORT_PATH.name


# =============================================================================================================== M3-R go_table (pure)
LOG_WEEKS = _weeks(FJW, 30)


def _logs():
    ung = _ungated(LOG_WEEKS)
    per = lambda n: {h: (_gate_log(n + h // 10, weeks=LOG_WEEKS), ung) for h in (5, 20, 60)}                    # noqa: E731
    return {"floor": {"a": per(60), "b": per(40), "c": per(50)}, "zero": {"a": per(50), "b": per(40), "c": per(45)}}


def _go(api, logs=None, invariants=True):
    dd = {h: LOG_WEEKS for h in (5, 20, 60)}
    fj = {h: FJW for h in (5, 20, 60)}
    return api.m3.go_table(_logs() if logs is None else logs, dd, fj, invariants)


def check_go_table_judged_cells_only(api):
    """M3-R1: one row per impact x arm x horizon in {5, 20, 60} = 18 rows. judged is True exactly for the FLOOR run at h = 20 and h = 60
    (go_all a bool); h = 5 (both runs) and every zero run are reported with their numbers, judged False, go_all None. Thresholds are
    identical for every arm (no multiplicity correction); an unknown arm, impact or horizon is refused."""
    t = _go(api)
    assert len(t) == 18 and set(zip(t.impact, t.arm, t.horizon)) == {(i, a, h) for i in ("floor", "zero") for a in "abc" for h in (5, 20, 60)}
    j = t[t["judged"].astype(bool)]
    assert set(zip(j.impact, j.horizon)) == {("floor", 20), ("floor", 60)} and len(j) == 6
    assert all(isinstance(x, (bool, np.bool_)) for x in j["go_all"])
    assert t[~t["judged"].astype(bool)]["go_all"].isna().all()
    assert np.allclose(t["c2_threshold"], 0.35)
    ung = _ungated(LOG_WEEKS)
    one = (_gate_log(40, weeks=LOG_WEEKS), ung)
    dd, fj = {20: LOG_WEEKS}, {20: FJW}
    for bad in ({"floor": {"d": {20: one}}}, {"atr": {"a": {20: one}}}, {"floor": {"a": {100: one}}}):
        with pytest.raises(ValueError):
            api.m3.go_table(bad, {**dd, 100: LOG_WEEKS}, {**fj, 100: FJW}, True)


def check_lag_and_blocks_in_go_table(api):
    """M3-R4: Newey-West lag and moving-block bootstrap block = ceil(h/5) weekly rows: 1 at h = 5, 4 at h = 20, 12 at h = 60. The GO table's
    nw_lag_weeks, Newey-West SE and weekly block-bootstrap CI equal vanta.m1's reference functions with that lag on the judged published
    FILLED rows (so a block of 4 used at h = 60 is caught)."""
    logs = _logs()
    t = _go(api, logs)
    for h, lag in ((5, 1), (20, 4), (60, 12)):
        r = t[(t.impact == "floor") & (t.arm == "a") & (t.horizon == h)].iloc[0]
        log = logs["floor"]["a"][h][0]
        a = log[(log["decision_date"] >= FJW) & log["published"].astype(bool) & (log["fill_status"] == "FILLED") & log["y_net"].notna()]
        wk = a.groupby("decision_date")["y_net"].mean().to_numpy()
        ci = m1.bootstrap_ci(m1.weekly_block_bootstrap(a["y_net"], m1.weekly_ids(a), 300, 11, lag))
        assert r.nw_lag_weeks == lag
        assert r.nw_se == pytest.approx(m1.newey_west_se(wk, lag))
        assert (r.boot_lo, r.boot_hi) == pytest.approx(ci)


# =============================================================================================================== M3-R report (text)
def _sections(md):
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _table_rows(body):
    """The FIRST pipe table of a section (text before its first '### ' subsection)."""
    body = body.split("\n### ")[0]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    head = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in ln.strip("|").split("|")]), _text=ln) for ln in lines[2:]]


def _key(r):
    return (r["impact"], r["arm"], int(float(r["h"])))


def check_report_tables(md):
    """M3-R2 (PREREG sections 6, 9): the GO table holds exactly the floor-run arms a/b/c at h = 20 and 60 (6 rows, none marked NOT JUDGED);
    the NOT JUDGED table holds the h = 5 rows of both runs and the zero-impact h = 20 / 60 rows (12 rows), and EVERY one says NOT JUDGED.
    Horizon 5 and zero-impact rows are absent from the GO table."""
    sec = _sections(md)
    go_k = [k for k in sec if k.startswith("GO table")]
    nj_k = [k for k in sec if k.startswith("NOT JUDGED")]
    assert len(go_k) == 1 and len(nj_k) == 1, list(sec)
    go_rows, nj_rows = _table_rows(sec[go_k[0]]), _table_rows(sec[nj_k[0]])
    assert len(go_rows) == 6 and {_key(r) for r in go_rows} == {("floor", a, h) for a in "abc" for h in (20, 60)}
    assert not any("not judged" in r["_text"].lower() for r in go_rows)
    want = {("zero", a, h) for a in "abc" for h in (20, 60)} | {(i, a, 5) for i in ("floor", "zero") for a in "abc"}
    assert len(nj_rows) == 12 and {_key(r) for r in nj_rows} == want
    assert all("NOT JUDGED" in r["_text"] for r in nj_rows)


def _first_paragraph(md):
    paras, cur = [], []
    for line in md.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    for p in paras:
        if not all(ln.lstrip().startswith("#") for ln in p):
            return " ".join(p)
    return ""


def check_report_header_and_footer(md):
    """M3-R3 (PREREG sections 1 and 9): the FIRST paragraph says the judged window was seen in M1, M1b and M2, that M3 is a re-test with a
    weaker pass, that the hold-to-horizon label choice adds a selection effect, and that only the forward paper log is fresh. The report ENDS
    with the footer: empirical number, not a guarantee, q = 0.30 an engineering target, the GO/NO-GO is the owner's; no verdict sentence."""
    first = _first_paragraph(md).lower()
    for needle in ("m1b", "m2", "re-test", "selection effect", "hold-to-horizon", "weaker", "paper", "fresh"):
        assert needle in first, needle
    last = md.strip().splitlines()[-1].lower()
    for needle in ("empirical", "not a guarantee", "engineering target", "owner"):
        assert needle in last, needle
    low = md.lower()
    assert "is a guarantee" not in low and "verdict is go" not in low and "verdict: go" not in low


def check_report_lag_rows(md):
    """M3-R4: the report prints the Newey-West lag next to each cell's numbers: (lag 4) on every h = 20 row and (lag 12) on every h = 60
    row of the comparator tables, and states the ceil(h/5) rule."""
    assert "ceil(h/5)" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "(lag" in ln]
    assert rows
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        h = int(float(cells[2]))
        assert f"(lag {ceil5(h)})" in ln, ln
    assert {int(float(ln.strip('|').split('|')[2])) for ln in rows} >= {20, 60}


def ceil5(h):
    return -(-h // 5)


def _fabricate_cells(out_dir):
    """Six finished cells (h 5/20/60 x floor/zero) in the layout scripts.run_m3 writes: meta_*.json and four log parquets each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ung = _ungated(LOG_WEEKS)
    counts = dict(n_weeks=30, n_age_only_fallback_weeks=3, share_age_only_fallback=0.1, n_state_missing_weeks=1, n_abstain_weeks=2)
    for h in (5, 20, 60):
        for imp in ("floor", "zero"):
            for arm, lg in (("a", _gate_log(60, weeks=LOG_WEEKS)), ("b", _gate_log(40, weeks=LOG_WEEKS)), ("c", _gate_log(50, weeks=LOG_WEEKS)),
                            ("ungated", ung)):
                lg = lg.assign(entry_date=lg["decision_date"] + pd.Timedelta(days=3), label_end_date=lg["decision_date"] + pd.Timedelta(days=30),
                               arm=arm)
                lg.to_parquet(out_dir / f"log_h{h}_{imp}_{arm}.parquet")
            meta = dict(horizon=h, impact=imp, first_scored_week=str(FJW.date()), first_judged_week=str(FJW.date()),
                        scored_weeks=[d.date().isoformat() for d in LOG_WEEKS], judged_counts={a: counts for a in "abc"},
                        counts={a: counts for a in "abc"}, fill_status={"FILLED": 100}, n_unscored_candidates=0, n_nan_score_candidates=0)
            (out_dir / f"meta_h{h}_{imp}.json").write_text(json.dumps(meta))


# =============================================================================================================== M3-P frozen parameters
def check_frozen_constants_equal_m2(api):
    """M3-P1 (PREREG section 5): q 0.30, cap 10, tau 26 weeks, Kish ESS floor 200, 52-week window, arms a/b/c, seeds 0-4, judged horizons
    (20, 60), reported-only horizon (5,), hold labels."""
    m3 = api.m3
    assert (m3.Q, m3.CAP, m3.TAU_WEEKS, m3.ESS_FLOOR, m3.WINDOW_WEEKS) == (0.30, 10, 26, 200, 52)
    assert tuple(m3.ARMS) == ("a", "b", "c") and tuple(m3.B1_SEEDS) == (0, 1, 2, 3, 4)
    assert tuple(m3.JUDGED_HORIZONS) == (20, 60) and tuple(m3.REPORTED_HORIZONS) == (5,) and m3.LABEL_VARIANT == "hold"
    assert list(m3.B1_HOLD_GRID) == list(bl.B1_PARAM_GRID)


FROZEN = {"q", "k", "k-s", "k-t", "ks", "kt", "tau", "tau-weeks", "cap", "ess-floor", "ess", "seed", "seeds", "window", "window-weeks",
          "liquidity-cliff", "cliff", "arms", "arm", "floor", "impact-floor", "oos-start", "first-judged-week", "folds", "variant",
          "params", "label", "labels", "atr", "label-variant", "grid", "embargo"}


def check_parser_frozen(parser):
    """M3-P2: --horizon accepts exactly 5, 20, 60; --impact exactly zero/floor; no option can change q, cap, tau, the ESS floor, seeds,
    window, arms, hyperparameters, grid, embargo or the label (there is no --variant: ATR is not part of M3, PREREG section 2)."""
    ch = {a.dest: a.choices for a in parser._actions}
    assert list(ch["horizon"]) == [5, 20, 60] and list(ch["impact"]) == ["zero", "floor"]
    opts = {s.lstrip("-") for a in parser._actions for s in a.option_strings}
    assert not (opts & FROZEN), sorted(opts & FROZEN)


def _toy_parser(*bugs):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, choices=[5, 20] if "no_h60" in bugs else [5, 20, 60])
    ap.add_argument("--impact", choices=["zero", "floor"])
    for b, opt in (("opt_q", "--q"), ("opt_cap", "--cap"), ("opt_tau", "--tau-weeks"), ("opt_seed", "--seeds"), ("opt_variant", "--variant"),
                   ("opt_params", "--params"), ("opt_k", "--k-s"), ("opt_embargo", "--embargo"), ("opt_grid", "--grid")):
        if b in bugs:
            ap.add_argument(opt, default=None)
    return ap


OVERRIDES = re.compile(r"\b(q|cap|tau_weeks|ess_floor|window_weeks)\s*=")


def check_sources_do_not_override_frozen(src_m3, src_run):
    """M3-P1/P2: the M3 sources never reference the ATR multipliers (K_S, K_T: no levels) and never pass q, cap, tau_weeks, ess_floor or
    window_weeks to the M2 gate machinery (the arms run with M2's own defaults); run_cell calls run_m2_arms(df, weeks) exactly."""
    for src in (src_m3, src_run):
        assert not re.search(r"\bK_[ST]\b", src), "ATR multiplier referenced"
    assert not OVERRIDES.search(src_run.split("def run_cell")[1].split("\ndef ")[0]), "run_cell overrides a frozen gate parameter"
    assert re.search(r"run_m2_arms\(\s*df\s*,\s*weeks\s*\)", src_run), "run_cell must call m2.run_m2_arms(df, weeks) with M2's defaults"


GOOD_M3, GOOD_RUN = "Q = m2.Q\nCAP = m2.CAP\n", "def run_cell(h):\n    logs = m2.run_m2_arms(df, weeks)\n\ndef other():\n    pass\n"


# =============================================================================================================== registry and tests
LABEL_MUTANTS = ["atr_labels", "target_only", "stop_only"]
TUNE_MUTANTS = ["reads_test_year", "wrong_fold", "shared_file", "atr_labels_tune", "own_grid", "features_all", "writes_m2_dir"]
PLACEHOLDER_MUTANTS = ["no_placeholder_guard", "fallback_20"]
FOLD_MUTANTS = ["no_fold_guard"]
OUTPUT_MUTANTS = ["writes_m2_dir", "labels_in_m2", "report_is_m2"]
GO_MUTANTS = ["judge_5", "judge_zero", "accept_any_arm"]
LAG_MUTANTS = ["lag4"]
CONST_MUTANTS = ["q25", "cap5", "tau13", "ess100", "win26", "four_arms", "seeds3", "judge_5", "label_atr"]
TABLE_MUTANTS = ["h5_in_go", "zero_in_go", "unmarked", "drop_h60_go", "nj_drops_h5"]
TEXT_MUTANTS = ["header_silent", "header_no_selection", "header_buried", "footer_guarantee", "footer_missing"]
PARSER_MUTANTS = ["no_h60", "opt_q", "opt_cap", "opt_tau", "opt_seed", "opt_variant", "opt_params", "opt_k", "opt_embargo", "opt_grid"]

WITH_MP = {
    "labels": (check_hold_labels_built_as_hold, LABEL_MUTANTS, False),
    "tune": (check_tuning_fold0_inner_split_hold_labels_per_horizon, TUNE_MUTANTS, True),
    "placeholder": (check_placeholder_refused_before_any_output, PLACEHOLDER_MUTANTS, True),
    "fold_guard": (check_judged_score_refuses_fold_overrides, FOLD_MUTANTS, True),
}


def _run_with_fixtures(check, api, mp, tmp_path, tmp_flag):
    if tmp_flag:
        check(api, mp, tmp_path)
    else:
        check(api, mp)


@pytest.mark.parametrize("name", list(WITH_MP))
def test_m3_real(name, monkeypatch, tmp_path):
    check, _, tmp_flag = WITH_MP[name]
    _run_with_fixtures(check, _real(), monkeypatch, tmp_path, tmp_flag)


@pytest.mark.parametrize("name", list(WITH_MP))
def test_m3_toy_passes(name, monkeypatch, tmp_path):
    check, _, tmp_flag = WITH_MP[name]
    _run_with_fixtures(check, toy_api(), monkeypatch, tmp_path, tmp_flag)


@pytest.mark.parametrize("name,bug", [(n, b) for n in WITH_MP for b in WITH_MP[n][1]])
def test_m3_mutant_fails(name, bug, monkeypatch, tmp_path):
    check, _, tmp_flag = WITH_MP[name]
    with pytest.raises((Exception, pytest.fail.Exception)) as ei:
        _run_with_fixtures(check, toy_api(bug), monkeypatch, tmp_path, tmp_flag)
    _no_attr_error(ei)


# ---- pure API checks (no fixtures)
PURE = {
    "go_table": (check_go_table_judged_cells_only, GO_MUTANTS),
    "lag_go_table": (check_lag_and_blocks_in_go_table, LAG_MUTANTS),
    "frozen": (check_frozen_constants_equal_m2, [b for b in CONST_MUTANTS]),
}


@pytest.mark.parametrize("name", list(PURE))
def test_m3_pure_real(name):
    PURE[name][0](_real())


@pytest.mark.parametrize("name", list(PURE))
def test_m3_pure_toy_passes(name):
    PURE[name][0](toy_api())


@pytest.mark.parametrize("name,bug", [(n, b) for n in PURE for b in PURE[n][1]])
def test_m3_pure_mutant_fails(name, bug):
    with pytest.raises((Exception, pytest.fail.Exception)) as ei:
        PURE[name][0](toy_api(bug))
    _no_attr_error(ei)


# ---- outputs disjoint from M2 (toy paths follow config, as the real ones do)
def test_m3_outputs_real():
    check_outputs_disjoint_from_m2(_real())


def test_m3_outputs_toy_passes():
    check_outputs_disjoint_from_m2(toy_api())


@pytest.mark.parametrize("bug", OUTPUT_MUTANTS)
def test_m3_outputs_mutant_fails(bug):
    with pytest.raises(AssertionError):
        check_outputs_disjoint_from_m2(toy_api(bug))


# ---- report text: the real report from fabricated cells, the toy text, and doctored toy texts
TEXT_CHECKS = {"tables": (check_report_tables, TABLE_MUTANTS), "header_footer": (check_report_header_and_footer, TEXT_MUTANTS),
               "lag_rows": (check_report_lag_rows, ["lag4_report"])}


@pytest.fixture
def real_report(monkeypatch, tmp_path):
    api = _real()
    import config
    from scripts import run_m1b
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed", raising=True)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs", raising=True)
    monkeypatch.setattr(run_m1b, "M1B_DIR", run_m1b.M1B_DIR)                       # report() reassigns it; restore afterwards
    monkeypatch.setattr(run_m1b, "load_nifty50_close", lambda: (_ for _ in ()).throw(FileNotFoundError("not in this test")))
    _fabricate_cells(api.run.out_dir_for("rt"))
    return api.run.report("rt", 50, invariants=(True, "test"))


@pytest.mark.parametrize("name", list(TEXT_CHECKS))
def test_m3_report_real(name, real_report):
    TEXT_CHECKS[name][0](real_report)


@pytest.mark.parametrize("name", list(TEXT_CHECKS))
def test_m3_report_toy_passes(name):
    TEXT_CHECKS[name][0](toy_report())


@pytest.mark.parametrize("name,bug", [(n, b) for n in TEXT_CHECKS for b in TEXT_CHECKS[n][1]])
def test_m3_report_mutant_fails(name, bug):
    with pytest.raises(AssertionError):
        TEXT_CHECKS[name][0](toy_report(bug))


# ---- CLI
def test_m3_parser_real():
    check_parser_frozen(_real().run.build_parser())


def test_m3_parser_toy_passes():
    check_parser_frozen(_toy_parser())


@pytest.mark.parametrize("bug", PARSER_MUTANTS)
def test_m3_parser_mutant_fails(bug):
    with pytest.raises(AssertionError):
        check_parser_frozen(_toy_parser(bug))


# ---- source scan
def test_m3_sources_real():
    api = _real()
    check_sources_do_not_override_frozen(inspect.getsource(api.m3), inspect.getsource(api.run))


def test_m3_source_scan_toy_passes():
    check_sources_do_not_override_frozen(GOOD_M3, GOOD_RUN)


@pytest.mark.parametrize("m3src,runsrc", [
    ("K_S = 1.5\n", GOOD_RUN),
    (GOOD_M3, "from vanta.m2 import K_T\n" + GOOD_RUN),
    (GOOD_M3, GOOD_RUN.replace("run_m2_arms(df, weeks)", "run_m2_arms(df, weeks, cap=5)")),
    (GOOD_M3, GOOD_RUN.replace("run_m2_arms(df, weeks)", "run_m2_arms(df, weeks, q=0.25)")),
    (GOOD_M3, GOOD_RUN.replace("run_m2_arms(df, weeks)", "run_m2_arms(df, weeks, ess_floor=100)")),
    (GOOD_M3, "def run_cell(h):\n    pass\n"),
])
def test_m3_source_scan_mutant_fails(m3src, runsrc):
    with pytest.raises(AssertionError):
        check_sources_do_not_override_frozen(m3src, runsrc)
