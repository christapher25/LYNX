# VANTA interfaces assumed by the invariant tests

Source of truth for names and signatures that `tests/invariants/vanta/` imports or calls on VANTA modules that do not exist yet.
The tests skip (`pytest.importorskip`) until the module exists, then run against it. If an implementation needs to differ from
anything here, change this file and the test together, via the lead.

Legend. **SPEC** = defined by `docs/vanta/VANTA_SPEC.md` (the names, enums and formulas). **INVENTED** = chosen by the tests; the
spec is silent, so the implementation must match it or the lead must rule otherwise. The spec defines no module paths or
function names at all, so every `vanta.*` module and function name below is INVENTED.

Support files (not VANTA interfaces, they hold the reference and deliberately broken toys): `tests/invariants/vanta/vanta_toys_labels.py`,
`tests/invariants/vanta/vanta_toys_gate.py`. The toys' only external dependency is the existing `costs.model.get_rates(as_of: date)`
(fields `stt_delivery_buy`, `stt_delivery_sell`, `stamp_duty_buy`).

## Import inventory (verified by grep over `tests/invariants/vanta/test_*.py`)

| Import in tests | Resolved name | Used by |
|---|---|---|
| `pytest.importorskip("vanta.labels").simulate_trade` | `vanta.labels.simulate_trade` | N1, N2, N3, N4, N6 |
| `pytest.importorskip("vanta.walkforward").purge_training_rows` | `vanta.walkforward.purge_training_rows` | N11 |
| `pytest.importorskip("vanta.gains").build_fold_gains` | `vanta.gains.build_fold_gains` | N10 |
| `pytest.importorskip("vanta.gate")` (module used as `api`) | `vanta.gate.regime_age_weights`, `conformal_pvalues`, `bh_select`, `select_calibration`, `validate_calibration`, `run_gate` | N5, N8, N9, N13 |

No test imports anything else from `vanta`. There are **no tests yet for N7, N12, N14** (not interfaces here; add when written).

## 1. `vanta.labels.simulate_trade` (Stage 13A; N1, N2, N3, N4, N6)

```python
simulate_trade(symbol, bars, decision_date, horizon, k_s, k_t, atr, hits, hit_coverage,
               cost_fn=None, delisted_after=None) -> Mapping   # INVENTED signature; row keys are SPEC
```

| Parameter | Type / unit | Notes |
|---|---|---|
| `symbol` | str | |
| `bars` | `pd.DataFrame`, `DatetimeIndex` of consecutive sessions | INVENTED columns: `raw_open, raw_high, raw_low, raw_close` (unadjusted; used ONLY for lock detection) and `open, high, low, close` (Stage 4 adjusted; used for entry, stop, target, fills, gross return). |
| `decision_date` | `pd.Timestamp`, must be in `bars.index` | Session t. Entry is the next row of `bars` (t+1). |
| `horizon` | int sessions (5, 20, 60) | Time exit = close of row t+h. |
| `k_s`, `k_t` | float | stop = entry − k_s·atr, target = entry + k_t·atr (SPEC). |
| `atr` | float, **adjusted-price units** (not a percent) | ATR(14) at t's close, same scale as `open/high/low/close`. Tests halve the adjusted columns and pass `atr=1.0` for `2.0`. |
| `hits` | `pd.DataFrame` columns `date` (Timestamp), `symbol` (str), `side` (`"UPPER"` or `"LOWER"`) | INVENTED. One row per (session, symbol, side) the symbol appeared on the price-band hit list. |
| `hit_coverage` | `set[pd.Timestamp]` | INVENTED. Sessions for which a published hit list exists. Used for `lock_source`. |
| `cost_fn` | `None` or callable `(symbol, entry_date, exit_date, entry_price, exit_price) -> (cost_entry, cost_exit, cost_impact)` | INVENTED. Each cost is a **fraction of notional** (same unit as the return). `None` means use the Stage 6 cost model at trade-date rates (N6 depends on this default). Tests pass a zero-cost function for N1-N4. |
| `delisted_after` | `None` or date | In the signature only; **no test exercises it**. |

Return: a mapping (tests index with `r["key"]`; a dict or a `pd.Series` both work). Keys are the SPEC `vanta_labels` columns the tests read:
`fill_status`, `entry_price_raw`, `entry_date`, `exit_date`, `exit_price`, `exit_reason`, `lock_delay_sessions`, `gross_return`,
`cost_entry`, `cost_exit`, `cost_impact`, `y_net`, `label_end_date`.

Behaviour the tests pin down (each SPEC unless marked):

- **N1.** Upper lock at entry = (symbol, entry session, `"UPPER"`) in `hits` **and** `raw_open == raw_high`, both required. A LOWER hit does not count. Detection uses raw prices. Result `fill_status == "NO_FILL_UPPER_LOCK"` and `entry_price_raw`, `exit_price`, `gross_return`, `y_net` are all NaN. The test helper `isnan` (`vanta_toys_labels.py:159`) also accepts None, but the implementation should return NaN per the data contract. Otherwise `fill_status == "FILLED"` and `entry_price_raw` is the raw open of t+1.
- **N2.** Lower-locked at close = (symbol, session, `"LOWER"`) in `hits` **and** `raw_close == raw_low`. If the stop triggers on session d and d is lower-locked, exit at the **adjusted close of the first later session that is not lower-locked**, `exit_reason == "STOP_DELAYED_LOWER_LOCK"`, `lock_delay_sessions` = number of sessions from d to that exit session (d, d+1 locked and d+2 free gives 2), `exit_date == label_end_date == ` that session. Not locked gives `exit_reason == "STOP"`, price = stop, delay 0.
- **N3.** Stop and target both inside one day's range gives exit at the stop, `exit_reason == "STOP"`. Also applies on the entry day itself (entry day's high/low are checked against stop/target). Target only gives `"TARGET"` at the target price.
- **N4.** For sessions after the entry day: open <= stop gives `exit_reason == "STOP_GAP"`, `exit_price` = that open (unless that session is lower-locked: then `"STOP_DELAYED_LOWER_LOCK"` at the first non-locked close, see ambiguity 7); open >= target gives `exit_reason == "TARGET"`, `exit_price` = the target (not the open). `gross_return = exit_price/entry − 1` on adjusted prices.
- **N6.** With `cost_fn=None`, the same price path priced wholly before (2004-09-21..27) vs wholly after (2004-10-05..11) the 2004-10-01 STT introduction gives `cost_entry + cost_exit` larger by at least 0.2% (0.1% STT each leg), so `y_net` falls by that much. `y_net == gross_return − cost_entry − cost_exit − cost_impact`. `exit_reason == "TIME"` for a quiet path.

INVENTED / not asserted (implementation free, flagged to the lead):

- A gap exactly at the stop (open == stop): the test asserts only `exit_price == 98`, not the reason (toy labels it `STOP_GAP`).
- Straddling a rate change: which date's rates apply is deliberately untested (toy uses the entry date for both legs).
- `lock_source` (`HIT_LIST` iff every session from entry to exit is in `hit_coverage`, else `NONE`), `delisted_after`/`DELISTED`, and a horizon running past the end of `bars` are implemented in the toy but **not asserted by any test**.

## 2. `vanta.walkforward.purge_training_rows` (N11)

```python
purge_training_rows(labels_df, test_start, embargo, sessions) -> pd.DataFrame   # INVENTED signature
```

- `labels_df`: DataFrame with at least `label_end_date` (Timestamp) and `decision_date`. Not mutated.
- `test_start`: `pd.Timestamp`. `embargo`: **int, in sessions** (5, 20, 60). `sessions`: sorted `pd.DatetimeIndex` trading calendar containing `test_start`.
- Cutoff = the session `embargo` positions before `test_start`. Keep rows with `label_end_date < cutoff`, drop `>=`. Return the kept rows, original order and columns.
- SPEC gives the rule (`label_end_date ≥ test_start − embargo` is dropped). The boundary is SPEC, not invented: `VANTA_SPEC.md:77` drops `label_end_date ≥ test_start − embargo`, so a row ending exactly on the cutoff session is dropped. INVENTED: only the unit of `embargo` (sessions, not calendar days; decided in `logs/agent_team_decisions.md` Q2) and the `sessions` parameter. The test pins calendar-day arithmetic as wrong (a holiday sits in the window).

## 3. `vanta.gains.build_fold_gains` (N10)

```python
build_fold_gains(labels, train_mask, horizon) -> pd.Series   # INVENTED signature
```

- `labels`: DataFrame with `y_net`, `atr_pct`. `train_mask`: boolean array-like aligned positionally to `labels`. `horizon`: int.
- Gain `g = y_net / (atr_pct * sqrt(horizon))` (SPEC). Winsorise **all** rows (train and test) at the 2nd and 98th percentiles of the **training rows' gains only** (SPEC). Return a Series indexed like `labels`; the tests use `g[mask]`, `g[~mask]`, `.to_numpy()`.
- INVENTED: percentile interpolation. The test needs numpy's default linear method (train gains 0..50 give bounds exactly 1 and 49). The spec says only "2nd and 98th percentile".

## 4. `vanta.gate` (Stage 13C; N5, N8, N9, N13)

The tests pass the **module** `vanta.gate` as `api` and call the six functions below, so all six must be module-level names. Common keyword arguments `model_version: str`, `candidate_filter_version: str` are the "current" tags (strings compared by equality).

Calibration table columns (SPEC `vanta_calibration`): `decision_date` (Timestamp), `symbol`, `horizon`, `score`, `model_version`, `fold_id`, `is_oos` (bool), `regime_posterior` (a length-3 list of floats per row), `hmm_version`, `y_net` (NaN until matured), `matured_at` (Timestamp or NaT), `candidate_filter_version`. The tests may add an extra `published` column to a calibration table; the gate must ignore it (N8 trap).

```python
regime_age_weights(age_weeks, pi_cal, pi_T, tau_weeks) -> np.ndarray
```
`age_weeks` (n,) float; `pi_cal` (n,3) array-like (a list of lists works); `pi_T` length-3 sequence; `tau_weeks` float. Returns `exp(-age/tau) * exp(-‖pi_cal_i − pi_T‖₁)` (SPEC).

```python
conformal_pvalues(cal_scores, cal_y, cal_w, test_scores, test_w=1.0) -> np.ndarray
```
Lists or arrays accepted. For each test score `s_j`: `(test_w + Σ_i w_i·1[y_i ≤ 0 and s_i ≥ s_j]) / (test_w + Σ_i w_i)` (SPEC). Ties count (`≥`), `y == 0` counts as a loss. Returns shape `(len(test_scores),)`.

```python
bh_select(pvals, q) -> np.ndarray[bool]
```
Step-up Benjamini-Hochberg over unsorted input; the result is aligned to the input order and compares equal to a list of bools. Empty input gives an empty array. (SPEC: BH at q.)

```python
select_calibration(cal, T, *, model_version, candidate_filter_version, window_weeks=52) -> pd.DataFrame
```
Returns the subset of `cal` satisfying the SPEC query rule: `matured_at <= T` (NaT excluded, equality included) AND `decision_date >= T − window_weeks` AND `model_version ==` AND `candidate_filter_version ==` AND `is_oos`. Rows are excluded, not raised on. All other columns kept.

```python
validate_calibration(cal, T, *, model_version, candidate_filter_version) -> None
```
Raises `ValueError` if any row of `cal` has `matured_at > T` (or null), a different `model_version`, a different `candidate_filter_version`, or `is_oos` false. Returns None on a clean set. INVENTED: the test requires BOTH behaviours (exclude in `select_calibration`, raise in `validate_calibration`) for "rejected" in N5/N9.

```python
run_gate(cands, cal, T, pi_T, *, model_version, candidate_filter_version,
         tau_weeks=26, q=0.3, cap=10, ess_floor=200) -> pd.DataFrame
```
`cands`: DataFrame with `symbol`, `score`. `T`: Timestamp. `pi_T`: length-3 tuple. Internally: `select_calibration`, age in weeks `= (T − decision_date).days / 7` (fractional; INVENTED reading of "age in weeks"), `regime_age_weights`, `conformal_pvalues`, `bh_select`, then keep at most `cap` BH members by ascending p-value, ties by **higher score first** (the spec is silent beyond "ties broken by score"; higher score first was confirmed by the user on 2026-10-27, `logs/agent_team_decisions.md` Q4). `ess_floor=0` disables the Kish fallback, which is what every test uses; the fallback is implemented in the toy but **not asserted**.

Returned DataFrame: one row per candidate, **same order as `cands`**, index reset (tests rely on order). Columns the tests read:

| Column | Type | Meaning |
|---|---|---|
| `symbol`, `score` | | copied from `cands` |
| `p_value` | float in (0, 1] | |
| `in_bh_set` | bool | |
| `bh_threshold` | float | INVENTED definition: `q·k/m` with k = BH set size, m = candidates; NaN when k = 0. The test asserts `in_bh_set == (p_value <= bh_threshold)`, so any definition with that property passes. |
| `published` | bool | subset of `in_bh_set`, at most `cap` true |
| `cap_binding` | bool | INVENTED: the same value on every row, true iff BH set size > `cap` |
| `q`, `n_calibration`, `effective_n` | float / int / float | `n_calibration` = rows in the selected calibration set (tests assert 120 and 100); `effective_n` = Kish ESS `(Σw)²/Σw²` |

Additional `run_gate` log columns (all INVENTED, alongside `cap_binding`): `score_missing` (bool; candidate score non-finite: p = 1, never in the BH set or published, but still counted in m), `age_only_fallback` (bool; Kish ESS below `ess_floor`, or a non-finite regime posterior, so age-only weights were used) and `effective_n_age_only` (Kish ESS of the age-only weights). `effective_n` is NaN when the weights are non-finite (a NaN posterior); consumers must tolerate NaN. `bh_threshold` and k are computed on the set after masking missing-score rows, with m including those rows. `build_fold_gains` maps non-finite gains (e.g. `atr_pct == 0`, negative or NaN ATR) to NaN: never clipped, never used in the winsorisation fit.

SPEC-required `vanta_gate_log` columns that the tests do not read (`decision_date`, `horizon`, version tags, `y_net`, `lost`) are not part of this interface yet.

## Ambiguities for the lead

1. **Hit-list schema.** Tests use `hits[date, symbol, side]` plus a `hit_coverage` set. The existing Stage 10 circuits block exposes `hit_upper_circuit` / `hit_lower_circuit` flags per symbol-row with NaN for no list. Either the real `simulate_trade` adapts, or a loader builds the `hits`/`hit_coverage` pair; the tests do not care which, but the adaptation point must be decided.
2. **Gap exactly at the stop** and **rate changes inside a trade** are unspecified; tests avoid asserting them.
3. **`bh_threshold` / `cap_binding` definitions** are ours (above). Spec lists the columns but not their definitions.
4. **Percentile method** for winsorisation is not specified (numpy linear assumed).
5. **`atr` units.** The spec defines `atr_pct = ATR(14)/close`; the tests pass absolute `atr` in adjusted price units to `simulate_trade`. The `atr_pct` column of `vanta_labels` is the caller's/store's concern.
6. **Kish fallback** (ESS < 200 gives age-only weights, logged) has no test and no log column defined.
7. **Gap-down rule: RESOLVED (user decision 2026-10-29).** Where `STOP_GAP` and `STOP_DELAYED_LOWER_LOCK` collide: a gap-down session (open <= stop, after the entry day) that is lower-locked (lower hit and raw close == raw low, the existing definition) exits at the close of the first non-locked session with `exit_reason = "STOP_DELAYED_LOWER_LOCK"` and `lock_delay_sessions` = sessions from the gap day to that exit (>= 1). An unlocked gap-down exits at the open with `"STOP_GAP"`, delay 0. Tested in `check_n4_gap_down_lock` (test_vanta_label_fills.py). Note: the "locked" test for a gap day is the close-based one; a day that opens at the lower limit but trades off it by the close counts as unlocked (STOP_GAP at the open); the tests do not assert that case.

Resolved, no longer open: tie order in publication (higher score first, user-confirmed Q4); `embargo` unit (sessions, Q2).

## 5. `vanta.m1` (Stage 13M; M1 pure functions, approved by the lead 2026-10-30)

Tests: `tests/invariants/vanta/test_vanta_m1.py`; reference and broken toys: `vanta_toys_m1.py`. All names are INVENTED (spec silent).
Constants: `Q=0.30`, `CAP=10`, `REFERENCE_NOTIONAL=100_000`, `HORIZONS=(5,20)`, `K_S_GRID=(1.0,1.5,2.0)`, `K_T_GRID=(1.5,2.0,3.0)`.
Gate-log frame `log`: `decision_date` (Timestamp), `symbol`, `published` (bool), `fill_status` (str), `y_net` (float; NaN until matured / on no fill).

| Function | Contract |
|---|---|
| `realised_fdr(log) -> dict` | keys `fdr, n_published, n_filled, n_no_fill, n_losers, n_unmatured`. Published rows only. `n_no_fill` = published with `fill_status != "FILLED"`; `n_filled` = published FILLED with finite `y_net`; `n_unmatured` = published FILLED with NaN `y_net`; loser = `y_net <= 0`; `fdr = n_losers/n_filled` (NaN if 0). |
| `fdr_by_year(log) -> dict[int,float]` | same rule per `decision_date.year`; years with `n_filled == 0` omitted (INVENTED). |
| `publication_rate(log, decision_dates=None) -> float` | weeks (unique decision_date) with >= 1 published / total weeks; denominator is `decision_dates` if given, else unique dates in `log`. |
| `newey_west_se(x, lags) -> float` | Bartlett: `gamma_l = (1/n) sum (x_t-m)(x_{t-l}-m)`; `var = (g0 + 2 sum_{l=1..L}(1-l/(L+1)) g_l)/n`; `[1,-1,1,-1]` gives 0.5 (L=0), 0.25 (L=1). Lag rule not tested. |
| `weekly_block_bootstrap(y, week_ids, n_boot, seed) -> ndarray` | pooled per-signal mean of each whole-week resample. |
| `bootstrap_ci(draws, level=0.95) -> (lo, hi)` | `np.percentile` at 2.5 / 97.5. |
| `weekly_block_bootstrap_diff(y_a, week_a, y_b, week_b, n_boot, seed) -> ndarray` | `mean_a - mean_b`, same resampled weeks for both arms, weeks drawn from the union. |
| `choose_k(trades, start="2010-01-01", end="2012-12-31") -> (k_s, k_t)` | `trades` cols `decision_date, k_s, k_t, y_net`; mean `y_net` per combo over rows with `start <= decision_date <= end` and finite `y_net` (NaN ignored, not zero); argmax; ties pick the first combo in grid order (k_s then k_t ascending), pinned by the lead. |
| `go_report(log, oos_start=Timestamp("2013-01-01")) -> {"in_sample": m, "oos": m}` | `m = dict(fdr, n_filled, n_no_fill, publication_rate, fdr_by_year)`, each on its own slice (`decision_date < oos_start` vs `>=`). |
| `v4_gate_reconstruction(cands, cal, T, *, model_version, candidate_filter_version, q=0.30, cap=10) -> DataFrame` | cols `symbol, score, fitted, bound, published`, order of `cands`. Calibration = SPEC query rule plus finite `y_net`. `fitted` = linear OLS of `y_net` on `score` (INVENTED). `shortfall = fitted - y`; `bound = fitted - np.quantile(shortfall, 1-q)` (numpy linear interpolation, INVENTED); published iff `bound > 0` (strict), at most `cap` by score descending. A RECONSTRUCTION of the v4 gate. |
| `impact_floor_per_side(turnover_rank) -> float` | 1..200 gives 0.0005, 201..700 gives 0.0020 (the user's assumption); rank > 700 or NaN gives NaN (INVENTED; the cost run must flag these trades, never default them); rank < 1 raises ValueError. |
| `turnover_ranks(trailing_turnover: Series) -> Series` | 1 = highest turnover, `method="min"`, NaN stays NaN. |
| `fdr_report_text(stats, q) -> str` | contains "empirical", "NO_FILL" and the `n_no_fill` number; never describes q as a guarantee (only "not a guarantee" / "no guarantee" phrasing). |

## 6. `vanta.m1b` (Stage 13M, corrected M1; proposed by tests-13a 2026-10-30, all names INVENTED)

Tests: `tests/invariants/vanta/test_vanta_m1b.py`; reference and broken toys: `vanta_toys_m1b.py`. Stats, constants (`Q`, `CAP`, `LIQUIDITY_CLIFF_RANK`), k-choice and GO report are reused from `vanta.m1`, unchanged.

| Function | Contract |
|---|---|
| `percentile_rank(values) -> ndarray` | One week. Finite values only; `pct = average_rank / n_finite` (ties share the average rank; range (0,1]). `[30,10,40,20]` gives `[.75,.25,1,.5]`; `[1,2,2,3]` gives `[.25,.625,.625,1]`. NaN/inf stays NaN and does not count in n (never 0). All-NaN week returns all NaN; a single finite value scores 1.0 (lead, 2026-10-30). |
| `weekly_scores(frame, momentum_col="momentum") -> Series` | Aligned to `frame.index`; `percentile_rank` applied separately per `decision_date`. Never raw momentum, never pooled across weeks. |
| `eligible_candidates(week, cliff=200) -> DataFrame` | `week` cols `symbol, banned, hit_upper, hit_lower, liquidity_rank, momentum`. Keeps EVERY row with not banned, not hit_upper, not hit_lower, `liquidity_rank` finite and `<= cliff` (inclusive), finite momentum. No top-N truncation; row order preserved. Flags may be nullable bool: a missing/NA flag means "list unavailable" and the row is EXCLUDED (block policy; lead 2026-10-30). Never `astype(bool)` an NA flag. |
| `calibration_rows(scored, T, window_weeks=52) -> DataFrame` | `scored` cols `decision_date, symbol, score, y_net, matured_at, published`. Keeps `matured_at <= T`, `decision_date >= T - window_weeks weeks` (both inclusive), finite `y_net`. Does not filter on `published` (calibration is all scored candidates, FCC). |
| `select_published(frame, cap=10) -> bool ndarray` | `frame` cols `symbol, score, in_bh_set`; aligned to `frame`. True iff `in_bh_set` and among the top `cap` by score descending (stable). Never more than `cap`, never outside the BH set. |

## 7. `vanta.baselines` (Stage 14 baselines A0, A1, B1; queue row 15c; all names INVENTED, proposed by tests-15c 2026-10-31)

Tests: `tests/invariants/vanta/test_baselines_invariants.py` (reference toy and 54 deliberately broken variants live in that file) and `tests/test_baselines.py` (library and source-scan tests; `catboost` / `xgboost` are not installed, so those skip). Model protocol everywhere: `factory(seed: int)` returns an object with `fit(X: ndarray, y: ndarray)` and `predict(X) -> ndarray`; X is a float ndarray with NaN for missing, never filled.

Constants. `B1_SEEDS = (0, 1, 2, 3, 4)` (M2_PREREG section 3). `B1_PARAMS: dict` (module-level, recorded before the first OOS run, no seed key). `MIN_NAMES` is the harness value.

| Function | Contract |
|---|---|
| `fit_ensemble(factory, seeds, X, y) -> list` | One `factory(s)` call per seed, in order, each fitted on the SAME `(X, y)`. |
| `predict_ensemble(models, X_new) -> ndarray` | Plain mean over the models' predictions (not median). Deterministic for a fixed seed list. |
| `make_fit_fn(features, feature_cols, factory, seeds=B1_SEEDS, target="net") -> fit_fn(rows, fold)` | Plugs into `wf.run_fold`. `features`: DataFrame with `decision_date`, `symbol` and the feature columns, point-in-time already (the caller built it through `PitReader`). `rows`: the purged training rows `run_fold` hands over (`decision_date, symbol, y_net, ...`); the fit uses exactly those and nothing else. Left-join on `(decision_date, symbol)`; a row whose label is NaN is DROPPED; a row with missing features is KEPT with NaN (no 0, no imputation). `target="binary"` (A1) trains on `y_net > 0` (1.0/0.0; a NaN stays missing and is dropped; `y_net == 0` is a loser). Returns an object with `.models` (one per seed), `.seeds`, `.train_keys` (DataFrame `decision_date, symbol` of the rows actually fitted) and `.feature_cols`. No state survives between calls: each fold gets fresh models. |
| `make_score_fn(features, feature_cols) -> score_fn(model, rows, fold, reader)` | Returns `predict_ensemble(model.models, X)` for `rows` (`decision_date, symbol`), same order as `rows`. Missing features go to the model as NaN. A score at date d uses only the `features` rows of date d (the test truncates or poisons later rows and requires equal scores): no cross-date normalisation, no next-date lookup. |
| `fold_rank_ic(scored, min_names=MIN_NAMES) -> dict` | `scored`: `decision_date, symbol, score, y_net`. Keys `per_date` (Series), `mean`, `n_dates`. Spearman per decision date over complete pairs (NaN labels/scores dropped, never 0); a date with fewer than `min_names` pairs is excluded, not counted as 0; `mean` is the unweighted mean over dates (NaN if none). |
| `top_n_per_date(scored, n=10) -> DataFrame` | Per decision date the `n` highest `score`, non-finite scores never selected, ties stable in input order, the label is never read. Rows keep their columns. |
| `net_per_signal(selected) -> dict` | `selected`: `decision_date, y_net` and optional `fill_status` (absent means every row is FILLED). Keys `mean`, `n_published`, `n_filled`, `n_no_fill`, `n_unmatured`. `mean` = pooled mean of `y_net` over FILLED rows with finite `y_net`; NO_FILL_* and FILLED-with-NaN (unmatured) are counted apart and never enter as 0. |
| `momentum_a0(close, dates) -> Series` | A0: `close[t-21] / close[t-252] - 1` by the symbol's own bar rows on ADJUSTED closes; NaN with fewer than 252 prior bars; identical (including the NaN pattern) to `vanta.m1.momentum_12_1`. |
| `fold_report(results, seeds=B1_SEEDS, n=10, min_names=MIN_NAMES) -> list[dict]` | `results`: `FoldResult`-like objects (`.fold` with `fold_id, train_start, val_start, test_start, test_end, embargo`; `.n_train`; `.scored`). One dict per fold with `fold_id, train_start, val_start, test_start, test_end, embargo, seeds, n_train, rank_ic, n_ic_dates, net_per_signal, n_published, n_filled, n_no_fill, n_unmatured`; `rank_ic` and the top-`n` net are computed on THAT fold's rows only. |
| `choose_hyperparameters(grid, model_for, features, labels, fold, sessions, seed=0, min_names=MIN_NAMES) -> dict` | Picks the `grid` entry (`model_for(params, seed)` builds a model) with the highest mean rank IC. Uses fold 0 only: fit rows = decisions in `[train_start, val_start)` with `label_end_date < session(val_start - embargo)`; evaluation rows = decisions in `[val_start, test_start)` with `label_end_date < session(test_start - embargo)`. Nothing from the test window, later folds, or unmatured boundary rows. Ties: first in grid order. |
| `b1_model_factory(**overrides)`, `a1_model_factory(**overrides)` | `factory(seed)` for CatBoost (`random_seed=seed`) and for the XGBoost A1 classifier (`predict` returns P(win) in [0,1]). Keyword overrides replace `B1_PARAMS` entries (tests use small `iterations`). Library tests skip while the libraries are missing. |

Source rule (scan in `tests/test_baselines.py`): `vanta/baselines.py` contains no `fillna(0`, `nan_to_num`, imputer or `interpolate`.

Open questions for the lead (tests avoid asserting these):
1. A1 "thresholded": the threshold and the "deterministic score" part (reuse `signals/scorer.py`?) are not specified; only the binary target (`y_net > 0`) is pinned.
2. Rows with ALL features missing at scoring time: score NaN (unavailable) or model output? Not asserted.
3. `wf.run_fold` hands `score_fn` and the `scored` frame only `decision_date, symbol, y_net`; NO_FILL candidates and `fill_status` are not carried. `net_per_signal` accepts `fill_status` when present; the implementer must decide how the published-signal frame gets it (M2 reports NO_FILL separately).
4. Hyperparameter tuning: `choose_hyperparameters` uses fold 0's inner split only. Using fold 1's validation year would read fold 0's OUT-OF-SAMPLE labels (they are the same year), which M2_PREREG section 3 forbids in spirit; "the earliest folds" (plural) is therefore read as fold 0 plus nothing else. Lead to confirm or give another inner scheme.

### 7a. Names added by the implementer (quant-15c, 2026-10-31)

- `vanta.baselines.B1_PARAMS_BY_HORIZON: dict[int, dict]` keys 5 and 20; template `B1_PARAMS`; placeholders until the tuning job has run (lead decision: tuned per horizon on fold 0's inner split). `B1_PARAM_GRID` (8 entries) and `A1_PARAMS`, `A1_PUBLISH_ABOVE = 0.5` are also module constants.
- `fold_report(..., publish_above=None)`: extra keyword; when set (A1) only scores strictly above it enter the top-n; rank IC always uses every scored row.
- `first_scored_and_judged_week(scored, oos_start, window_weeks=52) -> dict` keys `first_scored_week`, `first_judged_week`. Judged = first scored week >= oos_start with first_scored_week <= T - 52 weeks (the calibration window bound of vanta/m1b.py:68-74 and vanta/gate.py:59-74; same for all horizons; m1b has no code dropping early weeks, this is the lead's reading).
- `make_score_fn`: a row with every feature missing, or absent from the panel, scores NaN. `make_fit_fn` / `make_score_fn` raise on duplicate `(decision_date, symbol)` feature rows.
- Driver `scripts/run_baselines.py`: `indicator_panel(df, loader)` (Stage 9 indicators as of the decision date, computed from adjusted bars with `signals.indicators.compute_indicators`, no Postgres), `add_scorer_feature(ind)` (A1 `det_score`), `build_features(df, sessions)`. Features: Stage 12 blocks (no flows_dii) + `mom, liquidity_rank, univ_rank` + 9 scale-free indicators.

## 8. `vanta.m2` (Stage 13M, M2; row 15d; all names INVENTED, proposed by tests-m2 2026-11-01)

Tests: `tests/test_m2_invariants.py`, `tests/test_m2_driver.py` (reference toy in `tests/test_m2_toy.py`). Skip until the module exists.

- Constants: `Q, CAP, TAU_WEEKS, ESS_FLOOR, WINDOW_WEEKS, K_S, K_T, B1_SEEDS, IMPACT_FLOOR_TOP200, IMPACT_FLOOR_REST, ARMS, JUDGED_HORIZONS (5, 20), DIAGNOSTIC_HORIZONS (60)`.
- `nw_lag_weeks(h)`; `weekly_percentile(frame, score_col="b1")` (average ties, NaN stays NaN).
- `gate_arm(arm, cands, cal, T, pi_T, **gate_kwargs)` for arm in `ARMS` (a, b, c): the `run_gate` output columns plus `abstained`, `state_missing`.
- `run_m2_arms(cands, decision_dates, *, ess_floor=...)` -> dict a, b, c, ungated. `cands` needs `b1, y_net, fill_status, matured_at, regime_posterior, hmm_version`.
- `arm_counts(log)` (per week, not per row); `judged_window(scored, oos_start, window_weeks)`; `split_judged(log, first_judged_week)`.
- `go_criteria(log, ungated, decision_dates, h, *, first_judged_week, invariants_passed, n_boot, seed)` -> c1..c5 dicts, `nw_lag_weeks`, `nw_se`, `boot_ci95_block`, `all_pass`.
- `go_table(logs[impact][arm][h] = (log, ungated), decision_dates, first_judged_week, invariants_passed)`: 12 rows judged on the floor run, zero run reported; refuses h=60, unknown arms, unknown runs (ValueError).
- `diagnostics_table({(variant, h, arm): log})`: NOT judged (variants atr/hold, h 5/20/60); `diagnostics_13b(log, index_close, decision_dates)`.
- `hold_to_horizon_label`: the `simulate_trade` signature without `k_s`, `k_t` (A11).
- `report_header()`, `report_footer()`; `scripts.run_m2.build_parser()` exposes only `--horizon {5,20,60}`, `--impact {zero,floor}`, `--variant {atr,hold}`.
- Lead decisions: first judged week = `baselines.first_scored_and_judged_week`; arm (b) with no calibration rows abstains; NaN P(state 2) at T = missing -> arm (a) age-only handling, counted and flagged.

## 9. `vanta.m3`, `scripts.run_m3` (Stage 13M, M3; owner quant-m3; names proposed to test-m3 2026-11-02)

Authority: `docs/vanta/M3_PREREG.md`. Everything the pre-registration leaves unchanged is imported from `vanta.m2` / `vanta.baselines` / `vanta.m1`, not redefined.

`vanta.m3`:
- Constants: `JUDGED_HORIZONS (20, 60)`, `REPORTED_HORIZONS (5,)`, `ALL_HORIZONS (5, 20, 60)`, `IMPACTS`, `ARMS = m2.ARMS`, `Q, CAP, TAU_WEEKS, ESS_FLOOR, WINDOW_WEEKS` (= m2's), `MODEL_VERSION = "b1-hold@m3"`, `B1_HOLD_GRID` (is `baselines.B1_PARAM_GRID`).
- `B1_HOLD_PARAMS_BY_HORIZON` (keys 5, 20, 60). PLACEHOLDER (= `baselines.B1_PARAMS`) until the lead commits the tuned values from `data/processed/m3/b1hold_tuned_h{H}.json`.
- `hold_params(h)` (ValueError for another horizon; h=60 never falls back to h=20), `is_placeholder(h)`, `require_tuned_hold_params(h, mechanics=False)` (ValueError while placeholder; allowed with `mechanics=True`).
- `is_judged(impact, h)`: True only for `floor` at h in (20, 60); ValueError for another impact.
- `go_table(logs, decision_dates, first_judged_week, invariants_passed, n_boot, seed)`: `logs[impact][arm][h] = (gate_log, ungated_log)`, `decision_dates[h]`, `first_judged_week[h]` are dicts keyed by horizon. Criteria are `m2.go_criteria` (NW lag and bootstrap block ceil(h/5)); `go_all` is a bool only where `is_judged`, else None.
- `year_table(log)` (cols year, n_filled, n_losers, fdr, net_per_signal; filled published rows with a finite label), `publication_rate_by_year(log, decision_dates)`, `hold_label_source_fp()`, `report_header()` (M3_PREREG s1), `report_footer()`.

`scripts.run_m3` (outputs ONLY under `data/processed/m3[_tag]/`, never m2/, m1b/, baselines/):
- `--tune --horizon H`: B1 search on fold 0's inner split (labels and features built for decision dates before fold 0's test year only); writes `b1hold_tuned_h{H}.json`.
- `--score --horizon H`: walk-forward OOS B1 scores on hold labels; refused (SystemExit) while the horizon is a placeholder unless `--mechanics-check`; writes `scored_b1hold_h{H}.parquet`, `fold_report_b1hold_h{H}.json`, `report_b1hold_h{H}.md`.
- `--horizon H --impact {floor,zero}`: the cell (arms a, b, c and ungated): `log_h{H}_{impact}_{arm}.parquet`, `meta_h{H}_{impact}.json` (written last; skipped when its fingerprint matches).
- `--report`: `logs/m3_report.md` (also `m3_go_table.csv`, `m3_year_table.csv`).
- Helpers: `build_parser()`, `out_dir_for(tag)`, `stem(h, impact)`, `scored_name(h, mechanics)`, `hold_label_dir(h)`, `build_hold_labels(cands, h, sessions, terminal)`, `feature_panel(df, sessions, cand_name, out_dir)`, `_fingerprint`, `cell_is_current`, `report(tag, n_boot, invariants)`.
- A0 and A1 are not rerun for M3: A0 does not depend on labels (its M2 scores are unchanged), A1 is not part of the M3 pre-registration.

### 9a. `vanta.paper_log`, `scripts.run_paper_log` (forward paper log; contract = `tests/test_paper_log.py`, written by test-m3; implemented by quant-m3)

Files under `data/processed/paper_log/` (append-only JSON lines; the only way a file changes is one flushed append): `decisions.jsonl` (one row per candidate x horizon, no outcome field), `outcomes.jsonl` (matured, filled, finite outcomes), `unfilled.jsonl` (NO_FILL_*, NO_DATA), `runs.jsonl` (driver audit trail incl. refusals, code state, per-date block sha256), `models/` (live CatBoost models and annual HMM parameters, written once), `cache/`.

`vanta.paper_log`: `PaperLogError`, `StaleDataError`; `append_decisions(log_dir, decision_date, candidates, *, model_version, code_version, data_asof) -> "APPENDED"|"NOOP"` (candidates need `symbol`, `published`; optional `horizon` makes (symbol, horizon) the key; different content for a date on file raises; stale or unknown `data_asof` raises before anything is created); `append_outcomes(log_dir, labels, asof) -> int` (matured_at <= asof and finite y_net only, never twice; an outcome without a decision raises); `append_unfilled(...)`; `calibration_summary(log_dir, asof, window_weeks=52, min_rows=30)`; `calibration_frame(log_dir, horizon, asof)` (gate format, matured logged rows only, logged percentile and logged regime posterior); `calibration_status(log_dir, horizon, T)` (SUFFICIENT iff the first logged decision of the horizon is >= 52 weeks before T and a matured outcome is usable); `publications(horizon, cands, cal, T, pi_T, status)` (arms a, b, c via `m2.gate_arm`; not run, flags None, when INSUFFICIENT_CALIBRATION; ungated top 10 always); `refit_date(T, sessions)` (quarterly), `training_window`, `latest_complete_decision_date`, `freshness_problems`, `param_hash`, `code_version`, `code_version_string`, `read_decisions`, `read_outcomes`, `pending_keys`.
`scripts.run_paper_log`: `--stage {all,outcomes,models,decide}`, `--decision-date`, `--allow-late`, `--check`; exit 0 ok / NOOP, 2 placeholder params, 3 refused (reasons logged and in runs.jsonl).
