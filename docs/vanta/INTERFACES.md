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
