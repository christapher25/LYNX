# VANTA specification (extracted from the v6.0 architecture)

Source: `docs/LYNX-v6-Final-Architecture-and-Setup-Documentation.pdf` (text copy: `docs/LYNX-v6-Final-Architecture-and-Setup-Documentation.txt`), Sections 9, 10.2, 11.5, 12.3, 15.1 and Appendix A. If this file and the v6 document disagree, the v6 document wins.

VANTA is the decision pipeline of LYNX: net labels → ranking → selection gate → ATR trade levels → tiers. It is a named pipeline of established methods, not a new algorithm. It replaces PRISM's labels, ranking loss and publish gate; the encoders, graph, data layer and language layer stay.

## Components and kill rules

| Part | Component | Kept only if |
|---|---|---|
| A | NEL, net executable labels | Net return per published signal beats gross labels (A2) |
| B | VN-ListNet, volatility-normalised ranking with soft regime gating | Beats plain ListNet (A3) and is not beaten by B1 (CatBoost) through the same gate; otherwise B1 becomes the ranker and the result is reported |
| C | RW-CS, regime-weighted conformal selection | Its realised FDR by regime is closer to q than age-only and uniform weights (A16) |
| C | FCC, full-candidate calibration | Always kept: a correctness rule (N8). A17 measures the bias published-only calibration would add |
| D | ATR trade levels | A11 shows stops and targets do not destroy expected value |

Every component's settings are chosen on validation folds only and recorded in the version tags.

## Part A: NEL, net executable labels (Stage 13A)

For every (stock i, decision date t) the system simulates the exact trade that would have been published at t.

| Rule | Specification |
|---|---|
| Decision time | After the close of t. The weekly decision date is the last session of the week. |
| Entry | Raw open of session t+1. **No fill** if the stock opens locked at its upper circuit: the symbol is on that day's price-band hit list as an **upper** hit **and** raw open equals raw high. |
| Stop | Entry minus k_s × ATR(14) at t. If price gaps through the stop at the open, the fill is at the **open** (exception: see next row, gap-down into a lock). |
| Gap-down into a lock | Resolves the Stop and Lower-circuit lock rows (user decision 2026-10-29). If the gap-down session (open <= stop) is itself lower-locked, the exit is at the close of the first session that is not lower-locked, `exit_reason = STOP_DELAYED_LOWER_LOCK`, delay recorded. If the gap-down session is not locked, the exit is at the open, `exit_reason = STOP_GAP`. |
| Target | Entry plus k_t × ATR(14) at t. If price gaps through the target, the fill is at the **target** (conservative, no windfall). |
| Same-day ambiguity | If stop and target both sit inside one day's range, assume the **stop** hit first. |
| Lower-circuit lock | A session is lower-locked at close if the symbol is on the hit list as a **lower** hit and raw close equals raw low. A stop triggered inside a lock exits at the **close of the first session that is not lower-locked**, and the delay is recorded. |
| Time exit | Close of session t+h. |
| Corporate actions | Returns use Stage 4 adjusted prices. Lock detection always uses **raw** prices against the band. |
| Delisting | Exit at the last traded price (the Stage 5 terminal value). |
| Cost | Stock-specific round trip from the Stage 6 cost model at **trade-date** rates, plus impact cost. Entry and exit legs stored separately. |
| Label | y = payoff minus round-trip cost (a net return). |

- Horizons h: 5, 20 and 60 sessions, daily bars.
- k_s and k_t are chosen per horizon on validation folds only, then frozen per fold and versioned.
- Labels are policy-free: no incumbency inside labels. Every row is a full round trip; cost_entry is stored separately; the accounting layer charges the entry leg once per position (N12).
- No-fill handling is a training configuration (default: a penalty value), not a stored value.
- Lock detection uses only the hit lists (inferred bands measured 87.8% accurate). Rows with no hit-list coverage (`lock_source = NONE`) are excluded from primary results.

## Part B: VN-ListNet (Stage 22, after encoders)

1. Warm start: ridge regression or a small MLP on the net labels, on encoder outputs (raw feature matrix for baselines).
2. Fine-tune: ListNet (Plackett-Luce) loss over each decision date's candidate list. Gain g = y / (ATR% × √h), winsorised at the 2nd and 98th percentiles **of the training fold only**.
3. Soft regime gating: score = Σ_k π_t(k) · head_k(z), with π_t the filtered regime posterior. The weights are posterior probabilities, not learned parameters.

## Part C: RW-CS and FCC, the publish gate (Stage 13C)

**Calibration set at decision date T:** out-of-sample (score, realised net return) pairs for **all scored candidates** from the last 52 weeks, not only published ones (N8). A row enters only after its horizon has passed and its outcome is known (N5). For h = 60 the effective window is shorter.

**Weights:** w_i = exp(−age_i / τ) × exp(−‖π_i − π_T‖₁). Age in weeks; π is the 3-state filtered posterior; L1 distance. The test point carries weight 1. τ is chosen per horizon on validation folds from {13, 26, 52} weeks (default 26), frozen and recorded in `gate_version`.

**p-value** for candidate j (null: the trade loses after cost, y ≤ 0):

p_j = (1 + Σ_i w_i · 1[y_i ≤ 0 and s_i ≥ s_j]) / (1 + Σ_i w_i)

(Clipped-score conformal p-value of Jin and Candès, conservative tie handling, weighted.) Computation: sort calibration scores once per date, running sum of weights of losing rows: O((n + m) log n).

**Selection:**
1. Benjamini-Hochberg at q = 0.3 across all candidates for date T.
2. Keep at most 10 names, ascending p-value, ties broken by score.
3. Zero published names is valid.

**q is an engineering target, never a guarantee.** Measured as realised FDR = mean weekly false-discovery proportion (zero-publish weeks count as 0), reported next to the pooled loss share.

**Fallback:** log the Kish effective sample size (Σw)² / Σw² for every decision. Below a threshold (proposed 200) the gate falls back to age-only weights and logs that it did.

## Part D: trade levels

Published levels are the same ATR multiples as the labels: entry at next open, stop = entry − k_s·ATR, target = entry + k_t·ATR. Quantile heads may appear only as a displayed range (A11), with the quantile-crossing check.

## Walk-forward rules the tests rely on

- Rolling 4 years train / 1 validate / 1 test, rolled yearly.
- Embargo ≥ label horizon, per horizon (5, 20, 60), a config parameter. **Purging:** drop training rows with `label_end_date ≥ test_start − embargo`. Multi-horizon training uses the longest horizon.
- Normalisation, winsorisation, k_s/k_t, the HMM and gain scaling are fitted on the training fold only.

## Invariant tests N1–N14

| ID | Invariant | Test |
|---|---|---|
| N1 | No fill on a locked open | A day on the hit list as upper with open equal to high gives `NO_FILL_UPPER_LOCK`, no exit price and no payoff. |
| N2 | Stops respect lower-circuit locks | Stop triggers on day d. Days d and d+1 are lower-locked at close, d+2 is not. Exit is at d+2's close with `STOP_DELAYED_LOWER_LOCK` and a delay of 2. |
| N3 | Same-day ambiguity resolves to the stop | Both levels inside one day's range gives exit at the stop. |
| N4 | Gap fills | A gap below the stop fills at the open. A gap above the target fills at the target. Exception: a gap-down session that is itself lower-locked exits at the first non-locked close (`STOP_DELAYED_LOWER_LOCK`), see Gap-down into a lock. |
| N5 | The calibration set never holds an unmatured outcome | Every calibration row has `matured_at ≤ T`. An injected row with `matured_at = T+1` is caught. Adding outcomes after T never changes p-values at T. |
| N6 | Labels use the cost model of the trade date | Output changes across a known STT rate-change date. |
| N7 | Regime weights use filtered posteriors | The posterior at t computed on data up to t equals the posterior computed on the full series. Appending the future changes nothing. |
| N8 | Calibration covers all scored candidates | Calibration row count for a week equals candidates scored, not names published. |
| N9 | Calibration and test share model version, candidate filter and OOS status | A mismatched row is rejected. |
| N10 | Normalisation and winsorisation come from the training fold only | Changing test-fold labels does not change training gains. |
| N11 | Purging | No training label with `label_end_date ≥ test_start − embargo`. |
| N12 | Cost is charged once for incumbents | A name selected in weeks w and w+1 shows one entry leg and one exit leg in the ledger. |
| N13 | p-value mechanics | p ∈ (0, 1] and non-increasing in score. Equal weights reduce to the unweighted conformal p-value. Published names ⊆ BH set, at most 10, with `cap_binding` logged. |
| N14 | WFP checkpoint lineage | Every ancestor of a fold's encoder checkpoint has a data cutoff ≤ that fold's test start − embargo. A lineage with a later cutoff is rejected. |

Stage checks: 13A must pass N1–N4, N6, N11 (plus a coverage report by year and 20 sampled trades reconciled by hand). 13B must pass N7. 13C must pass N5, N8–N10, N13 (and on synthetic exchangeable data, realised FDR ≤ q).

## Data contracts (Parquet, append-only; every row carries the version tags)

**vanta_labels** (partitioned by horizon and year): `symbol` str; `decision_date` date (features and ATR as of t's close); `horizon` int (5, 20, 60); `label_version` str (encodes k_s, k_t, tie rule, lock rules); `cost_model_version` str; `entry_date` date (t+1); `entry_price_raw` float (NaN if no fill); `fill_status` enum {FILLED, NO_FILL_UPPER_LOCK, NO_FILL_NO_TRADE, NO_DATA}; `atr_pct` float (ATR(14)/close at t); `k_s`, `k_t` float; `stop_price`, `target_price` float; `exit_date` date; `exit_price` float; `exit_reason` enum {TARGET, STOP, STOP_GAP, STOP_DELAYED_LOWER_LOCK, TIME, DELISTED}; `lock_delay_sessions` int (0 unless delayed by a lower lock); `lock_source` enum {HIT_LIST, NONE} (NONE excluded from primary results); `gross_return` float (adjusted-price return); `cost_entry`, `cost_exit`, `cost_impact` float; `y_net` float (gross minus all costs); `label_end_date` date (= exit_date, used for purging). Winsorisation and the gain g are computed at training time from fold-only statistics and never stored.

**vanta_calibration:** `decision_date`, `symbol`, `horizon`, `score` (out-of-sample), `model_version`, `fold_id`, `is_oos` (must be true), `regime_posterior` float[3] (filtered π at decision_date), `hmm_version`, `y_net` (null until matured), `matured_at` (label end close plus publication lag; null until matured), `candidate_filter_version`. Query rule at T: `matured_at ≤ T AND decision_date ≥ T − 52 weeks AND model_version == current AND candidate_filter_version == current AND is_oos`. A newly trained model writes its out-of-fold scores for the past 52 weeks before the gate runs.

**vanta_gate_log** (immutable, one row per decision date × candidate): `decision_date`, `symbol`, `horizon`, `score`, `p_value`, `bh_threshold`, `in_bh_set`, `published`, `cap_binding`, `q`, `n_calibration`, `effective_n`, all version tags; `y_net` and `lost` added once matured. Realised FDR and pooled loss share are computed from this table only.

**Version tags:** `{component}@{fold_id}:{train_end}:{git_sha}:{config_hash}`, e.g. `vanta-ranker@f07:2018-12-31:a1b2c3d:9f2e01`. Tagged: label_version, cost_model_version, encoder_version (incl. WFP parent), model_version, hmm_version, gate_version (q, cap, τ, distance), candidate_filter_version. Registered in MLflow.

## Milestone M1 (Stage 13M)

Momentum 12-1 scores recomputed weekly, through NEL and the gate, on real NSE data, before any new model. Horizons 5 and 20. k_s ∈ {1.0, 1.5, 2.0}, k_t ∈ {1.5, 2.0, 3.0} chosen on the first three years only, then frozen. Window starts at the first price-band hit list on disk. Gate arms: age-only weights and age × regime weights. Compared with the ungated weekly top 10 and the v4 per-stock gate.

Pre-registered GO criteria: (1) N1–N13 pass on real data; (2) realised FDR ≤ q + 0.05 (0.35) over the window; (3) no calendar year above 0.5; (4) the gate publishes in ≥ 20% of weeks; (5) gated net return per signal ≥ ungated top 10, with a weekly block-bootstrap 95% CI. If 2 or 3 fail, fix the gate before any model is trained.
