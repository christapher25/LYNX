# M2 pre-registration (Stage 13M, replaces M1 as the gate test)

Written 2026-10-31 and committed BEFORE any Stage 14 run. Authority: the project owner's decisions of 2026-10-31 (amendment in the v6 document, section 11.5). Nothing below may be added to, removed or changed after any M2 or Stage 14 result has been seen. If something here proves impossible to run as written, the run stops and the owner decides; the file is not edited to fit the result.

## 1. Status of the evidence

The 2013-onward window was seen in M1 and again in M1b (both NO-GO: `logs/m1_report.md`, `logs/m1b_report.md`). M2 is therefore a re-test on a window already inspected, with a new score, not a fresh out-of-sample test. A pass here is weaker evidence than a pass on unseen data. The only truly fresh test is the forward paper-trading log.

## 2. What is being tested

Whether the VANTA gate (Part C) controls the realised false-discovery rate when the score carries real signal. The score is the Stage 14 B1 baseline, so any failure now belongs to the gate or to B1, not to momentum.

## 3. Score

- B1 = CatBoost on the same features and splits as the Stage 14 baselines, trained on net labels (NEL label simulator output, Stage 13A), for h = 5 and h = 20, as specified in v6 section 11.2 and Stage 14.
- Only walk-forward OUT-OF-SAMPLE scores are used: a score for decision date t comes from a model whose training labels were purged and embargoed before the test window per the Stage 13 harness (no pretraining or fitting across a fold boundary; N11 holds).
- B1 score for a candidate = the mean of the predictions of 5 CatBoost models trained with seeds 0, 1, 2, 3, 4 (fixed here; the seed list is not tuned).
- Hyperparameters of B1 are chosen once, using only the inner training folds of the earliest folds, and recorded in the Stage 14 code before the first full out-of-sample run. They are not changed after OOS results are seen.
- The gate consumes the weekly cross-sectional percentile rank of the B1 score among that week's candidates (average-rank ties), the same transform M1b applied to momentum, so calibration and test weeks stay comparable.
- The ungated comparator is the 10 candidates with the highest B1 percentile each week.

## 4. Unchanged from M1b (exactly)

- Candidates: every point-in-time universe name passing the F&O ban, upper and lower circuit and liquidity-cliff (top 200 by trailing turnover) gates that week; a missing ban or circuit flag excludes the name.
- Gates NOT applied (as in M1b): corporate-action frequency, pledging, news, volatility, drawdown, event proximity.
- Costs: the Stage 6 cost model plus the pessimistic impact floor (0.05% per side for rank 1-200, 0.20% for 201-700). The floor run is the judged run; the impact-zero run is reported.
- q = 0.30; publish at most 10 per week; Benjamini-Hochberg across all candidates at T.
- Calibration: all matured scored candidates of the last 52 weeks; maturity = the next session after the label ends.
- k_s = k_t = 1.5 (the values frozen in M1 on 2010-2012; not re-selected).
- Gate effective-sample-size floor: 200 (as in M1b).
- Loser = y_net <= 0 after costs; realised FDR counts FILLED published trades only; NO_FILL and NO_DATA are reported separately.
- Horizons h = 5 and h = 20, weekly decisions.
- Evaluation window: decision weeks from 2013-01-01 for which a B1 out-of-sample score exists and at least 52 weeks of matured calibration exist. The first judged week is reported. Earlier weeks are in-sample for reporting only and are not judged.
- Newey-West lag ceil(h/5) weekly rows; weekly block bootstrap (moving blocks of ceil(h/5) weeks at h = 20), both reported beside every headline mean.

## 5. The five GO criteria (fixed, judged separately for each arm, horizon h = 5 and h = 20, floor run, evaluation window)

1. All invariants N1-N13 pass (the invariant suite on the code that produced the run).
2. Realised FDR over the evaluation window <= q + 0.05 = 0.35.
3. No calendar year has realised FDR above 0.5.
4. The gate publishes in at least 20% of weeks.
5. Net return per published signal >= the ungated top 10 (same B1 score), reported with a weekly block-bootstrap 95% confidence interval of the difference.

No multiplicity correction is applied across the three arms and two horizons; each is reported with its own numbers and the GO/NO-GO is the owner's call. A table shows every arm x horizon against every criterion.

## 6. Gate arms: exactly three

All three use the age weights of the M1b age gate. "Filtered state" means the 13B filtered (forward-algorithm) posterior only; state 2 is the most stressed state (highest mean log India VIX), state index as in `vanta/regime.py`.

- **(a) Age x regime**, exactly `vanta.gate.run_gate` as in M1b: calibration weights exp(-age_weeks/tau) * exp(-L1(pi_i, pi_T)) with tau = 26 weeks and pi the filtered posterior; where no causal HMM fold or posterior exists, or a posterior is non-finite, or the Kish effective sample size of these weights is below 200, it falls back to age-only weights exp(-age_weeks/tau) (counted and reported as a share of weeks, as in M1b).
- **(b) Regime-conditional calibration**: calibration uses only matured rows (52-week window) from weeks whose argmax filtered state equals the argmax filtered state at T, with age-only weights exp(-age_weeks/tau), tau = 26. If the Kish effective sample size of those weights is below 200, the gate publishes nothing that week (abstains; counted). If the posterior at T is missing, the week falls back to arm (a)'s age-only calibration (counted and flagged, not treated as "no state").
- **(c) Arm (a) plus abstain**: as (a), and the gate publishes nothing in any week where the filtered probability of state 2 at T is >= 0.5. If that probability is missing at T, the week is handled as in (a) (age-only fallback), counted and flagged.

No other arm, variant, threshold, window length, k, q, floor or seed may be added or changed after results are seen. The v4 per-stock gate reconstruction is dropped from M2.

## 7. Diagnostics (reported, NOT judged; none enters a GO criterion)

The 13b diagnostics for every arm and the ungated comparator: weekly false-discovery proportion against the Nifty 50 weekly return (correlation; FDR in up versus down weeks), and realised FDR with loser = underperformed the Nifty 50 after costs. Also reported: B1 rank IC and net per published signal per fold (Stage 14), publication rate by year, the age-only fallback share, and the count of abstentions per arm.

## 8. Reporting

`logs/m2_report.md`. First paragraph: states section 1. Then each criterion with its number for each arm and horizon, comparators, diagnostics, not-applied gates, and the statement that realised FDR is an empirical number, not a guarantee, and q = 0.30 an engineering target. Then STOP; the GO/NO-GO is the owner's.
