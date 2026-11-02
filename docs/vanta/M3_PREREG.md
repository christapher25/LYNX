# M3 pre-registration (Stage 13M, replaces M2 as the gate test)

Written 2026-11-02 and committed BEFORE any M3 training, tuning or run. Authority: the project owner's decisions of 2026-11-02 (v6 amendment of the same date). Nothing below may be added to, removed or changed after any M3 result has been seen. If something here proves impossible to run as written, the run stops and the owner decides; the file is not edited to fit the result.

## 1. Status of the evidence

The 2016-onward judged window was seen in M1, M1b and M2, so M3 is a re-test, not a fresh out-of-sample test. The choice of hold-to-horizon labels was itself made after the M2 diagnostics (A11) on that same window, which adds a selection effect: a pass here is weaker evidence than a pass on unseen data. The forward paper log (`data/processed/paper_log/`, started 2026-11-02) is the only fresh test.

## 2. What is being tested

Whether the VANTA gate (Part C) controls the realised false-discovery rate when the score is trained on, and the trade is simulated as, hold to horizon. Per the A11 kill rule (v6 amendment 2026-11-02), ATR stop and target exits are not part of M3.

## 3. Labels

- Hold-to-horizon net labels: entry at the next session's open, exit at the horizon-end close (no stop, no target), same cost model, same impact floor, same circuit/lock rules, same NO_FILL and NO_DATA rules as the Stage 13A simulator (`vanta.m2.hold_to_horizon_label`, which calls the same simulator with infinite stop and target distances). Loser = y_net <= 0 after costs.
- Horizons: 20 and 60 are JUDGED. Horizon 5 is REPORTED, NOT JUDGED.

## 4. Score

- B1 = CatBoost, same features, folds (4 y train / 1 y validation / 1 y test, purged and embargoed per the Stage 13 harness, embargo at least the label horizon per horizon) and 5 seeds (0, 1, 2, 3, 4, mean) as in M2, RETRAINED on the hold-to-horizon net labels. N11 holds; no fitting across a fold boundary.
- Hyperparameters are tuned once per horizon, on fold 0's inner split only (training and validation of fold 0; fold 0's test year and everything later untouched), with the same search grid and selection rule as the M2 tuning (`vanta/baselines.py`, tuning driver). h = 60 is tuned on its own fold 0 and does not reuse h = 20. h = 5 is tuned the same way. The chosen values are committed in the code before the first out-of-sample run and are not changed afterwards.
- The gate consumes the weekly cross-sectional percentile rank of the B1 score among that week's candidates (average-rank ties). The ungated comparator is the 10 highest B1 percentile candidates each week.

## 5. Unchanged from M2 (exactly)

Candidates and gates (applied and not applied), costs and impact floor (floor run judged, zero run reported), q = 0.30, cap 10, Benjamini-Hochberg across all candidates at T, 52-week calibration window with maturity = the next session after the label ends, k_s and k_t are not used (no levels), Kish ESS floor 200, tau = 26 weeks, weekly decisions, filled-trades-only FDR, Newey-West lag ceil(h/5) weekly rows and weekly moving-block bootstrap with blocks of ceil(h/5) weeks (so 4 weeks at h = 20 and 12 weeks at h = 60), exactly the three M2 gate arms (a) age x regime, (b) regime-conditional calibration, (c) arm (a) plus abstain when the filtered P(state 2) at T >= 0.5, with filtered (forward-algorithm) posteriors only.

**Judged window:** the M2 rule: judged weeks run from the first B1 test fold (about 2016) onward, limited to weeks where a B1 out-of-sample score exists and 52 weeks of matured calibration exist; the first judged week is reported per horizon. Earlier weeks are reported only.

## 6. The five GO criteria (unchanged; judged separately for each arm at h = 20 and h = 60, floor run)

1. All invariants N1-N13 pass.
2. Realised FDR over the judged window <= 0.35.
3. No calendar year has realised FDR above 0.5.
4. The gate publishes in at least 20% of weeks.
5. Net return per published signal >= the ungated top 10 (same B1 score), with a weekly block-bootstrap 95% confidence interval of the difference.

No multiplicity correction across arms and horizons. A table shows every arm x horizon against every criterion. h = 5 rows are shown in a separate table, marked NOT JUDGED.

## 7. Reported, NOT judged

- Per-year realised FDR and per-year net per signal for every arm and the ungated comparator.
- FDR and net against the Nifty 50 (loser = underperformed the Nifty 50 over the same holding window, after costs), and the up-week versus down-week split (weekly false-discovery proportion against the Nifty 50 weekly return: correlation, FDR in up versus down weeks).
- The ungated comparator for every judged and reported cell.
- B1 rank IC and net per published signal per fold, publication rate by year, age-only fallback share, abstention counts, NO_FILL and NO_DATA counts, unscored counts.

## 8. Fixed

No arm, variant, threshold, window length, q, floor, seed or label variant may be added or changed after results are seen. A different label (for example the ATR levels, or a different holding rule) would be a new pre-registration, M4.

## 9. Reporting

`logs/m3_report.md`. First paragraph: states section 1. Then each criterion with its number for each arm and horizon, the comparators, the reported-not-judged items, the gates not applied, and the statement that realised FDR is an empirical number, not a guarantee, and q = 0.30 an engineering target. Then STOP; the GO/NO-GO is the owner's.
