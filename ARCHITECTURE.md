# LYNX Architecture — condensed reference (v4)

Working reference for Claude Code sessions. The canonical, full-detail
document is `docs/LYNX-v4-Technical-Architecture-Plan.txt` — if anything
here is ambiguous or seems to conflict with it, that file wins.

## PRISM — the six components

1. **Cross-market SSL pretraining.** Masked patch reconstruction (5-day
   patches, 40% masked) over India (~1,500 tickers) + US SignalForge corpus
   (~800 tickers), ~1M training sequences. Encoder: patch transformer,
   6 layers, d_model 256, 8 heads, ~5M params. **Must split by time, not
   ticker** — pretraining corpus for a fold testing year Y must end before
   Y minus embargo, or the encoder has seen the future of that fold.
2. **Multi-horizon task heads.** One shared trunk, four ranking heads
   (intraday/5d/20d/60d). LambdaLoss w/ NDCG weighting, trained @50,
   evaluated @10. Horizon weights: intraday 0.2, 5d 0.4, 20d 0.2, 60d 0.2.
3. **Heterogeneous temporal graph.** R-GAT, 2 layers. Six edge types:
   same_sector, same_promoter_group (India-specific — Tata/Reliance/Adani/
   Birla/Bajaj/JSW/Vedanta, ~35 groups), supply_chain (KG-derived),
   corr_knn (top-8 by 120d correlation), index_comembership, co_mention
   (GDELT-derived). Rebuilt monthly, trailing data only.
4. **Regime-specialised heads.** 3-state Gaussian HMM over India VIX, FII
   net flow, Nifty 500 breadth, USDINR change. Shared trunk, hard-routed
   separate output heads per regime state — **not** learned MoE gating.
   **Forward algorithm only at inference, never Viterbi.**
5. **Conformal abstention.** Mondrian-binned (by regime) split conformal,
   rolling 250-day window, α=0.1. Publish only if
   `conformal_lower_bound(i) > round_trip_cost(i)`. 0–10 names/day.
6. **Quantile trade levels.** Pinball loss, τ=0.1/0.5/0.9. Entry: intraday
   09:25 VWAP, swing/long-term next-day open. Stop-loss:
   `min(entry×(1+q10), entry − 1.5×ATR)`. Target: `entry×(1+q50)`, floored
   at 1.5R.

## Nine system layers

| L | Layer | Cadence | Note |
|---|---|---|---|
| 0 | Data | Nightly+bulk | bhavcopy, F&O, delivery%, flows, macro, GDELT, announcements |
| 1 | SSL pretraining | Per fold, offline | Encoder checkpoint |
| 2 | Features + scorer | Nightly | Deterministic scorer as feature + fallback, cross-sectionally z-scored |
| 3 | Agent extraction | Nightly | LangGraph, extractive only, never generative judgments |
| 4 | Knowledge graph | Nightly write, monthly rebuild | Neo4j |
| 5 | PRISM core | Nightly | **Only decision-maker** |
| 6 | Conformal | Nightly | Abstention, publish set |
| 7 | Explanation | Nightly | SHAP + LLM narration, grounded, hallucination forbidden |
| 8 | Delivery | Continuous | FastAPI, web + mobile (see Mobile section below) |
| 9 | Self-improvement | Weekly | Outcome recording, retrain, recalibration |

## 16-step pipeline (steps 1–2 start/collect are folded into Stage 0)

3 Corporate action adjustment → 4 Feature engineering/cross-sectional
z-scoring → 5 Deterministic override gates → 6 Agentic text extraction
(Qwen3-4B QLoRA + LangGraph) → 7 Knowledge graph construction (Neo4j) → 8
Time-series encoding (masked patches) → 9 Multi-hop graph fusion (R-GAT) →
10 Regime routing (HMM, forward-only) → 11 Multi-horizon ranking
(LambdaLoss) → 12 Trade sizing (pinball loss + ATR stop) → 13 Conformal
calibration + cost-aware abstention → 14 Explanation (SHAP + LLM) → 15
Delivery/logging/walk-forward validation → 16 Stop.

## Technology stack

Data: NSE bhavcopy (scraped), Kite Connect/Dhan/SmartAPI (intraday),
GDELT 2.0 GKG via BigQuery, NSE/BSE announcements, NSDL/RBI/MOSPI.
ML: PyTorch, PyTorch Geometric, LightGBM, hmmlearn (forward algorithm
only), MAPIE/custom split conformal, SHAP, Optuna.
Language: Qwen3-4B (confirmed, fits 4GB local GPU) + LoRA/QLoRA, LangChain,
LangGraph.
Memory: Neo4j, ChromaDB/pgvector, Groq API (explanation, off decision path).
Platform: Prefect, MLflow, PostgreSQL + Parquet + Redis, Docker Compose.
Compute: Kaggle T4/Colab (pretraining, LLM fine-tune — never co-resident
with local inference), local RTX 3050 Ti 4GB (supervised fine-tune,
nightly inference).
Delivery: FastAPI, Next.js + TradingView Lightweight Charts (web), React
Native/Expo (mobile — see below), WebAuthn + Argon2id (passkeys), Razorpay.
Deliberately absent: Kafka, Kubernetes, vector-DB SaaS, feature-store SaaS.

## Build order — 26 stages (update status inline as work progresses)

**Phase 0 — Foundation** [DONE]
1. Env/config/canonical paths — 2. Repo scaffold/CI/invariant test skeleton

**Phase 1 — Data foundation** [DONE]
3. Bhavcopy ingestion+backfill — 4. Corporate action engine — 5. Universe+PIT
constituents — 6. Transaction cost model (do early) — 7. Momentum baseline
GO/NO-GO gate (**GATE VERDICT = GO**, 4.69% net annualized spread — re-run 2026-09-30 against the refreshed
Stage 3 data, verdict unchanged, was ~4.60%)

**Phase 2 — Features** [Stages 8–9 done, Stage 10 in progress, 11–12 not started]
8. Index+macro pipeline — 9. Technical indicators + deterministic scorer
[DONE] — **10. India block (delivery/F&O/flows/pledging/circuits) (in progress: combined
checkpoint `scripts/verify_stage10.py` 4/5 PASS — delivery, F&O, pledging, circuits;
flows FAILS on a documented, un-backfillable publication-timing gap, not a code defect
— detail in CLAUDE.md)**
— 11. Override gates+circuit/ban filters — 12. PIT feature store

**Phase 3 — Harness and baselines** [not started]
13. Walk-forward harness (before any model) — 14. Baseline A0+A1
(SignalForge port) — 15. Baseline A2 (LightGBM lambdarank)

**Phase 4 — Language and knowledge** [not started]
16. Announcement ingestion+entity resolution — 17. GDELT BigQuery
extraction — 18. LLM extraction pipeline+LoRA fine-tune — 19. KG schema+
writers — 20. KG feature emission+graph edge derivation

**Phase 5 — PRISM** [not started]
21. SSL pretraining — 22. Multi-horizon heads+R-GAT+fine-tune — 23. Regime
HMM+specialised heads — 24. Conformal+quantile levels

**Phase 6 — Science and production** [not started]
25. Ablations A0–A12+era analysis (the thesis) — 26. Pipeline/API/clients/
outcome loop/deployment (mobile store submission is part of this stage)

## Invariants — never violate

Carried, unchanged: LLM never decides; no trade execution; every scheduled
job wrapped so one failure never stops the scheduler; secrets only via
`.env`→`config.py`; walk-forward on historical constituents only; every
signal stores its full feature snapshot as JSON; every signal carries a
disclaimer; one DB path/one model path in `config.py`.

New in v3: agents write to KG only, never to the signals table; KG facts
extractive with `source_span`; pretraining respects fold boundary; embargo
≥ label horizon per horizon (intraday 1, swing 5, long-term 60 — see
`config.py EMBARGO_DAYS`); NLP inference and model training never
co-resident on the 4GB card.

## The twelve silent failures (each needs its own invariant test)

1. Unadjusted corporate actions — test: no adjusted return outside price
   band without an action record
2. Graph leakage — test: every graph snapshot built from trailing windows only
3. Normalisation leakage — test: cross-sectional z-score per date, no
   full-history statistics
4. Viterbi at inference — test: assert forward algorithm only
5. Pretraining leakage — test: encoder checkpoint cutoff < fold test start − embargo
6. Graph fusion future-edge leak (Step 9) — test: synthetic future-dated
   edge must not change prediction vs. no-edge case
7. Regime routing future leak (Step 10) — test: regime label at time t uses
   only observations ≤ t
8. Horizon label off-by-one (Step 11) — test: hand-labelled toy sequence,
   each horizon head matches manually computed forward return
9. Quantile crossing (Step 12) — test: assert τ0.1 ≤ τ0.5 ≤ τ0.9 on every
   published signal; reject publication if violated. **Most likely of the
   new failures to actually occur** — pinball losses trained independently
   have no structural monotonicity guarantee.
10. Conformal calibration window leak (Step 13) — test: every calibration
    date strictly precedes earliest test-fold date
11. Cost-aware abstention wrong-side cost (Step 13) — test: cost function
    matches position side, output changes across STT regime-change dates
12. Ungrounded explanation (Step 14) — test: every named driver in
    generated text appears in SHAP feature list or retrieved evidence

## Mobile delivery (new in v4)

Trigger: supervisor requirement at Phase I interim presentation — LYNX
must ship on App Store and Play Store, not web-only. React Native/Expo,
one codebase two stores. FastAPI backend unchanged — mobile is a second
client of the same API, no decision logic in the client.
Sequencing: **backend first** (Layers 0–7 + Layer 8 API surface, fully
verified) — **then** web and mobile clients on top.
Store notes: both Apple and Google require an explicit financial-content
declaration at submission — ties into the unresolved SEBI RA registration
question (paid tiers + entry/target/SL = textbook recommendation trigger;
resolve before either store submission, not just before public launch).
Offline: cache last published signal set locally, show with timestamp; no
on-device computation ever (PRISM only runs server-side).
Descope order note: mobile is **no longer droppable** — it was v3's first
descope item, that line is removed in v4 since it's now mandatory.

## Team

Aaron Abraham Mathew, Abel Tijo, Anitta Stalen, Christapher Thomas Rajesh
(Reg. STC23AM020 — architecture/dev lead, individually responsible for
literature review base paper + first 4 related papers). Supervisor:
Dr Renisha P Salim.