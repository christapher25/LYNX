# LYNX

An AI decision-support system that ranks the NSE top-700 equity universe across three horizons
(intraday, swing, long-term) and produces entry / target / stop-loss levels with explanations.

**LYNX never executes trades. It is signalling and information only.**

Final-year capstone project, B.Tech CSE (AI & ML), St. Thomas College of Engineering and Technology (APJ Abdul Kalam Technological University).

## Status

Stages 1–9 are complete and verified. Stage 10 (India-specific feature blocks) is in progress. Stages 11–26 have not started.
See `CLAUDE.md` for the detailed build status and `ARCHITECTURE.md` for the condensed architecture.
The canonical plan is `docs/LYNX-v4-Technical-Architecture-Plan.txt`.

## Layout

| Path | Contents |
|---|---|
| `config.py` | Single source of configuration (paths, `DATABASE_URL`, embargo) |
| `data/` | Ingesters (bhavcopy, delivery, indices, F&O, pledging, circuits), trading calendar |
| `features/blocks/` | Feature blocks (delivery, flows, breadth, pledging, circuits) |
| `signals/` | Technical indicator engine and deterministic scorer |
| `scripts/` | Stage checkpoint verifiers |
| `tests/` | Unit and invariant tests |
| `requirements/` | Dependency files (`base.txt`, `local.txt`, `cloud.txt`) |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements/base.txt
cp .env.example .env   # then fill in values; never commit .env
docker compose up -d   # Postgres, Redis, Neo4j
pytest tests/invariants/
```

Raw and processed data (`data/raw`, `data/processed`) are not stored in git.
