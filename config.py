"""
LYNX canonical configuration. Every path, every environment-dependent
value, lives here and nowhere else. If a module needs to know where
something is, it imports from this file — it never constructs a path
itself.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; never commits secrets

# ── Root ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data ─────────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
CONSTITUENTS_DIR = DATA_ROOT / "constituents"

BHAVCOPY_DIR = RAW_DATA_DIR / "bhavcopy"
FNO_BHAVCOPY_DIR = RAW_DATA_DIR / "fno_bhavcopy"
FNO_BAN_DIR = RAW_DATA_DIR / "fno_ban"
NSDL_FPI_DIR = RAW_DATA_DIR / "nsdl_fpi"
SEC_LIST_DIR = RAW_DATA_DIR / "sec_list"             # daily security list: the price band per security per date
PRICE_BAND_HITS_DIR = RAW_DATA_DIR / "price_band_hits"   # PR-zip bh file: securities that hit their band that day
SHP_DIR = RAW_DATA_DIR / "shp"                       # per-symbol shareholding-pattern filings with promoter encumbrance
FII_DII_DIR = RAW_DATA_DIR / "fii_dii"
DELIVERY_DIR = RAW_DATA_DIR / "delivery"
ANNOUNCEMENTS_DIR = RAW_DATA_DIR / "announcements"

# Ingestion audit trail (one JSONL per dataset) and run logs.
MANIFEST_DIR = RAW_DATA_DIR / "_manifests"
LOG_DIR = PROJECT_ROOT / "logs"
# Measured publication times of daily sources (data/publication_monitor.py): append-only first-seen evidence.
PUBLICATION_LOG_DIR = RAW_DATA_DIR / "_publication"

CORPORATE_ACTIONS_LOG = PROCESSED_DATA_DIR / "corporate_actions.parquet"
PIT_FEATURE_STORE = PROCESSED_DATA_DIR / "pit_features.parquet"

# Stage 4 (adjusted prices, one file per symbol) and Stage 5 (PIT universe,
# one file per year) live as Parquet, not in Postgres.
ADJUSTED_PRICES_DIR = PROCESSED_DATA_DIR / "adjusted"
UNIVERSE_DIR = PROCESSED_DATA_DIR / "universe"
INDICATORS_CHECKPOINT = PROCESSED_DATA_DIR / "indicators_checkpoint.json"

# Stage 10 feature blocks: one Parquet file per symbol per block, row-aligned to that symbol's
# Stage 4 adjusted-price rows (same shape as ADJUSTED_PRICES_DIR).
FEATURE_BLOCKS_DIR = PROCESSED_DATA_DIR / "features"
DELIVERY_FEATURES_DIR = FEATURE_BLOCKS_DIR / "delivery"
FLOW_FEATURES_DIR = FEATURE_BLOCKS_DIR / "flows"
BREADTH_FEATURES_DIR = FEATURE_BLOCKS_DIR / "breadth"
PLEDGING_FEATURES_DIR = FEATURE_BLOCKS_DIR / "pledging"
PLEDGING_FEATURES_CHECKPOINT = PROCESSED_DATA_DIR / "pledging_features_checkpoint.json"
CIRCUIT_FEATURES_DIR = FEATURE_BLOCKS_DIR / "circuits"
CIRCUIT_FEATURES_CHECKPOINT = PROCESSED_DATA_DIR / "circuit_features_checkpoint.json"
DELIVERY_FEATURES_CHECKPOINT = PROCESSED_DATA_DIR / "delivery_features_checkpoint.json"

# ── Database ─────────────────────────────────────────────────────
# One path. Every module reads this, none construct their own.
# There is deliberately NO default. The old fallback (localhost:5432) is the native
# PostgreSQL_For_Odoo service, not the LYNX container (published on 5433), so a missing
# .env silently connected to the wrong database. The check is lazy (on access, not on
# import) because most stages are Parquet-only and import config for paths alone, and CI
# has no .env.
def _require_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Define it in .env (see .env.example); there is no default "
            "by design. On this machine it must point at the LYNX container on host port 5433."
        )
    return url


def __getattr__(name: str):
    """PEP 562: `config.DATABASE_URL` / `from config import DATABASE_URL` resolves (and can raise) lazily."""
    if name == "DATABASE_URL":
        return _require_database_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db_connection():
    """Open a psycopg2 connection to DATABASE_URL. Caller owns closing it."""
    url = _require_database_url()  # first: a missing URL must raise even where psycopg2 isn't installed
    import psycopg2  # lazy: keeps config importable where psycopg2 isn't installed (Kaggle/Colab)

    return psycopg2.connect(url)

# ── Models ───────────────────────────────────────────────────────
MODELS_ROOT = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_ROOT / "checkpoints"
GRAPH_SNAPSHOT_DIR = MODELS_ROOT / "graph" / "snapshots"

# ── Cost model ───────────────────────────────────────────────────
COST_RATES_FILE = PROJECT_ROOT / "costs" / "rates.yaml"

# ── Walk-forward ─────────────────────────────────────────────────
# Per-horizon embargo, in trading days. Never hardcode this elsewhere —
# Correction 2 in the v3 doc exists because a hardcoded embargo was
# too short for its label horizon.
EMBARGO_DAYS = {
    "intraday": 1,
    "swing": 5,
    "longterm": 60,
}

# ── Hardware ─────────────────────────────────────────────────────
LOCAL_GPU_VRAM_GB = 4.3
LLM_MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"

# ── Directory creation ───────────────────────────────────────────
def ensure_directories() -> None:
    """Call once at startup. Creates every directory this config
    references, if it doesn't already exist."""
    for path in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, CONSTITUENTS_DIR,
        BHAVCOPY_DIR, FNO_BHAVCOPY_DIR, FNO_BAN_DIR, DELIVERY_DIR, ANNOUNCEMENTS_DIR,
        CHECKPOINT_DIR, GRAPH_SNAPSHOT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATABASE_URL: {_require_database_url().rsplit('@', 1)[-1]}")  # host/db only, never the credentials
    print(f"EMBARGO_DAYS: {EMBARGO_DAYS}")
    print("All directories created/confirmed.")