# scripts/check_unexplained_vs_universe.py
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

ADJUSTED_DIR = PROJECT_ROOT / "data" / "processed" / "adjusted"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "processed" / "universe"

# All symbols that appear in ANY yearly universe snapshot
in_universe = set()
for f in UNIVERSE_DIR.glob("*.parquet"):
    df = pd.read_parquet(f, columns=["SYMBOL"])
    in_universe.update(df["SYMBOL"])

print(f"Symbols ever in-universe: {len(in_universe)}")

MAX_MOVE = 0.20
flagged_symbols = {}
for f in ADJUSTED_DIR.glob("*.parquet"):
    symbol = f.stem
    df = pd.read_parquet(f, columns=["DATE", "ADJ_CLOSE", "ADJUSTMENT_APPLIED"])
    returns = df["ADJ_CLOSE"].pct_change()
    bad = df[(returns.abs() > MAX_MOVE) & (~df["ADJUSTMENT_APPLIED"])]
    if not bad.empty:
        flagged_symbols[symbol] = len(bad)

total_flagged = len(flagged_symbols)
in_universe_flagged = sum(1 for s in flagged_symbols if s in in_universe)

print(f"Symbols with unexplained moves: {total_flagged}")
print(f"Of those, ever in-universe:     {in_universe_flagged}")
print(f"\nIn-universe flagged symbols (these actually matter):")
for s, n in sorted(flagged_symbols.items(), key=lambda x: -x[1]):
    if s in in_universe:
        print(f"  {s}: {n} unexplained moves")