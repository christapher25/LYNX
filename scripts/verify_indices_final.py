import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

INDICES_DIR = PROJECT_ROOT / "data" / "raw" / "indices"

files = sorted(INDICES_DIR.glob("*.parquet"))
print(f"Files: {len(files)}, {files[0].stem} to {files[-1].stem}\n")

# Confirm Nifty 50/Bank present from day 1, VIX only from its real start
for label, f in [("First file", files[0]), ("Last file", files[-1])]:
    df = pd.read_parquet(f)
    names = set(df["INDEX_NAME"])
    print(f"{label} ({f.stem}): Nifty 50={('Nifty 50' in names)}  "
          f"Nifty Bank={('Nifty Bank' in names)}  India VIX={('India VIX' in names)}")

# When does VIX actually start appearing, precisely
vix_first = None
for f in files:
    df = pd.read_parquet(f, columns=["INDEX_NAME"])
    if "India VIX" in set(df["INDEX_NAME"]):
        vix_first = f.stem
        break
print(f"\nIndia VIX first appears: {vix_first}")