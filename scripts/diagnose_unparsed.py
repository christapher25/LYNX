# scripts/diagnose_unparsed.py
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "corporate_actions"

frames = []
for f in OUTPUT_DIR.glob("*.parquet"):
    frames.append(pd.read_parquet(f))

df = pd.concat(frames, ignore_index=True)
failed = df[df["parse_confidence"] == "failed"]

counts = failed["raw_subject"].value_counts()
print(f"Total actions: {len(df)}")
print(f"Failed: {len(failed)} ({100*len(failed)/len(df):.1f}%)")
print(f"\nTop 40 unparsed subject patterns:\n")
for subject, count in counts.head(40).items():
    print(f"{count:5d}  {subject}")