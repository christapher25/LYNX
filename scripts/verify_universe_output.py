import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

UNIVERSE_DIR = PROJECT_ROOT / "data" / "processed" / "universe"

files = sorted(UNIVERSE_DIR.glob("*.parquet"))
print(f"Files: {len(files)}, years {files[0].stem}-{files[-1].stem}\n")

violations = 0
for f in files:
    year = int(f.stem)
    cutoff = pd.Timestamp(f"{year}-01-01")
    df = pd.read_parquet(f)

    n_delisted = df["IS_DELISTED"].sum()
    print(f"{year}: {len(df):4d} symbols | {n_delisted:3d} delisted")

    # Self-consistency check: no symbol's FIRST_DATE can be >= this
    # universe's own cutoff — if the trailing-window ranking is correct,
    # this is structurally impossible. A violation here is a real bug.
    bad = df[df["FIRST_DATE"] >= cutoff]
    if not bad.empty:
        violations += len(bad)
        print(f"  VIOLATION: {len(bad)} symbols with FIRST_DATE >= {year} "
              f"cutoff but ranked into the {year} universe")
        print(f"    {bad[['SYMBOL', 'FIRST_DATE']].head().to_string(index=False)}")

    # Sanity: non-delisted rows must have null TERMINAL_VALUE
    bad_terminal = df[(~df["IS_DELISTED"]) & (df["TERMINAL_VALUE"].notna())]
    if not bad_terminal.empty:
        print(f"  VIOLATION: {len(bad_terminal)} active symbols carry a TERMINAL_VALUE")

print(f"\n{'PASS — no point-in-time violations' if violations == 0 else f'FAIL — {violations} violations'}")