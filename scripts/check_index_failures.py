import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

manifest_path = PROJECT_ROOT / "data" / "raw" / "_manifests" / "indices.jsonl"

latest = {}
with manifest_path.open() as f:
    for line in f:
        rec = json.loads(line)
        latest[rec["trade_date"]] = rec

failed = {d: r for d, r in latest.items() if r["status"] == "failed"}
print(f"Total failed (deduplicated): {len(failed)}\n")

error_counts = {}
for d, r in sorted(failed.items()):
    err = r.get("error", "")[:80]
    error_counts[err] = error_counts.get(err, 0) + 1

print("Error patterns:")
for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:3d}x  {err}")

print("\nSample failed dates with full errors:")
for d, r in sorted(failed.items())[:10]:
    print(f"  {d}: {r.get('error', '')}")