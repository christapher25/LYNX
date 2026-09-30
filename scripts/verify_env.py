import sys, torch, importlib

REQUIRED = ["torch", "torch_geometric", "lightgbm", "hmmlearn", "shap",
            "mapie", "pandas", "numpy", "pyarrow", "psycopg2",
            "sklearn", "dotenv", "requests"]

print(f"Python: {sys.version.split()[0]}")
assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"

for pkg in REQUIRED:
    m = importlib.import_module(pkg)
    print(f"  {pkg}: {getattr(m, '__version__', 'ok')}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram:.1f} GB")
    assert vram > 3.5, "Expected ~4GB card"

print("ALL CHECKS PASSED")