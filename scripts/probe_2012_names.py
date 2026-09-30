import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
r = requests.get(
    "https://nsearchives.nseindia.com/content/indices/ind_close_all_21022012.csv",
    headers=HEADERS, timeout=15,
)
lines = r.text.strip().split("\n")
print(f"Total rows: {len(lines) - 1}\n")
print("First 15 index names in 2012 file:")
for line in lines[1:16]:
    print(f"  {line.split(',')[0]}")

print("\nSearching for anything containing NIFTY, BANK, VIX, CNX (case-insensitive):")
for line in lines:
    name = line.split(",")[0]
    if any(k in name.upper() for k in ["NIFTY", "BANK", "VIX", "CNX"]):
        print(f"  {name}")