import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{}.csv"

# Check today's file for which indices are actually present
r = requests.get(URL.format("04092026"), headers=HEADERS, timeout=15)
lines = r.text.strip().split("\n")
print(f"Total indices in today's file: {len(lines) - 1}")
print("\nSearching for VIX / BANK / NIFTY 500:")
for line in lines:
    name = line.split(",")[0]
    if any(k in name.upper() for k in ["VIX", "BANK", "NIFTY 500", "NIFTY 50"]):
        print(f"  {line[:80]}")

# Historical depth probe — same pattern as every other date-boundary
# search tonight
print("\n\nHistorical depth probe:")
TEST_DATES = [
    "02012024", "02012020", "02012015", "02012010", "02012005",
    "02012000", "03011997", "03111994",
]
for d in TEST_DATES:
    try:
        resp = requests.get(URL.format(d), headers=HEADERS, timeout=10)
        ok = resp.status_code == 200 and len(resp.content) > 200
        print(f"  {d}: {'OK' if ok else f'MISSING ({resp.status_code})'}")
    except Exception as e:
        print(f"  {d}: ERROR {e}")