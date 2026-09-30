import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Narrow the Jul-Oct 2019 window, month by month
TEST_DATES = [
    "01072019", "15072019",
    "01082019", "14082019",
    "02092019", "16092019",
    "01102019", "15102019",
]

for d in TEST_DATES:
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        ok = r.status_code == 200 and len(r.text) > 1000
        status = f"OK ({len(r.text):,} bytes)" if ok else f"MISSING ({r.status_code})"
    except Exception as e:
        status = f"ERROR: {e}"
    print(f"{d}: {status}")