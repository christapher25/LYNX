import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Late Sept to early Oct 2019, trading days only
TEST_DATES = [
    "17092019", "20092019", "24092019", "27092019",
    "30092019", "01102019", "03102019", "04102019",
]

for d in TEST_DATES:
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        ok = r.status_code == 200 and len(r.text) > 1000
        status = f"OK ({len(r.text):,} bytes)" if ok else f"MISSING ({r.status_code})"
    except Exception as e:
        status = "ERROR (retry)"
    print(f"{d}: {status}")