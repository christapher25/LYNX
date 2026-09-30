import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Multiple dates per year around the suspected boundary
TEST_DATES = [
    # 2020 - confirmed working, check early months
    "02012020", "03022020", "02032020",
    # 2019 - suspected boundary year, spread across it
    "01022019", "01042019", "01072019", "01102019", "02122019",
    # 2018 - should fail if boundary is 2019/2020
    "01022018", "02072018", "03122018",
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