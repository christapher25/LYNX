import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Probe one date per year, working backwards
TEST_DATES = [
    "01072024", "03012023", "03012022", "04012021", "02012020",
    "02012019", "02012018", "02012017", "01012016", "01012015",
    "02012014", "02012013", "02012012", "03012011", "04012010",
]

for d in TEST_DATES:
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        status = "OK" if r.status_code == 200 and len(r.text) > 1000 else f"MISSING ({r.status_code})"
    except Exception as e:
        status = f"ERROR: {e}"
    print(f"{d}: {status}")