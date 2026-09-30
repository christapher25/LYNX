import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

CANDIDATES = [
    "https://nsearchives.nseindia.com/content/indices/ind_close_all_04092026.csv",
    "https://archives.nseindia.com/content/indices/ind_close_all_04092026.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
]

for url in CANDIDATES:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        ok = r.status_code == 200 and len(r.content) > 200
        print(f"{url}\n  -> {'OK' if ok else 'MISS'} ({r.status_code}, {len(r.content)} bytes)")
        if ok:
            print(f"  First 300 chars: {r.text[:300]}")
    except Exception as e:
        print(f"{url}\n  -> ERROR: {e}")
    print()