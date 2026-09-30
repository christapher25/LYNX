import requests
from datetime import date, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0"}
NEW_URL = ("https://nsearchives.nseindia.com/content/cm/"
           "BhavCopy_NSE_CM_0_0_0_{}_F_0000.csv.zip")

# Walk back from today until we find the most recent available file
today = date.today()
print(f"Today: {today}\n")

for days_back in range(0, 15):
    d = today - timedelta(days=days_back)
    if d.weekday() >= 5:
        print(f"{d} ({d.strftime('%a')}): weekend, skipping")
        continue
    url = NEW_URL.format(d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        ok = r.status_code == 200 and len(r.content) > 500
        status = f"OK ({len(r.content):,} bytes)" if ok else f"MISSING ({r.status_code})"
    except Exception as e:
        status = f"ERROR: {e}"
    print(f"{d} ({d.strftime('%a')}): {status}")