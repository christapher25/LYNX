import requests
from datetime import date, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0"}
URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{}.csv"


def check(d: date) -> bool:
    resp = requests.get(URL.format(d.strftime("%d%m%Y")), headers=HEADERS, timeout=10)
    return resp.status_code == 200 and len(resp.content) > 200


# Binary search between confirmed-missing 2010-01-02 and confirmed-OK 2015-01-02
lo, hi = date(2010, 1, 2), date(2015, 1, 2)
print(f"Searching between {lo} (missing) and {hi} (OK)...")

while (hi - lo).days > 1:
    mid = lo + (hi - lo) // 2
    # snap to nearest weekday to avoid wasting probes on weekends
    while mid.weekday() >= 5:
        mid += timedelta(days=1)
    ok = check(mid)
    print(f"  {mid}: {'OK' if ok else 'MISSING'}")
    if ok:
        hi = mid
    else:
        lo = mid

print(f"\nBoundary: last MISSING = {lo}, first OK = {hi}")