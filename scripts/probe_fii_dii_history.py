"""Does fiidiiTradeReact accept date params, or is it today-only?"""

import requests
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
URL = "https://www.nseindia.com/api/fiidiiTradeReact"
REFERER = "https://www.nseindia.com/reports/fii-dii"


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass
    return s


def try_params(session, params, label):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    resp = session.get(URL, params=params, headers={"Referer": REFERER}, timeout=20)
    print(f"URL: {resp.url}")
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text[:200])
        return
    try:
        data = resp.json()
        print(f"Records: {len(data) if isinstance(data, list) else 'dict'}")
        if isinstance(data, list) and data:
            dates = {r.get("date") for r in data}
            print(f"Distinct dates returned: {sorted(dates)[:10]}")
    except json.JSONDecodeError:
        print("Not JSON:", resp.text[:200])


if __name__ == "__main__":
    s = get_session()
    try_params(s, {}, "BASELINE: no params")
    try_params(s, {"from": "01-09-2026", "to": "20-09-2026"}, "TEST: from/to")
    try_params(s, {"from_date": "01-09-2026", "to_date": "20-09-2026"}, "TEST: from_date/to_date")
    try_params(s, {"date": "15-09-2026"}, "TEST: single date")