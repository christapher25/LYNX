"""
Probe: NSE's own FII/DII trading activity endpoint.

NSE's fii-dii report page (nseindia.com/reports/fii-dii) is a dynamic
page like corporate-actions was — testing the likely underlying API
endpoint before falling back to any third-party aggregator, which would
be a step down from every other primary source used tonight.
"""

import requests
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass
    return session


def try_endpoint(session, url, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"URL: {url}")
    resp = session.get(
        url, headers={"Referer": "https://www.nseindia.com/reports/fii-dii"}, timeout=20
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Response: {resp.text[:300]}")
        return
    try:
        data = resp.json()
        print(f"Type: {type(data)}")
        if isinstance(data, list) and data:
            print(f"Records: {len(data)}")
            print(json.dumps(data[0], indent=2))
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print(json.dumps(data, indent=2)[:1000])
    except json.JSONDecodeError:
        print("Not JSON:", resp.text[:300])


if __name__ == "__main__":
    session = get_session()

    # A few plausible endpoint names based on NSE's established /api/
    # naming pattern (confirmed working for corporate-actions earlier)
    try_endpoint(session, "https://www.nseindia.com/api/fiidiiTradeReact",
                 "TEST 1: fiidiiTradeReact")
    try_endpoint(session, "https://www.nseindia.com/api/fii-dii-trading-activity",
                 "TEST 2: fii-dii-trading-activity")