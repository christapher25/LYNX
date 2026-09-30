"""
Probe: NSE historical index data (NIFTY 50, BANK NIFTY, INDIA VIX).

Testing the likely endpoint pattern based on how corporate-actions'
endpoint was structured (same domain, same /api/ prefix) — not assumed
correct, verified here before any ingester is built against it.
"""

import requests
import json

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass
    return session


def try_index(session, index_name, from_date, to_date, label):
    url = "https://www.nseindia.com/api/historical/indicesHistory"
    params = {"indexType": index_name, "from": from_date, "to": to_date}
    resp = session.get(
        url, params=params,
        headers={"Referer": "https://www.nseindia.com/reports-indices-historical-index-data"},
        timeout=20,
    )

    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"URL: {resp.url}")
    print(f"Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"Response: {resp.text[:400]}")
        return

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("Not valid JSON:", resp.text[:400])
        return

    print(f"Top-level type: {type(data)}")
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        print(json.dumps(data, indent=2)[:2000])


if __name__ == "__main__":
    session = get_session()

    try_index(session, "NIFTY 50", "01-01-2024", "31-01-2024", "TEST 1: NIFTY 50, Jan 2024")
    try_index(session, "NIFTY BANK", "01-01-2024", "31-01-2024", "TEST 2: BANK NIFTY, Jan 2024")
    try_index(session, "INDIA VIX", "01-01-2024", "31-01-2024", "TEST 3: INDIA VIX, Jan 2024")