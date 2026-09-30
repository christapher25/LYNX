"""
Probe 2: does the NSE corporate-actions endpoint support a historical
date range, or is it locked to a recent/upcoming default window?

The first probe returned only 20 records for Reliance, clustered around
2026 — that's consistent with either interpretation, so this test asks
explicitly for deep history and checks what actually comes back.
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
    session.get("https://www.nseindia.com", timeout=15)
    return session


def try_date_range(session, symbol, from_date, to_date, label):
    url = "https://www.nseindia.com/api/corporates-corporateActions"
    params = {
        "index": "equities",
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date,
    }
    resp = session.get(
        url, params=params,
        headers={"Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions"},
        timeout=15,
    )

    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"Response: {resp.text[:300]}")
        return

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("Not valid JSON:", resp.text[:300])
        return

    if isinstance(data, list):
        print(f"Records returned: {len(data)}")
        if data:
            dates = [r.get("exDate", "?") for r in data]
            print(f"Earliest exDate in results: {min(dates)}")
            print(f"Latest exDate in results:   {max(dates)}")
            print(f"\nSample record: {json.dumps(data[0], indent=2)}")
    else:
        print("Unexpected shape:", type(data), str(data)[:300])


if __name__ == "__main__":
    session = get_session()

    # Test 1: explicitly request a deep historical window
    try_date_range(session, "RELIANCE", "01-01-2000", "31-12-2005",
                    "TEST 1: Reliance, 2000-2005 (deep history)")

    # Test 2: a company with well-known, well-documented old splits/bonuses,
    # to cross-check against known public record
    try_date_range(session, "TCS", "01-01-2005", "31-12-2010",
                    "TEST 2: TCS, 2005-2010")

    # Test 3: no date params at all, to see the true default window size
    try_date_range(session, "RELIANCE", "", "",
                    "TEST 3: Reliance, no date params (baseline)")