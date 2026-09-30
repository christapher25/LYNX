"""
Probe: verify NSE's corporate actions API before building the real ingester.

Corporate actions live on a different NSE surface than bhavcopy/delivery —
the dynamic API (nseindia.com/api/...) rather than the static archive
(nsearchives.nseindia.com). The dynamic API requires an established browser
session (real cookies) before it responds, and is more prone to blocking
under repeated requests than the archive files.

This script does the minimum needed to see the REAL response structure —
one company, one request — before any production code is written against
an assumed format.
"""

import requests
import json

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_session() -> requests.Session:
    """
    NSE's dynamic API rejects requests with no prior session — it expects
    the cookies a real browser would have picked up from loading the site
    first. This visits the homepage to collect those cookies before the
    real request.
    """
    session = requests.Session()
    session.headers.update(BASE_HEADERS)

    resp = session.get("https://www.nseindia.com", timeout=15)
    print(f"Homepage visit: {resp.status_code}, "
          f"{len(session.cookies)} cookies collected")

    return session


def probe_corporate_actions(session: requests.Session, symbol: str = "RELIANCE"):
    """
    Try the endpoint pattern used by NSE's own corporate-actions page.
    We're checking: does it respond, what shape is the data, does it
    give history or only upcoming events.
    """
    url = "https://www.nseindia.com/api/corporates-corporateActions"
    params = {"index": "equities", "symbol": symbol}

    print(f"\nRequesting: {url}")
    print(f"Params: {params}")

    resp = session.get(url, params=params, headers={"Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions"}, timeout=15)

    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")

    if resp.status_code != 200:
        print(f"Response text (first 500 chars): {resp.text[:500]}")
        return

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("Response was not valid JSON.")
        print(f"Raw text (first 500 chars): {resp.text[:500]}")
        return

    print(f"\nResponse type: {type(data)}")
    if isinstance(data, list):
        print(f"Number of records: {len(data)}")
        if data:
            print("\nFirst record:")
            print(json.dumps(data[0], indent=2))
            print("\nField names across first record:")
            print(list(data[0].keys()))
    elif isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        print(json.dumps(data, indent=2)[:1500])


if __name__ == "__main__":
    session = get_session()
    probe_corporate_actions(session, symbol="RELIANCE")