"""
Probe: niftyindices.com index historical data — the confirmed real
endpoint, not the nseindia.com guess that 503'd.

Confirmed via three independent sources: POST to an ASP.NET
Backpage.aspx ScriptService, body is {"cinfo": "<single-quoted string>"}
(not standard JSON nesting — an ASP.NET PageMethods convention), response
comes back wrapped in a "d" key that itself needs a second json.loads().

Documented trap: a non-browser User-Agent doesn't error, it makes the
server accept the POST and never send a response body — silent hang,
not a clean failure. Timeout is bounded explicitly here so that failure
mode is visible rather than looking like a frozen script.
"""

import requests
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}

BASE_URL = "https://www.niftyindices.com"
ENDPOINT = f"{BASE_URL}/Backpage.aspx/getHistoricaldatatabletoString"
REFERER = f"{BASE_URL}/reports/historical-data"


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(BASE_URL, timeout=20)
    print(f"Homepage warm-up: {resp.status_code}, {len(session.cookies)} cookies")
    return session


def fetch_index_history(session, index_name, start_date, end_date):
    """start_date/end_date format: 'DD-Mon-YYYY' e.g. '01-Jan-2024'."""
    # NOTE the single quotes INSIDE this string — this is not a mistake,
    # it's the exact ASP.NET convention the three sources independently
    # confirmed. Standard double-quoted JSON here would not match what
    # the server expects.
    cinfo = (
        f"{{'name':'{index_name}','startDate':'{start_date}',"
        f"'endDate':'{end_date}','indexName':'{index_name}'}}"
    )
    body = {"cinfo": cinfo}

    print(f"\nPOST {ENDPOINT}")
    print(f"Body: {body}")

    try:
        resp = session.post(
            ENDPOINT, json=body,
            headers={"Referer": REFERER},
            timeout=25,   # bounded — the documented hang trap means an
                          # unbounded wait here would be indistinguishable
                          # from a frozen script
        )
    except requests.exceptions.Timeout:
        print("TIMEOUT — likely the documented UA/header trap, not a dead server.")
        return

    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Response: {resp.text[:400]}")
        return

    try:
        outer = resp.json()
    except json.JSONDecodeError:
        print("Not valid JSON:", resp.text[:400])
        return

    print(f"Outer keys: {list(outer.keys())}")
    if "d" not in outer:
        print("No 'd' key — response shape differs from documented pattern.")
        print(json.dumps(outer, indent=2)[:1000])
        return

    inner = json.loads(outer["d"])
    print(f"Inner data type: {type(inner)}")
    if isinstance(inner, list) and inner:
        print(f"Records: {len(inner)}")
        print(f"First record: {json.dumps(inner[0], indent=2)}")
        print(f"Fields: {list(inner[0].keys())}")


if __name__ == "__main__":
    session = get_session()
    fetch_index_history(session, "NIFTY 50", "01-Jan-2024", "31-Jan-2024")