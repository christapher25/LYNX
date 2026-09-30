"""
Probe: NSDL FPI/FII data — the source NSE's own fii-dii page names as
authoritative for FINAL figures (NSE's own numbers are provisional).

NSE explicitly links: https://www.fpi.nsdl.co.in/Reports/Latest.aspx
Checking whether that surface exposes historical data in a fetchable
form, before concluding FII flows have no primary-source history path.
"""

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

CANDIDATES = [
    ("NSDL FPI latest reports", "https://www.fpi.nsdl.co.in/Reports/Latest.aspx"),
    ("NSDL FPI home", "https://www.fpi.nsdl.co.in/"),
    ("NSDL FPI fortnightly", "https://www.fpi.nsdl.co.in/web/Reports/ReportItem.aspx?ReportId=31"),
]

for label, url in CANDIDATES:
    print(f"\n{'='*60}\n{label}\n{url}\n{'='*60}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        print(f"Status: {r.status_code}, {len(r.content)} bytes")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        if r.status_code == 200:
            text = r.text
            # Look for signs of downloadable data or a date-range form
            for marker in [".csv", ".xls", "download", "FromDate", "ToDate", "DatePicker"]:
                if marker.lower() in text.lower():
                    print(f"  found marker: {marker}")
            print(f"\n  First 400 chars:\n  {text[:400]}")
    except Exception as e:
        print(f"ERROR: {e}")