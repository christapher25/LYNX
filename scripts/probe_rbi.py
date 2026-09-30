import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
r = requests.get("https://www.rbi.org.in/scripts/referenceratearchive.aspx",
                  headers=HEADERS, timeout=20)
print(f"Status: {r.status_code}, {len(r.content)} bytes")
text = r.text
for marker in [".csv", ".xls", "download", "__VIEWSTATE", "FromDate", "btnGo", "GridView"]:
    if marker.lower() in text.lower():
        print(f"  found: {marker}")
print(f"\nFirst 500 chars:\n{text[:500]}")