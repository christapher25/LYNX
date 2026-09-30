import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Old-format bhavcopy: one date per year, working backwards
TEST_DATES = [
    ("2024", "JUN", "03"), ("2020", "JAN", "02"), ("2016", "JAN", "01"),
    ("2012", "JAN", "02"), ("2008", "JAN", "02"), ("2005", "JAN", "03"),
    ("2002", "JAN", "01"), ("2000", "JAN", "03"), ("1998", "JAN", "02"),
    ("1996", "JAN", "02"), ("1995", "JAN", "02"), ("1994", "NOV", "03"),
]

for yyyy, mon, dd in TEST_DATES:
    url = (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
           f"{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        ok = r.status_code == 200 and len(r.content) > 1000
        status = f"OK ({len(r.content):,} bytes)" if ok else f"MISSING ({r.status_code})"
    except Exception:
        status = "ERROR (retry)"
    print(f"{dd}-{mon}-{yyyy}: {status}")