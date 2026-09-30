import requests
from datetime import date, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0"}
URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{}.csv"


def get_names(d: date) -> set[str]:
    resp = requests.get(URL.format(d.strftime("%d%m%Y")), headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return set()
    lines = resp.text.strip().split("\n")[1:]
    return {line.split(",")[0] for line in lines}


# Check a spread of dates across the rebrand-suspected window
CHECK_DATES = [
    date(2012, 2, 21), date(2013, 1, 2), date(2014, 1, 2),
    date(2015, 1, 2), date(2015, 6, 1), date(2015, 9, 1),
    date(2016, 1, 4), date(2017, 1, 2),
]

for d in CHECK_DATES:
    names = get_names(d)
    has_new_nifty = "Nifty 50" in names
    has_old_nifty = "S&P CNX Nifty" in names
    has_vix = "India VIX" in names
    print(f"{d}: 'Nifty 50'={has_new_nifty}  'S&P CNX Nifty'={has_old_nifty}  "
          f"'India VIX'={has_vix}  (total indices: {len(names)})")