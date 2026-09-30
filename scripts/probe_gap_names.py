import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{}.csv"

for label, d in [("2014-06-02", "02062014"), ("2015-06-01", "01062015")]:
    r = requests.get(URL.format(d), headers=HEADERS, timeout=15)
    lines = r.text.strip().split("\n")[1:]
    names = sorted(line.split(",")[0] for line in lines)
    print(f"\n{'='*50}\n{label} — {len(names)} indices\n{'='*50}")
    for n in names:
        if any(k in n.upper() for k in ["NIFTY", "CNX", "BANK", "50"]):
            print(f"  {n}")