import requests, zipfile, io
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}

URLS = [
    ("1995-01-02", "https://nsearchives.nseindia.com/content/historical/EQUITIES/1995/JAN/cm02JAN1995bhav.csv.zip"),
    ("2000-01-03", "https://nsearchives.nseindia.com/content/historical/EQUITIES/2000/JAN/cm03JAN2000bhav.csv.zip"),
    ("2010-01-04", "https://nsearchives.nseindia.com/content/historical/EQUITIES/2010/JAN/cm04JAN2010bhav.csv.zip"),
]

for label, url in URLS:
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, nrows=3)
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(df.head(2))
    except Exception as e:
        print(f"FAILED: {e}")