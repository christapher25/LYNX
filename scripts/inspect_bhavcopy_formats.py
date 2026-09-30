import requests
import zipfile
import io
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_and_peek(url, label):
    print(f"\n{'='*60}\n{label}\n{url}\n{'='*60}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f, nrows=5)
                print(f"Columns: {list(df.columns)}")
                print(df.head())
    except Exception as e:
        print(f"FAILED: {e}")


fetch_and_peek(
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/2024/JUN/cm03JUN2024bhav.csv.zip",
    "OLD FORMAT (before July 2024)",
)

fetch_and_peek(
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20240710_F_0000.csv.zip",
    "NEW FORMAT (after July 2024)",
)