import requests
import io
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_and_peek(url, label):
    print(f"\n{'='*60}\n{label}\n{url}\n{'='*60}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), nrows=5)
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    except Exception as e:
        print(f"FAILED: {e}")


fetch_and_peek(
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_03062024.csv",
    "DELIVERY DATA - pre-cutover date",
)

fetch_and_peek(
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_10072024.csv",
    "DELIVERY DATA - post-cutover date",
)