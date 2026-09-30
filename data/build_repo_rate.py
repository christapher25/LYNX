"""
Stage 8 — RBI repo rate history.

Sparse, event-driven series (~50 changes across 24 years, MPC meets
bi-monthly) — hand-curated from published sources, not scraped, matching
the same date-versioned-table pattern used for costs/model.py's rate table.

SOURCING:
  2002-04-06 through 2023-04-06: Reuters/Business Standard timeline
  (business-standard.com, "As RBI keeps repo rate unchanged, here's a
  timeline of it since June 2000"), a single continuous cited table.
  High confidence.

  2023-2026: no single continuous table found. Reconstructed by
  triangulating multiple independent news sources (Business Standard,
  Moneycontrol, Drishti IAS, ChartForest, 5paisa), cross-checked for
  arithmetic consistency (cumulative bps matches independently stated
  totals). High confidence on dates and direction; the exact bps split
  between the April 2025 and June 2025 moves is the one point inferred
  from arithmetic rather than read directly off one source — flagged
  explicitly below, worth a final check if this table is used for
  anything more precise than regime-state classification.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "repo_rate.parquet"

# (effective_date, rate_pct, confidence, note)
REPO_RATE_HISTORY: list[tuple[date, float, str, str]] = [
    (date(2002, 11, 12), 7.50, "high", "Reuters/Business Standard timeline"),
    (date(2003, 3, 7),   7.10, "high", ""),
    (date(2003, 3, 19),  7.00, "high", ""),
    (date(2004, 3, 31),  6.00, "high", ""),
    (date(2005, 10, 26), 6.25, "high", ""),
    (date(2006, 1, 24),  6.50, "high", ""),
    (date(2006, 6, 8),   6.75, "high", ""),
    (date(2006, 7, 25),  7.00, "high", ""),
    (date(2006, 10, 30), 7.25, "high", ""),
    (date(2007, 1, 31),  7.50, "high", ""),
    (date(2007, 3, 30),  7.75, "high", ""),
    (date(2008, 6, 11),  8.00, "high", ""),
    (date(2008, 6, 24),  8.50, "high", ""),
    (date(2008, 7, 29),  9.00, "high", ""),
    (date(2008, 10, 20), 8.00, "high", "start of GFC easing cycle"),
    (date(2008, 11, 3),  7.50, "high", ""),
    (date(2008, 12, 8),  6.50, "high", ""),
    (date(2009, 1, 2),   5.50, "high", ""),
    (date(2009, 3, 4),   5.00, "high", ""),
    (date(2009, 4, 21),  4.75, "high", ""),
    (date(2010, 3, 19),  5.00, "high", "start of post-GFC hiking cycle"),
    (date(2010, 4, 20),  5.25, "high", ""),
    (date(2010, 7, 2),   5.50, "high", ""),
    (date(2010, 7, 27),  5.75, "high", ""),
    (date(2010, 9, 16),  6.00, "high", ""),
    (date(2010, 11, 2),  6.25, "high", ""),
    (date(2011, 1, 25),  6.50, "high", ""),
    (date(2011, 3, 17),  6.75, "high", ""),
    (date(2011, 5, 3),   7.25, "high", ""),
    (date(2011, 6, 16),  7.50, "high", ""),
    (date(2011, 7, 26),  8.00, "high", ""),
    (date(2011, 9, 16),  8.25, "high", ""),
    (date(2011, 10, 25), 8.50, "high", ""),
    (date(2012, 4, 17),  8.00, "high", ""),
    (date(2013, 1, 29),  7.75, "high", ""),
    (date(2013, 3, 19),  7.50, "high", ""),
    (date(2013, 5, 3),   7.25, "high", ""),
    (date(2013, 9, 20),  7.50, "high", ""),
    (date(2013, 10, 29), 7.75, "high", ""),
    (date(2014, 1, 28),  8.00, "high", ""),
    (date(2015, 1, 15),  7.75, "high", ""),
    (date(2015, 3, 4),   7.50, "high", ""),
    (date(2015, 6, 2),   7.25, "high", ""),
    (date(2015, 9, 29),  6.75, "high", ""),
    (date(2016, 4, 5),   6.50, "high", ""),
    (date(2016, 10, 4),  6.25, "high", ""),
    (date(2017, 8, 2),   6.00, "high", ""),
    (date(2018, 6, 6),   6.25, "high", ""),
    (date(2018, 8, 1),   6.50, "high", ""),
    (date(2019, 2, 7),   6.25, "high", ""),
    (date(2019, 4, 4),   6.00, "high", ""),
    (date(2019, 6, 6),   5.75, "high", ""),
    (date(2019, 8, 7),   5.40, "high", ""),
    (date(2019, 10, 4),  5.15, "high", ""),
    (date(2020, 3, 27),  4.40, "high", "COVID emergency cut"),
    (date(2020, 5, 22),  4.00, "high", "COVID emergency cut"),
    (date(2022, 5, 4),   4.40, "high", "start of post-COVID hiking cycle"),
    (date(2022, 6, 8),   4.90, "high", ""),
    (date(2022, 8, 5),   5.40, "high", ""),
    (date(2022, 9, 30),  5.90, "high", ""),
    (date(2022, 12, 7),  6.25, "high", ""),
    (date(2023, 2, 8),   6.50, "high", ""),
    # --- 2023-2026 segment, reconstructed by triangulation, see docstring ---
    (date(2025, 2, 7),   6.25, "high", "first cut since May 2020, confirmed explicitly"),
    (date(2025, 4, 9),   6.00, "high", "confirmed explicitly"),
    (date(2025, 6, 6),   5.50, "medium", "inferred: cumulative -100bps since Feb "
                                          "confirmed, minus Feb+Apr's -50bps leaves "
                                          "-50bps here; matches known jumbo-cut pattern"),
    (date(2025, 12, 5),  5.25, "high", "confirmed explicitly, brings 2025 total to "
                                        "-125bps matching independently stated figure"),
    # 2026: confirmed unchanged at every meeting through August 2026
    # (Feb, June, Aug all explicitly reported as holds at 5.25%)
]


def build_daily_series(start: date, end: date) -> pd.DataFrame:
    """
    Forward-fill the sparse change-table into a daily series — between
    MPC meetings the rate simply stays at its last-set value. Same
    construction pattern as costs/model.py's date-versioned rate lookup.
    """
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"DATE": dates})

    change_dates = [d for d, _, _, _ in REPO_RATE_HISTORY]
    change_rates = [r for _, r, _, _ in REPO_RATE_HISTORY]

    df["REPO_RATE"] = pd.NA
    for i, cd in enumerate(change_dates):
        df.loc[df["DATE"] >= pd.Timestamp(cd), "REPO_RATE"] = change_rates[i]

    df = df.dropna(subset=["REPO_RATE"]).reset_index(drop=True)
    df["REPO_RATE"] = df["REPO_RATE"].astype(float)
    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    daily = build_daily_series(date(1996, 1, 1), date.today())
    daily.to_parquet(OUTPUT_PATH, index=False)

    print(f"Repo rate: {len(REPO_RATE_HISTORY)} change events, "
          f"{daily['DATE'].min().date()} to {daily['DATE'].max().date()}")
    print(f"Written to {OUTPUT_PATH}")

    medium_confidence = [e for e in REPO_RATE_HISTORY if e[2] == "medium"]
    if medium_confidence:
        print(f"\n{len(medium_confidence)} entries at 'medium' confidence "
              f"(inferred via arithmetic, not read directly from one source):")
        for d, r, c, note in medium_confidence:
            print(f"  {d}: {r}% — {note}")


if __name__ == "__main__":
    main()