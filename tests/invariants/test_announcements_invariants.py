"""
tests/invariants/test_announcements_invariants.py

Two silent-failure modes, each proved impossible by running the checker on the real pipeline output AND on a deliberately
broken variant that must be caught:

  A. Look-ahead keying. An announcement is keyed by when it was PUBLISHED, never by the event date its text describes
     ("results for the quarter ended...", "meeting held on December 24, 2009"). Keying on the event date would hand the
     model news before it existed.
  B. Guessed tickers. A record that cannot be resolved unambiguously carries NO ticker and an explicit reason. A resolver
     that picks "the first candidate" would silently attach news to the wrong stock.
"""

from datetime import date

import pandas as pd

from data import ingest_announcements as ia
from data.entity_resolution import SymbolMaster, resolve

MASTER = SymbolMaster.from_frame(pd.DataFrame({
    "SYMBOL": ["TCS", "OLDCO", "NEWCO"], "ISIN": ["INE467B01029", "INE111A01011", "INE111A01011"],
    "FIRST_DATE": pd.to_datetime(["2004-08-25", "2005-01-01", "2015-06-01"]),
    "LAST_DATE": pd.to_datetime(["2026-09-29", "2015-05-29", "2026-09-29"]),
}))


def payload():
    def r(symbol, isin, name, an_dt, text, seq):
        return {"symbol": symbol, "sm_isin": isin, "sm_name": name, "an_dt": an_dt, "exchdisstime": "-", "desc": "x",
                "attchmntText": text, "attchmntFile": "-", "seq_id": seq, "hasXbrl": False}
    return [
        r("TCS", "INE467B01029", "Tata Consultancy Services Limited", "02-Jan-2020 18:00:00",
          "minutes of the meeting held on December 24, 2019", "1"),
        r("-", "INE111A01011", "Shared Isin Limited", "02-Jan-2001 10:00:00", "outside both windows", "2"),   # ambiguous
        r("-", "-", "Nobody Limited", "02-Jan-2020 10:00:00", "no identifiers", "3"),                         # unmatched
        r("-", "INE111A01011", "Shared Isin Limited", "02-Jan-2020 10:00:00", "dated window picks NEWCO", "4"),
    ]


def build():
    return ia.build_chunk_frame(payload(), date(2001, 1, 1), date(2020, 1, 31), MASTER)


def test_real_pipeline_output_satisfies_every_invariant():
    df = build()
    assert ia.announcement_invariant_violations(df, MASTER) == {}
    assert df["TICKER"].notna().sum() == 2 and (df["RESOLUTION_METHOD"] == "unresolved").sum() == 2


def test_publication_time_not_event_date_keys_the_record():
    df = build()
    tcs = df[df["SEQ_ID"] == "1"].iloc[0]
    assert tcs["PUBLISHED_DATE"] == pd.Timestamp("2020-01-02")          # not the 2019-12-24 meeting in the text


def test_unresolved_records_have_no_ticker_and_a_reason():
    df = build()
    unres = df[df["RESOLUTION_METHOD"] == "unresolved"]
    assert len(unres) == 2 and unres["TICKER"].isna().all() and unres["UNRESOLVED_REASON"].notna().all()
    assert set(unres["UNRESOLVED_REASON"]) == {"isin_ambiguous", "unmatched"}


def test_mutant_guessing_resolver_is_caught():
    df = build()
    broken = df.copy()
    mask = broken["RESOLUTION_METHOD"] == "unresolved"
    broken.loc[mask, "TICKER"] = "OLDCO"                                  # "first candidate" guess; method still says unresolved
    assert "ticker_on_unresolved" in ia.announcement_invariant_violations(broken, MASTER)

    broken2 = df.copy()
    broken2.loc[mask, "RESOLUTION_METHOD"] = "isin"                      # relabelled as resolved but ticker still null
    assert "resolved_without_ticker" in ia.announcement_invariant_violations(broken2, MASTER)


def test_mutant_unresolved_without_reason_is_caught():
    broken = build()
    broken.loc[broken["RESOLUTION_METHOD"] == "unresolved", "UNRESOLVED_REASON"] = pd.NA
    assert "unresolved_without_reason" in ia.announcement_invariant_violations(broken, MASTER)


def test_mutant_ticker_not_in_master_is_caught():
    broken = build()
    broken.loc[broken["TICKER"].notna(), "TICKER"] = "FAKE"
    assert "ticker_not_in_master" in ia.announcement_invariant_violations(broken, MASTER)


def test_mutant_event_date_keying_is_caught():
    broken = build()
    broken["PUBLISHED_DATE"] = pd.Timestamp("2019-12-24")                # keyed on the event date in the text
    assert "published_date_mismatch" in ia.announcement_invariant_violations(broken, MASTER)


def test_mutant_publication_earlier_than_source_stamp_is_caught():
    broken = build()
    broken["PUBLISHED_AT"] = broken["PUBLISHED_AT"] - pd.Timedelta(hours=3)
    broken["PUBLISHED_DATE"] = broken["PUBLISHED_AT"].dt.tz_localize(None).dt.normalize()
    assert "published_before_source_stamp" in ia.announcement_invariant_violations(broken, MASTER)


def test_mutant_naive_timestamp_is_caught():
    broken = build()
    broken["PUBLISHED_AT"] = broken["PUBLISHED_AT"].dt.tz_localize(None)
    assert "published_at_not_tz_aware" in ia.announcement_invariant_violations(broken, MASTER)


def test_resolver_cannot_attach_a_ticker_from_an_ambiguous_isin():
    df = pd.DataFrame([{"SYMBOL_RAW": None, "ISIN": "INE111A01011", "COMPANY_NAME": "x", "PUBLISHED_DATE": pd.Timestamp("2001-01-01")}])
    assert pd.isna(resolve(df, MASTER, {}).iloc[0]["TICKER"])
