"""
tests/test_fno_ingest.py

Tests for data/ingest_fno_bhavcopy.py — Stage 10, F&O ingestion. The silent-failure modes for a
derivatives file: a futures contract reading as "strike 0 / option type XX", rupees mistaken for
lakhs across the format cutover, an untraded contract's zero price rewritten or invented, and a
contract silently appearing twice.

The raw frames below have exactly the shapes NSE serves (legacy 15 columns + a trailing empty one;
UDiFF 34 columns), reduced to four representative contracts.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.ingest_fno_bhavcopy import (
    FIRST_AVAILABLE,
    OUTPUT_COLS,
    UDIFF_CUTOVER,
    build_url,
    parse_legacy,
    parse_udiff,
    validate,
)

TRADE_DATE = date(2024, 7, 5)

# (instrument, symbol, expiry, strike, option, open, high, low, close, settle, contracts, value_rupees, oi, chg_oi)
CONTRACTS = [
    ("FUTIDX", "NIFTY", "2024-07-25", None, None, 24400.0, 24500.0, 24350.0, 24480.0, 24481.5, 90000, 5_500_000_000.0, 1_200_000, -5_000),
    ("FUTSTK", "ESCORTS", "2024-08-29", None, None, 4130.95, 4260.0, 4130.95, 4179.4, 4179.4, 150, 173_088_410.0, 31_350, -1_375),
    ("OPTSTK", "HDFCLIFE", "2024-09-26", 680.0, "CE", 0.0, 0.0, 0.0, 10.85, 12.65, 0, 0.0, 0, 0),
    ("OPTIDX", "NIFTY", "2024-07-25", 24000.0, "PE", 30.0, 41.5, 22.0, 25.0, 25.6, 5000, 987_654_321.0, 700_000, 12_000),
]
UDIFF_TYPE = {"FUTIDX": "IDF", "FUTSTK": "STF", "OPTSTK": "STO", "OPTIDX": "IDO"}


def _legacy_raw(trade_date_text="05-JUL-2024", contracts=CONTRACTS) -> pd.DataFrame:
    rows = []
    for inst, sym, expiry, strike, opt, o, h, l, c, s, n, val, oi, doi in contracts:
        rows.append({
            "INSTRUMENT": inst, "SYMBOL": sym,
            "EXPIRY_DT": pd.Timestamp(expiry).strftime("%d-%b-%Y"),
            "STRIKE_PR": 0.0 if strike is None else strike, "OPTION_TYP": "XX" if opt is None else opt,
            "OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "SETTLE_PR": s, "CONTRACTS": n,
            "VAL_INLAKH": np.floor(val / 100_000 * 100) / 100, "OPEN_INT": oi, "CHG_IN_OI": doi,   # legacy TRUNCATES to 0.01 lakh
            "TIMESTAMP": trade_date_text, "Unnamed: 15": np.nan,
        })
    return pd.DataFrame(rows)


def _udiff_raw(trade_date_iso="2024-07-05", contracts=CONTRACTS) -> pd.DataFrame:
    rows = []
    for inst, sym, expiry, strike, opt, o, h, l, c, s, n, val, oi, doi in contracts:
        rows.append({
            "TradDt": trade_date_iso, "BizDt": trade_date_iso, "Sgmt": "FO", "Src": "NSE",
            "FinInstrmTp": UDIFF_TYPE[inst], "FinInstrmId": 1, "ISIN": np.nan, "TckrSymb": sym, "SctySrs": np.nan,
            "XpryDt": expiry, "FininstrmActlXpryDt": expiry,
            "StrkPric": np.nan if strike is None else strike, "OptnTp": np.nan if opt is None else opt,
            "FinInstrmNm": f"{sym}FUT", "OpnPric": o, "HghPric": h, "LwPric": l, "ClsPric": c, "LastPric": c,
            "PrvsClsgPric": c, "UndrlygPric": 4130.75, "SttlmPric": s, "OpnIntrst": oi, "ChngInOpnIntrst": doi,
            "TtlTradgVol": n, "TtlTrfVal": val, "TtlNbOfTxsExctd": n, "SsnId": "F1", "NewBrdLotQty": 275,
            "Rmks": np.nan, "Rsvd1": np.nan, "Rsvd2": np.nan, "Rsvd3": np.nan, "Rsvd4": np.nan,
        })
    return pd.DataFrame(rows)


def _row(df: pd.DataFrame, symbol: str, instrument: str) -> pd.Series:
    return df[(df["SYMBOL"] == symbol) & (df["INSTRUMENT"] == instrument)].iloc[0]


@pytest.fixture(params=["legacy", "udiff"])
def parsed(request) -> pd.DataFrame:
    return parse_legacy(_legacy_raw()) if request.param == "legacy" else parse_udiff(_udiff_raw())


class TestBothEras:
    def test_output_schema_is_the_declared_columns_in_order(self, parsed):
        assert list(parsed.columns) == OUTPUT_COLS

    def test_futures_have_no_strike_and_no_option_type_rather_than_zero_or_XX(self, parsed):
        """Legacy files encode 'not an option' as strike 0 / type XX; a 0 strike is a false price level."""
        future = _row(parsed, "NIFTY", "FUTIDX")
        assert np.isnan(future["STRIKE"])
        assert pd.isna(future["OPTION_TYPE"])
        assert not (parsed["OPTION_TYPE"] == "XX").any()

    def test_options_keep_their_strike_and_type(self, parsed):
        option = _row(parsed, "HDFCLIFE", "OPTSTK")
        assert option["STRIKE"] == 680.0 and option["OPTION_TYPE"] == "CE"
        assert _row(parsed, "NIFTY", "OPTIDX")["OPTION_TYPE"] == "PE"

    def test_instrument_codes_use_the_legacy_names(self, parsed):
        assert set(parsed["INSTRUMENT"]) == {"FUTIDX", "FUTSTK", "OPTSTK", "OPTIDX"}

    def test_value_is_in_lakhs_of_rupees(self, parsed):
        """
        UDiFF reports exact rupees, legacy reports lakhs TRUNCATED to 0.01 (Rs 1,000); both must land on lakhs
        (1 lakh = 100,000), so legacy sits within 0.01 lakh at or below the exact value.
        """
        assert _row(parsed, "ESCORTS", "FUTSTK")["VALUE_LAKHS"] == pytest.approx(1730.88410, abs=0.01)

    def test_expiry_and_trade_date_are_real_timestamps(self, parsed):
        row = _row(parsed, "ESCORTS", "FUTSTK")
        assert row["EXPIRY_DATE"] == pd.Timestamp("2024-08-29")
        assert (parsed["DATE"] == pd.Timestamp("2024-07-05")).all()

    def test_open_interest_and_contract_counts_carry_over(self, parsed):
        row = _row(parsed, "NIFTY", "FUTIDX")
        assert (row["OPEN_INT"], row["CHG_IN_OI"], row["CONTRACTS"]) == (1_200_000, -5_000, 90_000)

    def test_zero_prices_on_an_untraded_contract_are_kept_as_reported(self, parsed):
        """
        NSE writes OPEN/HIGH/LOW = 0 for a contract that did not trade. Ingestion is faithful to the file;
        masking a zero price is the feature layer's job (as Stage 9's prepare_ohlcv does for equities).
        The close/settle values, which NSE does compute, must survive.
        """
        row = _row(parsed, "HDFCLIFE", "OPTSTK")
        assert (row["OPEN"], row["HIGH"], row["LOW"]) == (0.0, 0.0, 0.0)
        assert (row["CLOSE"], row["SETTLE_PRICE"]) == (10.85, 12.65)

    def test_a_blank_source_cell_stays_nan_never_zero(self):
        for parse, raw, col in [(parse_legacy, _legacy_raw(), "OPEN_INT"), (parse_udiff, _udiff_raw(), "OpnIntrst")]:
            raw = raw.copy()
            raw[col] = raw[col].astype(float)
            raw.loc[0, col] = np.nan
            assert np.isnan(parse(raw).loc[0, "OPEN_INT"])


class TestEraSpecificFields:
    def test_udiff_only_fields_are_populated_from_udiff(self):
        row = _row(parse_udiff(_udiff_raw()), "ESCORTS", "FUTSTK")
        assert row["UNDERLYING_PRICE"] == 4130.75 and row["LOT_SIZE"] == 275 and row["NUM_TRADES"] == 150

    def test_those_fields_are_nan_not_zero_in_legacy_where_nse_never_published_them(self):
        out = parse_legacy(_legacy_raw())
        assert out[["UNDERLYING_PRICE", "LOT_SIZE", "NUM_TRADES"]].isna().all().all()

    def test_legacy_single_digit_day_dates_parse(self):
        """Early files write '9-NOV-2001', not '09-NOV-2001'."""
        raw = _legacy_raw(trade_date_text="9-NOV-2001", contracts=[
            ("FUTIDX", "NIFTY", "2001-11-29", None, None, 986.0, 994.0, 983.6, 990.05, 990.05, 4369, 864_510_000.0, 1_029_200, 800)])
        assert parse_legacy(raw).loc[0, "DATE"] == pd.Timestamp("2001-11-09")

    def test_legacy_trailing_empty_column_is_ignored(self):
        assert "Unnamed: 15" not in parse_legacy(_legacy_raw()).columns

    def test_legacy_column_names_with_stray_whitespace_are_tolerated(self):
        raw = _legacy_raw()
        raw.columns = [f" {c} " for c in raw.columns]
        assert len(parse_legacy(raw)) == len(CONTRACTS)


class TestValidation:
    def test_a_clean_frame_passes(self, parsed):
        assert len(validate(parsed, TRADE_DATE)) == len(CONTRACTS)

    def test_file_dated_for_another_day_is_rejected(self, parsed):
        with pytest.raises(ValueError, match="requested"):
            validate(parsed, date(2024, 7, 8))

    def test_more_than_one_date_in_a_file_is_rejected(self, parsed):
        parsed = parsed.copy()
        parsed.loc[0, "DATE"] = pd.Timestamp("2024-07-04")
        with pytest.raises(ValueError, match="one date"):
            validate(parsed, TRADE_DATE)

    def test_unparseable_dates_are_rejected(self):
        with pytest.raises(ValueError, match="date"):
            validate(parse_legacy(_legacy_raw(trade_date_text="not-a-date")), TRADE_DATE)

    def test_empty_file_is_rejected(self, parsed):
        with pytest.raises(ValueError, match="zero rows"):
            validate(parsed.iloc[:0], TRADE_DATE)

    def test_the_same_contract_twice_is_rejected(self, parsed):
        with pytest.raises(ValueError, match="duplicate"):
            validate(pd.concat([parsed, parsed.iloc[[1]]], ignore_index=True), TRADE_DATE)

    def test_negative_open_interest_or_volume_is_rejected(self, parsed):
        for col in ("OPEN_INT", "CONTRACTS"):
            bad = parsed.copy()
            bad.loc[0, col] = -1
            with pytest.raises(ValueError, match="negative"):
                validate(bad, TRADE_DATE)

    def test_an_option_without_a_strike_or_type_is_rejected(self, parsed):
        for col in ("STRIKE", "OPTION_TYPE"):
            bad = parsed.copy()
            bad.loc[bad["INSTRUMENT"] == "OPTSTK", col] = np.nan
            with pytest.raises(ValueError, match="option"):
                validate(bad, TRADE_DATE)

    def test_a_future_carrying_a_strike_is_rejected(self, parsed):
        bad = parsed.copy()
        bad.loc[bad["INSTRUMENT"] == "FUTSTK", "STRIKE"] = 100.0
        with pytest.raises(ValueError, match="future"):
            validate(bad, TRADE_DATE)

    def test_contract_expired_before_it_traded_is_rejected(self, parsed):
        bad = parsed.copy()
        bad.loc[0, "EXPIRY_DATE"] = pd.Timestamp("2024-07-01")
        with pytest.raises(ValueError, match="expir"):
            validate(bad, TRADE_DATE)

    def test_an_unrecognised_instrument_is_rejected_not_passed_through(self, parsed):
        bad = parsed.copy()
        bad.loc[0, "INSTRUMENT"] = "FUTXYZ"
        with pytest.raises(ValueError, match="instrument"):
            validate(bad, TRADE_DATE)

    def test_a_contract_expiring_today_is_allowed(self, parsed):
        ok = parsed.copy()
        ok.loc[0, "EXPIRY_DATE"] = pd.Timestamp(TRADE_DATE)
        validate(ok, TRADE_DATE)


class TestUrls:
    def test_legacy_url_uses_upper_case_month_and_zero_padded_day(self):
        url, era = build_url(date(2024, 6, 3))
        assert era == "legacy"
        assert url.endswith("/content/historical/DERIVATIVES/2024/JUN/fo03JUN2024bhav.csv.zip")

    def test_udiff_url_from_the_cutover_date(self):
        url, era = build_url(UDIFF_CUTOVER)
        assert era == "udiff"
        assert url.endswith("/content/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip")

    def test_the_last_legacy_day_and_first_udiff_day_are_adjacent_trading_days(self):
        assert build_url(date(2024, 7, 5))[1] == "legacy"
        assert build_url(date(2024, 7, 8))[1] == "udiff"

    def test_first_available_is_the_measured_index_futures_launch(self):
        """Probed 2026-09-24: 2000-06-09 -> 404, 2000-06-12 -> 200 (NSE's first index-futures day)."""
        assert FIRST_AVAILABLE == date(2000, 6, 12)


# --- fetch_day, exercised with a fake session (no network) ------------------------------------------

import io
import zipfile

import requests

from data.ingest_fno_bhavcopy import fetch_day


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.requested = []

    def get(self, url, timeout=None):
        self.requested.append(url)
        return self.response


def _zip_of(csv_text: str, member: str = "fo12JUN2000bhav.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_text)
    return buffer.getvalue()


class TestFetchDay:
    def test_a_tiny_valid_archive_is_parsed_not_mistaken_for_a_placeholder(self):
        """
        NSE's first F&O bhavcopies (2000-06-12: three futures rows, 386 bytes) are tiny and valid. Stage 3's
        equity ingester treats anything under 500 bytes as an empty placeholder; copying that here would
        silently discard the first days of the derivatives archive.
        """
        one_contract = [("FUTIDX", "NIFTY", "2000-06-29", None, None, 1461.0, 1472.0, 1451.0, 1452.75, 1451.15, 62, 18_120_000.0, 4000, 4000)]
        raw = _legacy_raw(trade_date_text="12-JUN-2000", contracts=one_contract).drop(columns=["Unnamed: 15"])
        payload = _zip_of(raw.to_csv(index=False))
        assert len(payload) < 500                                        # the size Stage 3's cutoff would have rejected

        df, era, nbytes = fetch_day(_FakeSession(_FakeResponse(200, payload)), date(2000, 6, 12))
        assert era == "legacy" and nbytes == len(payload)
        assert len(df) == 1 and df.loc[0, "SYMBOL"] == "NIFTY"

    def test_404_means_no_trading_that_day(self):
        session = _FakeSession(_FakeResponse(404, b"<html>not found</html>"))
        assert fetch_day(session, date(2020, 1, 1)) == (None, "legacy", 0)

    def test_a_200_that_is_not_a_zip_fails_the_day_instead_of_being_ingested(self):
        with pytest.raises(ValueError, match="not a zip"):
            fetch_day(_FakeSession(_FakeResponse(200, b"<html>maintenance page</html>")), date(2020, 1, 2))

    def test_an_empty_archive_fails_the_day(self):
        buffer = io.BytesIO()
        zipfile.ZipFile(buffer, "w").close()
        with pytest.raises(ValueError, match="empty"):
            fetch_day(_FakeSession(_FakeResponse(200, buffer.getvalue())), date(2020, 1, 2))

    def test_a_server_error_propagates_so_the_day_is_recorded_as_failed_and_retried(self):
        with pytest.raises(requests.HTTPError):
            fetch_day(_FakeSession(_FakeResponse(503)), date(2020, 1, 2))

    def test_the_requested_url_matches_the_era(self):
        session = _FakeSession(_FakeResponse(404))
        fetch_day(session, date(2024, 7, 8))
        assert session.requested == [build_url(date(2024, 7, 8))[0]]


# --- variants found by surveying the live archive (2026-09-24), absent from any hand-picked sample ------

class TestHistoricalVariants:
    """
    A survey of ~50 real dates across 2000-2026 showed the legacy format was not uniform. Each case
    below was a real failure of the first version of this parser.
    """

    def test_option_type_column_was_named_OPTIONTYPE_in_some_years(self):
        """Seen 2003-07-16, 2005-01-17, 2007-01-16; OPTION_TYP the rest of the time."""
        raw = _legacy_raw().rename(columns={"OPTION_TYP": "OPTIONTYPE"})
        pd.testing.assert_frame_equal(parse_legacy(raw), parse_legacy(_legacy_raw()))

    def test_american_style_stock_options_keep_their_CA_PA_codes(self):
        """
        Stock options were American-style (CA/PA) until ~2011, then European (CE/PE). Keeping the raw code
        preserves the distinction; call vs put is the first letter either way.
        """
        contracts = [
            ("OPTSTK", "RELIANCE", "2001-11-29", 1000.0, "CA", 10.0, 11.0, 9.0, 10.5, 10.5, 5, 500_000.0, 2000, 100),
            ("OPTSTK", "RELIANCE", "2001-11-29", 1000.0, "PA", 8.0, 9.0, 7.0, 8.5, 8.5, 3, 300_000.0, 1500, -50),
        ]
        out = parse_legacy(_legacy_raw(trade_date_text="09-NOV-2001", contracts=contracts))
        assert out["OPTION_TYPE"].tolist() == ["CA", "PA"] and (out["STRIKE"] == 1000.0).all()
        validate(out, date(2001, 11, 9))

    @pytest.mark.parametrize("instrument,symbol,option_code", [
        ("FUTINT", "NSE10Y06", None),        # interest-rate futures, 2003-2009
        ("UNDINT", "NSE10Y06", "FF"),        # their underlying rows: option-type code 'FF', strike 0
        ("FUTIVX", "INDIAVIX", None),        # India VIX futures, 2014-2018
    ])
    def test_non_equity_instruments_are_ingested_rather_than_failing_the_day(self, instrument, symbol, option_code):
        contracts = [(instrument, symbol, "2014-07-22", None, option_code, 0.0, 0.0, 0.0, 101.15, 101.30, 0, 0.0, 15_400, -550)]
        out = parse_legacy(_legacy_raw(trade_date_text="16-JUL-2014", contracts=contracts))
        validate(out, date(2014, 7, 16))
        assert out.loc[0, "INSTRUMENT"] == instrument
        assert np.isnan(out.loc[0, "STRIKE"]) and pd.isna(out.loc[0, "OPTION_TYPE"])   # 'FF'/'XX' are not option types

    def test_a_genuinely_unknown_instrument_still_fails_the_day(self):
        contracts = [("FUTZZZ", "X", "2014-07-22", None, None, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 0.0, 0, 0)]
        with pytest.raises(ValueError, match="instrument"):
            validate(parse_legacy(_legacy_raw(trade_date_text="16-JUL-2014", contracts=contracts)), date(2014, 7, 16))


class TestTwoDigitYearGlitch:
    """
    On 2020-07-13 NSE's equity bhavcopy wrote dates as '13-Jul-20' instead of '13-JUL-2020' (Stage 3's parser
    rejected all 2,001 rows and lost the day). The same glitch may affect the F&O file for that day, and the
    legacy F&O parser must not lose a real trading day to a date-format quirk.
    """

    def test_two_digit_year_dates_parse_in_both_the_trade_date_and_the_expiry(self):
        raw = _legacy_raw(trade_date_text="13-Jul-20", contracts=[
            ("FUTIDX", "NIFTY", "2020-07-30", None, None, 10_800.0, 10_900.0, 10_750.0, 10_850.0, 10_851.0, 90_000, 5_500_000_000.0, 1_200_000, 5_000)])
        raw["EXPIRY_DT"] = "30-Jul-20"
        out = parse_legacy(raw)
        assert out.loc[0, "DATE"] == pd.Timestamp("2020-07-13")
        assert out.loc[0, "EXPIRY_DATE"] == pd.Timestamp("2020-07-30")
        validate(out, date(2020, 7, 13))

    def test_four_and_two_digit_years_can_coexist_in_one_file(self):
        raw = _legacy_raw(trade_date_text="13-JUL-2020", contracts=CONTRACTS[:2])
        raw.loc[1, "EXPIRY_DT"] = "27-Aug-20"
        assert parse_legacy(raw)["EXPIRY_DATE"].notna().all()

    def test_a_genuinely_unparseable_date_still_fails_the_day(self):
        """Tolerance is for the known glitch only; garbage must still be rejected by validate()."""
        with pytest.raises(ValueError, match="date"):
            validate(parse_legacy(_legacy_raw(trade_date_text="13/07/2020")), date(2020, 7, 13))

    def test_a_two_digit_year_cannot_smuggle_in_the_wrong_day(self):
        """The parsed date is still checked against the requested day."""
        out = parse_legacy(_legacy_raw(trade_date_text="13-Jul-20"))
        with pytest.raises(ValueError, match="requested"):
            validate(out, date(2020, 7, 14))


# --- corrupted legacy CSV lines found by the full backfill (2002 files) ------------------------------------

from data.ingest_fno_bhavcopy import repair_legacy_csv

HEADER = "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,"
REC_A = "OPTSTK,BAJAJAUTO,30-May-2002,440,CA,0,0,0,46.65,66.2,0,0,0,0,19-MAR-2002,"
REC_B = "OPTSTK,INFOSYSTCH,25-Apr-2002,3600,CA,0,0,0,575.7,495.6,0,0,0,0,19-MAR-2002,"
REC_C = "OPTSTK,VSNL,30-May-2002,190,CA,0,0,0,8.5,10.35,0,0,0,0,19-MAR-2002,"
REC_D = "FUTSTK,ACC,28-Mar-2002,0,XX,150,155,149,152,152,10,15.2,5000,100,19-MAR-2002,"


def _read(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), low_memory=False)


class TestCorruptedLegacyLines:
    """
    NSE's 2002 files contain two kinds of damage that made the first ingest fail 12 days:
      * a second copy of the header line in the middle of the file (two files concatenated);
      * two or more records fused onto one line because a newline was lost. Record 1's trailing comma
        is the separator, so a fused line has 15*k + 1 fields where 16 are expected.
    Both are recoverable exactly; anything else must still fail the day rather than be guessed at.
    """

    CLEAN = "\n".join([HEADER, REC_A, REC_B, REC_C, REC_D]) + "\n"

    def test_a_clean_file_passes_through_unchanged_with_nothing_repaired(self):
        text, report = repair_legacy_csv(self.CLEAN)
        assert text == self.CLEAN and report == {"header_repeats": 0, "fused_lines": 0, "footers": 0}

    def test_fused_records_are_split_into_the_records_they_were(self):
        fused = "\n".join([HEADER, REC_A + REC_B, REC_C, REC_D]) + "\n"           # newline lost between A and B
        assert _read(fused).shape[0] != 4                                          # sanity: pandas alone cannot read it
        text, report = repair_legacy_csv(fused)
        assert report["fused_lines"] == 1
        pd.testing.assert_frame_equal(_read(text), _read(self.CLEAN))               # identical to the undamaged file

    def test_three_records_fused_into_one_line_are_recovered(self):
        text, report = repair_legacy_csv("\n".join([HEADER, REC_A + REC_B + REC_C, REC_D]) + "\n")
        assert report["fused_lines"] == 1 and len(_read(text)) == 4

    def test_a_repeated_header_line_is_dropped_and_counted(self):
        with_repeat = "\n".join([HEADER, REC_A, HEADER, REC_B, REC_C, REC_D]) + "\n"
        text, report = repair_legacy_csv(with_repeat)
        assert report["header_repeats"] == 1
        pd.testing.assert_frame_equal(_read(text), _read(self.CLEAN))

    def test_only_an_exact_copy_of_the_header_counts_as_a_repeat(self):
        """A row that merely starts with 'INSTRUMENT' is data and is left for validation to judge."""
        odd = "\n".join([HEADER, REC_A, REC_B.replace("OPTSTK", "INSTRUMENT", 1)]) + "\n"
        _, report = repair_legacy_csv(odd)
        assert report["header_repeats"] == 0

    def test_a_line_with_an_irregular_field_count_fails_the_day_instead_of_being_guessed_at(self):
        broken = "\n".join([HEADER, REC_A, "OPTSTK,BAJAJAUTO,30-May-2002,440,CA,0,0,0,46.65", REC_B]) + "\n"
        with pytest.raises(ValueError, match="field"):
            repair_legacy_csv(broken)

    def test_a_field_count_that_is_not_a_whole_number_of_records_fails_the_day(self):
        with pytest.raises(ValueError, match="field"):
            repair_legacy_csv("\n".join([HEADER, REC_A + REC_B + "X,Y,Z,"]) + "\n")

    def test_blank_lines_are_ignored_not_counted_as_damage(self):
        text, report = repair_legacy_csv(HEADER + "\n\n" + REC_A + "\n\n")
        assert report == {"header_repeats": 0, "fused_lines": 0, "footers": 0} and len(_read(text)) == 1

    def test_files_without_a_trailing_comma_are_left_alone(self):
        """Early files (2001) have 15 columns and no trailing comma; the fused-line rule is defined only for the 16-token layout."""
        header15, rec = HEADER.rstrip(","), REC_A.rstrip(",")
        text, report = repair_legacy_csv("\n".join([header15, rec]) + "\n")
        assert report == {"header_repeats": 0, "fused_lines": 0, "footers": 0} and len(_read(text)) == 1

    def test_the_repaired_file_parses_and_validates_end_to_end(self):
        fused = "\n".join([HEADER, REC_A + REC_B, REC_C, REC_D]) + "\n"
        out = parse_legacy(_read(repair_legacy_csv(fused)[0]))
        assert len(validate(out, date(2002, 3, 19))) == 4

    def test_fetch_day_reports_the_repair_in_the_era_label_for_the_manifest(self):
        fused = "\n".join([HEADER, REC_A + REC_B, REC_C, REC_D, HEADER]) + "\n"
        df, era, _ = fetch_day(_FakeSession(_FakeResponse(200, _zip_of(fused, "fo19MAR2002bhav.csv"))), date(2002, 3, 19))
        assert len(df) == 4
        assert era.startswith("legacy") and "fused=1" in era and "header=1" in era


# --- rows that expired before the trade date (NSE kept the just-expired series in the next day's file) --------

from data.ingest_fno_bhavcopy import drop_expired_contracts


def _with_expiries(trade_date_text: str, expiries: list[str]) -> pd.DataFrame:
    contracts = [("FUTSTK", f"SYM{i}", e, None, None, 100.0, 101.0, 99.0, 100.5, 100.5, 10, 1_000_000.0, 5000, 100)
                 for i, e in enumerate(expiries)]
    return parse_legacy(_legacy_raw(trade_date_text=trade_date_text, contracts=contracts))


class TestStaleExpiredContracts:
    """
    2002-09-27's file is a real day (it lists 334 contracts the previous file lacks) but also still carries the
    707 September contracts that expired on 2002-09-26, with volume and open interest, which is impossible after
    expiry. Rejecting the day would discard ~1,400 valid contracts; the expired rows are provably stale and are
    dropped, counted, and recorded in the manifest. The guards stop a systematic date fault from emptying a file.
    """

    def test_contracts_that_expired_the_day_before_are_dropped_and_counted(self):
        df = _with_expiries("27-SEP-2002", ["2002-09-26", "2002-09-26", "2002-10-31", "2002-11-28", "2002-12-26"])
        kept, dropped = drop_expired_contracts(df, date(2002, 9, 27))
        assert dropped == 2 and len(kept) == 3
        assert (kept["EXPIRY_DATE"] >= pd.Timestamp("2002-09-27")).all()
        validate(kept, date(2002, 9, 27))                       # the strict validator now accepts the day

    def test_a_contract_expiring_on_the_trade_date_itself_is_kept(self):
        df = _with_expiries("26-SEP-2002", ["2002-09-26", "2002-10-31", "2002-11-28"])
        kept, dropped = drop_expired_contracts(df, date(2002, 9, 26))
        assert dropped == 0 and len(kept) == 3

    def test_nothing_is_touched_when_nothing_expired(self):
        df = _with_expiries("27-SEP-2002", ["2002-10-31", "2002-11-28"])
        kept, dropped = drop_expired_contracts(df, date(2002, 9, 27))
        assert dropped == 0
        pd.testing.assert_frame_equal(kept, df)

    def test_a_contract_that_expired_long_ago_means_a_date_fault_not_a_stale_row(self):
        df = _with_expiries("27-SEP-2002", ["2002-09-26", "2001-01-25", "2002-10-31", "2002-11-28"])
        with pytest.raises(ValueError, match="long before"):
            drop_expired_contracts(df, date(2002, 9, 27))

    def test_a_file_that_would_lose_half_its_rows_fails_instead_of_being_emptied(self):
        df = _with_expiries("27-SEP-2002", ["2002-09-26", "2002-09-26", "2002-10-31", "2002-11-28"])   # 2 of 4
        with pytest.raises(ValueError, match="half"):
            drop_expired_contracts(df, date(2002, 9, 27))

    def test_it_never_drops_a_row_that_validate_would_have_accepted(self):
        df = _with_expiries("27-SEP-2002", ["2002-09-27", "2002-10-31", "2002-11-28"])
        kept, dropped = drop_expired_contracts(df, date(2002, 9, 27))
        assert dropped == 0 and len(validate(kept, date(2002, 9, 27))) == 3

    def test_fetch_day_drops_them_end_to_end_and_records_the_count_in_the_era(self):
        contracts = [("FUTSTK", f"SYM{i}", e, None, None, 100.0, 101.0, 99.0, 100.5, 100.5, 10, 1_000_000.0, 5000, 100)
                     for i, e in enumerate(["2002-09-26", "2002-10-31", "2002-11-28", "2002-12-26"])]
        raw = _legacy_raw(trade_date_text="27-SEP-2002", contracts=contracts).drop(columns=["Unnamed: 15"])
        df, era, _ = fetch_day(_FakeSession(_FakeResponse(200, _zip_of(raw.to_csv(index=False), "fo27SEP2002bhav.csv"))),
                               date(2002, 9, 27))
        assert len(df) == 3 and "expired=1" in era


# --- files that are the same day's data pasted together more than once -------------------------------------

from data.ingest_fno_bhavcopy import collapse_identical_duplicates


class TestRepeatedCopiesOfTheSameDay:
    """
    Several 2003 files are one day's data concatenated 2x (2003-04-17) or 6x (2003-04-23, with header lines and
    fused records inside), and every copy is identical on every column. Identical copies are redundant and
    collapse losslessly; copies that DISAGREE mean the file is untrustworthy and must still fail the day.
    """

    def _frame(self):
        return parse_legacy(_legacy_raw(contracts=CONTRACTS))

    def test_a_file_repeated_twice_collapses_to_one_copy(self):
        one = self._frame()
        doubled = pd.concat([one, one], ignore_index=True)
        out, removed = collapse_identical_duplicates(doubled)
        assert removed == len(one)
        pd.testing.assert_frame_equal(out.reset_index(drop=True), one)

    def test_a_file_repeated_six_times_collapses_to_one_copy(self):
        one = self._frame()
        out, removed = collapse_identical_duplicates(pd.concat([one] * 6, ignore_index=True))
        assert removed == 5 * len(one) and len(out) == len(one)

    def test_the_first_occurrence_is_kept_and_order_is_preserved(self):
        one = self._frame()
        mixed = pd.concat([one.iloc[:2], one, one.iloc[2:]], ignore_index=True)
        out, _ = collapse_identical_duplicates(mixed)
        assert out["SYMBOL"].tolist() == one["SYMBOL"].tolist()

    def test_futures_with_no_strike_collapse_too(self):
        """NaN strike / option type must not stop identical futures rows from matching each other."""
        one = self._frame()
        assert one["STRIKE"].isna().any()
        assert len(collapse_identical_duplicates(pd.concat([one, one], ignore_index=True))[0]) == len(one)

    def test_copies_that_disagree_fail_the_day_instead_of_picking_one(self):
        one = self._frame()
        other = one.copy()
        other.loc[0, "OPEN_INT"] = one.loc[0, "OPEN_INT"] + 1000
        with pytest.raises(ValueError, match="disagree"):
            collapse_identical_duplicates(pd.concat([one, other], ignore_index=True))

    def test_a_clean_frame_is_returned_untouched(self):
        one = self._frame()
        out, removed = collapse_identical_duplicates(one)
        assert removed == 0
        pd.testing.assert_frame_equal(out, one)

    def test_fetch_day_handles_a_doubled_file_with_a_repeated_header_and_records_it(self):
        body = [REC_A, REC_B, REC_C, REC_D]
        doubled = "\n".join([HEADER, *body, HEADER, *body]) + "\n"
        df, era, _ = fetch_day(_FakeSession(_FakeResponse(200, _zip_of(doubled, "fo19MAR2002bhav.csv"))), date(2002, 3, 19))
        assert len(df) == 4 and not df.duplicated(["SYMBOL", "INSTRUMENT", "EXPIRY_DATE", "STRIKE", "OPTION_TYPE"]).any()
        assert "header=1" in era and "dupes=4" in era

    def test_fetch_day_still_fails_when_the_pasted_copies_disagree(self):
        altered = REC_B.replace(",0,0,0,0,19-MAR-2002", ",0,0,0,500,19-MAR-2002")        # OPEN_INT differs in the 2nd copy
        text = "\n".join([HEADER, REC_A, REC_B, HEADER, REC_A, altered]) + "\n"
        with pytest.raises(ValueError, match="disagree"):
            fetch_day(_FakeSession(_FakeResponse(200, _zip_of(text, "fo19MAR2002bhav.csv"))), date(2002, 3, 19))


class TestTrailingFooterNote:
    """
    NSE's 2012-05-14 file ends with a stray '*Updated' line after the last data row (a re-publication marker).
    It is not data. Only that exact text, as the final line, is dropped; any other odd line still fails the day.
    """

    CLEAN = TestCorruptedLegacyLines.CLEAN

    def test_a_trailing_updated_footer_is_dropped_and_counted(self):
        text, report = repair_legacy_csv(self.CLEAN + "\n*Updated\n")
        assert report["footers"] == 1
        pd.testing.assert_frame_equal(_read(text), _read(self.CLEAN))

    def test_the_footer_match_is_exact_but_not_fussy_about_case_or_spacing(self):
        assert repair_legacy_csv(self.CLEAN + "  *UPDATED  \n")[1]["footers"] == 1

    def test_the_same_text_in_the_middle_of_the_file_is_not_a_footer(self):
        broken = "\n".join([HEADER, REC_A, "*Updated", REC_B]) + "\n"
        with pytest.raises(ValueError, match="field"):
            repair_legacy_csv(broken)

    def test_any_other_trailing_note_still_fails_the_day(self):
        with pytest.raises(ValueError, match="field"):
            repair_legacy_csv(self.CLEAN + "*Revised again\n")

    def test_fetch_day_records_the_footer_in_the_era(self):
        text = self.CLEAN + "\n*Updated\n"
        df, era, _ = fetch_day(_FakeSession(_FakeResponse(200, _zip_of(text, "fo19MAR2002bhav.csv"))), date(2002, 3, 19))
        assert len(df) == 4 and "footer=1" in era
