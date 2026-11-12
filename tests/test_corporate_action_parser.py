"""R21-1: tests for data/corporate_action_parser.py (face-value splits without a factor; non-equity bonus must not adjust)."""
from __future__ import annotations

import random

import pytest

from data.corporate_action_parser import ActionType, adjustment_factor, parse_subject

# the twelve wordings measured on the real files (plus the one that already parsed); {a} = from, {b} = to
WORDINGS = [
    "Face Value Split (Sub-Division) - From Rs {a}/- Per Share To Re {b}/- Per Share",
    "Face Value Split (Sub-Division) - From Rs {a}/- Per Share To Rs {b}/- Per Share",
    "Fv Split Rs.{a}/- To Rs.{b}/",
    "Fv Split Rs.{a}/- To Re.{b}/",
    "Face Value Split From Rs.{a}/- To Re.{b}/-",
    "Face Value Split From Rs {a} To Re {b}",
    "Fv Split Rs.{a} To Rs.{b}",
    "Fv Split Rs.{a}/- To Re.{b}/-",
    "Face Value Split From Rs {a}/- Per Share To Re {b}/- Per Share",
    "Face Value Split Rs {a} To Rs {b}",
    "Fv Split Rs.{a} To Re.{b}",
    "Fv Split Rs.{a}/- To Rs.{b}/-",
    "Fv Split Rs.{a}/- To Rs.{b}/Rd Date Revised",
]


def parsed(subject, fv=None):
    p = parse_subject(subject, fv)
    return p, adjustment_factor(p)


# T1 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wording", WORDINGS)
@pytest.mark.parametrize("a,b", [(10, 1), (10, 2), (5, 1), (2, 1), (10, 5)])
def test_t1_every_wording_parses_with_the_exact_factor(wording, a, b):
    p, f = parsed(wording.format(a=a, b=b))
    assert p.action_type == ActionType.SPLIT and p.parse_confidence == "high"
    assert (p.face_value_from, p.face_value_to) == (float(a), float(b))
    assert f == b / a


@pytest.mark.parametrize("subject,factor", [
    ("Face Value Split Rs.10/- To Re.1/- Per Share", 0.1),                                         # SBIN
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 0.1),         # TATASTEEL / EICHERMOT
    ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share", 0.2),          # KOTAKBANK
])
def test_t1_real_subjects(subject, factor):
    p, f = parsed(subject)
    assert p.action_type == ActionType.SPLIT and p.parse_confidence == "high"
    assert f == pytest.approx(factor)


@pytest.mark.parametrize("fn", [str.lower, str.upper])
def test_t1_case_does_not_matter(fn):
    p, f = parsed(fn("Fv Split Rs.10/- To Re.1/-"))
    assert p.parse_confidence == "high" and f == 0.1


# T2 -----------------------------------------------------------------------------------------------
def test_t2_decimals_parse():
    p, f = parsed("Fv Split Rs.2.50 To Re.0.50")
    assert p.parse_confidence == "high" and (p.face_value_from, p.face_value_to) == (2.5, 0.5)
    assert f == 0.5 / 2.5


@pytest.mark.parametrize("subject", [
    "Fv Split Rs.10/- To Rs.10/-",          # equal values
    "Fv Split Rs.10/- To Re.0/-",           # zero
    "Fv Split Rs.0/- To Re.1/-",            # zero from
    "Face Value Split From Rs 10/-",        # one number only
    "Face Value Split",                     # none
])
def test_t2_unusable_pairs_are_low_and_have_no_factor(subject):
    p, f = parsed(subject)
    assert p.action_type == ActionType.SPLIT
    assert p.parse_confidence == "low" and f is None


def test_t2_consolidation_gives_a_factor_above_one():
    p, f = parsed("Face Value Split From Re 1/- To Rs 10/-")
    assert p.parse_confidence == "high" and f == 10.0
    p, f = parsed("Fv Split Rs.1/- To Rs.10/-")
    assert f == 10.0


# T3 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("subject", [
    "Bonus Preference Shares 21:1",
    "Scheme Of Arrangement - Bonus Debentures 6:1",
    "Scheme Of Arrangement - Bonus Debentures 1:1",
    "Scheme Of Arrangement - Bonus Debentures 1:1 (Purpose Revised)",
    "Bonus Warrants 1:2",
    "Bonus Ncd 1:1",
    "Bonus Bonds 1:10",
    "Bonus Pref. Shares 1:1",
    "Bonus Pref Shares 1:1",
    "BONUS DEBENTURES 1:1",
])
def test_t3_non_equity_bonus_never_gets_a_factor(subject):
    p, f = parsed(subject)
    assert p.action_type == ActionType.BONUS
    assert p.parse_confidence == "low" and f is None


def test_t3_non_equity_bonus_may_record_the_ratio():
    p, _ = parsed("Bonus Preference Shares 21:1")
    assert (p.ratio_new, p.ratio_old) == (21.0, 1.0)


@pytest.mark.parametrize("subject,factor", [("Bonus 1:1", 0.5), ("Bonus Issue 3:2", 0.4), ("Bonus - 1:2", 1 / 1.5)])
def test_t3_equity_bonus_keeps_its_factor(subject, factor):
    p, f = parsed(subject)
    assert p.action_type == ActionType.BONUS and p.parse_confidence == "high"
    assert f == pytest.approx(factor)


@pytest.mark.parametrize("subject", ["Bondada Engineering Bonus 1:1", "Bonus 1:1 Bondada", "Prefecture Infra Bonus 1:1"])
def test_t3_bond_and_pref_match_only_whole_words(subject):
    p, f = parsed(subject)
    assert p.parse_confidence == "high" and f == 0.5


# bonus branch still runs first ----------------------------------------------------------------------
def test_split_subject_that_also_says_bonus_keeps_the_bonus_branch():
    p, f = parsed("Bonus 1:1 And Face Value Split From Rs 10/- To Rs 5/-")
    assert p.action_type == ActionType.BONUS and f == 0.5


# T4 regression: parses that were high before are unchanged ------------------------------------------------
REGRESSION = [
    # (subject, face value at record, type, confidence, ratio_new, ratio_old, fv_from, fv_to, dividend_amount, dividend_percent, factor)
    ("Dividend - Rs 6 Per Share", "10", "dividend", "high", None, None, None, None, 6.0, None, None),
    ("Dividend-Rs.7.5/- Per Sh.", "10", "dividend", "high", None, None, None, None, 7.5, None, None),
    ("Agm/Dividend-500%", "1", "dividend", "high", None, None, None, None, 5.0, 500.0, None),
    ("Bonus 1:1", None, "bonus", "high", 1.0, 1.0, None, None, None, None, 0.5),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", None, "split", "high", None, None, 10.0, 2.0,
     None, None, 0.2),
    ("Something Totally Unrecognized XYZ", None, "unknown", "failed", None, None, None, None, None, None, None),
    ("Annual General Meeting", None, "no_action", "high", None, None, None, None, None, None, None),
    (" Annual General Meeting", None, "no_action", "high", None, None, None, None, None, None, None),
    ("Agm", None, "no_action", "high", None, None, None, None, None, None, None),
    ("Agm/Div-Rs.1.50 Per Share", "10", "dividend", "high", None, None, None, None, 1.5, None, None),
    ("Agm/Div-Rs.1.50/- Per Sh", "10", "dividend", "high", None, None, None, None, 1.5, None, None),
    ("Agm/Div.-Re.1/- Per Share", "10", "dividend", "high", None, None, None, None, 1.0, None, None),
    ("Int Div-Rs.2/- Per Share", "10", "dividend", "high", None, None, None, None, 2.0, None, None),
    ("2nd Int Div-Rs.2/- Pr Shr", "10", "dividend", "high", None, None, None, None, 2.0, None, None),
    ("1st Int Div-Rs1.20 Per Shpurpose Revised", "10", "dividend", "high", None, None, None, None, 1.2, None, None),
    ("Int Div-Re.0.50 Per Sh   Purpose Revised", "10", "dividend", "high", None, None, None, None, 0.5, None, None),
    ("Demerger", None, "unknown", "failed", None, None, None, None, None, None, None),
    (" Scheme Of Arrangement & Reconstruction", None, "unknown", "failed", None, None, None, None, None, None, None),
    ("Agm                      Bc Revised", None, "no_action", "high", None, None, None, None, None, None, None),
    (" Annual General Meeting/Divided - Rs 35/- Per Share", "10", "dividend", "high", None, None, None, None, 35.0, None, None),
    ("Agm/Div-Int-200%+Fin-200%", "10", "dividend", "high", None, None, None, None, 20.0, 200.0, None),
    ("Bonus Issue 3:2", None, "bonus", "high", 3.0, 2.0, None, None, None, None, 0.4),
    ("Bonus - 1:2", None, "bonus", "high", 1.0, 2.0, None, None, None, None, 0.6666666666666666),
    ("Rights 1:5", None, "rights", "high", 1.0, 5.0, None, None, None, None, None),
    ("Buyback", None, "buyback", "high", None, None, None, None, None, None, None),
    ("Face Value Split From Rs 10/- To Rs 5/-", None, "split", "high", None, None, 10.0, 5.0, None, None, 0.5),
    ("Bonus", None, "bonus", "low", None, None, None, None, None, None, None),
    ("Split", None, "split", "low", None, None, None, None, None, None, None),
]


@pytest.mark.parametrize("row", REGRESSION, ids=[r[0].strip()[:40] for r in REGRESSION])
def test_t4_existing_parses_are_unchanged(row):
    subject, fv, typ, conf, rn, ro, ff, ft, amt, pct, factor = row
    p = parse_subject(subject, fv)
    assert (p.action_type.value, p.parse_confidence, p.ratio_new, p.ratio_old, p.face_value_from, p.face_value_to,
            p.dividend_amount, p.dividend_percent, adjustment_factor(p)) == (typ, conf, rn, ro, ff, ft, amt, pct, factor)


# T5 property ---------------------------------------------------------------------------------------------
def test_t5_random_pairs_in_every_wording():
    rng = random.Random(21)
    values = [1, 2, 5, 10, 100, 0.5, 2.5, 0.1, 1.25, 4, 20]
    n = 0
    while n < 200:
        a, b = rng.choice(values), rng.choice(values)
        if a == b:
            continue
        wording = WORDINGS[n % len(WORDINGS)]
        p, f = parsed(wording.format(a=a, b=b))
        assert p.parse_confidence == "high", (wording, a, b)
        assert f == float(b) / float(a), (wording, a, b)
        n += 1


def test_no_network_or_file_access_in_the_parser():
    import data.corporate_action_parser as mod
    src = open(mod.__file__, encoding="utf-8").read()
    for word in ("requests", "open(", "read_parquet", "urllib", "socket"):
        assert word not in src


# A dividend amount in front of the split text must not become the "from" value (found by the read-only run on the real files) ---
@pytest.mark.parametrize("subject,factor", [
    ("Annual General Meeting / Dividend - Rs 1.35/- Per Share / Face Value Split - From Rs 10/- Per Share To Rs 2/- Per Share", 0.2),
    ("Interim Dividend - Rs 1.50/- Per Share / Face Value Split - From Rs 10/- Per Share To Rs 5/- Per Share (Purpose Revised)", 0.5),
    ("Interim Dividend Rs.3/- Per Share And Face Value Split From Rs.5/- To Rs.2/- (Purpose Revised)", 0.4),
    ("Interim Dividend Rs 2/- Per Share And Face Value Split From Rs 10/- To Rs 2/-", 0.2),
    ("Interim Dividend Rs 2/- Per Share And Face Value Split From Rs 10/- To Re 2/-", 0.2),   # the Re wording after a dividend
    ("Dividend Rs 1.35 Fv Split Rs.10/- To Re.2/-", 0.2),                                      # no "from" at all, dividend first
])
def test_dividend_amount_before_the_split_text_is_not_the_from_value(subject, factor):
    p, f = parsed(subject)
    assert p.action_type == ActionType.SPLIT and p.parse_confidence == "high"
    assert f == pytest.approx(factor)


def test_amounts_before_the_split_keyword_still_parse_as_before():
    # parsed by the original "from rs N ... to rs N" pattern; the wider pattern only looks after the keyword
    p, f = parsed("From Rs 10/- To Rs 2/- Face Value Split")
    assert p.action_type == ActionType.SPLIT and p.parse_confidence == "high" and f == 0.2
