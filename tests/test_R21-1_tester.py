"""TESTER edge cases for R21-1 (corporate-action parser). Pure string tests, no data."""
import pytest

from data.corporate_action_parser import adjustment_factor, parse_subject


def f(subject):
    return adjustment_factor(parse_subject(subject))


@pytest.mark.parametrize("subject,expected", [
    ("FACE VALUE SPLIT FROM RS.10/- TO RE.1/-", 0.1),
    ("face value split rs 10 to re 2", 0.2),
    ("Fv Split Rs.10/-To Re.1/-", 0.1),                       # no space before "To"
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 0.1),
    ("Sub-Division/Split From Rs 5/- To Re 1/- Per Share Rd Date Revised", 0.2),
    ("Face Value Split Rs 2.50 To Re 0.50", 0.2),
])
def test_wordings_with_odd_spacing_and_case(subject, expected):
    assert f(subject) == pytest.approx(expected)


@pytest.mark.parametrize("subject", [
    "Face Value Split Rs 0 To Rs 1", "Face Value Split Rs 10 To Rs 0", "Face Value Split Rs 10 To Rs 10",
    "Face Value Split From Rs 10", "Face Value Split", "Split Us 64 Into 2 Parts", "Face Value Split Rs.10/- To",
])
def test_unusable_split_subjects_never_get_a_factor(subject):
    assert f(subject) is None


def test_first_pair_after_the_split_keyword_wins():
    assert f("Face Value Split Rs 10 To Rs 2 / Dividend Rs 1 To Rs 3") == pytest.approx(0.2)


def test_dividend_amount_before_the_keyword_never_becomes_the_from_value():
    assert f("Dividend - Rs 1.35/- Per Share / Face Value Split Rs 10/- To Rs 2/-") == pytest.approx(0.2)
    assert f("Interim Dividend Rs 4 Per Share Face Value Split Rs.10 To Re.1") == pytest.approx(0.1)


@pytest.mark.parametrize("subject", ["", " ", "None"])
def test_empty_like_subjects_do_not_crash_and_have_no_factor(subject):
    assert f(subject) is None


def test_bonus_branch_still_wins_over_split_wording():
    p = parse_subject("Bonus 1:1 And Face Value Split Rs 10 To Rs 5")
    assert str(getattr(p.action_type, "value", p.action_type)) == "bonus"


@pytest.mark.parametrize("subject", [
    "Bonus Debentures 1:1", "Bonus Preference Shares 2:1", "Bonus Warrants 1:2", "Bonus Pref Shares 1:1",
    "Bonus NCDs 1:1", "Bonus Ncd 1:1", "Bonus Bonds 1:1", "BONUS DEBENTURE 3:1",
])
def test_non_equity_bonus_wordings_get_no_factor(subject):
    assert f(subject) is None


@pytest.mark.parametrize("subject,expected", [
    ("Bonus 1:1", 0.5), ("Bonus Issue 3:2", 0.4), ("Bonus 1:1 Bondada Engineering", 0.5),
    ("Bonus 2:1 Prefecture Ltd", pytest.approx(1/3, rel=1e-4)),
])
def test_equity_bonus_keeps_its_factor_and_look_alike_words_are_not_matched(subject, expected):
    assert f(subject) == pytest.approx(expected, rel=1e-4)


def test_non_equity_bonus_still_records_the_ratio():
    p = parse_subject("Scheme Of Arrangement - Bonus Debentures 6:1")
    assert (p.ratio_new, p.ratio_old) == (6.0, 1.0) and p.parse_confidence == "low"


def test_consolidation_via_split_keyword_gives_factor_above_one():
    assert f("Face Value Split Rs 1 To Rs 10") == pytest.approx(10.0)
