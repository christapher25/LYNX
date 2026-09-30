"""data/corporate_action_parser.py — v2, fixes based on real NSE data"""

import re
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    BONUS = "bonus"
    SPLIT = "split"
    DIVIDEND = "dividend"
    RIGHTS = "rights"
    BUYBACK = "buyback"
    NO_ACTION = "no_action"   # AGM/book-closure notices — no price impact
    UNKNOWN = "unknown"


@dataclass
class ParsedAction:
    action_type: ActionType
    raw_subject: str
    ratio_new: float | None = None
    ratio_old: float | None = None
    face_value_from: float | None = None
    face_value_to: float | None = None
    dividend_amount: float | None = None
    dividend_percent: float | None = None
    parse_confidence: str = "high"


_BONUS_RATIO = re.compile(r"bonus.*?(\d+)\s*(?::|for|/)\s*(\d+)", re.IGNORECASE)

_SPLIT_FACE_VALUE = re.compile(
    r"from\s+rs\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*.*?to\s+rs\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Rs./Re., optional missing space, decimal optional, "Per Sh"/"Per Share"/"Pr Shr" all accepted
_DIVIDEND_RUPEE = re.compile(
    r"(?:div|divided).*?r[se]\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*(?:per\s*sh|pr\s*shr)",
    re.IGNORECASE,
)
# fallback: rupee amount present anywhere after div/divided, no "per share" required
_DIVIDEND_RUPEE_LOOSE = re.compile(
    r"(?:div|divided).*?r[se]\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)

_DIVIDEND_PERCENT = re.compile(r"(?:div|divided).*?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)

_RIGHTS_RATIO = re.compile(r"rights.*?(\d+)\s*(?::|for|/)\s*(\d+)", re.IGNORECASE)

# Pure administrative notices — no dividend/bonus/split attached
_NO_ACTION_PATTERNS = (
    re.compile(r"^\s*annual\s*general\s*meeting\s*$", re.IGNORECASE),
    re.compile(r"^\s*agm\s*$", re.IGNORECASE),
    re.compile(r"^\s*agm\s+bc\s+revised\s*$", re.IGNORECASE),
    re.compile(r"^\s*annual\s*book\s*closure\s*$", re.IGNORECASE),
)


def _is_no_action(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text).strip()
    return any(p.match(stripped) for p in _NO_ACTION_PATTERNS)


def parse_subject(subject: str, face_val_at_record: str | None = None) -> ParsedAction:
    text = subject.strip()
    lower = text.lower()

    if _is_no_action(text):
        return ParsedAction(ActionType.NO_ACTION, subject, parse_confidence="high")

    if "bonus" in lower:
        m = _BONUS_RATIO.search(text)
        if m:
            return ParsedAction(
                ActionType.BONUS, subject,
                ratio_new=float(m.group(1)), ratio_old=float(m.group(2)),
                parse_confidence="high",
            )
        return ParsedAction(ActionType.BONUS, subject, parse_confidence="low")

    if "split" in lower or "sub-division" in lower or "subdivision" in lower:
        m = _SPLIT_FACE_VALUE.search(text)
        if m:
            return ParsedAction(
                ActionType.SPLIT, subject,
                face_value_from=float(m.group(1)), face_value_to=float(m.group(2)),
                parse_confidence="high",
            )
        return ParsedAction(ActionType.SPLIT, subject, parse_confidence="low")

    if "rights" in lower:
        m = _RIGHTS_RATIO.search(text)
        if m:
            return ParsedAction(
                ActionType.RIGHTS, subject,
                ratio_new=float(m.group(1)), ratio_old=float(m.group(2)),
                parse_confidence="high",
            )
        return ParsedAction(ActionType.RIGHTS, subject, parse_confidence="low")

    if "buyback" in lower or "buy back" in lower:
        return ParsedAction(ActionType.BUYBACK, subject, parse_confidence="high")

    if "demerger" in lower or "scheme of" in lower or "arrangement" in lower:
        # Company-specific ratio, not expressible from subject text alone —
        # correctly unparseable, never guess.
        return ParsedAction(ActionType.UNKNOWN, subject, parse_confidence="failed")

    if "div" in lower:  # catches div, dividend, divided (typo variant seen in real data)
        m = _DIVIDEND_RUPEE.search(text)
        if not m:
            m = _DIVIDEND_RUPEE_LOOSE.search(text)
        if m:
            return ParsedAction(
                ActionType.DIVIDEND, subject,
                dividend_amount=float(m.group(1)), parse_confidence="high",
            )

        m_pct = _DIVIDEND_PERCENT.search(text)
        if m_pct:
            pct = float(m_pct.group(1))
            amount = None
            if face_val_at_record:
                try:
                    amount = round(float(face_val_at_record) * pct / 100, 4)
                except ValueError:
                    pass
            return ParsedAction(
                ActionType.DIVIDEND, subject,
                dividend_percent=pct, dividend_amount=amount,
                parse_confidence="high" if amount is not None else "low",
            )

        return ParsedAction(ActionType.DIVIDEND, subject, parse_confidence="low")

    return ParsedAction(ActionType.UNKNOWN, subject, parse_confidence="failed")


def adjustment_factor(action: ParsedAction) -> float | None:
    if action.parse_confidence in ("failed", "low"):
        return None
    if action.action_type == ActionType.BONUS:
        if action.ratio_new is not None and action.ratio_old is not None:
            return action.ratio_old / (action.ratio_old + action.ratio_new)
        return None
    if action.action_type == ActionType.SPLIT:
        if action.face_value_from and action.face_value_to:
            return action.face_value_to / action.face_value_from
        return None
    return None


if __name__ == "__main__":
    cases = [
        ("Dividend - Rs 6 Per Share", "10"),
        ("Dividend-Rs.7.5/- Per Sh.", "10"),
        ("Agm/Dividend-500%", "1"),
        ("Bonus 1:1", None),
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", None),
        ("Something Totally Unrecognized XYZ", None),
        ("Annual General Meeting", None),
        (" Annual General Meeting", None),
        ("Agm", None),
        ("Agm/Div-Rs.1.50 Per Share", "10"),
        ("Agm/Div-Rs.1.50/- Per Sh", "10"),
        ("Agm/Div.-Re.1/- Per Share", "10"),
        ("Int Div-Rs.2/- Per Share", "10"),
        ("2nd Int Div-Rs.2/- Pr Shr", "10"),
        ("1st Int Div-Rs1.20 Per Shpurpose Revised", "10"),
        ("Int Div-Re.0.50 Per Sh   Purpose Revised", "10"),
        ("Demerger", None),
        (" Scheme Of Arrangement & Reconstruction", None),
        ("Agm                      Bc Revised", None),
        (" Annual General Meeting/Divided - Rs 35/- Per Share", "10"),
        ("Agm/Div-Int-200%+Fin-200%", "10"),
    ]
    for subj, fv in cases:
        r = parse_subject(subj, fv)
        print(f"{r.parse_confidence:7s} {r.action_type.value:10s} {subj!r} -> amt={r.dividend_amount} pct={r.dividend_percent}")