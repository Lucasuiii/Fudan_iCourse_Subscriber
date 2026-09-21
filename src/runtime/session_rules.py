"""Private per-course schedule allowlists for lecture processing."""

from __future__ import annotations

import re
from datetime import date


SessionRule = tuple[int, int, int]
SessionRules = dict[str, frozenset[SessionRule] | None]
SessionOverrideDates = frozenset[date]

_WEEKDAYS = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}
_RULE_RE = re.compile(
    r"周([一二三四五六日天])\s*第\s*(\d+)"
    r"(?:\s*[-—–~～至]\s*(\d+))?\s*节"
)
_PERIOD_RE = re.compile(
    r"第\s*(\d+)(?:\s*[-—–~～至]\s*(\d+))?\s*节"
)
_DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


class SessionRulesError(ValueError):
    """Raised for malformed rules without echoing their secret contents."""


def parse_course_session_rules(raw: str) -> SessionRules:
    """Parse private rules such as ``12345=周一第1-2节|周三第6-8节``.

    Blank input means no filtering.  ``全部`` (also ``ALL`` or ``*``) keeps
    every lecture for that course.  Error messages intentionally contain only
    a line number because workflow logs are public in a public fork.
    """
    parsed: SessionRules = {}
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise SessionRulesError(
                f"Invalid COURSE_SESSION_RULES at line {line_number}"
            )
        course_id, selection = (part.strip() for part in line.split("=", 1))
        if not course_id or not selection or course_id in parsed:
            raise SessionRulesError(
                f"Invalid COURSE_SESSION_RULES at line {line_number}"
            )

        tokens = [part.strip() for part in re.split(r"[|｜]", selection)]
        if any(not token for token in tokens):
            raise SessionRulesError(
                f"Invalid COURSE_SESSION_RULES at line {line_number}"
            )
        if len(tokens) == 1 and tokens[0].upper() in {"全部", "ALL", "*"}:
            parsed[course_id] = None
            continue

        rules: set[SessionRule] = set()
        for token in tokens:
            match = _RULE_RE.fullmatch(token)
            if not match:
                raise SessionRulesError(
                    f"Invalid COURSE_SESSION_RULES at line {line_number}"
                )
            start = int(match.group(2))
            end = int(match.group(3) or start)
            if start < 1 or end < start:
                raise SessionRulesError(
                    f"Invalid COURSE_SESSION_RULES at line {line_number}"
                )
            rules.add((_WEEKDAYS[match.group(1)], start, end))
        parsed[course_id] = frozenset(rules)
    return parsed


def parse_session_override_dates(raw: str) -> SessionOverrideDates:
    """Parse one-off dates that bypass recurring per-course allowlists.

    This is intended for make-up classes and timetable changes.  Values may
    be separated by commas, pipes, or newlines.  The value is kept in a
    separate secret so adding an exception never requires reading or
    replacing the existing write-only ``COURSE_SESSION_RULES`` secret.
    """
    parsed: set[date] = set()
    tokens = re.split(r"[,|｜\s]+", raw.strip()) if raw.strip() else []
    for index, token in enumerate(tokens, start=1):
        match = _DATE_RE.fullmatch(token)
        if not match:
            raise SessionRulesError(
                f"Invalid COURSE_SESSION_OVERRIDE_DATES item {index}"
            )
        try:
            parsed.add(date(*(int(part) for part in match.groups())))
        except ValueError as exc:
            raise SessionRulesError(
                f"Invalid COURSE_SESSION_OVERRIDE_DATES item {index}"
            ) from exc
    return frozenset(parsed)


def lecture_is_selected(
    course_id: str, lecture: dict, rules: SessionRules,
    override_dates: SessionOverrideDates = frozenset(),
) -> bool:
    """Return whether a lecture matches its course's configured allowlist.

    Courses absent from ``rules`` (or explicitly set to ``全部``) are
    unrestricted.  Configured courses fail closed when their date or period
    cannot be parsed, preventing an unexpected model call.
    """
    course_id = str(course_id)
    if course_id not in rules or rules[course_id] is None:
        return True

    sub_title = str(lecture.get("sub_title") or "")
    date_text = str(lecture.get("date") or "")
    date_match = _DATE_RE.search(date_text) or _DATE_RE.search(sub_title)
    if not date_match:
        return False

    try:
        lecture_date = date(*(int(part) for part in date_match.groups()))
    except ValueError:
        return False
    if lecture_date in override_dates:
        return True

    period_match = _PERIOD_RE.search(sub_title)
    if not period_match:
        return False
    start = int(period_match.group(1))
    end = int(period_match.group(2) or start)
    return (lecture_date.weekday(), start, end) in rules[course_id]
