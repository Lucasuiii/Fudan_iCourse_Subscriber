import unittest

from src.runtime.session_rules import (
    SessionRulesError,
    lecture_is_selected,
    parse_course_session_rules,
    parse_course_session_exclusions,
    parse_session_override_dates,
)


class SessionRulesTests(unittest.TestCase):
    def test_monday_afternoon_exclusion_preserves_other_sessions_and_courses(self):
        exclusions = parse_course_session_exclusions('123=周一第6-10节')
        for day, period, expected in [('2026-09-28','9-10',False), ('2026-09-28','6-8',False),
                                     ('2026-09-28','1-2',True), ('2026-09-28','11-13',True),
                                     ('2026-09-29','9-10',True)]:
            with self.subTest(day=day, period=period):
                self.assertEqual(lecture_is_selected('123',{'date':day,'sub_title':f'第{period}节'}, {},
                                                     exclusions=exclusions),expected)
        self.assertTrue(lecture_is_selected('456',{}, {},exclusions=exclusions))

    def test_exclusions_take_precedence_over_allowlist_date_overrides(self):
        exclusions = parse_course_session_exclusions('123=周一第6-10节')
        self.assertFalse(lecture_is_selected('123',{'date':'2026-09-28','sub_title':'第9-10节'},
            parse_course_session_rules('123=周二第9-10节'),parse_session_override_dates('2026-09-28'),
            exclusions=exclusions))
        self.assertFalse(lecture_is_selected('123',{'date':'2026-09-28','sub_title':'第5-6节'}, {},
                                            exclusions=exclusions))

    def test_exclusion_invalid_lecture_fails_closed_and_parser_hides_values(self):
        exclusions = parse_course_session_exclusions('123=周一第6-10节')
        for lecture in ({}, {'date':'2026-02-30','sub_title':'第9-10节'}, {'date':'2026-09-28','sub_title':'第10-9节'}):
            self.assertFalse(lecture_is_selected('123',lecture,{},exclusions=exclusions))
        with self.assertRaises(SessionRulesError) as caught:parse_course_session_exclusions('private=invalid')
        self.assertIn('COURSE_SESSION_EXCLUSIONS',str(caught.exception))
        self.assertNotIn('private',str(caught.exception))

    def test_blank_rules_allow_every_course(self):
        rules = parse_course_session_rules("")
        self.assertTrue(lecture_is_selected("123", {}, rules))

    def test_course_not_listed_is_unrestricted(self):
        rules = parse_course_session_rules("123=周一第1-2节")
        self.assertTrue(lecture_is_selected("456", {}, rules))

    def test_all_rule_is_unrestricted(self):
        rules = parse_course_session_rules("123=全部")
        self.assertTrue(lecture_is_selected("123", {}, rules))

    def test_matches_weekday_and_period(self):
        rules = parse_course_session_rules(
            "123=周一第1-2节|周三第6-8节"
        )
        self.assertTrue(lecture_is_selected(
            "123",
            {"date": "2026-09-14", "sub_title": "2026-09-14第1-2节"},
            rules,
        ))
        self.assertTrue(lecture_is_selected(
            "123",
            {"date": "", "sub_title": "2026-09-16第6-8节"},
            rules,
        ))

    def test_configured_course_fails_closed(self):
        rules = parse_course_session_rules("123=周一第1-2节")
        self.assertFalse(lecture_is_selected(
            "123",
            {"date": "2026-09-14", "sub_title": "课次名称无法识别"},
            rules,
        ))
        self.assertFalse(lecture_is_selected(
            "123",
            {"date": "2026-09-15", "sub_title": "2026-09-15第1-2节"},
            rules,
        ))

    def test_one_off_date_bypasses_recurring_allowlist(self):
        rules = parse_course_session_rules("123=周一第1-2节")
        overrides = parse_session_override_dates("2026-09-20")
        self.assertTrue(lecture_is_selected(
            "123",
            {"date": "2026-09-20", "sub_title": "2026-09-20第6-8节"},
            rules,
            overrides,
        ))
        self.assertFalse(lecture_is_selected(
            "123",
            {"date": "2026-09-21", "sub_title": "2026-09-21第6-8节"},
            rules,
            overrides,
        ))

    def test_override_dates_support_separators_and_hide_secret(self):
        parsed = parse_session_override_dates(
            "2026-09-20,2026-10-01|2026-10-02"
        )
        self.assertEqual(len(parsed), 3)
        secret = "not-a-date"
        with self.assertRaisesRegex(SessionRulesError, r"item 1") as ctx:
            parse_session_override_dates(secret)
        self.assertNotIn(secret, str(ctx.exception))

    def test_invalid_rule_reports_only_line_number(self):
        secret = "123=周八第1-2节"
        with self.assertRaisesRegex(SessionRulesError, r"line 1") as ctx:
            parse_course_session_rules(secret)
        self.assertNotIn(secret, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
