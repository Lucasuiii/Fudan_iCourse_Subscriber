import json
import unittest

from scripts.private_action_request import parse_request


REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"


class PrivateActionRequestTests(unittest.TestCase):
    def _envelope(self, **extra):
        return json.dumps({
            "request_id": REQUEST_ID,
            "course_ids": "course-1,course:2",
            **extra,
        })

    def test_single_run_exports_only_course_selection(self):
        self.assertEqual(
            parse_request("single-run", REQUEST_ID, self._envelope()),
            {"COURSE_IDS": "course-1,course:2"},
        )

    def test_export_and_delete_keep_optional_lecture_selection(self):
        envelope = self._envelope(sub_ids="lecture.1,lecture_2")
        self.assertEqual(
            parse_request("export", REQUEST_ID, envelope),
            {
                "EXPORT_COURSE_ID": "course-1,course:2",
                "EXPORT_SUB_IDS": "lecture.1,lecture_2",
            },
        )
        self.assertEqual(
            parse_request("delete", REQUEST_ID, envelope),
            {
                "COURSE_IDS_INPUT": "course-1,course:2",
                "SUB_IDS_INPUT": "lecture.1,lecture_2",
            },
        )

    def test_rejects_stale_or_mismatched_request(self):
        stale = self._envelope().replace(REQUEST_ID, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_request("single-run", REQUEST_ID, stale)

    def test_rejects_shell_metacharacters(self):
        with self.assertRaisesRegex(ValueError, "invalid format"):
            parse_request(
                "delete", REQUEST_ID,
                self._envelope(sub_ids="x'; echo leaked"),
            )

    def test_rejects_non_uuid_request_id(self):
        with self.assertRaisesRegex(ValueError, "request ID"):
            parse_request("single-run", "not-a-uuid", self._envelope())


if __name__ == "__main__":
    unittest.main()
