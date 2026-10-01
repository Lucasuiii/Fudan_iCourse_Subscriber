import json
import tempfile
import unittest
from pathlib import Path

from src.ai.course_glossary import course_terms, terminology_reference


class CourseGlossaryTests(unittest.TestCase):
    def test_normalized_course_title_and_no_cross_course_leakage(self):
        self.assertIn("扰动", course_terms("数值算法与案例分析 Ⅰ"))
        self.assertIn("行列式", course_terms("高等代数 I"))
        self.assertIn("哈希表", course_terms("数据结构（H）"))
        self.assertIn("BFS", course_terms("数据结构 (H)"))
        self.assertEqual(len(course_terms("数据结构（H）")), 30)
        self.assertEqual(course_terms("数据结构"), [])
        self.assertEqual(course_terms("高等代数II"), [])
        self.assertEqual(course_terms("其他课程"), [])
        self.assertEqual(terminology_reference("其他课程"), "")

    def test_invalid_entries_are_filtered_and_terms_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glossary.json"
            path.write_text(json.dumps([{"courses": ["A"], "terms":
                ["矩阵", "矩阵", "", 12, "<injection>", "换\n行", "x" * 41]
                + [f"词{i}" for i in range(100)]}]), encoding="utf-8")
            terms = course_terms("A", path)
            self.assertEqual(terms[0], "矩阵")
            self.assertEqual(len(terms), 30)
            self.assertNotIn("<injection>", terms)
            self.assertEqual(terms.count("矩阵"), 1)

    def test_bad_or_missing_config_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glossary.json"
            self.assertEqual(course_terms("A", path), [])
            path.write_text("broken json", encoding="utf-8")
            self.assertEqual(course_terms("A", path), [])
            path.write_text('{}', encoding="utf-8")
            self.assertEqual(course_terms("A", path), [])
