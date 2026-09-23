import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.rerun_date import prepare, verify
from src.data.database import Database


class RerunDateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "icourse.db"
        self.manifest = Path(self.tmp.name) / "targets.json"
        db = Database(str(self.db_path))
        for course_id, sub_id, run_date in (
            ("course-a", "sub-a", "2026-09-22"),
            ("course-b", "sub-b", "2026-09-22"),
            ("course-a", "other-day", "2026-09-21"),
        ):
            db.upsert_course(course_id, course_id, "teacher")
            db.insert_lecture(sub_id, course_id, sub_id, run_date)
            db.update_transcript(sub_id, "original transcript")
            db.update_summary(sub_id, "original summary", "old/model")
            db.mark_processed(sub_id)
            db.mark_emailed(sub_id)
        db.conn.close()

    def test_prepare_and_verify_only_exact_date(self):
        env_path = Path(self.tmp.name) / "github-env"
        prepare(self.db_path, self.manifest, "2026-09-22", 2,
                "course-a,course-b", env_path)
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(set(manifest["ids"]), {"sub-a", "sub-b"})
        self.assertEqual(env_path.read_text().strip(),
                         "RERUN_TARGET_IDS=sub-a,sub-b")
        self.assertTrue(self.db_path.with_suffix(".pre-rerun.db").exists())

        with sqlite3.connect(self.db_path) as conn:
            untouched = conn.execute(
                "SELECT summary FROM lectures WHERE sub_id = 'other-day'"
            ).fetchone()[0]
            self.assertEqual(untouched, "original summary")
            for sub_id in manifest["ids"]:
                row = conn.execute(
                    "SELECT transcript, summary, processed_at, emailed_at "
                    "FROM lectures WHERE sub_id = ?", (sub_id,)
                ).fetchone()
                self.assertEqual(row, (None, None, None, None))

        with self.assertRaises(ValueError):
            verify(self.db_path, self.manifest)

        with sqlite3.connect(self.db_path) as conn:
            for sub_id in manifest["ids"]:
                conn.execute(
                    """UPDATE lectures SET transcript = 'new transcript',
                       summary = 'new summary', summary_model = 'new/model',
                       processed_at = '2026-09-23T02:00:00',
                       emailed_at = '2026-09-23T02:01:00'
                       WHERE sub_id = ?""", (sub_id,),
                )
        self.assertEqual(verify(self.db_path, self.manifest), 2)

    def test_wrong_count_does_not_reset_database(self):
        with self.assertRaises(ValueError):
            prepare(self.db_path, self.manifest, "2026-09-22", 3,
                    "course-a,course-b")
        self.assertFalse(self.manifest.exists())
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute(
                "SELECT summary FROM lectures WHERE sub_id = 'sub-a'"
            ).fetchone()[0], "original summary")

    def test_incomplete_target_is_rejected_before_reset(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE lectures SET emailed_at = NULL WHERE sub_id = 'sub-b'")
        with self.assertRaises(ValueError):
            prepare(self.db_path, self.manifest, "2026-09-22", 2,
                    "course-a,course-b")
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute(
                "SELECT summary FROM lectures WHERE sub_id = 'sub-a'"
            ).fetchone()[0], "original summary")


if __name__ == "__main__":
    unittest.main()
