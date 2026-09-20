import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.validate_db import validate_database
from src.data.database import Database
from src.data.schema import SCHEMA_SQL


class DatabaseSafetyTests(unittest.TestCase):
    def test_accepts_complete_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            conn = sqlite3.connect(path)
            conn.executescript(SCHEMA_SQL)
            conn.close()
            validate_database(str(path))

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            path.touch()
            with self.assertRaises(ValueError):
                validate_database(str(path))

    def test_rejects_non_sqlite_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            path.write_bytes(b"not a sqlite database")
            with self.assertRaises(sqlite3.Error):
                validate_database(str(path))

    def test_rejects_sqlite_with_incomplete_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE unrelated (id INTEGER)")
            conn.close()
            with self.assertRaises(ValueError):
                validate_database(str(path))

    def test_unusable_transcript_can_be_cleared_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            db = Database(str(path))
            db.upsert_course("course", "课程", "教师")
            db.insert_lecture("lecture", "course", "课次", "2026-09-20")
            db.update_transcript("lecture", "unusable")
            db.clear_transcript("lecture")
            self.assertIsNone(db.get_lecture("lecture")["transcript"])
            db.conn.close()

    def test_ppt_status_counts_include_persisted_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icourse.db"
            db = Database(str(path))
            db.upsert_course("course", "课程", "教师")
            db.insert_lecture("lecture", "course", "课次", "2026-09-20")
            with db.conn:
                db.conn.executemany(
                    """INSERT INTO ppt_pages
                       (sub_id, page_num, created_sec, ocr_status)
                       VALUES ('lecture', ?, ?, ?)""",
                    [(1, 0, "failed"), (2, 30, "failed"),
                     (3, 60, "dedup_dropped")],
                )
            self.assertEqual(
                db.get_ppt_status_counts("lecture"),
                {"failed": 2, "dedup_dropped": 1},
            )
            db.conn.close()
