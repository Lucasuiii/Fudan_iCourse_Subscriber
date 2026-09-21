import os
import sqlite3
import tempfile
import unittest

from scripts.verify_db_equivalence import verify_equivalent


class VerifyDatabaseEquivalenceTests(unittest.TestCase):
    def _database(self, rows, *, extra_table=False):
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO items VALUES (?, ?)", rows)
            if extra_table:
                conn.execute("CREATE TABLE extra (id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_accepts_equivalent_databases(self):
        left = self._database([(1, "alpha"), (2, "beta")])
        right = self._database([(2, "beta"), (1, "alpha")])
        verify_equivalent(left, right)

    def test_rejects_changed_rows(self):
        left = self._database([(1, "alpha")])
        right = self._database([(1, "changed")])
        with self.assertRaisesRegex(ValueError, "row contents differ"):
            verify_equivalent(left, right)

    def test_rejects_changed_table_set(self):
        left = self._database([(1, "alpha")])
        right = self._database([(1, "alpha")], extra_table=True)
        with self.assertRaisesRegex(ValueError, "table sets differ"):
            verify_equivalent(left, right)


if __name__ == "__main__":
    unittest.main()
