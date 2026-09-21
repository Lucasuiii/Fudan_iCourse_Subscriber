import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "frontend/js/app.js").read_text(encoding="utf-8")
CRYPTO = (ROOT / "frontend/js/crypto.js").read_text(encoding="utf-8")
HTML = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
DB_JS = (ROOT / "frontend/js/db.js").read_text(encoding="utf-8")
SCHEMA_JS = (ROOT / "frontend/js/schema.js").read_text(encoding="utf-8")


class FrontendSessionPrivacyTests(unittest.TestCase):
    def test_database_key_is_used_directly(self):
        self.assertIn("function _icsBuildStoragePassword", CRYPTO)
        self.assertIn("secrets.dbkey", CRYPTO)
        self.assertIn("value.length < 32", CRYPTO)
        self.assertIn("buildStoragePassword: _icsBuildStoragePassword", CRYPTO)
        self.assertIn("ICS.crypto.buildStoragePassword(creds)", APP)
        self.assertNotIn("ICS.crypto.buildPasswordV2(creds)", APP)

    def test_setup_no_longer_requests_uis_credentials(self):
        self.assertIn('x-model="setup.dbkey"', HTML)
        self.assertNotIn('x-model="setup.stuid"', HTML)
        self.assertNotIn('x-model="setup.uispsw"', HTML)
        self.assertNotIn("{key: 'stuid'", HTML)
        self.assertNotIn("{key: 'uispsw'", HTML)

    def test_credentials_are_session_only(self):
        self.assertIn(
            'sessionStorage.getItem(_LS + "creds")', APP
        )
        self.assertIn(
            'sessionStorage.setItem(_LS + "creds"', APP
        )
        self.assertNotIn(
            'localStorage.setItem(_LS + "creds"', APP
        )
        self.assertIn('localStorage.removeItem(_LS + "creds")', APP)

    def test_decrypted_database_cache_is_memory_only(self):
        self.assertIn("var _sessionBlobCache = new Map()", APP)
        self.assertNotIn("function _idbPut", APP)
        self.assertNotIn("function _idbGet", APP)
        self.assertIn("indexedDB.deleteDatabase(_legacyIdbName)", APP)

    def test_deleted_lectures_are_hidden_by_frontend_queries(self):
        self.assertIn("deleted_at TEXT", SCHEMA_JS)
        self.assertIn("deleted_at IS NULL", DB_JS)


if __name__ == "__main__":
    unittest.main()
