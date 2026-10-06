import subprocess
from pathlib import Path
import unittest


class FrontendSecurityTests(unittest.TestCase):
    def test_real_dom_search_and_markdown_payloads(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['node', 'scripts/test_frontend_security.cjs'], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
