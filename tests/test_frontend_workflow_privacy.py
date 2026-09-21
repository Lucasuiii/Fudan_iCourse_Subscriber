import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITHUB_JS = (ROOT / "frontend/js/github.js").read_text(encoding="utf-8")


def _function_source(name: str, next_marker: str) -> str:
    pattern = rf"async function {re.escape(name)}\b.*?(?={re.escape(next_marker)})"
    match = re.search(pattern, GITHUB_JS, flags=re.DOTALL)
    if not match:
        raise AssertionError(f"Could not find function {name}")
    return match.group(0)


class FrontendWorkflowPrivacyTests(unittest.TestCase):
    def test_single_run_selection_uses_dedicated_secret(self):
        source = _function_source(
            "_triggerSingleRunWorkflow", "async function _triggerDeleteWorkflow"
        )
        self.assertIn("SINGLE_RUN_REQUEST", source)
        dispatch_inputs = re.search(
            r"const inputs = \{(.*?)\};", source, flags=re.DOTALL
        ).group(1)
        self.assertNotIn("course_ids", dispatch_inputs)
        self.assertIn("request_id", dispatch_inputs)
        self.assertIn('useOfficial ? "true" : "false"', source)

        workflow = (ROOT / ".github/workflows/single_run.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("secrets.SINGLE_RUN_REQUEST", workflow)
        self.assertIn("scripts/private_action_request.py single-run", workflow)

    def test_export_selection_is_not_workflow_metadata(self):
        source = _function_source("_triggerExportWorkflow", "window.ICS.github")
        self.assertIn("EXPORT_REQUEST", source)
        dispatch_inputs = re.search(
            r"const payload = \{.*?inputs: \{(.*?)\}\s*,?\s*\};",
            source,
            flags=re.DOTALL,
        ).group(1)
        self.assertNotIn("course_id", dispatch_inputs)
        self.assertNotIn("sub_ids", dispatch_inputs)
        self.assertIn("request_id", dispatch_inputs)

        workflow = (ROOT / ".github/workflows/export.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(workflow, r"(?m)^\s+(course_id|sub_ids):\s*$")

    def test_delete_selection_is_not_workflow_metadata(self):
        source = _function_source(
            "_triggerDeleteWorkflow", "async function _triggerExportWorkflow"
        )
        self.assertIn("DELETE_REQUEST", source)
        self.assertIn("inputs: { request_id: requestId }", source)
        self.assertNotIn("inputs: { course_ids", source)

        workflow = (ROOT / ".github/workflows/delete_course.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(workflow, r"(?m)^\s+(course_ids|sub_ids):\s*$")


if __name__ == "__main__":
    unittest.main()
