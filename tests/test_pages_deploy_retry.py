from __future__ import annotations

import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/cloud-bot.yml"
FIRST = "Publish GitHub Pages dashboard"
WAIT = "Wait before retrying the publish"
RETRY = "Retry GitHub Pages dashboard publish"


class PagesDeployRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = WORKFLOW.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.steps = re.split(r"\n(?=      - name: )", self.text)

    def step(self, name: str) -> str:
        matches = [block for block in self.steps if block.startswith(f"      - name: {name}\n")]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_first_attempt_is_allowed_to_fail(self) -> None:
        block = self.step(FIRST)
        self.assertIn("id: deployment\n", block)
        self.assertIn("continue-on-error: true", block)
        self.assertIn("uses: actions/deploy-pages@v4", block)

    def test_retry_runs_only_after_a_failed_first_attempt(self) -> None:
        for name in (WAIT, RETRY):
            self.assertIn("if: steps.deployment.outcome == 'failure'", self.step(name))
        retry = self.step(RETRY)
        self.assertIn("id: deployment_retry", retry)
        self.assertIn("uses: actions/deploy-pages@v4", retry)

    def test_second_failure_still_fails_the_run(self) -> None:
        self.assertNotIn("continue-on-error", self.step(RETRY))

    def test_steps_run_in_order_and_retry_is_last(self) -> None:
        names = [re.match(r"      - name: (.*)", block).group(1) for block in self.steps[1:]]
        self.assertEqual(names[-3:], [FIRST, WAIT, RETRY])

    def test_environment_url_uses_whichever_attempt_published(self) -> None:
        self.assertIn(
            "url: ${{ steps.deployment_retry.outputs.page_url || steps.deployment.outputs.page_url }}",
            self.text,
        )

    def test_artifact_upload_precedes_every_publish_attempt(self) -> None:
        upload = self.text.index("Upload dashboard artifact")
        self.assertLess(upload, self.text.index(f"- name: {FIRST}"))
        self.assertEqual(self.text.count("actions/deploy-pages@v4"), 2)


if __name__ == "__main__":
    unittest.main()
