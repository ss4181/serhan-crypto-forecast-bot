from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from crypto_forecaster.pages_dispatch import DISPATCH_URL, dispatch_workflow


class PagesDispatchTests(unittest.TestCase):
    def test_dispatch_uses_only_main_ref_and_expected_minimum_permission_api(self) -> None:
        captured = {}

        class Response(BytesIO):
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def opener(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        dispatch_workflow("fine-grained-secret", opener=opener)
        request = captured["request"]
        self.assertEqual(request.full_url, DISPATCH_URL)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer fine-grained-secret")
        self.assertEqual(request.get_header("X-github-api-version"), "2022-11-28")
        self.assertEqual(request.data, b'{"ref": "main"}')
        self.assertEqual(captured["timeout"], 20)

    def test_dispatch_rejects_missing_token_without_network_call(self) -> None:
        with patch("crypto_forecaster.pages_dispatch.urlopen") as opener:
            with self.assertRaisesRegex(ValueError, "missing or malformed"):
                dispatch_workflow("  ", opener=opener)
            opener.assert_not_called()

    def test_http_error_does_not_leak_token(self) -> None:
        token = "private-token-value"

        def opener(_request, *, timeout):
            raise HTTPError(DISPATCH_URL, 401, "Unauthorized", {}, None)

        with self.assertRaises(RuntimeError) as raised:
            dispatch_workflow(token, opener=opener)
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn(token, str(raised.exception))

    def test_network_error_is_sanitized(self) -> None:
        def opener(_request, *, timeout):
            raise URLError("no route")

        with self.assertRaisesRegex(RuntimeError, "could not reach GitHub API"):
            dispatch_workflow("private-token-value", opener=opener)


if __name__ == "__main__":
    unittest.main()
