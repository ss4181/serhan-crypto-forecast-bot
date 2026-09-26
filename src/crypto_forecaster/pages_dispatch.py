"""Trigger the public dashboard publisher through GitHub's workflow API."""
from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOKEN_ENV = "TRADE3_GITHUB_ACTIONS_TOKEN"
DISPATCH_URL = (
    "https://api.github.com/repos/ss4181/serhan-crypto-forecast-bot/"
    "actions/workflows/cloud-bot.yml/dispatches"
)


def dispatch_workflow(token: str, *, opener=urlopen) -> None:  # type: ignore[no-untyped-def]
    """Request one Pages refresh; never expose the bearer token in errors."""
    if not token or not token.strip() or "\n" in token or "\r" in token:
        raise ValueError("GitHub Actions token is missing or malformed")
    request = Request(
        DISPATCH_URL,
        data=json.dumps({"ref": "main"}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "trade3-pages-dispatch/1.0",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=20) as response:
            if getattr(response, "status", 204) != 204:
                raise RuntimeError(
                    f"GitHub workflow dispatch returned HTTP {response.status}"
                )
    except HTTPError as error:
        raise RuntimeError(
            f"GitHub workflow dispatch rejected (HTTP {error.code}); "
            "check token expiry and Actions: write permission"
        ) from None
    except URLError:
        raise RuntimeError("GitHub workflow dispatch could not reach GitHub API") from None


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "").strip()
    try:
        dispatch_workflow(token)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Trade3 Pages dispatch failed: {error}", file=sys.stderr)
        return 1
    print("Trade3 Pages workflow dispatch accepted by GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
