"""Trigger the public dashboard publisher through GitHub's workflow API."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .persistence import atomic_write_json

TOKEN_ENV = "TRADE3_GITHUB_ACTIONS_TOKEN"
HEARTBEAT_ENV = "TRADE3_PAGES_HEARTBEAT_PATH"
DISPATCH_URL = (
    "https://api.github.com/repos/ss4181/serhan-crypto-forecast-bot/"
    "actions/workflows/cloud-bot.yml/dispatches"
)
RUNS_URL = (
    "https://api.github.com/repos/ss4181/serhan-crypto-forecast-bot/"
    "actions/workflows/cloud-bot.yml/runs?branch=main&status=success&per_page=1"
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


def latest_publish_ms(token: str, *, opener=urlopen) -> int:  # type: ignore[no-untyped-def]
    """Return when the newest successful publish finished, per GitHub's API."""
    request = Request(
        RUNS_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "trade3-pages-dispatch/1.0",
        },
    )
    try:
        with opener(request, timeout=20) as response:
            payload = json.loads(response.read(2_000_000).decode("utf-8"))
        finished = str(payload["workflow_runs"][0]["updated_at"]).replace("Z", "+00:00")
        return int(datetime.fromisoformat(finished).timestamp() * 1000)
    except HTTPError as error:
        raise RuntimeError(f"GitHub run lookup rejected (HTTP {error.code})") from None
    except URLError:
        raise RuntimeError("GitHub run lookup could not reach GitHub API") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("GitHub run lookup returned an unexpected payload") from None


def record_publish_heartbeat(token: str, path: Path, *, opener=urlopen) -> None:  # type: ignore[no-untyped-def]
    """Publish proof for the bot, which runs as a different user than this timer."""
    atomic_write_json(
        path, {"version": 1, "published_at_ms": latest_publish_ms(token, opener=opener)}
    )
    path.chmod(0o644)


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "").strip()
    heartbeat = os.environ.get(HEARTBEAT_ENV, "").strip()
    if heartbeat:
        # Only the stale-dashboard alarm depends on this, so it must not block a refresh.
        try:
            record_publish_heartbeat(token, Path(heartbeat))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"Trade3 Pages heartbeat failed: {error}", file=sys.stderr)
    try:
        dispatch_workflow(token)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Trade3 Pages dispatch failed: {error}", file=sys.stderr)
        return 1
    print("Trade3 Pages workflow dispatch accepted by GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
