"""Push the dashboard snapshot to the Serhan / Lab data gate.

Kept out of the CLI so that `serve` on an always-on host keeps the panel as
fresh as the GitHub Actions path does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


INGEST_URL_ENV = "PROJECT_HUB_INGEST_URL"
INGEST_TOKEN_ENV = "PROJECT_HUB_INGEST_TOKEN"
SITES_BYPASS_ENV = "OAI_SITES_BYPASS_TOKEN"
ACCEPTED_STATUS = 202


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def write_snapshot(report_dir: Path, snapshot: dict[str, object]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "cloud_snapshot.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def hub_configured() -> bool:
    return bool(os.environ.get(INGEST_URL_ENV, "").strip()) or bool(
        os.environ.get(INGEST_TOKEN_ENV, "").strip()
    )


def post_snapshot(snapshot: dict[str, object], *, opener=None) -> bool:  # type: ignore[no-untyped-def]
    endpoint = os.environ.get(INGEST_URL_ENV, "").strip()
    token = os.environ.get(INGEST_TOKEN_ENV, "").strip()
    if not endpoint and not token:
        return False
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{INGEST_URL_ENV} gecerli bir HTTPS adresi degil")
    if len(token) < 32:
        raise ValueError(f"{INGEST_TOKEN_ENV} tanimli veya yeterince guclu degil")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "btc-eth-probability-bot/0.1",
    }
    bypass = os.environ.get(SITES_BYPASS_ENV, "").strip()
    if bypass:
        headers["OAI-Sites-Authorization"] = f"Bearer {bypass}"
    request = Request(
        endpoint,
        data=json.dumps(
            snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    call = opener or build_opener(_NoRedirect()).open
    try:
        with call(request, timeout=20.0) as response:
            status = response.getcode()
            response.read(64 * 1024)
    except Exception:
        raise RuntimeError("Proje paneli guncellenemedi") from None
    if status != ACCEPTED_STATUS:
        raise RuntimeError("Proje paneli guncellemeyi kabul etmedi")
    return True


__all__ = [
    "INGEST_TOKEN_ENV",
    "INGEST_URL_ENV",
    "hub_configured",
    "post_snapshot",
    "write_snapshot",
]
