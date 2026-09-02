"""Build a redacted, static signal dashboard payload for GitHub Pages."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .outcomes import load_ledger, pending_dir
from .scalping import load_scalp_target_ledger

SCHEMA = "trade3-signal-dashboard-v1"
SOURCE_STATUSES = frozenset({"fresh", "stale"})


def build_dashboard_payload(
    settings: Settings,
    *,
    now: datetime | None = None,
    limit: int = 2_000,
    source_status: str = "fresh",
) -> dict[str, Any]:
    """Return only allow-listed signal/outcome fields suitable for public hosting."""
    if limit < 1:
        raise ValueError("Dashboard limit pozitif olmali")
    if source_status not in SOURCE_STATUSES:
        raise ValueError("Dashboard veri durumu fresh veya stale olmali")
    current = now or datetime.now(UTC)
    signals: list[dict[str, Any]] = []
    regular_settled = load_ledger(settings.outcome_state_dir, limit=limit)
    for row in regular_settled[-limit:]:
        signals.append(
            {
                "kind": "regular",
                "signalId": str(row.get("signal_id", "")),
                "symbol": str(row.get("symbol", "")),
                "interval": str(row.get("interval", "")),
                "direction": str(row.get("direction", "")),
                "tier": str(row.get("tier", "")),
                "score": None,
                "families": [],
                "probabilityUp": _number(row.get("probability")),
                "probabilityDown": (
                    1.0 - _number(row.get("probability"))
                    if _number(row.get("probability")) is not None
                    else None
                ),
                "sourcePrice": _number(row.get("source_price")),
                "sourceTimeMs": _integer(row.get("source_close_time_ms")),
                "status": str(row.get("resolution", "SONUC")),
                "success": row.get("correct") is True,
                "netBps": _number(row.get("net_bps")),
                "targetPercent": None,
                "notified": True,
            }
        )
    for path in sorted(pending_dir(settings.outcome_state_dir).glob("*.json"))[-limit:]:
        row = _read_json(path)
        if not row:
            continue
        signals.append(
            {
                "kind": "regular",
                "signalId": str(row.get("signal_id", path.stem)),
                "symbol": str(row.get("symbol", "")),
                "interval": str(row.get("interval", "")),
                "direction": str(row.get("direction", "")),
                "tier": str(row.get("tier", "")),
                "score": None,
                "families": [],
                "probabilityUp": _number(row.get("probability")),
                "probabilityDown": (
                    1.0 - _number(row.get("probability"))
                    if _number(row.get("probability")) is not None
                    else None
                ),
                "sourcePrice": _number(row.get("source_price")),
                "sourceTimeMs": _integer(row.get("source_close_time_ms")),
                "status": "BEKLEMEDE",
                "success": None,
                "netBps": None,
                "targetPercent": None,
                "notified": True,
            }
        )
    scalp_rows = load_scalp_target_ledger(settings.scalp_state_dir, limit=limit)
    for row in scalp_rows[-limit:]:
        signals.append(
            {
                "kind": "scalp-target",
                "signalId": str(row.get("setup_id", "")),
                "symbol": str(row.get("spot_symbol", "")),
                "interval": "5m",
                "direction": str(row.get("direction", "")),
                "tier": "KURULUM",
                "score": _number(row.get("score")),
                "families": row.get("families", []),
                "probabilityUp": row.get("probability_up", {}),
                "probabilityDown": row.get("probability_down", {}),
                "sourcePrice": _number(row.get("source_price")),
                "sourceTimeMs": _integer(row.get("bar_close_time_ms")),
                "status": "HEDEF ULAŞTI" if row.get("hit") is True else "HEDEF ULAŞMADI",
                "success": row.get("hit") is True,
                "netBps": None,
                "targetPercent": _number(row.get("target_percent")),
                "notified": bool(row.get("notification_sent", False)),
            }
        )
    signals.sort(key=lambda row: int(row.get("sourceTimeMs") or 0), reverse=True)
    signals = signals[:limit]
    latest_signal_ms = max(
        (int(row["sourceTimeMs"]) for row in signals if row.get("sourceTimeMs")),
        default=None,
    )
    scalp_targets = [row for row in signals if row["kind"] == "scalp-target"]
    notified = [row for row in scalp_targets if row["notified"]]
    hit_count = sum(row["success"] is True for row in scalp_targets)
    notified_hit_count = sum(row["success"] is True for row in notified)
    return {
        "schema": SCHEMA,
        "generatedAtUtc": current.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sourceStatus": source_status,
        "latestSignalAtUtc": _milliseconds_to_utc_text(latest_signal_ms),
        "summary": {
            "signalCount": len(signals),
            "settledCount": sum(row["success"] is not None for row in signals),
            "pendingCount": sum(row["success"] is None for row in signals),
            "scalpTargetCount": len(scalp_targets),
            "scalpTargetHits": hit_count,
            "scalpTargetHitRate": hit_count / len(scalp_targets) if scalp_targets else None,
            "notifiedScalpTargetCount": len(notified),
            "notifiedScalpTargetHits": notified_hit_count,
            "notifiedScalpTargetHitRate": (
                notified_hit_count / len(notified) if notified else None
            ),
        },
        "signals": signals,
    }


def write_dashboard_payload(
    settings: Settings,
    output: Path,
    *,
    now: datetime | None = None,
    limit: int = 2_000,
    source_status: str = "fresh",
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload(
        settings, now=now, limit=limit, source_status=source_status
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _milliseconds_to_utc_text(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        moment = datetime.fromtimestamp(value / 1_000, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "SCHEMA",
    "SOURCE_STATUSES",
    "build_dashboard_payload",
    "write_dashboard_payload",
]
