"""Build a redacted, static signal dashboard payload for GitHub Pages."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .measurement import deadline_ms, measurement_summary
from .outcomes import load_ledger, pending_dir
from .scalping import (
    SCALP_TARGET_TOUCH_PERCENTS,
    load_pending_scalp_brackets,
    load_pending_scalp_targets,
    load_scalp_bracket_ledger,
    load_scalp_target_ledger,
)

SCHEMA = "trade3-signal-dashboard-v1"
SOURCE_STATUSES = frozenset({"fresh", "stale"})
HISTORY_LIMIT = 20_000


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
    now_ms = int(current.timestamp() * 1_000)
    history_limit = max(HISTORY_LIMIT, limit)
    signals: list[dict[str, Any]] = []
    source_counts: list[int] = []
    regular_settled = load_ledger(settings.outcome_state_dir, limit=history_limit)
    source_counts.append(len(regular_settled))
    for row in regular_settled:
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
    regular_pending = sorted(pending_dir(settings.outcome_state_dir).glob("*.json"))
    source_counts.append(len(regular_pending))
    for path in regular_pending[-history_limit:]:
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
    scalp_pending = load_pending_scalp_targets(settings.scalp_state_dir, limit=history_limit)
    source_counts.append(len(scalp_pending))
    for row in scalp_pending:
        for percent in SCALP_TARGET_TOUCH_PERCENTS:
            if percent in row.get("outcome_recorded_percents", []):
                continue
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
                    "status": "BEKLEMEDE",
                    "success": None,
                    "netBps": None,
                    "targetPercent": percent,
                    "targetPrice": float(row["source_price"]) * (1 + (1 if row["direction"] == "YUKARI" else -1) * percent / 100),
                    "horizonHours": int(row["horizon_ms"]) / 3_600_000,
                    "notified": bool(row.get("notification_sent", False)),
                    "strategy": str(row.get("strategy_label", "")),
                    "confidence": str(row.get("confidence", "")),
                    "successProbability": _number(row.get("success_probability")),
                    "expectedNetBps": _number(row.get("expected_net_bps")),
                    "qualityPercentile": _number(row.get("quality_percentile")),
                }
            )
    scalp_rows = load_scalp_target_ledger(settings.scalp_state_dir, limit=history_limit)
    source_counts.append(len(scalp_rows))
    for row in scalp_rows:
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
                "status": "HEDEF ULAŞTI" if row.get("hit") is True else "HEDEF ULAŞMADI" if row.get("hit") is False else "VERİ EKSİK",
                "success": row.get("hit") if type(row.get("hit")) is bool else None,
                "netBps": None,
                "targetPercent": _number(row.get("target_percent")),
                "targetPrice": _number(row.get("target_price")),
                "touchTimeMs": _integer(row.get("touch_close_time_ms")),
                "horizonHours": (_number(row.get("horizon_ms")) or 0) / 3_600_000,
                "notified": bool(row.get("notification_sent", False)),
                "strategy": str(row.get("strategy_label", "")),
                "confidence": str(row.get("confidence", "")),
                "successProbability": _number(row.get("success_probability")),
                "expectedNetBps": _number(row.get("expected_net_bps")),
                "qualityPercentile": _number(row.get("quality_percentile")),
            }
        )
    bracket_pending = load_pending_scalp_brackets(settings.scalp_state_dir, limit=history_limit)
    source_counts.append(len(bracket_pending))
    for row in bracket_pending:
        signals.append(
            {
                "kind": "scalp-bracket",
                "signalId": str(row.get("setup_id", "")),
                "symbol": str(row.get("spot_symbol", "")),
                "interval": "5m",
                "direction": str(row.get("direction", "")),
                "tier": "KURULUM",
                "score": _number(row.get("raw_score")),
                "families": row.get("families", []),
                "probabilityUp": None,
                "probabilityDown": None,
                "sourcePrice": _number(row.get("source_price")),
                "sourceTimeMs": _integer(row.get("bar_close_time_ms")),
                "status": "TP/SL BEKLEMEDE",
                "horizonHours": (_number(row.get("horizon_minutes")) or 0) / 60,
                "success": None,
                "netBps": None,
                "targetPercent": (
                    _number(row.get("target_bps")) / 100.0
                    if _number(row.get("target_bps")) is not None
                    else None
                ),
                "notified": bool(row.get("notification_sent", False)),
                "strategy": str(row.get("strategy_label", "")),
                "confidence": str(row.get("confidence", "")),
                "successProbability": _number(row.get("success_probability")),
                "expectedNetBps": _number(row.get("expected_net_bps")),
                "qualityPercentile": _number(row.get("quality_percentile")),
                "stopPercent": (
                    _number(row.get("stop_bps")) / 100.0
                    if _number(row.get("stop_bps")) is not None
                    else None
                ),
            }
        )
    bracket_rows = load_scalp_bracket_ledger(settings.scalp_state_dir, limit=history_limit)
    source_counts.append(len(bracket_rows))
    for row in bracket_rows:
        signals.append(
            {
                "kind": "scalp-bracket",
                "signalId": str(row.get("setup_id", "")),
                "symbol": str(row.get("spot_symbol", "")),
                "interval": "5m",
                "direction": str(row.get("direction", "")),
                "tier": "KURULUM",
                "score": _number(row.get("raw_score")),
                "families": row.get("families", []),
                "probabilityUp": None,
                "probabilityDown": None,
                "sourcePrice": _number(row.get("source_price")),
                "sourceTimeMs": _integer(row.get("bar_close_time_ms")),
                "status": str(row.get("resolution", "SONUÇ")),
                "horizonHours": (_number(row.get("horizon_minutes")) or 0) / 60,
                "touchTimeMs": _integer(row.get("exit_time_ms")),
                "entryPrice": _number(row.get("entry_price")),
                "targetPrice": _number(row.get("target_price")),
                "stopPrice": _number(row.get("stop_price")),
                "success": row.get("resolution") == "TARGET",
                "netBps": _number(row.get("net_bps")),
                "targetPercent": (
                    _number(row.get("target_bps")) / 100.0
                    if _number(row.get("target_bps")) is not None
                    else None
                ),
                "notified": bool(row.get("notification_sent", False)),
                "strategy": str(row.get("strategy_label", "")),
                "confidence": str(row.get("confidence", "")),
                "successProbability": _number(row.get("success_probability")),
                "expectedNetBps": _number(row.get("expected_net_bps")),
                "qualityPercentile": _number(row.get("quality_percentile")),
                "stopPercent": (
                    _number(row.get("stop_bps")) / 100.0
                    if _number(row.get("stop_bps")) is not None
                    else None
                ),
                "mfeBps": _number(row.get("mfe_bps")),
                "maeBps": _number(row.get("mae_bps")),
                "elapsedMinutes": _number(row.get("elapsed_minutes")),
            }
        )
    # A durable append may precede removal of pending state during a restart.
    # Prefer the settled evidence; never count both copies as separate trials.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(signals):
        key = (row["kind"], row["signalId"] or index, row.get("targetPercent"))
        previous = unique.get(key)
        if previous is None or previous["success"] is None or row["success"] is not None:
            unique[key] = row
    signals = list(unique.values())
    for row in signals:
        if row["kind"] not in {"scalp-target", "scalp-bracket"}:
            continue
        deadline = deadline_ms(row)
        row["deadlineTimeMs"] = deadline
        row["cohortMatured"] = deadline <= now_ms if deadline is not None else None
        if row["success"] is None and deadline is not None and deadline <= now_ms:
            row["status"] = "VERİ EKSİK"
    history_complete = all(count < history_limit for count in source_counts)
    measurements = measurement_summary(signals, now_ms=now_ms, history_complete=history_complete)
    signals.sort(key=lambda row: int(row.get("sourceTimeMs") or 0), reverse=True)
    latest_signal_ms = max(
        (int(row["sourceTimeMs"]) for row in signals if row.get("sourceTimeMs")),
        default=None,
    )
    scalp_targets = [row for row in signals if row["kind"] == "scalp-target"]
    settled_scalp_targets = [
        row for row in scalp_targets if row["success"] is not None
    ]
    pending_scalp_targets = [row for row in scalp_targets if row["success"] is None]
    notified = [row for row in settled_scalp_targets if row["notified"]]
    hit_count = sum(row["success"] is True for row in settled_scalp_targets)
    notified_hit_count = sum(row["success"] is True for row in notified)
    scalp_brackets = [row for row in signals if row["kind"] == "scalp-bracket"]
    settled_scalp_brackets = [
        row for row in scalp_brackets if row["success"] is not None
    ]
    bracket_wins = sum(row["success"] is True for row in settled_scalp_brackets)
    return {
        "schema": SCHEMA,
        "generatedAtUtc": current.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sourceStatus": source_status,
        "latestSignalAtUtc": _milliseconds_to_utc_text(latest_signal_ms),
        "measurements": measurements,
        "displayedCount": min(len(signals), limit),
        "historyLimitReached": not history_complete,
        "summary": {
            "signalCount": len(signals),
            "settledCount": sum(row["success"] is not None for row in signals),
            "pendingCount": sum(row["success"] is None for row in signals),
            "scalpTargetCount": len(scalp_targets),
            "settledScalpTargetCount": len(settled_scalp_targets),
            "pendingScalpTargetCount": len(pending_scalp_targets),
            "scalpTargetHits": hit_count,
            # Deprecated pooled rates mixed horizons and early wins. Keep the
            # keys nullable for old clients; use measurements.audiences instead.
            "scalpTargetHitRate": None,
            "notifiedScalpTargetCount": len(notified),
            "notifiedScalpTargetHits": notified_hit_count,
            "notifiedScalpTargetHitRate": None,
            "targetLevels": {
                str(int(level)): {
                    "hits": sum(r["success"] is True for r in scalp_targets if r.get("targetPercent") == level),
                    "misses": sum(r["success"] is False for r in scalp_targets if r.get("targetPercent") == level),
                    "pending": sum(r["success"] is None for r in scalp_targets if r.get("targetPercent") == level),
                }
                for level in SCALP_TARGET_TOUCH_PERCENTS
            },
            "scalpBracketCount": len(scalp_brackets),
            "settledScalpBracketCount": len(settled_scalp_brackets),
            "scalpBracketWins": bracket_wins,
            "scalpBracketWinRate": None,
        },
        "signals": signals[:limit],
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
    output.write_text(
        dashboard_payload_text(
            settings, now=now, limit=limit, source_status=source_status
        ),
        encoding="utf-8",
    )
    return output


def dashboard_payload_text(
    settings: Settings,
    *,
    now: datetime | None = None,
    limit: int = 2_000,
    source_status: str = "fresh",
) -> str:
    """Serialize the public allow-list for a file or restricted SSH command."""
    payload = build_dashboard_payload(
        settings, now=now, limit=limit, source_status=source_status
    )
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
    except (TypeError, ValueError, OverflowError):
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
    "dashboard_payload_text",
    "write_dashboard_payload",
]
