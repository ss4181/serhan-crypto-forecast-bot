"""Prospective, isolated comparison of consensus and fixed-strategy horizons.

This module never sends messages or promotes a policy. Both arms pass through
the same candidate gates; cooldown, top-K and delivery are outside this study.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import Settings
from .measurement import _wilson_interval
from .persistence import atomic_write_json
from .scalping import (
    CONSENSUS_DIRECTION_POLICY,
    SCALP_STEP_MS,
    STRATEGY_HORIZON_DIRECTION_POLICY,
    ScalpScanReport,
    filter_scalp_notification_report,
    load_pending_scalp_targets,
    load_scalp_bracket_ledger,
    load_scalp_setup_forward_ledger,
    load_scalp_target_ledger,
    record_scalp_bracket_setups,
    record_scalp_setup_forward_setups,
    record_scalp_target_setups,
    settle_scalp_bracket_outcomes,
    settle_scalp_setup_forward,
    settle_scalp_target_outcomes,
)
from .universe import UniverseManifest

VERSION = "strategy-horizon-shadow-v1"
LIMIT = 100_000
ARM_POLICIES = {
    "baseline": CONSENSUS_DIRECTION_POLICY,
    "proposed": STRATEGY_HORIZON_DIRECTION_POLICY,
}


def _filter_options(settings: Settings) -> dict[str, Any]:
    mapping = {
        "minimum_score": "minimum_alert_score",
        "minimum_quality_percentile": "minimum_quality_percentile",
        "minimum_direction_probability": "minimum_direction_probability",
        "minimum_expected_net_bps": "minimum_expected_net_bps",
        "minimum_calibration_samples": "minimum_calibration_samples",
        "maximum_spread_bps": "maximum_spread_bps",
        "minimum_quote_volume_24h_usdt": "minimum_quote_volume_24h_usdt",
        "maximum_abs_funding_bps": "maximum_abs_funding_bps",
        "maximum_bar_volatility_bps": "maximum_bar_volatility_bps",
        "live_families": "live_families",
    }
    for regime in ("transition", "off"):
        mapping[f"{regime}_alerts_enabled"] = f"{regime}_alerts_enabled"
        for field in (
            "score",
            "quality_percentile",
            "direction_probability",
            "expected_net_bps",
            "calibration_samples",
        ):
            mapping[f"{regime}_minimum_{field}"] = (
                f"{regime}_minimum_{'alert_score' if field == 'score' else field}"
            )
    return {key: getattr(settings, f"scalp_{attr}") for key, attr in mapping.items()}


def _root(settings: Settings) -> Path:
    return settings.scalp_state_dir / "experiments" / VERSION


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def run_horizon_shadow(
    settings: Settings,
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    ledger: Iterable[dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Freeze decisions now and measure only future closed candles thereafter."""
    if not settings.scalp_horizon_shadow_enabled:
        return {}
    now_ms = int(now.timestamp() * 1000)
    options = _filter_options(settings)
    config = {
        "version": VERSION,
        "universe": manifest.version,
        "filters": options,
        "bracketMinutes": settings.scalp_bracket_horizon_minutes,
        "milestoneHours": settings.scalp_milestone_horizon_hours,
        "minimumCoverage": settings.scalp_minimum_coverage,
    }
    profile_id = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    root = _root(settings)
    profile = root / f"profile-{profile_id}"
    metadata = _read(profile / "config.json")
    if not metadata:
        metadata = {**config, "startedAtMs": now_ms}
        atomic_write_json(profile / "config.json", metadata)

    # Continue settling old profiles after settings change; do not combine their
    # performance with the current profile's thresholds.
    for previous in root.glob("profile-*"):
        for arm in ARM_POLICIES:
            directory = previous / arm
            settle_scalp_target_outcomes(directory, settings.scalp_data_dir, now=now)
            settle_scalp_bracket_outcomes(directory, settings.scalp_data_dir, now=now)
            settle_scalp_setup_forward(directory, settings.scalp_data_dir, now=now)

    # The ordinary family ledger settles all horizons together after 60m.
    # Reject future or incomplete evidence even when this function is replayed.
    historical = tuple(
        row
        for row in ledger
        if isinstance(row, dict)
        and _number(row.get("exit_time_ms")) is not None
        and float(row["exit_time_ms"]) <= report.evaluated_at_ms
        and _number(row.get("bar_close_time_ms")) is not None
        and float(row["bar_close_time_ms"]) + 3_600_000 <= report.evaluated_at_ms
        and row.get("resolution") != "DATA_MISSING"
    )
    expected_close = now_ms // SCALP_STEP_MS * SCALP_STEP_MS - 1
    healthy = (
        report.coverage >= settings.scalp_minimum_coverage
        and report.newest_close_time_ms == expected_close
        and abs(now_ms - report.evaluated_at_ms) < SCALP_STEP_MS
    )
    last_scan = _read(profile / "scan.json")
    already_recorded = report.newest_close_time_ms <= (last_scan.get("barCloseMs") or 0)
    scan_counts: dict[str, Any] = (
        last_scan.get("counts", {}) if already_recorded else {}
    )
    for arm, policy in ARM_POLICIES.items():
        if already_recorded:
            continue
        diagnostics: dict[str, int] = {}
        if healthy:
            filtered = filter_scalp_notification_report(
                report,
                **options,
                ledger=historical,
                direction_policy=policy,
                diagnostics=diagnostics,
            )
            args = {
                "manifest": manifest,
                "top_k": max(len(filtered.observations), 1),
                "ledger": historical,
                "notification_sent": False,
                "direction_policy": policy,
            }
            directory = profile / arm
            record_scalp_target_setups(
                directory,
                filtered,
                **args,
                milestone_horizon_hours=settings.scalp_milestone_horizon_hours,
            )
            record_scalp_bracket_setups(
                directory,
                filtered,
                **args,
                horizon_minutes=settings.scalp_bracket_horizon_minutes,
            )
            record_scalp_setup_forward_setups(directory, filtered, **args)
        else:
            diagnostics["data_unhealthy"] = 1
        scan_counts[arm] = diagnostics
    if healthy and not already_recorded:
        atomic_write_json(
            profile / "scan.json",
            {"barCloseMs": report.newest_close_time_ms, "counts": scan_counts},
        )

    data = {arm: _arm_data(profile / arm) for arm in ARM_POLICIES}
    baseline_keys = set(data["baseline"]["candidates"])
    proposed_keys = set(data["proposed"]["candidates"])
    summary = {
        "version": VERSION,
        "mode": "shadow",
        "profileId": profile_id,
        "startedAtMs": metadata["startedAtMs"],
        "evaluatedAtMs": now_ms,
        "autoPromotion": False,
        "dataHealthy": healthy,
        "lastScan": scan_counts,
        "arms": {
            "baseline": _arm_metrics(data["baseline"], baseline_keys, settings, now_ms),
            "proposed": _arm_metrics(data["proposed"], proposed_keys, settings, now_ms),
            "additional": _arm_metrics(
                data["proposed"], proposed_keys - baseline_keys, settings, now_ms
            ),
        },
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def _event(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("perpetual_symbol"),
        row.get("bar_close_time_ms"),
        row.get("direction"),
    )


def _arm_data(directory: Path) -> dict[str, Any]:
    targets = load_scalp_target_ledger(directory, limit=LIMIT)
    pending = load_pending_scalp_targets(directory, limit=LIMIT)
    brackets = load_scalp_bracket_ledger(directory, limit=LIMIT)
    forward = load_scalp_setup_forward_ledger(directory, limit=LIMIT)
    return {
        "candidates": {_event(row): row for row in (*targets, *pending)},
        "targets": targets,
        "brackets": brackets,
        "forward": forward,
        "complete": all(
            len(rows) < LIMIT for rows in (targets, pending, brackets, forward)
        ),
    }


def _metric(
    rows: Iterable[dict[str, Any]],
    candidates: dict[tuple[Any, ...], dict[str, Any]],
    *,
    now_ms: int,
    maturity_ms: int,
    field: str,
    net_field: str | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    mature = {key for key in candidates if int(key[1]) + maturity_ms <= now_ms}
    outcomes = {_event(row): row for row in rows if _event(row) in mature}
    resolved = [row for row in outcomes.values() if type(row.get(field)) is bool]
    missing = len(mature) - len(resolved)
    wins = sum(row[field] for row in resolved)
    ready = bool(resolved) and not missing and complete
    net = [_number(row.get(net_field)) for row in resolved] if net_field else []
    return {
        "resolved": len(resolved),
        "wins": wins,
        "missing": missing,
        "pending": len(candidates) - len(mature),
        "rate": wins / len(resolved) if ready else None,
        "wilson95": _wilson_interval(wins, len(resolved)) if ready else None,
        "meanNetBps": sum(net) / len(net)
        if ready and net and all(v is not None for v in net)
        else None,
    }


def _arm_metrics(
    data: dict[str, Any], keys: set[tuple[Any, ...]], settings: Settings, now_ms: int
) -> dict[str, Any]:
    candidates = {
        key: value for key, value in data["candidates"].items() if key in keys
    }
    common = {"candidates": candidates, "now_ms": now_ms, "complete": data["complete"]}
    return {
        "candidates": len(candidates),
        "historyComplete": data["complete"],
        "targets": {
            str(level): _metric(
                (row for row in data["targets"] if row.get("target_percent") == level),
                **common,
                maturity_ms=settings.scalp_milestone_horizon_hours * 3_600_000,
                field="hit",
            )
            for level in (2, 3, 5)
        },
        "bracket": _metric(
            data["brackets"],
            **common,
            maturity_ms=settings.scalp_bracket_horizon_minutes * 60_000,
            field="success",
            net_field="net_bps",
        ),
        # All three forward outcomes are persisted together after 60 minutes.
        "forward": {
            str(horizon): _metric(
                (
                    row
                    for row in data["forward"]
                    if row.get("horizon_minutes") == horizon
                ),
                **common,
                maturity_ms=3_600_000,
                field="success",
                net_field="directional_net_bps",
            )
            for horizon in (15, 30, 60)
        },
    }


def _number(value: Any) -> float | None:
    return (
        float(value) if type(value) in (int, float) and math.isfinite(value) else None
    )


def load_horizon_shadow_summary(settings: Settings) -> dict[str, Any]:
    """Allow-list public aggregates; never export raw state or recipient data."""
    source = _read(_root(settings) / "summary.json")
    if source.get("version") != VERSION or source.get("mode") != "shadow":
        return {}

    def metric(row: Any) -> dict[str, Any]:
        row = row if isinstance(row, dict) else {}
        interval = row.get("wilson95")
        return {
            **{
                key: _number(row.get(key))
                for key in (
                    "resolved",
                    "wins",
                    "missing",
                    "pending",
                    "rate",
                    "meanNetBps",
                )
            },
            "wilson95": [_number(v) for v in interval]
            if isinstance(interval, list) and len(interval) == 2
            else None,
        }

    def mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    arms = {}
    for arm in (*ARM_POLICIES, "additional"):
        row = mapping(mapping(source.get("arms")).get(arm))
        arms[arm] = {
            "candidates": _number(row.get("candidates")),
            "historyComplete": row.get("historyComplete") is True,
            "targets": {
                str(level): metric(mapping(row.get("targets")).get(str(level)))
                for level in (2, 3, 5)
            },
            "bracket": metric(row.get("bracket")),
            "forward": {
                str(h): metric(mapping(row.get("forward")).get(str(h)))
                for h in (15, 30, 60)
            },
        }
    return {
        "version": VERSION,
        "mode": "shadow",
        "autoPromotion": False,
        "enabled": settings.scalp_horizon_shadow_enabled,
        "dataHealthy": source.get("dataHealthy") is True,
        "startedAtMs": _number(source.get("startedAtMs")),
        "evaluatedAtMs": _number(source.get("evaluatedAtMs")),
        "arms": arms,
    }


def format_horizon_shadow_status(
    settings: Settings, *, now: datetime | None = None
) -> str:
    if not settings.scalp_horizon_shadow_enabled:
        return "🧪 Ufuk karşılaştırması kapalı."
    summary = load_horizon_shadow_summary(settings)
    if not summary:
        return "🧪 Ufuk karşılaştırması: ilk sessiz ölçüm bekleniyor."
    counts = {key: int(row["candidates"] or 0) for key, row in summary["arms"].items()}
    current = int((now or datetime.now(UTC)).timestamp() * 1000)
    stale = current - (summary["evaluatedAtMs"] or 0) > 15 * 60_000
    status = (
        "kayıt eski"
        if stale
        else "sessiz ölçüm"
        if summary["dataHealthy"]
        else "veri yetersiz"
    )
    return (
        f"🧪 Ufuk karşılaştırması · {status}\n"
        f"Mevcut {counts['baseline']} · Yeni {counts['proposed']} · Ek {counts['additional']} aday\n"
        "Yeni kural bildirim göndermez; sonuçlar panodaki karşılaştırmada."
    )
