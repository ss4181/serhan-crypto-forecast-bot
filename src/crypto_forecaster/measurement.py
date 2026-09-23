"""Descriptive, maturity-aware metrics over the dashboard's public records.

Touch frequency, first-barrier outcome and positive net return are different
events. No probabilities or independent-trade claims are inferred here.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def deadline_ms(row: dict[str, Any]) -> int | None:
    source = finite_number(row.get("sourceTimeMs"))
    horizon = finite_number(row.get("horizonHours"))
    if source is None or source <= 0 or horizon is None or horizon <= 0:
        return None
    end = source + horizon * 3_600_000
    return int(end) if math.isfinite(end) else None


def measurement_summary(
    records: Iterable[dict[str, Any]], *, now_ms: int, history_complete: bool = True
) -> dict[str, Any]:
    rows = list(records)
    audiences: dict[str, Any] = {}
    for audience in ("all", "notified", "muted"):
        selected = [row for row in rows if audience == "all" or
                    bool(row.get("notified")) == (audience == "notified")]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in selected:
            kind = row.get("kind")
            if kind not in {"scalp-target", "scalp-bracket", "scalp-forward"}:
                continue
            level = finite_number(row.get("targetPercent")) if kind == "scalp-target" else None
            horizon = finite_number(row.get("horizonHours"))
            horizon = horizon if horizon is not None and horizon > 0 else None
            values = row.get("families")
            families = sorted({str(value) for value in values if value}) if isinstance(values, list) else []
            if not families:
                families = ["Bilinmiyor"]
            base = (
                kind, level, horizon,
                str(row.get("direction") or "UNKNOWN"),
                str(row.get("regimeState") or "UNKNOWN"),
                str(row.get("strategy") or "Bilinmiyor"),
                str(row.get("policyVersion") or "legacy"),
                str(row.get("symbol") or "Bilinmiyor"),
            )
            for family in families:
                groups.setdefault((*base, family), []).append(row)
        cohorts = []
        for (kind, level, horizon, direction, regime, strategy, policy, symbol, family), group in sorted(
            groups.items(),
            key=lambda item: (
                item[0][0], item[0][1] or 0, item[0][2] or 0,
                item[0][3], item[0][4], item[0][5], item[0][6], item[0][7], item[0][8],
            ),
        ):
            known = [r for r in group if deadline_ms(r) is not None]
            mature = [r for r in known if deadline_ms(r) <= now_ms]
            open_rows = [r for r in known if deadline_ms(r) > now_ms]
            resolved = [r for r in mature if _resolved(r)]
            missing = len(mature) - len(resolved)
            unknown = len(group) - len(known)
            ready = bool(resolved) and not missing and not unknown and history_complete
            wins = sum(_won(r) for r in resolved)
            distinct_days = set()
            for row in resolved:
                source_ms = finite_number(row.get("sourceTimeMs"))
                if source_ms is not None and source_ms > 0:
                    distinct_days.add(
                        datetime.fromtimestamp(source_ms / 1000, tz=UTC).date().isoformat()
                    )
            wilson = _wilson_interval(wins, len(resolved)) if ready else None
            cohort = {
                "kind": kind, "targetPercent": level, "horizonHours": horizon,
                "direction": direction, "regime": regime, "strategy": strategy,
                "policyVersion": policy,
                "symbol": symbol,
                "family": family,
                "recordCount": len(group), "maturedCount": len(mature),
                "resolvedCount": len(resolved), "openCount": len(open_rows),
                "distinctUtcDays": len(distinct_days),
                "unresolvedCount": missing, "unknownDeadlineCount": unknown,
                "earlyHits": sum(_won(r) for r in open_rows if _resolved(r)),
                "hits": wins, "misses": len(resolved) - wins,
                "hitRate": wins / len(resolved) if ready else None,
                "hitRateWilson95": wilson,
                "rateAvailable": ready,
                "rolling": _rolling_rates(resolved),
            }
            if kind == "scalp-bracket":
                net = [v for r in resolved if (v := finite_number(r.get("netBps"))) is not None]
                net_ready = ready and len(net) == len(resolved)
                cohort.update({
                    "stops": sum(r.get("status") == "STOP" for r in resolved),
                    "timeExits": sum(r.get("status") == "TIME_EXIT" for r in resolved),
                    "netSampleCount": len(net),
                    "missingNetCount": len(resolved) - len(net),
                    "positiveNetCount": sum(v > 0 for v in net),
                    "positiveNetRate": sum(v > 0 for v in net) / len(net) if net_ready else None,
                    "meanNetBps": sum(net) / len(net) if net_ready else None,
                })
            cohorts.append(cohort)
        audiences[audience] = cohorts
    return {
        "version": "matured-cohorts-v1", "asOfMs": now_ms,
        "historyComplete": history_complete, "audiences": audiences,
    }


def _resolved(row: dict[str, Any]) -> bool:
    if row.get("kind") in {"scalp-target", "scalp-forward"}:
        return type(row.get("success")) is bool
    return row.get("status") in {"TARGET", "STOP", "TIME_EXIT"}


def _won(row: dict[str, Any]) -> bool:
    return row.get("success") is True if row.get("kind") in {"scalp-target", "scalp-forward"} else row.get("status") == "TARGET"


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float] | None:
    """Return a 95% Wilson interval; this does not correct cross-coin clustering."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rolling_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rolling settled outcomes with uncertainty; never count open/missing rows."""
    ordered = sorted(
        rows,
        key=lambda row: finite_number(row.get("sourceTimeMs")) or 0,
        reverse=True,
    )
    result: dict[str, Any] = {}
    for size in (10, 50, 100):
        sample = ordered[:size]
        wins = sum(_won(row) for row in sample)
        result[str(size)] = {
            "count": len(sample),
            "successes": wins,
            "rate": wins / len(sample) if sample else None,
            "wilson95": _wilson_interval(wins, len(sample)),
        }
    return result
