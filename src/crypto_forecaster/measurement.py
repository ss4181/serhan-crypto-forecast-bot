"""Descriptive, maturity-aware metrics over the dashboard's public records.

Touch frequency, first-barrier outcome and positive net return are different
events. No probabilities or independent-trade claims are inferred here.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


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
        groups: dict[tuple[str, float | None, float | None], list[dict[str, Any]]] = {}
        for row in selected:
            kind = row.get("kind")
            if kind not in {"scalp-target", "scalp-bracket"}:
                continue
            level = finite_number(row.get("targetPercent")) if kind == "scalp-target" else None
            horizon = finite_number(row.get("horizonHours"))
            horizon = horizon if horizon is not None and horizon > 0 else None
            groups.setdefault((kind, level, horizon), []).append(row)
        cohorts = []
        for (kind, level, horizon), group in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1] or 0, item[0][2] or 0)
        ):
            known = [r for r in group if deadline_ms(r) is not None]
            mature = [r for r in known if deadline_ms(r) <= now_ms]
            open_rows = [r for r in known if deadline_ms(r) > now_ms]
            resolved = [r for r in mature if _resolved(r)]
            missing = len(mature) - len(resolved)
            unknown = len(group) - len(known)
            ready = bool(resolved) and not missing and not unknown and history_complete
            wins = sum(_won(r) for r in resolved)
            cohort = {
                "kind": kind, "targetPercent": level, "horizonHours": horizon,
                "recordCount": len(group), "maturedCount": len(mature),
                "resolvedCount": len(resolved), "openCount": len(open_rows),
                "unresolvedCount": missing, "unknownDeadlineCount": unknown,
                "earlyHits": sum(_won(r) for r in open_rows if _resolved(r)),
                "hits": wins, "misses": len(resolved) - wins,
                "hitRate": wins / len(resolved) if ready else None,
                "rateAvailable": ready,
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
    if row.get("kind") == "scalp-target":
        return type(row.get("success")) is bool
    return row.get("status") in {"TARGET", "STOP", "TIME_EXIT"}


def _won(row: dict[str, Any]) -> bool:
    return row.get("success") is True if row.get("kind") == "scalp-target" else row.get("status") == "TARGET"
