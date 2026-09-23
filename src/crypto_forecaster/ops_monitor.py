"""Throttled operational alarms for the read-only scalp observer."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import Settings
from .persistence import atomic_write_json

ALERT_REPEAT_MS = 6 * 60 * 60 * 1000


def scalp_health_incidents(
    settings: Settings,
    *,
    evaluated_at_ms: int,
    fresh: int,
    attempted: int,
    eligible: int,
    websocket_connected: bool,
    delivery_status: str | None,
    dashboard_url: str | None = None,
) -> list[dict[str, str]]:
    """Return incidents that should be sent now, preserving their throttle."""
    path = settings.scalp_state_dir / "ops-alerts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("incidents", {}) if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        state = {}
    if not isinstance(state, dict):
        state = {}

    problems: dict[str, str] = {}
    if attempted <= 0 or fresh / attempted < settings.scalp_minimum_coverage:
        problems["binance_coverage"] = f"Binance verisi yetersiz: {fresh}/{attempted} piyasa taze."
    if not websocket_connected:
        problems["websocket"] = "Binance 5m WebSocket bağlı değil; REST yedeği kullanılıyor."
    cache_files = list(settings.scalp_data_dir.glob("*_5m_futures.csv"))
    if not cache_files:
        problems["cache"] = "Scalp 5m cache'leri bulunamadı. Binance veri yenilemeyi kontrol edin."
    else:
        try:
            oldest_age = max(evaluated_at_ms - int(path.stat().st_mtime * 1000) for path in cache_files)
            if oldest_age > 15 * 60_000:
                problems["cache"] = f"Scalp cache'lerinden en az biri {oldest_age // 60_000} dakikadır yenilenmedi."
        except OSError:
            problems["cache"] = "Scalp cache yaşı kontrol edilemedi."
    if delivery_status in {"ERROR", "PARTIAL", "FAILED"}:
        problems["telegram"] = f"Telegram teslimat durumu: {delivery_status}."
    if eligible == 0:
        since = state.get("no_signal_since_ms")
        if not isinstance(since, int) or since <= 0:
            state["no_signal_since_ms"] = evaluated_at_ms
        elif evaluated_at_ms - since >= 30 * 60_000:
            problems["no_eligible_signal"] = "30 dakikadır bildirime uygun sinyal yok (sessiz filtreler dahil)."
    else:
        state.pop("no_signal_since_ms", None)

    try:
        disk_path = settings.scalp_state_dir if settings.scalp_state_dir.exists() else settings.scalp_state_dir.parent
        usage = shutil.disk_usage(disk_path if disk_path.exists() else Path.cwd())
        free_percent = 100.0 * usage.free / max(usage.total, 1)
        if usage.free < 1_000_000_000 or free_percent < 10.0:
            problems["disk"] = f"Disk alanı azalıyor: {usage.free / 1e9:.1f} GB boş (%{free_percent:.1f})."
    except OSError:
        problems["disk_check"] = "Disk alanı kontrol edilemedi."

    dashboard = settings.report_dir / "scalp-data.json"
    try:
        age_ms = evaluated_at_ms - int(dashboard.stat().st_mtime * 1000)
        if age_ms > 30 * 60_000:
            problems["dashboard"] = f"Yerel scalp dashboard verisi {age_ms // 60_000} dakikadır yenilenmedi."
    except OSError:
        problems["dashboard"] = "Yerel scalp dashboard çıktısı bulunamadı."
    if dashboard_url:
        try:
            request = Request(dashboard_url, headers={"User-Agent": "trade3-health/1.0"})
            with urlopen(request, timeout=3.0) as response:
                published = json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))
            generated = datetime.fromisoformat(str(published["generatedAtUtc"]))
            published_age = evaluated_at_ms - int(generated.timestamp() * 1000)
            if published_age < -60_000 or published_age > 30 * 60_000:
                problems["public_dashboard"] = f"GitHub Pages dashboard verisi eski ({max(0, published_age) // 60_000} dk)."
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            problems["public_dashboard"] = "GitHub Pages dashboard verisi okunamadı; yayın/Actions durumunu kontrol edin."

    due: list[dict[str, str]] = []
    for code, message in problems.items():
        prior = state.get(code, {})
        last_sent = prior.get("last_sent_ms", 0) if isinstance(prior, dict) else 0
        if not isinstance(last_sent, int) or evaluated_at_ms - last_sent >= ALERT_REPEAT_MS:
            due.append({"code": code, "message": message})
    # Persist the no-signal timer and last observed incidents. Successful-send
    # times are written separately, so Telegram errors cannot eat the alarm.
    atomic_write_json(path, {"version": 1, "incidents": state})
    return due


def mark_health_alert_sent(
    settings: Settings, code: str, *, sent_at_ms: int | None = None
) -> None:
    path = settings.scalp_state_dir / "ops-alerts.json"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        incidents = payload.get("incidents", {}) if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        incidents = {}
    if not isinstance(incidents, dict):
        incidents = {}
    row = incidents.get(code)
    if not isinstance(row, dict):
        row = {}
    row["last_sent_ms"] = sent_at_ms or int(datetime.now(UTC).timestamp() * 1000)
    incidents[code] = row
    atomic_write_json(path, {"version": 1, "incidents": incidents})
