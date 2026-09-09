"""Explain a quiet scalp sender without relaxing any notification gate."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings, local_text

SCHEMA = "scalp-notification-status-v1"
REASONS = {
    "no_setup": "çoklu teyit yok",
    "direction_unclear": "yön net değil",
    "calibration_missing": "geçmiş ölçüm eksik",
    "probability_low": "net-pozitif geçmiş oranı düşük",
    "expected_net_low": "net beklenti düşük",
    "score_low": "ham skor düşük",
    "quality_low": "aile içi kalite düşük",
}
DELIVERIES = {
    "NO_CANDIDATE": "uygun aday yok; gönderim denenmedi",
    "PENDING": "gönderim sonucu henüz kaydedilmedi",
    "SENT": "Telegram teslimatı tamamlandı",
    "PARTIAL": "Telegram teslimatı kısmi",
    "DEDUPLICATED": "tekrar gönderim engellendi",
    "UNCERTAIN": "Telegram teslimatı belirsiz",
    "REJECTED": "Telegram gönderimi reddetti",
    "ERROR": "gönderim sırasında hata",
}


def write_notification_status(
    state_dir: Path, *, evaluated_at_ms: int, fresh: int, attempted: int,
    counts: dict[str, int], delivery_status: str,
) -> None:
    """Atomically replace operational state; no recipients or secrets are stored."""
    payload = {
        "schema": SCHEMA, "evaluated_at_ms": evaluated_at_ms,
        "fresh": fresh, "attempted": attempted,
        "counts": {key: int(counts.get(key, 0)) for key in (*REASONS, "eligible")},
        "delivery_status": delivery_status if delivery_status in DELIVERIES else "ERROR",
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=state_dir, delete=False,
            prefix="notification-status-", suffix=".tmp",
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_dir / "notification_status.json")
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def format_notification_status(settings: Settings, *, now: datetime | None = None) -> str:
    if not settings.scalp_observation_enabled:
        return "📡 Scalp gözlemi bu süreçte kapalı."
    try:
        row = json.loads((settings.scalp_state_dir / "notification_status.json").read_text(encoding="utf-8"))
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise ValueError("schema")
        timestamp = row["evaluated_at_ms"]
        counts = row["counts"]
        values = [timestamp, row["fresh"], row["attempted"]]
        values.extend(counts[key] for key in (*REASONS, "eligible"))
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("counts")
        current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
        age_ms = current_ms - timestamp
        stamp = local_text(timestamp, with_seconds=False)
        if timestamp <= 0 or age_ms < -60_000:
            raise ValueError("clock")
        stale = age_ms > 15 * 60_000
        total = sum(counts[key] for key in (*REASONS, "eligible"))
        rejected = [f"{counts[key]} {label}" for key, label in REASONS.items() if counts[key]]
        delivery = DELIVERIES.get(row.get("delivery_status"), "teslimat kaydı bilinmiyor")
        coverage = f"{row['fresh']}/{row['attempted']} taze piyasa"
        if row["attempted"] <= 0 or row["fresh"] / row["attempted"] < settings.scalp_minimum_coverage:
            coverage += " ⚠️ kapsam yetersiz"
        return "\n".join([
            "📡 SCALP BİLDİRİM DURUMU" + (" ⚠️ KAYIT ESKİ" if stale else ""),
            f"Son tarama: {stamp}",
            f"{coverage} • {counts['eligible']}/{total} coin/mum bildirime uygun",
            "İlk elenme nedeni: " + ("; ".join(rejected) if rejected else "yok"),
            "Son taramada: " + delivery,
            f"Az örnekte ham skor ≥{settings.scalp_minimum_alert_score:g}; yön/teyit yine gerekli.",
            f"Yeterli örnekte: kalite ≥%{settings.scalp_minimum_quality_percentile * 100:g}, "
            f"net-pozitif oran ≥%{settings.scalp_minimum_direction_probability * 100:g}, "
            f"net beklenti ≥{settings.scalp_minimum_expected_net_bps:g} bps.",
        ])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
        return "📡 Scalp bildirim durum kaydı henüz yok veya okunamıyor; servis günlüğünü kontrol edin."
