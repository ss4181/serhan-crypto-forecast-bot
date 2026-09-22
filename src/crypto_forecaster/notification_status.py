"""Explain a quiet scalp sender without relaxing any notification gate."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    regime_state: str = "UNKNOWN", radar_count: int = 0,
    setup_count: int = 0, candidates: Iterable[dict[str, Any]] = (),
) -> None:
    """Atomically replace operational state; no recipients or secrets are stored."""
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        symbol = str(candidate.get("symbol", "")).strip().upper()
        direction = str(candidate.get("direction", "")).strip().upper()
        try:
            price = float(candidate.get("price"))
            score = float(candidate.get("score"))
        except (TypeError, ValueError):
            continue
        if (not symbol or direction not in {"YUKARI", "AŞAĞI"}
                or not math.isfinite(price) or price <= 0
                or not math.isfinite(score)):
            continue
        row: dict[str, Any] = {
            "symbol": symbol, "direction": direction,
            "price": price, "score": score,
            "families": str(candidate.get("families", "")),
        }
        for key in ("horizon", "sample_count"):
            value = candidate.get(key)
            if type(value) is int and value >= 0:
                row[key] = value
        for key in ("success_probability", "expected_net_bps"):
            value = candidate.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                row[key] = float(value)
        candidate_rows.append(row)
        if len(candidate_rows) >= 5:
            break
    payload = {
        "schema": SCHEMA, "evaluated_at_ms": evaluated_at_ms,
        "fresh": fresh, "attempted": attempted,
        "counts": {key: int(counts.get(key, 0)) for key in (*REASONS, "eligible")},
        "delivery_status": delivery_status if delivery_status in DELIVERIES else "ERROR",
        "regime_state": regime_state if regime_state in {"BULL", "TRANSITION", "OFF", "UNKNOWN"} else "UNKNOWN",
        "radar_count": max(int(radar_count), 0),
        "setup_count": max(int(setup_count), 0),
        "candidates": candidate_rows,
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
        regime_labels = {"BULL": "BOĞA", "TRANSITION": "GEÇİŞ", "OFF": "KAPALI", "UNKNOWN": "VERİ YETERSİZ"}
        regime = regime_labels.get(row.get("regime_state"), "VERİ YETERSİZ")
        radar = row.get("radar_count", 0)
        setups = row.get("setup_count", 0)
        if type(radar) is not int or radar < 0:
            radar = 0
        if type(setups) is not int or setups < 0:
            setups = 0
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        candidate_lines: list[str] = []
        for candidate in candidates[:5]:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_lines.append(
                    f"{candidate['symbol']} • {candidate['direction']} • "
                    f"{float(candidate['price']):g} • skor {float(candidate['score']):.2f}"
                )
            except (KeyError, TypeError, ValueError):
                continue
        return "\n".join([
            "📡 SCALP BİLDİRİM DURUMU" + (" ⚠️ KAYIT ESKİ" if stale else ""),
            f"Son tarama: {stamp}",
            f"Kaynak: Binance USD-M PERP • Rejim: {regime}",
            f"{coverage} • Radar {radar} • Kurulum {setups}",
            f"Bu taramada: {counts['eligible']}/{total} coin/mum bildirime uygun",
            "İlk elenme nedeni: " + ("; ".join(rejected) if rejected else "yok"),
            "Son taramada: " + delivery,
            "Güncel uygun adaylar: " + ("yok" if not candidate_lines else "") ,
            *candidate_lines,
            f"Az örnekte ham skor ≥{settings.scalp_minimum_alert_score:g}; yön/teyit yine gerekli.",
            (f"Yeterli örnekte: kalite ≥%{settings.scalp_minimum_quality_percentile * 100:g}, "
            f"net-pozitif oran ≥%{settings.scalp_minimum_direction_probability * 100:g}, "
            f"net beklenti ≥{settings.scalp_minimum_expected_net_bps:g} bps."),
            (f"Geçiş kapısı: {'açık' if settings.scalp_transition_alerts_enabled else 'kapalı'}; "
             f"skor ≥{settings.scalp_transition_minimum_alert_score:g}, "
             f"kalite ≥%{settings.scalp_transition_minimum_quality_percentile * 100:g}, "
             f"yön ≥%{settings.scalp_transition_minimum_direction_probability * 100:g}, "
             f"net ≥{settings.scalp_transition_minimum_expected_net_bps:g} bps, "
             f"n ≥{settings.scalp_transition_minimum_calibration_samples}."),
        ])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
        return "📡 Scalp bildirim durum kaydı henüz yok veya okunamıyor; servis günlüğünü kontrol edin."
