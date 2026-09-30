"""Experimental LONG alerts: private owner only, receipt-backed and opt-in."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from .config import Settings, local_text
from .long_scout import (
    DAY,
    HOUR,
    LABELS,
    STEP,
    TARGETS,
    VERSION,
    _publish,
    _read,
    _scout_lock,
    root,
)
from .telegram import TelegramNotifier, is_primary, telegram_configured

FRESH_MS = 15 * 60_000


def alert_id(record: dict, target: int | None = None) -> str:
    return sha256(
        f"owner-long|{VERSION}|{record['id']}|{target or 'initial'}".encode()
    ).hexdigest()


def initial_text(record: dict) -> str:
    price, stop = record["referencePrice"], record["stopPct"]
    return "\n".join(
        (
            f"🧪 DENEYSEL LONG ADAYI • {record['symbol']}",
            f"{LABELS[record['strategy']]} • Yerleşik boğa",
            f"Referans: ${price:.8g} • {local_text(record['sourceCloseMs'], with_seconds=False)}",
            f"Güç: {record['score']:.0f}/100 (olasılık değil)",
            f"Gerekçe: {record['reason']}",
            f"Araştırma stopu: ${price * (1 - stop / 100):.8g} (−%{stop:.1f})",
            "Hedefler: "
            + " · ".join(f"%{k} ${price * (1 + k / 100):.8g}" for k in TARGETS),
            "Dokunuşlar bu referansa göre, teslim sonrası tam 5m mumlarında izlenir.",
            "Araştırma bildirimi • Emir verilmez • Kârlılık doğrulanmadı.",
        )
    )


def target_text(record: dict, level: int, result: dict) -> str:
    price = record["ownerAlert"]["referencePrice"]
    ordering = (
        "Hedef/stop aynı mumda; önce hangisi bilinmiyor."
        if result.get("sameBarAmbiguous")
        else "Stop sonrası dokunuş; başarılı işlem/kâr değildir."
        if result.get("firstBeforeStop") is False
        else "Araştırma stopundan önce dokundu."
    )
    return "\n".join(
        (
            f"🎯 DENEYSEL LONG • {record['symbol']} • +%{level} DOKUNDU",
            f"Referans ${price:.8g} → hedef ${price * (1 + level / 100):.8g}",
            f"{LABELS[record['strategy']]} • {local_text(result['touchAtMs'], with_seconds=False)}",
            ordering,
            "Kapalı 5m mumunun tepe fiyatı; gerçekleşmiş kâr veya emir dolumu değildir.",
        )
    )


def deliver_long_notifications(
    settings: Settings,
    *,
    now: datetime | None = None,
    notifier: TelegramNotifier | None = None,
    started_at_ms: int = 0,
) -> list[tuple[str, str]]:
    """Never backfill old candidates, broadcast, or send orphan target alerts."""
    if (
        not settings.long_scout_alerts_enabled
        or not settings.long_scout_enabled
        or not is_primary()
    ):
        return []
    if notifier is None and not telegram_configured():
        return []
    current = int((now or datetime.now(UTC)).timestamp() * 1000)
    sender = notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)
    receipts = settings.telegram_state_dir / "long-scout"
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts.chmod(0o700)
    events = []
    with _scout_lock(root(settings) / "worker.lock") as acquired:
        if not acquired:
            return []
        state = _read(root(settings) / "state.json")
        if state.get("version") != VERSION:
            return []
        # One real configuration acknowledgement, never a fabricated coin signal.
        activation = sha256(f"owner-long|{VERSION}|enabled".encode()).hexdigest()
        if not sender.owner_delivery_receipt(activation, receipts):
            result = sender.deliver_owner_once(
                signal_id=activation,
                state_dir=receipts,
                text="🧪 TRADE3 • LONG bildirimleri açıldı\nYalnız sana özel deneysel bildirim: yerleşik boğada yüksek skorlu kırılım/geri test adayları ve %2/%3/%5/%10/%20 dokunuşları.\nSkor olasılık değildir; emir verilmez. Henüz 1m tetikleyici yok.",
            )
            events.append(("etkinleştirme", result.status))
        records = sorted(
            state.get("records", []), key=lambda r: (r["recordedAtMs"], -r["score"])
        )
        last_by_coin = {}
        for record in records:
            proof = sender.owner_delivery_receipt(alert_id(record), receipts)
            if proof:
                last_by_coin[record["symbol"]] = max(
                    last_by_coin.get(record["symbol"], 0), proof["delivered_at_ms"]
                )
        changed = False
        for record in records:
            identifier = alert_id(record)
            proof = sender.owner_delivery_receipt(identifier, receipts)
            fresh = (
                record["recordedAtMs"] >= started_at_ms
                and 0 <= current - record["recordedAtMs"] <= FRESH_MS
                and 0 <= current - record["sourceCloseMs"] <= FRESH_MS
            )
            if (
                not proof
                and fresh
                and record.get("stage") == "TEYİT"
                and record.get("strategy") in {"LB1", "LP1"}
                and record["score"] >= settings.long_scout_minimum_score
                and state.get("regime", {}).get("state") == "BULL_CONFIRMED"
                and state.get("regime", {}).get("dataHealthy") is True
                and 0 <= current - state.get("scannedAtMs", 0) < HOUR
                and current - last_by_coin.get(record["symbol"], 0) >= DAY
            ):
                result = sender.deliver_owner_once(
                    signal_id=identifier, text=initial_text(record), state_dir=receipts
                )
                events.append((f"{record['symbol']} aday", result.status))
                proof = sender.owner_delivery_receipt(identifier, receipts)
                if proof:
                    last_by_coin[record["symbol"]] = proof["delivered_at_ms"]
            if not proof:
                continue  # UNCERTAIN/rejected/another owner's delivery is NOT proof.
            delivered = proof["delivered_at_ms"]
            if record.get("ownerAlert", {}).get("deliveredAtMs") != delivered:
                # Recovery after a crash between receipt and ledger writes.
                record["ownerAlert"] = {
                    "deliveredAtMs": delivered,
                    "referencePrice": record["referencePrice"],
                    "trackingStartMs": (delivered // STEP + 1) * STEP,
                    "outcome": {},
                }
                changed = True
            outcome = record["ownerAlert"].get("outcome", {})
            if not outcome.get("complete"):
                continue
            for level in TARGETS:
                target = outcome.get("targets", {}).get(str(level), {})
                touch = target.get("touchAtMs")
                if (
                    target.get("hit") is not True
                    or not isinstance(touch, int)
                    or touch - STEP + 1 < record["ownerAlert"]["trackingStartMs"]
                    or sender.owner_delivery_receipt(alert_id(record, level), receipts)
                ):
                    continue
                result = sender.deliver_owner_once(
                    signal_id=alert_id(record, level),
                    text=target_text(record, level, target),
                    state_dir=receipts,
                )
                events.append((f"{record['symbol']} %{level}", result.status))
        if changed:
            _publish(settings, state, current)
    return events
