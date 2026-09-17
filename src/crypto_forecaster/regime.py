"""Confirmed market regime, persisted by the primary scalp scanner only.

Policy v1 is an operational anti-chatter rule, not a fitted return predictor.
Only consecutive, distinct, complete five-minute candles can confirm a change.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings, local_text
from .telegram import (
    TelegramDelivery,
    TelegramNotifier,
    digest_signal_id,
    is_primary,
    telegram_channel_keyboard,
    telegram_configured,
)

if TYPE_CHECKING:
    from .scalping import BullRegime

SCHEMA = "scalp-regime-v1"
STEP_MS = 300_000
ENTRY_CONFIRMATIONS = 3
EXIT_CONFIRMATIONS = 2
LABELS = {"BULL": "BOĞA", "TRANSITION": "GEÇİŞ", "OFF": "KAPALI", "UNKNOWN": "VERİ YETERSİZ"}


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA
                or value.get("state") not in LABELS
                or value.get("candidate") not in (*LABELS, None)
                or not isinstance(value.get("policy_key"), str)
                or value.get("raw_state") not in LABELS
                or type(value.get("required")) is not int
                or not 0 <= value["required"] <= ENTRY_CONFIRMATIONS
                or any(not _fraction(value.get(k)) for k in ("breadth", "trend_fraction"))
                or any(type(value.get(k)) is not int or value[k] < 0
                       for k in ("bar_close_ms", "count", "updated_at_ms"))):
            return None
        if ((value["candidate"] is None and value["count"] != 0)
                or (value["candidate"] is not None and not 0 < value["count"] < value["required"])):
            return None
        event = value.get("event")
        if event is not None and not _valid_event(event):
            return None
        return value
    except (FileNotFoundError, UnicodeError, ValueError):
        return None


def _fraction(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _valid_event(event: object) -> bool:
    return (isinstance(event, dict) and event.get("from") in LABELS
            and event.get("to") in LABELS and isinstance(event.get("reason"), str)
            and isinstance(event.get("id"), str) and len(event["id"]) == 64
            and isinstance(event.get("delivery_status"), str)
            and type(event.get("persistent_up")) is bool
            and all(type(event.get(k)) is int and event[k] >= 0
                    for k in ("bar_close_ms", "at_ms", "confirmations"))
            and all(_fraction(event.get(k)) for k in ("breadth", "trend_fraction")))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="regime-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_regime(
    settings: Settings, raw: BullRegime, *, bar_close_ms: int,
    evaluated_at_ms: int, healthy: bool, universe_key: str,
) -> BullRegime:
    """Return the effective regime used by B-family detectors and setup gates.

    Entry: >=60% breadth (configured entry threshold), persistent 4-week
    direction and >=3/4 major trend votes, for 3 consecutive candles.
    Hold band: breadth >= entry-10 percentage points, same major conditions.
    Ordinary exit: 2 consecutive failures. Breadth <35% or <=1/4 trend votes
    exits immediately. Missing data suspends setups immediately as UNKNOWN.
    """
    path = settings.scalp_state_dir / "regime_status.json"
    previous = _read(path)
    entry = settings.scalp_bull_breadth_threshold
    exit_breadth = max(0.35, entry - 0.10)
    policy_key = f"{SCHEMA}|{universe_key}|{entry:g}"
    if previous and previous.get("policy_key") != policy_key:
        previous = None
    healthy = healthy and raw.state in {"BULL", "TRANSITION", "OFF"} and all(
        _fraction(value) for value in (raw.score, raw.breadth, raw.trend_fraction)
    )
    baseline = "UNKNOWN" if not healthy else "TRANSITION" if raw.state == "BULL" else raw.state
    state = dict(previous) if previous else {
        "schema": SCHEMA, "policy_key": policy_key, "state": baseline,
        "candidate": None, "count": 0, "bar_close_ms": 0,
        "updated_at_ms": 0, "event": None,
    }
    old = state["state"]
    last = state["bar_close_ms"]
    if previous and bar_close_ms < last:
        # An older replay cannot roll operational state back or enable setups.
        return replace(raw, state="UNKNOWN")
    if healthy and previous and bar_close_ms == last:
        return replace(raw, state=old)

    consecutive = last > 0 and bar_close_ms == last + STEP_MS
    if not consecutive:
        state["candidate"], state["count"] = None, 0
    gap = bool(previous and last and bar_close_ms - last > STEP_MS)
    required = 0
    reason = "Koşullar kararlı."
    if not healthy:
        state["state"] = "UNKNOWN"
        state["candidate"], state["count"] = None, 0
        reason = "Son kapalı mum kapsamı veya BTC/ETH saatlik verisi yetersiz."
    else:
        # After a missed scan, a previous BULL must earn entry again. Never
        # join confirmations across missing candles, including after restart.
        if gap:
            state["state"] = "UNKNOWN"
        major_ok = raw.persistent_up and raw.trend_fraction >= 0.75
        strong = major_ok and raw.breadth >= entry
        hold = major_ok and raw.breadth >= exit_breadth
        severe = raw.breadth < 0.35 or raw.trend_fraction <= 0.25
        non_bull = "OFF" if raw.score < 0.55 else "TRANSITION"
        if severe:
            target, required = non_bull, 1
            reason = "Sert bozulma: genişlik <%35 veya trend teyidi ≤1/4."
        elif state["state"] == "BULL":
            target = "BULL" if hold else non_bull
            required = EXIT_CONFIRMATIONS
            reason = f"Genişlik <%{exit_breadth * 100:g} veya ana trend koşulları bozuldu."
        elif strong:
            target, required = "BULL", ENTRY_CONFIRMATIONS
            reason = f"Genişlik ≥%{entry * 100:g}, trend ≥3/4 ve kalıcı 4 haftalık yön pozitif."
        else:
            target, required = non_bull, EXIT_CONFIRMATIONS
            reason = "Boğa giriş koşulları tam değil."
        if target == state["state"]:
            state["candidate"], state["count"] = None, 0
            reason = "Koşullar kararlı."
        else:
            # Both OFF and TRANSITION count as failed BULL conditions; a
            # fluctuating raw score must not keep a weak BULL alive forever.
            same_failure = state["state"] == "BULL" and state["candidate"] in {"OFF", "TRANSITION"}
            state["count"] = state["count"] + 1 if state["candidate"] == target or same_failure else 1
            state["candidate"] = target
            if state["count"] >= required:
                state["state"] = target
                state["candidate"], state["count"] = None, 0

    state.update(
        bar_close_ms=bar_close_ms, updated_at_ms=evaluated_at_ms,
        raw_state=raw.state if raw.state in LABELS else "UNKNOWN",
        breadth=raw.breadth if _fraction(raw.breadth) else 0.0,
        trend_fraction=raw.trend_fraction if _fraction(raw.trend_fraction) else 0.0,
        persistent_up=raw.persistent_up, reason=reason, required=required,
        entry_breadth=entry, exit_breadth=exit_breadth,
    )
    if previous and state["state"] != old:
        # Save the event together with the decision before attempting Telegram.
        state["event"] = {
            "id": digest_signal_id(f"{policy_key}|{old}|{state['state']}", bar_close_ms),
            "from": old, "to": state["state"], "bar_close_ms": bar_close_ms,
            "at_ms": evaluated_at_ms, "breadth": state["breadth"],
            "trend_fraction": state["trend_fraction"], "persistent_up": raw.persistent_up,
            "reason": ("Ardışık 5m veri akışı kesildi; rejim yeniden teyit bekliyor."
                       if gap and state["state"] == "UNKNOWN" else reason),
            "confirmations": required if healthy and state["state"] != "UNKNOWN" else 0,
            "delivery_status": "PENDING",
        }
    _write(path, state)
    return replace(raw, state=state["state"])


def format_regime_change(event: dict) -> str:
    before, after = LABELS[event["from"]], LABELS[event["to"]]
    effect = (
        "B1/B2/B3 ve kurulum değerlendirmesi açık; diğer kalite filtreleri uygulanır."
        if event["to"] == "BULL" else
        "Yeni boğa kurulumları duraklatıldı; mevcut hedeflerin takibi sürüyor."
    )
    confirmation = f"{event['confirmations']} kapanış teyidi" if event["confirmations"] > 1 else "Anında koruma"
    return "\n".join([
        f"🧭 REJİM | {before} → {after}",
        local_text(event["bar_close_ms"] + 1, with_seconds=False),
        f"Genişlik: %{event['breadth'] * 100:.0f} • Trend: {round(event['trend_fraction'] * 4)}/4",
        f"4 haftalık yön: {'pozitif' if event['persistent_up'] else 'teyitsiz'} • {confirmation}",
        event["reason"], effect,
        "Piyasa koşulu bildirimidir; coin yönü veya başarı olasılığı değildir.",
    ])


def deliver_regime_change(
    settings: Settings, *, now: datetime | None = None,
    notifier: TelegramNotifier | None = None,
) -> TelegramDelivery | None:
    if not is_primary() or (notifier is None and not telegram_configured()):
        return None
    path = settings.scalp_state_dir / "regime_status.json"
    state = _read(path)
    event = state.get("event") if state else None
    if not isinstance(event, dict) or event.get("delivery_status") in {
        "SENT", "DEDUPLICATED", "UNCERTAIN", "EXPIRED",
    }:
        return None
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    if current_ms - event["at_ms"] > 15 * 60_000 or current_ms < event["at_ms"]:
        event["delivery_status"] = "EXPIRED"
        _write(path, state)
        return None
    delivery = (notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)).deliver_once(
        signal_id=event["id"], text=format_regime_change(event),
        state_dir=settings.telegram_state_dir / "regime",
        reply_markup=telegram_channel_keyboard(),
    )
    # A manual primary scan may have advanced the state while Telegram was
    # answering. Do not roll that newer decision back with our old snapshot.
    latest = _read(path)
    if latest and latest.get("event") and latest["event"]["id"] == event["id"]:
        latest["event"]["delivery_status"] = delivery.status
        _write(path, latest)
    return delivery


def format_regime_status(settings: Settings, *, now: datetime | None = None) -> str:
    state = _read(settings.scalp_state_dir / "regime_status.json")
    if state is None:
        return "🧭 Rejim: henüz teyit kaydı yok."
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    if not 0 <= current_ms - state["updated_at_ms"] <= 15 * 60_000:
        return "🧭 Rejim kaydı eski; güncel boğa teyidi olarak kullanmayın."
    pending = f" • {LABELS[state['candidate']]} teyidi {state['count']}/{state['required']}" if state["candidate"] else ""
    return (f"🧭 Rejim: {LABELS[state['state']]}{pending}\n"
            f"Genişlik %{state['breadth'] * 100:.0f} • Trend {round(state['trend_fraction'] * 4)}/4")
