"""Owner-only LONG +2%/-2% forward experiment.

Rules are frozen in research/long_2pct_experiment_protocol_20261108.md. The
module only reads the scan report and writes its own state, so the scalp
filters, ledgers and subscriber alerts behave exactly as before.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings, local_text
from .data import MarketDataError, load_cache
from .long_scout import _read
from .model import block_bootstrap_interval
from .persistence import atomic_write_json
from .scalping import (
    SCALP_SETTLEMENT_GRACE_DAYS,
    ScalpObservation,
    ScalpScanReport,
    _first_touch_bracket,
    _scalp_bracket_window,
    scalp_cache_path,
)
from .telegram import TelegramError, TelegramNotifier, telegram_configured

VERSION = "long-2pct-v1"
TARGET_BPS = 200.0
STOP_BPS = 200.0
HORIZON_MINUTES = 1_440
COST_BPS = 20.0
STRESS_COST_BPS = 30.0
BREAK_EVEN_WIN_RATE = 0.55
TESTED_FAMILIES = frozenset({"F1", "F2", "F3", "B1", "B2", "B3"})
MAX_SIGNAL_AGE_MS = 15 * 60_000
DAY_MS = 86_400_000
MIN_SETTLED = 100
MIN_SIGNAL_DAYS = 15
DECISION_DAYS = 28
HARD_STOP_DAYS = 56
WIN = "TARGET"
LOSS = "STOP"
TIME_EXIT = "TIME_EXIT"
DATA_MISSING = "DATA_MISSING"
OPEN = "OPEN"
DELIVERED = {"SENT", "DEDUPLICATED"}
# Nothing reached Telegram in these cases. UNCERTAIN may have, so it is never resent.
RETRYABLE = {None, "ERROR", "REDDEDILDI"}


def _state_path(settings: Settings) -> Path:
    return settings.scalp_state_dir / "experiments" / VERSION / "state.json"


def _load_state(settings: Settings, *, started_at_ms: int | None = None) -> dict[str, Any]:
    path = _state_path(settings)
    if not path.exists():
        return {"version": VERSION, "startedAtMs": started_at_ms, "records": []}
    payload = _read(path)
    if payload.get("version") != VERSION:
        raise ValueError("LONG %2 deney durumu dogrulanamadi")
    return payload


def select_candidates(
    report: ScalpScanReport,
    records: list[dict[str, Any]],
    *,
    daily_limit: int,
) -> list[tuple[str, tuple[ScalpObservation, ...]]]:
    """Return the protocol's picks for the newest bar, best detector score first."""
    if report.regime is None or report.regime.state != "BULL" or not report.observations:
        return []
    newest = report.newest_close_time_ms
    grouped: dict[str, list[ScalpObservation]] = {}
    for item in report.observations:
        if item.bar_close_time_ms == newest and item.family in TESTED_FAMILIES:
            grouped.setdefault(item.perpetual_symbol, []).append(item)
    open_symbols = {row["perpetual_symbol"] for row in records if row["status"] == OPEN}
    known = {row["id"] for row in records}
    day = newest // DAY_MS
    used = sum(1 for row in records if row["bar_close_time_ms"] // DAY_MS == day)
    ranked = sorted(
        (
            (symbol, tuple(items))
            for symbol, items in grouped.items()
            if len({item.family for item in items}) >= 2
            and any(item.alert_tier == "KURULUM" for item in items)
            and any(item.execution_eligible for item in items)
            and symbol not in open_symbols
            and record_id(symbol, newest) not in known
        ),
        key=lambda pair: (-max(item.score for item in pair[1]), pair[0]),
    )
    return ranked[: max(daily_limit - used, 0)]


def record_id(symbol: str, close_ms: int) -> str:
    return sha256(f"{VERSION}|{symbol}|{close_ms}".encode()).hexdigest()


def _new_record(symbol: str, items: tuple[ScalpObservation, ...], now_ms: int) -> dict[str, Any]:
    close_ms = items[0].bar_close_time_ms
    return {
        "id": record_id(symbol, close_ms),
        "perpetual_symbol": symbol,
        "families": sorted({item.family for item in items}),
        "score": max(item.score for item in items),
        "bar_close_time_ms": close_ms,
        "source_price": items[0].price,
        "direction": "YUKARI",
        "target_bps": TARGET_BPS,
        "stop_bps": STOP_BPS,
        "horizon_minutes": HORIZON_MINUTES,
        "round_trip_cost_bps": COST_BPS,
        "recorded_at_ms": now_ms,
        "status": OPEN,
        "outcome": None,
        "delivery": None,
        "result_delivery": None,
    }


def settle_records(records: list[dict[str, Any]], data_dir: Path, now_ms: int) -> int:
    """Resolve open records with complete candles; never infer across a gap."""
    frames: dict[str, pd.DataFrame] = {}
    settled = 0
    grace_ms = SCALP_SETTLEMENT_GRACE_DAYS * DAY_MS
    for record in records:
        if record["status"] != OPEN:
            continue
        symbol = record["perpetual_symbol"]
        if symbol not in frames:
            try:
                frames[symbol] = load_cache(scalp_cache_path(data_dir, symbol))
            except (MarketDataError, OSError, ValueError):
                frames[symbol] = pd.DataFrame()
        source_ms = int(record["bar_close_time_ms"])
        deadline_ms = source_ms + HORIZON_MINUTES * 60_000
        window = _scalp_bracket_window(frames[symbol], source_ms, min(now_ms, deadline_ms))
        outcome = _first_touch_bracket(record, window, deadline_reached=now_ms >= deadline_ms)
        if outcome is None:
            if now_ms - deadline_ms > grace_ms:
                record["status"] = DATA_MISSING
                settled += 1
            continue
        record["status"] = outcome["resolution"]
        record["outcome"] = {
            key: outcome[key]
            for key in (
                "entry_price",
                "target_price",
                "stop_price",
                "exit_price",
                "exit_time_ms",
                "ambiguous_same_bar",
                "gross_bps",
                "net_bps",
            )
        }
        settled += 1
    return settled


def initial_text(record: dict[str, Any], daily_limit: int, used_today: int) -> str:
    price = float(record["source_price"])
    return "\n".join(
        (
            f"🧪 DENEYSEL LONG %2 • {record['perpetual_symbol']}",
            f"Referans: ${price:.8g} • {local_text(record['bar_close_time_ms'], with_seconds=False)}",
            f"Hedef: ${price * 1.02:.8g} (+%2) • Stop: ${price * 0.98:.8g} (−%2)",
            f"Süre: 24 saat • Aileler: {', '.join(record['families'])} • Bugün {used_today}/{daily_limit}",
            "Ölçüm girişi sonraki 5m açılışıdır; seviyeler ona göre hesaplanır.",
            "Başabaş başarı %55. Üstünlüğü kanıtlanmadı • Emir verilmez.",
        )
    )


def result_text(record: dict[str, Any], summary: dict[str, Any]) -> str:
    outcome = record["outcome"]
    title = {WIN: "🎯 HEDEF", LOSS: "🛑 STOP", TIME_EXIT: "⏱ SÜRE DOLDU"}[record["status"]]
    rate = summary["win_rate"]
    score = f"%{rate * 100:.1f}" if rate is not None else "henüz yok"
    lines = [
        f"{title} • DENEYSEL LONG %2 • {record['perpetual_symbol']}",
        f"Giriş ${outcome['entry_price']:.8g} → Çıkış ${outcome['exit_price']:.8g}",
        f"Net: {outcome['net_bps']:+.1f} bps (20 bps maliyet sonrası)",
    ]
    if outcome["ambiguous_same_bar"]:
        lines.append("Hedef ve stop aynı mumda; ihtiyatla STOP sayıldı.")
    lines.append(f"Deney karnesi: {summary['wins']}/{summary['decided']} • {score}")
    return "\n".join(lines)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in records if row["status"] in {WIN, LOSS, TIME_EXIT}]
    wins = sum(row["status"] == WIN for row in done)
    decided = sum(row["status"] in {WIN, LOSS} for row in done)
    days = np.array([int(row["bar_close_time_ms"]) // DAY_MS for row in done], dtype=np.int64)
    net = np.array([float(row["outcome"]["net_bps"]) for row in done], dtype=np.float64)
    interval = block_bootstrap_interval(net, days) if len(np.unique(days)) >= 2 else None
    return {
        "signals": len(records),
        "open": sum(row["status"] == OPEN for row in records),
        "missing": sum(row["status"] == DATA_MISSING for row in records),
        "settled": len(done),
        "decided": decided,
        "wins": wins,
        "time_exits": len(done) - decided,
        "signal_days": len(np.unique(days)),
        "win_rate": wins / decided if decided else None,
        "mean_net_bps": float(net.mean()) if len(net) else None,
        "mean_net_stress_bps": float(net.mean()) - (STRESS_COST_BPS - COST_BPS) if len(net) else None,
        "net_ci_bps": interval,
    }


def verdict(summary: dict[str, Any], started_at_ms: int | None, now_ms: int) -> str:
    elapsed = (now_ms - started_at_ms) / DAY_MS if started_at_ms else 0.0
    if elapsed < DECISION_DAYS:
        return "SÜRÜYOR"
    if summary["settled"] < MIN_SETTLED or summary["signal_days"] < MIN_SIGNAL_DAYS:
        return "KAPAT" if elapsed >= HARD_STOP_DAYS else "VERİ YETERSİZ"
    if (
        summary["mean_net_bps"] <= 0.0
        or summary["win_rate"] is None
        or summary["win_rate"] < BREAK_EVEN_WIN_RATE
    ):
        return "BAŞARISIZ"
    interval = summary["net_ci_bps"]
    if summary["mean_net_stress_bps"] > 0.0 and interval is not None and interval[0] > 0.0:
        return "GEÇTİ"
    return "KARARSIZ"


def _deliver(
    sender: TelegramNotifier, identifier: str, text: str, receipts: Path
) -> str:
    try:
        return sender.deliver_owner_once(signal_id=identifier, text=text, state_dir=receipts).status
    except (TelegramError, OSError, ValueError):
        return "ERROR"


def run_long_2pct_experiment(
    settings: Settings,
    report: ScalpScanReport,
    *,
    now: datetime | None = None,
    notifier: TelegramNotifier | None = None,
) -> list[str]:
    """Settle, select and notify for one scan. Returns log lines."""
    if not settings.long_2pct_experiment_enabled:
        return []
    now_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    state = _load_state(settings, started_at_ms=now_ms)
    records: list[dict[str, Any]] = state["records"]
    events: list[str] = []
    settled = settle_records(records, settings.scalp_data_dir, now_ms)
    if settled:
        events.append(f"{settled} sonuc")

    fresh = 0 <= now_ms - report.newest_close_time_ms <= MAX_SIGNAL_AGE_MS
    if (
        fresh
        and report.coverage >= settings.scalp_minimum_coverage
        and not settings.scalp_alert_kill_switch
    ):
        for symbol, items in select_candidates(
            report, records, daily_limit=settings.long_2pct_daily_limit
        ):
            records.append(_new_record(symbol, items, now_ms))
    # Persist before any send: a crash then retries through the idempotent receipt.
    atomic_write_json(_state_path(settings), state)

    if notifier is None and not telegram_configured():
        return events
    sender = notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)
    receipts = settings.telegram_state_dir / VERSION
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    for record in records:
        close_day = record["bar_close_time_ms"] // DAY_MS
        if record["delivery"] in RETRYABLE and now_ms - record["recorded_at_ms"] <= MAX_SIGNAL_AGE_MS:
            position = sum(
                1
                for row in records
                if row["bar_close_time_ms"] // DAY_MS == close_day
                and (row["bar_close_time_ms"], row["id"]) <= (record["bar_close_time_ms"], record["id"])
            )
            record["delivery"] = _deliver(
                sender,
                record["id"],
                initial_text(record, settings.long_2pct_daily_limit, position),
                receipts,
            )
            events.append(f"{record['perpetual_symbol']} aday: {record['delivery']}")
        if (
            record["delivery"] in DELIVERED
            and record["status"] in {WIN, LOSS, TIME_EXIT}
            and record["result_delivery"] in RETRYABLE
            and now_ms - record["outcome"]["exit_time_ms"] <= DAY_MS
        ):
            identifier = sha256(f"{record['id']}|result".encode()).hexdigest()
            record["result_delivery"] = _deliver(
                sender, identifier, result_text(record, summarize(records)), receipts
            )
            events.append(f"{record['perpetual_symbol']} {record['status']}: {record['result_delivery']}")
    atomic_write_json(_state_path(settings), state)
    return events


def format_long_2pct_status(settings: Settings, *, now: datetime | None = None) -> str:
    now_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    state = _load_state(settings)
    summary = summarize(state["records"])
    started = state.get("startedAtMs")
    rate = summary["win_rate"]
    mean = summary["mean_net_bps"]
    interval = summary["net_ci_bps"]
    unsent = sum(
        row["delivery"] in DELIVERED
        and row["status"] in {WIN, LOSS, TIME_EXIT}
        and row["result_delivery"] not in DELIVERED
        for row in state["records"]
    )
    return "\n".join(
        (
            f"🧪 Deneysel LONG %2 • {'açık' if settings.long_2pct_experiment_enabled else 'kapalı'}",
            f"Başlangıç: {local_text(started, with_seconds=False) if started else 'henüz yok'}",
            (f"Sinyal {summary['signals']} • Açık {summary['open']} • Sonuçlanan {summary['settled']} "
            f"• Süre dolan {summary['time_exits']} • Veri eksik {summary['missing']}"),
            f"Başarı: {summary['wins']}/{summary['decided']}"
            + (f" (%{rate * 100:.1f}, başabaş %55)" if rate is not None else ""),
            "Ortalama net: "
            + (
                f"{mean:+.1f} bps @20 • {summary['mean_net_stress_bps']:+.1f} bps @30"
                if mean is not None
                else "henüz yok"
            ),
            "%95 günlük blok aralığı @20: "
            + (f"{interval[0]:+.1f} / {interval[1]:+.1f} bps" if interval else "henüz yok"),
            f"Sinyal günü: {summary['signal_days']} • Karar: {verdict(summary, started, now_ms)}",
            *([f"Teslim edilemeyen sonuç mesajı: {unsent}"] if unsent else []),
        )
    )
