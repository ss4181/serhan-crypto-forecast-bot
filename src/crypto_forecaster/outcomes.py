"""Live scorecard: what actually happened to the signals the bot sent.

The backtest measures the model; this measures the bot.  Every delivered signal
is parked in ``pending`` until its barrier resolves -- take profit, stop, or the
time limit -- and is then scored exactly the way the backtest scores it and
appended to an append-only ledger.

Scoring the next candle's close instead would answer a different question than
the model was asked, and a forward test that measures the wrong thing is worse
than none: it looks like evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import INTERVAL_LABELS, cache_path
from .data import load_cache


PENDING_SCHEMA = "signal-outcome-pending-v2"
LEDGER_SCHEMA = "signal-outcome-v2"
SETTLEMENT_GRACE_DAYS = 7


def pending_dir(state_dir: Path) -> Path:
    return state_dir / "pending"


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "ledger.jsonl"


def record_delivery(
    state_dir: Path,
    *,
    signal_id: str,
    symbol: str,
    interval: str,
    tier: str,
    direction: str,
    probability: float,
    source_price: float,
    source_close_time_ms: int,
    target_close_time_ms: int,
    delivered_at_ms: int,
    barrier_bps: float,
    horizon_ms: int,
) -> Path:
    directory = pending_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{signal_id}.json"
    payload = {
        "schema": PENDING_SCHEMA,
        "signal_id": signal_id,
        "symbol": symbol,
        "interval": interval,
        "tier": tier,
        "direction": direction,
        "probability": float(probability),
        "source_price": float(source_price),
        "source_close_time_ms": int(source_close_time_ms),
        "target_close_time_ms": int(target_close_time_ms),
        "delivered_at_ms": int(delivered_at_ms),
        # The barrier the model was actually asked about.  Scoring the next
        # candle's close instead would measure a different question than the
        # one the model answers, and a month of that is a month wasted.
        "barrier_bps": float(barrier_bps),
        "horizon_ms": int(horizon_ms),
    }
    if not path.exists():
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return path


def settle_pending(
    state_dir: Path,
    data_dir: Path,
    *,
    round_trip_cost_bps: float,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    directory = pending_dir(state_dir)
    if not directory.exists():
        return []
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    grace_ms = SETTLEMENT_GRACE_DAYS * 24 * 60 * 60 * 1000
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    settled: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_record(path)
        if record is None:
            path.unlink(missing_ok=True)
            continue
        source_ms = int(record["source_close_time_ms"])
        deadline_ms = source_ms + int(record["horizon_ms"])
        key = (str(record["symbol"]), str(record["interval"]))
        if key not in frames:
            frames[key] = _candles(data_dir, *key)
        window = _window(frames[key], source_ms, deadline_ms)
        outcome = _resolve(record, window, expired=current_ms >= deadline_ms)
        if outcome is None:
            # Still open, or the candles have not arrived yet.
            if current_ms - deadline_ms > grace_ms:
                path.unlink(missing_ok=True)
            continue
        settled.append(_settle(record, outcome, round_trip_cost_bps, current_ms))
        path.unlink(missing_ok=True)
    if settled:
        _append_ledger(state_dir, settled)
    return settled


def load_ledger(state_dir: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    path = ledger_path(state_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == LEDGER_SCHEMA:
            rows.append(payload)
    return rows


def scorecard(
    rows: Iterable[dict[str, Any]],
    *,
    days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    window_ms = days * 24 * 60 * 60 * 1000
    recent = [
        row
        for row in rows
        if isinstance(row.get("target_close_time_ms"), int)
        and current_ms - int(row["target_close_time_ms"]) <= window_ms
    ]
    return {
        "days": days,
        "overall": _aggregate(recent),
        "byTier": {
            tier: _aggregate([row for row in recent if row.get("tier") == tier])
            for tier in ("ISLEM", "GOZLEM")
        },
        "byModel": {
            key: _aggregate([row for row in recent if _model_key(row) == key])
            for key in sorted({_model_key(row) for row in recent})
        },
    }


def format_scorecard(card: dict[str, Any]) -> str:
    overall = card["overall"]
    lines = [
        f"📊 CANLI KARNE — son {card['days']} gun",
        "",
        "Gonderilen sinyallerin gercek sonucu. Backtest degil, botun kendi kaydi.",
        "",
        f"Toplam kapanan sinyal: {overall['count']}",
    ]
    if not overall["count"]:
        lines.append("Henuz kapanmis sinyal yok.")
        lines.append("")
        lines.append("Yalnizca arastirma bildirimidir; yatirim tavsiyesi veya emir degildir.")
        return "\n".join(lines)
    lines.extend(
        [
            f"Yon isabeti: %{overall['hitRate'] * 100:.1f}",
            f"Maliyet sonrasi ortalama: {overall['netBps']:+.2f} bps/sinyal",
            f"Maliyet sonrasi toplam: {overall['netBpsTotal']:+.1f} bps",
            "",
            "MODEL BAZINDA",
        ]
    )
    for key, stats in card["byModel"].items():
        if not stats["count"]:
            continue
        symbol, _, interval = key.partition("_")
        label = INTERVAL_LABELS.get(interval, interval)
        lines.append(
            f"• {symbol} {label}: n={stats['count']}, isabet %{stats['hitRate'] * 100:.1f}, "
            f"net {stats['netBps']:+.2f} bps"
        )
    lines.extend(
        [
            "",
            "Yalnizca arastirma bildirimidir; yatirim tavsiyesi veya emir degildir.",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _Outcome:
    """How the barrier trade ended, scored the way the backtest scores it."""

    gross_bps: float
    resolution: str
    correct: bool
    exit_price: float
    exit_time_ms: int


def _resolve(
    record: dict[str, Any], window: pd.DataFrame, *, expired: bool
) -> _Outcome | None:
    source_price = float(record["source_price"])
    barrier_bps = float(record["barrier_bps"])
    long_side = str(record["direction"]) == "YUKARI"
    upper = source_price * math.exp(barrier_bps / 10_000.0)
    lower = source_price * math.exp(-barrier_bps / 10_000.0)
    for close_time, high, low, close in window.itertuples(index=False):
        up = high >= upper
        down = low <= lower
        if not up and not down:
            continue
        if up and down:
            # One candle reached both levels; nothing says which came first, so
            # neither side may claim the good one.
            return _Outcome(-barrier_bps, "BELIRSIZ", False, float(close), int(close_time))
        favourable = up if long_side else down
        return _Outcome(
            barrier_bps if favourable else -barrier_bps,
            "HEDEF" if favourable else "STOP",
            favourable,
            upper if up else lower,
            int(close_time),
        )
    if not expired or window.empty:
        return None
    last = window.iloc[-1]
    exit_price = float(last["close"])
    move_bps = math.log(exit_price / source_price) * 10_000.0
    gross = move_bps if long_side else -move_bps
    return _Outcome(gross, "SURE", gross > 0, exit_price, int(last["close_time_ms"]))


def _settle(
    record: dict[str, Any],
    outcome: _Outcome,
    round_trip_cost_bps: float,
    settled_at_ms: int,
) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "signal_id": record["signal_id"],
        "symbol": record["symbol"],
        "interval": record["interval"],
        "tier": record.get("tier", "GOZLEM"),
        "direction": str(record["direction"]),
        "probability": float(record["probability"]),
        "source_price": float(record["source_price"]),
        "source_close_time_ms": int(record["source_close_time_ms"]),
        "target_close_time_ms": int(record["target_close_time_ms"]),
        "barrier_bps": float(record["barrier_bps"]),
        "resolution": outcome.resolution,
        "realized_close": outcome.exit_price,
        "exit_time_ms": outcome.exit_time_ms,
        "correct": outcome.correct,
        "gross_bps": outcome.gross_bps,
        "net_bps": outcome.gross_bps - float(round_trip_cost_bps),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "settled_at_ms": settled_at_ms,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"count": 0, "hitRate": 0.0, "grossBps": 0.0, "netBps": 0.0, "netBpsTotal": 0.0}
    hits = sum(1 for row in rows if row.get("correct") is True)
    gross = [float(row.get("gross_bps", 0.0)) for row in rows]
    net = [float(row.get("net_bps", 0.0)) for row in rows]
    return {
        "count": count,
        "hitRate": hits / count,
        "grossBps": sum(gross) / count,
        "netBps": sum(net) / count,
        "netBpsTotal": sum(net),
    }


def _model_key(row: dict[str, Any]) -> str:
    return f"{row.get('symbol', '?')}_{row.get('interval', '?')}"


def _candles(data_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    try:
        return load_cache(cache_path(data_dir, symbol, interval))
    except Exception:
        return pd.DataFrame(columns=["close_time_ms", "high", "low", "close"])


def _window(frame: pd.DataFrame, after_ms: int, until_ms: int) -> pd.DataFrame:
    """Candles a trade opened at `after_ms` would have lived through."""
    if frame.empty:
        return frame
    selected = frame[
        (frame["close_time_ms"] > after_ms) & (frame["close_time_ms"] <= until_ms)
    ]
    return selected.loc[:, ["close_time_ms", "high", "low", "close"]]


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != PENDING_SCHEMA:
        return None
    required = (
        "signal_id",
        "symbol",
        "interval",
        "direction",
        "source_price",
        "source_close_time_ms",
        "target_close_time_ms",
        "barrier_bps",
        "horizon_ms",
    )
    if any(key not in payload for key in required):
        return None
    try:
        float(payload["source_price"])
        int(payload["source_close_time_ms"])
        int(payload["target_close_time_ms"])
        float(payload["barrier_bps"])
        int(payload["horizon_ms"])
    except (TypeError, ValueError):
        return None
    return payload


def _append_ledger(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with ledger_path(state_dir).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


__all__ = [
    "format_scorecard",
    "load_ledger",
    "record_delivery",
    "scorecard",
    "settle_pending",
]
