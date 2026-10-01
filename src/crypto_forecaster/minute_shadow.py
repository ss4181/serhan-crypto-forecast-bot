"""Prospective 1m entry experiment, never a live notification/order strategy."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from threading import Event

import numpy as np
import pandas as pd

from .config import Settings
from .data import BinanceMarketDataClient, update_market_cache
from .long_scout import HOUR, STEP, _read, _scout_lock, profile, root
from .measurement import _wilson_interval
from .persistence import atomic_write_json

MINUTE = 60_000
VERSION = "long-minute-shadow-v1"


def trigger(bars: pd.DataFrame, now_ms: int) -> dict | None:
    x = bars.sort_values("open_time_ms").drop_duplicates("open_time_ms")
    x = x[x["close_time_ms"] < now_ms]
    if len(x) < 90 or not 0 <= now_ms - int(x.iloc[-1]["close_time_ms"]) <= 90_000:
        return None
    x = x.tail(90)
    if not bool((np.diff(x["open_time_ms"]) == MINUTE).all()):
        return None
    c, q = x["close"], x["quote_volume"]
    fast, slow = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    previous_high = float(x["high"].iloc[-31:-1].max())
    breakout = (float(c.iloc[-1]) / previous_high - 1) * 100
    reference_volume = float(q.iloc[-63:-3].mean()) * 3
    ratio = float(q.iloc[-3:].sum()) / reference_volume if reference_volume > 0 else 0
    last = x.iloc[-1]
    span = float(last["high"] - last["low"])
    position = float((last["close"] - last["low"]) / span) if span > 0 else 0
    if not (
        c.iloc[-1] > fast.iloc[-1] > slow.iloc[-1]
        and 0 < breakout <= 0.5
        and ratio >= 1.5
        and position >= 0.7
    ):
        return None
    return {
        "sourceCloseMs": int(last["close_time_ms"]),
        "price": float(last["close"]),
        "volumeRatio": ratio,
        "breakoutPct": breakout,
    }


def outcome(record: dict, bars: pd.DataFrame, now_ms: int, entry_step: int) -> dict:
    start = (record["recordedAtMs"] // entry_step + 1) * entry_step
    end = start + HOUR
    cutoff = min(end, now_ms // MINUTE * MINUTE)
    x = (
        bars[(bars["open_time_ms"] >= start) & (bars["close_time_ms"] < cutoff)]
        .sort_values("open_time_ms")
        .drop_duplicates("open_time_ms")
    )
    expected = np.arange(start, cutoff, MINUTE, dtype=np.int64)
    complete = (
        len(expected) > 0
        and np.array_equal(x["open_time_ms"].to_numpy(), expected)
        and bool((x["close_time_ms"] - x["open_time_ms"] == MINUTE - 1).all())
    )
    mature = now_ms >= end
    if not complete:
        return {"mature": mature, "complete": False}
    price = float(x.iloc[0]["open"])
    high = (x["high"].to_numpy() / price - 1) * 100
    low = (1 - x["low"].to_numpy() / price) * 100
    stop_indices = np.flatnonzero(low >= record["stopPct"] - 1e-9)
    stops = int(stop_indices[0]) if len(stop_indices) else None
    targets = {}
    for level in (2, 3, 5):
        hit_indices = np.flatnonzero(high >= level - 1e-9)
        hit = int(hit_indices[0]) if len(hit_indices) else None
        same = hit is not None and hit == stops
        first = (
            None
            if same
            else (stops is None or hit < stops)
            if hit is not None
            else False
            if mature or stops is not None
            else None
        )
        net = (
            level * 100 - record["costBps"]
            if first is True
            else -record["stopPct"] * 100 - record["costBps"]
            if stops is not None
            else float((x.iloc[-1]["close"] / price - 1) * 10000 - record["costBps"])
            if mature
            else None
        )
        targets[str(level)] = {
            "hit": True if hit is not None else False if mature else None,
            "firstBeforeStop": first,
            "netBps": None if same else net,
        }
    return {
        "mature": mature,
        "complete": True,
        "entryPrice": price,
        "entryOpenMs": start,
        "targets": targets,
    }


def run_minute_shadow(
    settings: Settings,
    summary: dict,
    *,
    now: datetime | None = None,
    client: BinanceMarketDataClient | None = None,
    stop: Event | None = None,
) -> dict:
    """Run one bounded prospective cycle, protected against overlapping writers."""
    if not settings.long_scout_minute_shadow_enabled:
        return {}
    with _scout_lock(root(settings) / "minute-shadow" / "worker.lock") as acquired:
        if not acquired:
            return load_minute_shadow_summary(settings)
        return _run(settings, summary, now=now, client=client, stop=stop)


def _run(settings: Settings, summary: dict, *, now, client, stop) -> dict:
    started = now or datetime.now(UTC)
    stamp = int(started.timestamp() * 1000)
    directory = root(settings) / "minute-shadow"
    state = _read(directory / "state.json")
    records = state.get("records", []) if state.get("version") == VERSION else []
    profile_id = sha256(f"{VERSION}|{profile(settings)[0]}".encode()).hexdigest()[:16]
    regime = summary.get("regime", {})
    ready = (
        regime.get("state") == "BULL_CONFIRMED"
        and regime.get("dataHealthy") is True
        and 0 <= stamp - summary.get("scannedAtMs", 0) < HOUR
    )
    candidates = (
        [
            r
            for r in summary.get("watchlist", [])
            if r["score"] >= settings.long_scout_minimum_score
        ][:10]
        if ready
        else []
    )
    pending = [r for r in records if stamp < r["recordedAtMs"] + 2 * HOUR]
    symbols = sorted({r["symbol"] for r in candidates + pending})
    market = client or BinanceMarketDataClient(
        market_name="futures", timeout_seconds=8, request_pause_seconds=0.2
    )
    if client is None:
        market.page_limit = 499
    errors = 0
    for symbol in symbols:
        if stop is not None and stop.is_set():
            return {}
        try:
            bars = update_market_cache(
                directory / "prices" / f"{symbol}_1m.csv",
                symbol,
                "1m",
                days=1,
                client=market,
                now=started,
            )
        except (OSError, RuntimeError, ValueError):
            errors += 1
            continue
        current_ms = (
            stamp if now is not None else int(datetime.now(UTC).timestamp() * 1000)
        )
        for record in records:
            if (
                record["symbol"] == symbol
                and current_ms < record["recordedAtMs"] + 2 * HOUR
            ):
                record["oneMinute"] = outcome(record, bars, current_ms, MINUTE)
                record["fiveMinute"] = outcome(record, bars, current_ms, STEP)
        candidate = next((r for r in candidates if r["symbol"] == symbol), None)
        signal = trigger(bars, current_ms) if candidate else None
        latest = max(
            (r["recordedAtMs"] for r in records if r["symbol"] == symbol), default=0
        )
        if (
            signal
            and current_ms - latest >= HOUR
            and sum(current_ms < r["recordedAtMs"] + 2 * HOUR for r in records) < 20
        ):
            records.append(
                {
                    **signal,
                    "symbol": symbol,
                    "strategy": candidate["strategy"],
                    "recordedAtMs": current_ms,
                    "regime": "BULL_CONFIRMED",
                    "profileId": profile_id,
                    "stopPct": candidate.get(
                        "stopPct", max(2, min(6, 2 * candidate["atrPct"]))
                    ),
                    "costBps": 2 * settings.scalp_taker_fee_bps
                    + candidate["spreadBps"]
                    + 2 * settings.scalp_slippage_bps_per_side,
                    "id": sha256(
                        f"{VERSION}|{symbol}|{signal['sourceCloseMs']}".encode()
                    ).hexdigest(),
                    "oneMinute": {},
                    "fiveMinute": {},
                }
            )
    records = records[-1000:]
    current_records = [r for r in records if r.get("profileId") == profile_id]
    paired = [
        r
        for r in current_records
        if all(
            r.get(k, {}).get("mature") and r[k].get("complete")
            for k in ("oneMinute", "fiveMinute")
        )
    ]
    result = {
        "version": VERSION,
        "mode": "shadow",
        "alerts": False,
        "profileId": profile_id,
        "evaluatedAtMs": stamp,
        "candidateCount": len(candidates),
        "errors": errors,
        "records": len(current_records),
        "pending": sum(
            stamp < r["recordedAtMs"] + HOUR + STEP for r in current_records
        ),
        "missing": sum(
            stamp >= r["recordedAtMs"] + HOUR + STEP
            and not all(
                r.get(k, {}).get("mature") and r[k].get("complete")
                for k in ("oneMinute", "fiveMinute")
            )
            for r in current_records
        ),
        "pairedN": len(paired),
        "oneMinute2PctHits": sum(
            r["oneMinute"]["targets"]["2"]["hit"] is True for r in paired
        ),
        "fiveMinute2PctHits": sum(
            r["fiveMinute"]["targets"]["2"]["hit"] is True for r in paired
        ),
        "entryMetrics": {
            key: {
                str(level): _target_metrics(paired, key, str(level))
                for level in (2, 3, 5)
            }
            for key in ("oneMinute", "fiveMinute")
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "state.json", {"version": VERSION, "records": records}
    )
    (directory / "state.json").chmod(0o600)
    atomic_write_json(directory / "public-summary.json", result)
    (directory / "public-summary.json").chmod(0o644)
    return result


def _target_metrics(records: list[dict], key: str, level: str) -> dict:
    targets = [r[key]["targets"][level] for r in records]
    first = [t for t in targets if t["firstBeforeStop"] is not None]
    net = [t["netBps"] for t in targets if t["netBps"] is not None]
    hits = sum(t["hit"] is True for t in targets)
    return {
        "n": len(targets),
        "hits": hits,
        "wilson95": _wilson_interval(hits, len(targets)) if targets else None,
        "firstN": len(first),
        "firstWins": sum(t["firstBeforeStop"] is True for t in first),
        "ambiguous": len(targets) - len(first),
        "meanNetBps": float(np.mean(net)) if net else None,
    }


def load_minute_shadow_summary(settings: Settings) -> dict:
    data = _read(root(settings) / "minute-shadow" / "public-summary.json")
    if data.get("version") != VERSION or data.get("mode") != "shadow":
        return {}
    return {
        key: data[key]
        for key in (
            "version",
            "mode",
            "alerts",
            "profileId",
            "evaluatedAtMs",
            "candidateCount",
            "errors",
            "records",
            "pending",
            "missing",
            "pairedN",
            "oneMinute2PctHits",
            "fiveMinute2PctHits",
            "entryMetrics",
        )
        if key in data
    }
