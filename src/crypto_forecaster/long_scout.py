"""Binance-only LONG discovery and prospective paper tracking, isolated from scalp.

Rules are fixed before outcomes: accumulation, volume breakout, trend retest.
Scores rank evidence, never represent probabilities. Optional private owner
alerts are experimental; no orders, model promotion or other-project writes.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings, local_text
from .data import BinanceMarketDataClient, FuturesMarketSnapshot, update_market_cache
from .measurement import _wilson_interval
from .persistence import atomic_write_json

VERSION = "long-discovery-v1"
HOUR = 3_600_000
DAY = 24 * HOUR
STEP = 300_000
TARGETS = (2, 3, 5, 10, 20)
LABELS = {"LA1": "Birikim", "LB1": "Hacimli kırılım", "LP1": "Trend geri testi"}
WATCH_FIELDS = (
    "symbol",
    "strategy",
    "stage",
    "score",
    "referencePrice",
    "sourceCloseMs",
    "contractAgeDays",
    "newListing",
    "relativeStrength7dPct",
    "relativeStrength24hPct",
    "volume4hRatio",
    "compressionRatio",
    "atrPct",
    "spreadBps",
    "fundingBps",
    "reason",
)


def root(settings: Settings) -> Path:
    return settings.scalp_state_dir / "experiments" / VERSION


def profile(settings: Settings) -> tuple[str, dict]:
    options = {
        "version": VERSION,
        **{
            name: getattr(settings, name)
            for name in (
                "long_scout_universe_limit",
                "long_scout_history_days",
                "long_scout_minimum_volume_usdt",
                "long_scout_maximum_spread_bps",
                "long_scout_maximum_abs_funding_bps",
                "long_scout_minimum_score",
                "long_scout_maximum_active",
                "scalp_taker_fee_bps",
                "scalp_slippage_bps_per_side",
            )
        },
    }
    return sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[
        :16
    ], options


def _read(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def select_universe(
    contracts: dict,
    snapshots: dict[str, FuturesMarketSnapshot],
    settings: Settings,
    now_ms: int,
) -> list[str]:
    """Screen every contract; zero cap scans all markets passing quality gates."""
    eligible = []
    for symbol, contract in contracts.items():
        snapshot = snapshots.get(symbol)
        age = (now_ms - contract["onboardDate"]) / DAY
        if snapshot is None or age < 14:
            continue
        volume = _finite(snapshot.quote_volume_24h_usdt)
        funding = _finite(snapshot.funding_rate_bps)
        spread = _finite(snapshot.spread_bps)
        if (
            volume is None
            or volume < settings.long_scout_minimum_volume_usdt
            or funding is None
            or abs(funding) > settings.long_scout_maximum_abs_funding_bps
            or spread is None
            or not 0 <= spread <= settings.long_scout_maximum_spread_bps
        ):
            continue
        eligible.append((symbol, age, volume))
    liquid = sorted(eligible, key=lambda row: (-row[2], row[0]))
    recent = sorted(
        (row for row in eligible if row[1] <= 365),
        key=lambda row: (row[1], -row[2], row[0]),
    )
    cap = settings.long_scout_universe_limit
    if cap == 0:
        return [row[0] for row in liquid]
    chosen = [row[0] for row in recent[: cap // 4]]
    chosen.extend(row[0] for row in liquid if row[0] not in chosen)
    return chosen[:cap]


def features(bars: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """All rolling measurements end at their own closed hour; no future filling."""
    x = bars.sort_values("open_time_ms").drop_duplicates("open_time_ms").copy()
    x = x.set_index("close_time_ms", drop=False)
    if x.empty:
        return x
    # Never bridge an exchange/data gap inside rolling windows.
    gaps = np.flatnonzero(np.diff(x.index.to_numpy()) != HOUR)
    if len(gaps):
        x = x.iloc[int(gaps[-1]) + 1 :].copy()
    c = x["close"]
    x["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()
    x["ema200"] = c.ewm(span=200, adjust=False, min_periods=200).mean()
    x["return24"] = (c / c.shift(24) - 1) * 100
    x["return7d"] = (c / c.shift(168) - 1) * 100
    reference = btc.drop_duplicates("close_time_ms").set_index("close_time_ms")["close"]
    reference = reference.reindex(x.index)  # Exact alignment, no forward/back fill.
    x["rs24"] = ((c / c.shift(24)) / (reference / reference.shift(24)) - 1) * 100
    x["rs7d"] = ((c / c.shift(168)) / (reference / reference.shift(168)) - 1) * 100
    q = x["quote_volume"]
    x["quote24"] = q.rolling(24).sum()
    x["volume4"] = q.rolling(4).sum() / (q.shift(4).rolling(168).mean() * 4).replace(
        0, np.nan
    )
    span24 = (x["high"].rolling(24).max() / x["low"].rolling(24).min() - 1) * 100
    x["compression"] = span24.shift(4) / span24.shift(24).rolling(168).median().replace(
        0, np.nan
    )
    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - c.shift()).abs(),
            (x["low"] - c.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atrPct"] = true_range.rolling(14).mean() / c * 100
    previous_high = x["high"].shift().rolling(168).max()
    x["breakoutPct"] = (c / previous_high - 1) * 100
    x["closePosition"] = (c - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    x["takerRatio"] = x["taker_buy_base"].rolling(4).sum() / x["volume"].rolling(
        4
    ).sum().replace(0, np.nan)
    x["trend"] = (
        (c > x["ema20"])
        & (x["ema20"] > x["ema50"])
        & (x["ema50"] > x["ema50"].shift(24))
    )
    x["macroTrend"] = (
        (c > x["ema50"])
        & (x["ema50"] > x["ema200"])
        & (x["ema50"] > x["ema50"].shift(24))
        & (x["return7d"] > 0)
    )
    x["retest"] = (
        (c.shift() <= x["ema20"].shift())
        & (c > x["ema20"])
        & (abs(c / x["ema20"] - 1) <= 0.02)
    )
    x["ready"] = np.arange(len(x)) >= 335  # Two weeks of hourly, contiguous history.
    return x


def bull_history(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """A slower LONG regime: BTC+ETH trend, >=60% breadth, six closed hours."""
    if "BTCUSDT" not in tables or "ETHUSDT" not in tables or len(tables) < 3:
        return pd.DataFrame(
            columns=["breadth", "coverage", "rawBull", "establishedBull"]
        )
    btc, eth = tables["BTCUSDT"], tables["ETHUSDT"]
    index = btc.index.intersection(eth.index).sort_values()
    ready = pd.DataFrame(
        {
            s: t["ready"].reindex(index).fillna(False).astype(bool)
            for s, t in tables.items()
        }
    )
    up = pd.DataFrame(
        {
            s: (t["return24"] > 0).reindex(index).fillna(False).astype(bool)
            for s, t in tables.items()
        }
    )
    coverage = ready.sum(axis=1) / len(tables)
    breadth = (up & ready).sum(axis=1) / ready.sum(axis=1).replace(0, np.nan)
    raw = (
        btc["macroTrend"].reindex(index).fillna(False)
        & eth["macroTrend"].reindex(index).fillna(False)
        & (coverage >= 0.8)
        & (breadth >= 0.6)
    )
    consecutive = (
        pd.Series(index.to_numpy(), index=index).diff().eq(HOUR).rolling(5).sum().eq(5)
    )
    return pd.DataFrame(
        {
            "breadth": breadth,
            "coverage": coverage,
            "rawBull": raw,
            "establishedBull": raw.rolling(6).sum().eq(6) & consecutive,
        }
    )


def candidate(
    row: pd.Series, symbol: str, contract: dict, settings: Settings
) -> dict | None:
    required = (
        "rs7d",
        "rs24",
        "volume4",
        "compression",
        "atrPct",
        "breakoutPct",
        "closePosition",
        "takerRatio",
    )
    if not row["ready"] or any(_finite(row[key]) is None for key in required):
        return None
    age = (int(row["close_time_ms"]) - contract["onboardDate"]) / DAY
    if (
        age < 14
        or not row["trend"]
        or row["rs7d"] < 3
        or row["rs24"] <= 0
        or row["quote24"] < settings.long_scout_minimum_volume_usdt
        or not 0.2 <= row["atrPct"] <= 6
        or row["return24"] > 15
        or not -10 <= row["return7d"] <= 40
    ):
        return None
    # Breakout/retest become paper entries; accumulation remains a watch item.
    if (
        0 < row["breakoutPct"] <= 3
        and row["volume4"] >= 1.8
        and row["closePosition"] >= 0.7
    ):
        strategy, stage, reason = (
            "LB1",
            "TEYİT",
            "7 günlük zirve üstünde hacimli kapanış",
        )
    elif (
        row["retest"]
        and -8 <= row["breakoutPct"] <= 0
        and row["volume4"] >= 1.2
        and row["takerRatio"] >= 0.53
    ):
        strategy, stage, reason = "LP1", "TEYİT", "Yükselen trendde EMA20 geri alındı"
    elif (
        row["compression"] <= 0.85
        and row["volume4"] >= 1.15
        and -10 <= row["breakoutPct"] <= 0
    ):
        strategy, stage, reason = (
            "LA1",
            "İZLE",
            "Daralan fiyat aralığında göreli güç ve hacim artışı",
        )
    else:
        return None

    def clip(value):
        return min(1.0, max(0.0, float(value)))

    parts = {
        "rs7d": 20 * clip(row["rs7d"] / 12),
        "rs24": 10 * clip(row["rs24"] / 4),
        "trend": 20.0,
        "volume": 20 * clip(row["volume4"] / 3),
        "compression": 15 * clip(1 - row["compression"]),
        "close": 15 * clip(row["closePosition"]),
    }
    score = sum(parts.values())
    if score < (50 if stage == "İZLE" else settings.long_scout_minimum_score):
        return None
    return {
        "symbol": symbol,
        "strategy": strategy,
        "stage": stage,
        "score": round(score, 2),
        "scoreParts": parts,
        "reason": reason,
        "direction": "LONG",
        "referencePrice": float(row["close"]),
        "sourceCloseMs": int(row["close_time_ms"]),
        "contractAgeDays": round(age, 1),
        "newListing": age <= 365,
        "relativeStrength7dPct": round(float(row["rs7d"]), 2),
        "relativeStrength24hPct": round(float(row["rs24"]), 2),
        "volume4hRatio": round(float(row["volume4"]), 2),
        "compressionRatio": round(float(row["compression"]), 2),
        "atrPct": float(row["atrPct"]),
        "stopPct": min(6.0, max(2.0, 2 * float(row["atrPct"]))),
    }


def paper_outcome(record: dict, bars: pd.DataFrame, cutoff_ms: int) -> dict:
    """Use only full 5m candles AFTER the frozen decision, starting next open.

    Early wins don't enter performance denominators until seven days expire.
    Missing data and same-bar target/stop order are explicitly unresolved.
    """
    start = record["entryOpenMs"]
    expiry = start + 7 * DAY
    end = min(expiry, cutoff_ms // STEP * STEP)
    future = (
        bars[(bars["open_time_ms"] >= start) & (bars["close_time_ms"] < end)]
        .sort_values("open_time_ms")
        .drop_duplicates("open_time_ms")
    )
    expected = np.arange(start, end, STEP, dtype=np.int64)
    complete = (
        len(future) == len(expected)
        and np.array_equal(future["open_time_ms"].to_numpy(), expected)
        and bool((future["close_time_ms"] - future["open_time_ms"] == STEP - 1).all())
    )
    mature = cutoff_ms >= expiry
    result = {
        "mature": mature,
        "complete": bool(complete),
        "entryPrice": None,
        "targets": {},
    }
    if future.empty or int(future.iloc[0]["open_time_ms"]) != start:
        return result
    # Private alerts measure targets against their explicitly reported price;
    # ordinary paper records retain next-open entry. Never rewrite market bars.
    entry = float(record.get("referenceEntryPrice", future.iloc[0]["open"]))
    highs = (future["high"].to_numpy() / entry - 1) * 100
    lows = (1 - future["low"].to_numpy() / entry) * 100
    stops = np.flatnonzero(lows >= record["stopPct"] - 1e-9)
    stop = int(stops[0]) if len(stops) else None
    result.update(
        entryPrice=entry,
        mfePct=float(max(0, highs.max())),
        maePct=float(max(0, lows.max())),
        lastNetBps=float(
            (future.iloc[-1]["close"] / entry - 1) * 10_000 - record["costBps"]
        ),
    )
    for level in TARGETS:
        hits = np.flatnonzero(highs >= level - 1e-9)
        hit = int(hits[0]) if len(hits) else None
        ambiguous = hit is not None and hit == stop
        first = (
            None
            if not complete or ambiguous
            else (stop is None or hit < stop)
            if hit is not None
            else False
            if mature or stop is not None
            else None
        )
        result["targets"][str(level)] = {
            "hit": (hit is not None if mature else True if hit is not None else None)
            if complete
            else None,
            "firstBeforeStop": first,
            "sameBarAmbiguous": ambiguous,
            "touchAtMs": int(future.iloc[hit]["close_time_ms"])
            if complete and hit is not None
            else None,
            "netBps": (
                level * 100 - record["costBps"]
                if first is True
                else -record["stopPct"] * 100 - record["costBps"]
                if stop is not None
                else result["lastNetBps"]
                if mature
                else None
            )
            if complete and not ambiguous
            else None,
        }
    return result


def _metrics(records: list[dict]) -> dict:
    resolved = [
        r
        for r in records
        if r.get("outcome", {}).get("mature")
        and r["outcome"].get("complete")
        and r["outcome"].get("entryPrice")
    ]
    metrics = {}
    for target in TARGETS:
        values = [r["outcome"]["targets"][str(target)] for r in resolved]
        hits = sum(v["hit"] is True for v in values)
        first = [
            v["firstBeforeStop"]
            for v in values
            if isinstance(v["firstBeforeStop"], bool)
        ]
        net = [v["netBps"] for v in values if _finite(v.get("netBps")) is not None]
        metrics[str(target)] = {
            "n": len(values),
            "hits": hits,
            "rate": hits / len(values) if values else None,
            "wilson95": list(_wilson_interval(hits, len(values))) if values else None,
            "firstN": len(first),
            "firstWins": sum(first),
            "ambiguous": len(values) - len(first),
            "meanNetBps": float(np.mean(net)) if net else None,
            "netN": len(net),
        }
    return {
        "records": len(records),
        "pending": sum(not r.get("outcome", {}).get("mature", False) for r in records),
        "missing": sum(
            r.get("outcome", {}).get("mature", False)
            and not r["outcome"].get("complete", False)
            for r in records
        ),
        "targets": metrics,
    }


def _publish(settings: Settings, state: dict, now_ms: int) -> dict:
    profile_id, _ = profile(settings)
    records = [r for r in state.get("records", []) if r.get("profileId") == profile_id]
    summary = {
        "version": VERSION,
        "mode": "shadow",
        "autoPromotion": False,
        "enabled": settings.long_scout_enabled,
        "ownerAlertsEnabled": settings.long_scout_alerts_enabled,
        "evaluatedAtMs": now_ms,
        "scannedAtMs": state.get("scannedAtMs"),
        "regime": state.get("regime", {}),
        "universeCount": state.get("universeCount", 0),
        "screenedCount": state.get("screenedCount", 0),
        "qualityExcludedCount": state.get("qualityExcludedCount", 0),
        "freshCount": state.get("freshCount", 0),
        "errorCount": len(state.get("errors", {})),
        "recordLimitReached": state.get("recordLimitReached", False),
        "profileId": profile_id,
        "watchlist": [
            {k: row.get(k) for k in WATCH_FIELDS}
            for row in state.get("watchlist", [])[:20]
        ],
        "performance": _metrics(records),
        "byStrategy": {
            s: _metrics([r for r in records if r["strategy"] == s])
            for s in ("LA1", "LB1", "LP1")
        },
        "tracked": [
            {
                "symbol": r["symbol"],
                "strategy": r["strategy"],
                "recordedAtMs": r["recordedAtMs"],
                "score": r["score"],
                "stopPct": r["stopPct"],
                "entryOpenMs": r["entryOpenMs"],
                "entryPrice": r.get("outcome", {}).get("entryPrice"),
                "outcome": r.get("outcome", {}),
            }
            for r in records[-100:][::-1]
        ],
    }
    atomic_write_json(root(settings) / "state.json", state)
    (root(settings) / "state.json").chmod(0o600)
    atomic_write_json(root(settings) / "public-summary.json", summary)
    (root(settings) / "public-summary.json").chmod(0o644)
    return summary


def settle_paper(
    settings: Settings,
    state: dict,
    client: BinanceMarketDataClient,
    now: datetime,
    stop: Event | None = None,
) -> None:
    now_ms = int(now.timestamp() * 1000)
    pending = [
        r
        for r in state.get("records", [])
        if not r.get("outcome", {}).get("mature", False)
        or (
            not r["outcome"].get("complete", False)
            and now_ms < r["entryOpenMs"] + 8 * DAY
        )
        or (
            r.get("ownerAlert")
            and (
                not r["ownerAlert"].get("outcome", {}).get("mature", False)
                or (
                    not r["ownerAlert"].get("outcome", {}).get("complete", False)
                    and now_ms < r["ownerAlert"]["trackingStartMs"] + 8 * DAY
                )
            )
        )
    ]
    for symbol in sorted({r["symbol"] for r in pending}):
        if stop is not None and stop.is_set():
            return
        try:
            bars = update_market_cache(
                root(settings) / "prices" / f"{symbol}_5m.csv",
                symbol,
                "5m",
                days=9,
                client=client,
                now=now,
            )
            for record in pending:
                if record["symbol"] == symbol:
                    record["outcome"] = paper_outcome(record, bars, now_ms)
                    if record.get("ownerAlert"):
                        alert = record["ownerAlert"]
                        alert["outcome"] = paper_outcome(
                            {
                                **record,
                                "entryOpenMs": alert["trackingStartMs"],
                                "referenceEntryPrice": alert["referencePrice"],
                            },
                            bars,
                            now_ms,
                        )
        except (OSError, RuntimeError, ValueError) as error:
            state.setdefault("errors", {})[symbol] = type(error).__name__
            for record in pending:
                if (
                    record["symbol"] == symbol
                    and now_ms >= record["entryOpenMs"] + 7 * DAY
                ):
                    record["outcome"] = {
                        **record.get("outcome", {}),
                        "mature": True,
                        "complete": False,
                    }
                if record["symbol"] == symbol and record.get("ownerAlert"):
                    alert = record["ownerAlert"]
                    if now_ms >= alert["trackingStartMs"] + 7 * DAY:
                        alert["outcome"] = {
                            **alert.get("outcome", {}),
                            "mature": True,
                            "complete": False,
                        }


def run_long_scout(
    settings: Settings,
    *,
    now: datetime | None = None,
    client: BinanceMarketDataClient | None = None,
    discover: bool = True,
    stop: Event | None = None,
) -> dict:
    """Hourly discovery; independent five-minute settlement. No outgoing messages."""
    if not settings.long_scout_enabled:
        return {}
    # The service worker and a manual CLI refresh share the same paper ledger.
    # An OS advisory lock is released even if either process crashes.
    with _scout_lock(root(settings) / "worker.lock") as acquired:
        if not acquired:
            return load_long_scout_summary(settings)
        return _run_long_scout_unlocked(
            settings, now=now, client=client, discover=discover, stop=stop
        )


@contextmanager
def _scout_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        os.chmod(path, 0o600)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError):
            pass
        try:
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_long_scout_unlocked(
    settings: Settings,
    *,
    now: datetime | None = None,
    client: BinanceMarketDataClient | None = None,
    discover: bool = True,
    stop: Event | None = None,
) -> dict:
    started = now or datetime.now(UTC)
    now_ms = int(started.timestamp() * 1000)
    market = client or BinanceMarketDataClient(
        market_name="futures", timeout_seconds=12, request_pause_seconds=0.2
    )
    if client is None:
        # Binance charges by requested page size. 499 keeps each scouting
        # request in the low-weight tier, including tiny incremental refreshes.
        market.page_limit = 499
    state = _read(root(settings) / "state.json")
    if state.get("version") != VERSION:
        state = {"version": VERSION, "records": [], "errors": {}}
    state["errors"] = {}
    settle_paper(settings, state, market, started, stop)
    if stop is not None and stop.is_set():
        return {}
    if not discover:
        return _publish(settings, state, now_ms)
    contracts = market.fetch_crypto_perpetual_contracts()
    snapshots = market.fetch_futures_market_snapshots()
    selected = select_universe(contracts, snapshots, settings, now_ms)
    # BTC/ETH are benchmark inputs; never paper-traded as discovery candidates.
    symbols = list(dict.fromkeys(["BTCUSDT", "ETHUSDT", *selected]))
    atomic_write_json(
        root(settings) / "universe.json",
        {
            "atMs": now_ms,
            "contracts": {s: contracts[s] for s in symbols if s in contracts},
            "selected": selected,
        },
    )
    tables = {}
    frames = {}
    for symbol in symbols:
        if stop is not None and stop.is_set():
            return {}
        try:
            frames[symbol] = update_market_cache(
                root(settings) / "prices" / f"{symbol}_1h.csv",
                symbol,
                "1h",
                days=settings.long_scout_history_days,
                client=market,
                now=started,
            )
        except (OSError, RuntimeError, ValueError) as error:
            state["errors"][symbol] = type(error).__name__
        if stop is not None and stop.wait(0.2):
            return {}
    if "BTCUSDT" in frames:
        tables = {s: features(frame, frames["BTCUSDT"]) for s, frame in frames.items()}
    frames.clear()  # Expanded universe: don't retain both raw and derived frames.
    history = bull_history(tables)
    expected = now_ms // HOUR * HOUR - 1
    fresh = sum(
        not table.empty
        and table.index[-1] == expected
        and bool(table.iloc[-1]["ready"])
        for table in tables.values()
    )
    healthy = fresh / max(1, len(symbols)) >= 0.8
    latest = history.loc[expected] if expected in history.index else None
    established = bool(latest is not None and latest["establishedBull"] and healthy)
    state["regime"] = {
        "state": "BULL_CONFIRMED" if established else "WAIT",
        "confirmHours": 6,
        "breadth": float(latest["breadth"])
        if latest is not None and _finite(latest["breadth"]) is not None
        else None,
        "dataHealthy": healthy,
        "sourceCloseMs": expected,
    }
    watchlist = []
    for symbol in selected:
        if (
            symbol in {"BTCUSDT", "ETHUSDT"}
            or symbol not in tables
            or expected not in tables[symbol].index
        ):
            continue
        row = candidate(
            tables[symbol].loc[expected], symbol, contracts[symbol], settings
        )
        if row is not None:
            row.update(
                spreadBps=snapshots[symbol].spread_bps,
                fundingBps=snapshots[symbol].funding_rate_bps,
            )
            if not established:
                row["stage"] = "BOĞA TEYİDİ BEKLİYOR"
            watchlist.append(row)
    watchlist.sort(key=lambda row: (-row["score"], row["symbol"]))
    state.update(
        watchlist=watchlist,
        scannedAtMs=now_ms,
        universeCount=len(selected),
        screenedCount=len(contracts),
        qualityExcludedCount=len(contracts)
        - len(
            select_universe(
                contracts,
                snapshots,
                replace(settings, long_scout_universe_limit=0),
                now_ms,
            )
        ),
        freshCount=fresh,
    )
    recorded_ms = int(datetime.now(UTC).timestamp() * 1000) if now is None else now_ms
    # Don't freeze new entries against hours that went stale during a backfill.
    timely = recorded_ms // HOUR * HOUR - 1 == expected
    active = sum(recorded_ms < r["entryOpenMs"] + 7 * DAY for r in state["records"])
    profile_id, options = profile(settings)
    priority = sorted(
        watchlist, key=lambda item: (item["stage"] != "TEYİT", -item["score"])
    )
    for row in priority[:5]:
        if (
            not established
            or not timely
            or row["stage"] not in {"TEYİT", "İZLE"}
            or active >= settings.long_scout_maximum_active
        ):
            continue
        previous = [
            r
            for r in state["records"]
            if r["symbol"] == row["symbol"]
            and r["strategy"] == row["strategy"]
            and r.get("profileId") == profile_id
        ]
        if previous and recorded_ms - max(r["recordedAtMs"] for r in previous) < DAY:
            continue
        snapshot = snapshots[row["symbol"]]
        record = {
            **row,
            "recordedAtMs": recorded_ms,
            "entryOpenMs": (recorded_ms // STEP + 1) * STEP,
            "costBps": 2 * settings.scalp_taker_fee_bps
            + snapshot.spread_bps
            + 2 * settings.scalp_slippage_bps_per_side,
            "profileId": profile_id,
            "profile": options,
            "outcome": {},
        }
        record["id"] = sha256(
            f"{VERSION}|{row['symbol']}|{row['sourceCloseMs']}".encode()
        ).hexdigest()
        state["records"].append(record)
        active += 1
    if len(state["records"]) > 2000:
        state["records"] = state["records"][-2000:]
        state["recordLimitReached"] = True
    return _publish(settings, state, recorded_ms)


def load_long_scout_summary(settings: Settings) -> dict:
    # Exporter reads this allowlisted, world-readable market-data summary only.
    summary = _read(root(settings) / "public-summary.json")
    if summary.get("version") != VERSION or summary.get("mode") != "shadow":
        return {
            "version": VERSION,
            "mode": "shadow",
            "enabled": settings.long_scout_enabled,
            "watchlist": [],
            "tracked": [],
        }
    fields = (
        "version",
        "mode",
        "autoPromotion",
        "enabled",
        "ownerAlertsEnabled",
        "evaluatedAtMs",
        "scannedAtMs",
        "regime",
        "universeCount",
        "screenedCount",
        "qualityExcludedCount",
        "freshCount",
        "errorCount",
        "recordLimitReached",
        "profileId",
        "watchlist",
        "performance",
        "byStrategy",
        "tracked",
    )
    return {
        **{key: summary[key] for key in fields if key in summary},
        "enabled": settings.long_scout_enabled,
    }


def format_long_scout(settings: Settings, *, now: datetime | None = None) -> str:
    if not settings.long_scout_enabled:
        return "🚀 LONG Radar kapalı."
    summary = load_long_scout_summary(settings)
    stamp = summary.get("scannedAtMs")
    if not stamp:
        return (
            "🚀 LONG Radar hazırlanıyor. Saatlik veri ve yeni listelemeler taranacak."
        )
    now_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    stale = not 0 <= now_ms - stamp < 2 * HOUR
    regime = summary.get("regime", {})
    label = (
        "YERLEŞİK BOĞA"
        if regime.get("state") == "BULL_CONFIRMED"
        else "BOĞA TEYİDİ BEKLENİYOR"
    )
    lines = [
        "🚀 TRADE3 • LONG Radar",
        f"{label} • {'VERİ ESKİ' if stale else local_text(stamp, with_seconds=False)}",
        f"{summary.get('universeCount', 0)} piyasa • saatlik tarama • 7 gün takip",
        f"Ön tarama {summary.get('screenedCount', summary.get('universeCount', 0))} • kalite nedeniyle elenen {summary.get('qualityExcludedCount', 0)}",
        "Özel deneysel bildirim: "
        + (
            "yalnız sahip için açık"
            if summary.get("ownerAlertsEnabled", settings.long_scout_alerts_enabled)
            else "kapalı"
        ),
    ]
    for row in summary.get("watchlist", [])[:5]:
        lines.extend(
            [
                f"\n{row['symbol']} · {LABELS.get(row['strategy'], row['strategy'])} · {row['stage']}",
                f"Güç {row['score']:.0f}/100 · BTC'ye göre 7g {row['relativeStrength7dPct']:+.1f}% · hacim {row['volume4hRatio']:.1f}×",
                f"Saatlik referans ${row['referencePrice']:.8g}"
                + (
                    f" · yeni kontrat ({row['contractAgeDays']:.0f}g)"
                    if row["newListing"]
                    else ""
                ),
            ]
        )
    if not summary.get("watchlist"):
        lines.append("\nŞu anda sabit kuralları karşılayan aday yok.")
    minute = _read(root(settings) / "minute-shadow" / "public-summary.json")
    if settings.long_scout_minute_shadow_enabled:
        lines.append(
            f"1m giriş deneyi: sessiz • {minute.get('pairedN', 0)} tamamlanmış 1m/5m karşılaştırması (canlı 1m bildirimi yok)"
        )
    performance = summary.get("performance", {})
    for record in summary.get("tracked", [])[:3]:
        outcome = record.get("outcome", {})
        touched = [
            str(k)
            for k in TARGETS
            if outcome.get("targets", {}).get(str(k), {}).get("hit") is True
        ]
        lines.append(
            f"\nTakip: {record['symbol']} · {LABELS.get(record['strategy'], record['strategy'])} · "
            + ("dokunan %" + "/".join(touched) if touched else "henüz hedef yok")
            + (" · 7g tamamlandı" if outcome.get("mature") else " · açık")
        )
    lines.append(
        f"\nSessiz takip: {performance.get('pending', 0)} açık · {performance.get('missing', 0)} veri eksik"
    )
    for level in (2, 3, 5):
        metric = performance.get("targets", {}).get(str(level), {})
        n = metric.get("n", 0)
        lines.append(
            f"%{level}: {metric.get('hits', 0)}/{n}"
            + (f" · %{100 * metric['rate']:.0f}" if n else " · örnek bekleniyor")
        )
    lines.append(
        "\nSkor başarı olasılığı değildir. LONG araştırması; ilk giriş sonraki tam 5m açılışında simüle edilir. Kontrat yaşı proje yaşı değildir."
    )
    lines.append(
        "LA1 birikim: sıkışma + göreli güç. LB1 kırılım: 7g zirve üstünde hacimli kapanış. LP1 geri test: yükselen trendde EMA20 geri alımı. Boğa teyidi: BTC/ETH trendi ve ≥%60 genişlik, 6 saat."
    )
    return "\n".join(lines)


class LongScoutWorker:
    """One background job; market I/O never blocks the candle/Telegram loop."""

    def __init__(self, settings: Settings, progress: Callable[[str], None]) -> None:
        self.settings, self.progress = settings, progress
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="long-scout"
        )
        self.future: Future | None = None
        self.last_discovery_hour = -1
        self.last_settle_bar = -1
        self.last_minute = -1
        self.cancel = Event()
        self.started_at_ms = int(datetime.now(UTC).timestamp() * 1000)

    def _run(self, *, discover: bool, settle: bool = True) -> dict:
        summary = (
            run_long_scout(self.settings, discover=discover, stop=self.cancel)
            if settle
            else load_long_scout_summary(self.settings)
        )
        if self.settings.long_scout_alerts_enabled and not self.cancel.is_set():
            from .long_notifications import deliver_long_notifications
            from .telegram import TelegramError

            try:
                for event, status in deliver_long_notifications(
                    self.settings, started_at_ms=self.started_at_ms
                ):
                    self.progress(f"LONG özel bildirim: {event}: {status}")
            except (TelegramError, OSError, ValueError) as error:
                self.progress(f"LONG özel bildirim hatası: {type(error).__name__}")
        if self.settings.long_scout_minute_shadow_enabled and not self.cancel.is_set():
            from .minute_shadow import run_minute_shadow

            try:
                run_minute_shadow(self.settings, summary, stop=self.cancel)
            except (OSError, RuntimeError, ValueError) as error:
                self.progress(f"LONG 1m sessiz deney hatası: {type(error).__name__}")
        return summary

    def tick(self, now: datetime) -> None:
        if self.future is not None:
            if not self.future.done():
                return
            try:
                summary = self.future.result()
                self.progress(
                    f"LONG Radar: {summary.get('universeCount', 0)} piyasa, {len(summary.get('watchlist', []))} aday, {summary.get('regime', {}).get('state', 'WAIT')}; sessiz takip"
                )
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
                self.progress(f"LONG Radar hatası: {type(error).__name__}: {error}")
                self.last_discovery_hour = -1  # Retry on the next five-minute boundary.
            self.future = None
        stamp = int(now.timestamp() * 1000)
        bar, hour, minute = stamp // STEP, stamp // HOUR, stamp // 60_000
        settle = bar != self.last_settle_bar
        if not settle and (
            not self.settings.long_scout_minute_shadow_enabled
            or minute == self.last_minute
        ):
            return
        discover = hour != self.last_discovery_hour
        self.future = self.executor.submit(self._run, discover=discover, settle=settle)
        self.last_minute = minute
        self.last_settle_bar = bar
        if discover:
            self.last_discovery_hour = hour

    def stop(self) -> None:
        self.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
