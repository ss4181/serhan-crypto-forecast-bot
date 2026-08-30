"""Research-only broad-universe 5m scalp observer.

The detector intentionally mirrors the three preregistered F1/F2/F3 families
that failed trade1's historical production gate.  Seeing one of these setups is
therefore an observation, never a trade candidate.  Every observation is
recorded and scored at fixed 15/30/60 minute time exits so future evidence can
be evaluated without changing the question after seeing the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .config import Settings, local_text, validate_market_symbol
from .data import BinanceMarketDataClient, MarketDataError, load_cache, update_market_cache
from .telegram import TelegramDelivery, TelegramNotifier, digest_signal_id
from .universe import UniverseEntry, UniverseManifest, load_trade1_universe


SCALP_INTERVAL = "5m"
SCALP_STEP_MS = 5 * 60 * 1000
SCALP_MINIMUM_BARS = 290
SCALP_PENDING_SCHEMA = "scalp-observation-pending-v1"
SCALP_LEDGER_SCHEMA = "scalp-observation-outcome-v1"
SCALP_SETTLEMENT_GRACE_DAYS = 2
FAMILY_LABELS = {
    "F1": "hacim momentumu",
    "F2": "kaskad tepki",
    "F3": "kirilim devami",
}
FAMILY_EVIDENCE = {
    "F1": "30-coin tarihsel testte maliyet sonrasi kapiyi gecemedi",
    "F2": "30-coin tarihsel testte net ortalama negatifti",
    "F3": "30-coin tarihsel testte net medyan negatifti",
}


@dataclass(frozen=True, slots=True)
class ScalpObservation:
    universe_version: str
    spot_symbol: str
    perpetual_symbol: str
    universe_group: str
    family: str
    score: float
    price: float
    bar_open_time_ms: int
    bar_close_time_ms: int
    details: tuple[str, ...]

    @property
    def signal_id(self) -> str:
        material = (
            f"{self.universe_version}|{self.perpetual_symbol}|{self.family}|"
            f"{self.bar_close_time_ms}"
        )
        return sha256(material.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScalpScanReport:
    universe_version: str
    attempted: int
    fresh: int
    stale: int
    errors: tuple[str, ...]
    observations: tuple[ScalpObservation, ...]
    evaluated_at_ms: int

    @property
    def coverage(self) -> float:
        return self.fresh / self.attempted if self.attempted else 0.0

    def top(self, limit: int) -> tuple[ScalpObservation, ...]:
        if limit < 1:
            raise ValueError("Scalp top-K en az 1 olmali")
        return tuple(
            sorted(
                self.observations,
                key=lambda item: (-item.score, item.spot_symbol, item.family),
            )[:limit]
        )


def scalp_cache_path(data_dir: Path, perpetual_symbol: str) -> Path:
    symbol = validate_market_symbol(perpetual_symbol)
    return data_dir / f"{symbol}_{SCALP_INTERVAL}_futures.csv"


def scan_scalp_frame(
    entry: UniverseEntry,
    frame: pd.DataFrame,
    *,
    universe_version: str,
) -> tuple[ScalpObservation, ...]:
    """Evaluate only the latest closed 5m bar, with no future values."""
    if len(frame) < SCALP_MINIMUM_BARS:
        return ()
    close = frame["close"].astype("float64")
    open_ = frame["open"].astype("float64")
    high = frame["high"].astype("float64")
    volume = frame["volume"].astype("float64")
    log_return = np.log(close).diff()
    sigma5 = log_return.rolling(288, min_periods=144).std()
    log_volume = np.log1p(volume)
    volume_mean = log_volume.rolling(288, min_periods=144).mean()
    volume_std = log_volume.rolling(288, min_periods=144).std()
    volume_z = (log_volume - volume_mean) / volume_std.replace(0.0, np.nan)
    return_30m = np.log(close / close.shift(6))
    high_12h = high.rolling(144, min_periods=144).max().shift(1)

    latest = len(frame) - 1
    price = float(close.iloc[latest])
    bar_open_ms = int(frame["open_time_ms"].iloc[latest])
    bar_close_ms = int(frame["close_time_ms"].iloc[latest])
    observations: list[ScalpObservation] = []

    z_now = float(volume_z.iloc[latest])
    up_bar = price > float(open_.iloc[latest])
    f1_condition = volume_z.ge(3.0).fillna(False)
    if _edge_is_new(f1_condition, latest) and up_bar:
        observations.append(
            _observation(
                entry,
                universe_version,
                family="F1",
                score=z_now / 3.0,
                price=price,
                bar_open_ms=bar_open_ms,
                bar_close_ms=bar_close_ms,
                details=(f"log-hacim z={z_now:+.2f}", "yukari kapanan bar"),
            )
        )

    ret_now = float(return_30m.iloc[latest])
    sigma30_now = float(sigma5.iloc[latest]) * math.sqrt(6.0)
    sigma30 = sigma5 * math.sqrt(6.0)
    f2_condition = (
        return_30m.le(-3.0 * sigma30)
        & sigma30.gt(0.0)
        & volume_z.ge(2.0)
    ).fillna(False)
    if (
        _edge_is_new(f2_condition, latest)
        and math.isfinite(ret_now)
        and math.isfinite(sigma30_now)
        and sigma30_now > 0
    ):
        shock_ratio = -ret_now / sigma30_now
        observations.append(
            _observation(
                entry,
                universe_version,
                family="F2",
                score=min(shock_ratio / 3.0, z_now / 2.0),
                price=price,
                bar_open_ms=bar_open_ms,
                bar_close_ms=bar_close_ms,
                details=(
                    f"30dk getiri %{ret_now * 100:+.2f} ({shock_ratio:.1f} sigma)",
                    f"log-hacim z={z_now:+.2f}",
                ),
            )
        )

    high_now = float(high_12h.iloc[latest])
    f3_condition = (close.gt(high_12h) & volume_z.ge(2.0)).fillna(False)
    if _edge_is_new(f3_condition, latest) and math.isfinite(high_now):
        breakout_bps = math.log(price / high_now) * 10_000.0
        volatility_bps = max(float(sigma5.iloc[latest]) * 10_000.0, 1e-9)
        observations.append(
            _observation(
                entry,
                universe_version,
                family="F3",
                score=z_now / 2.0 + min(breakout_bps / volatility_bps, 1.0),
                price=price,
                bar_open_ms=bar_open_ms,
                bar_close_ms=bar_close_ms,
                details=(
                    f"12s zirvesinin {breakout_bps:+.1f} bps ustu",
                    f"log-hacim z={z_now:+.2f}",
                ),
            )
        )
    return tuple(observations)


def refresh_and_scan_scalp_universe(
    settings: Settings,
    *,
    manifest: UniverseManifest | None = None,
    entries: tuple[UniverseEntry, ...] | None = None,
    client: BinanceMarketDataClient | None = None,
    now: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> ScalpScanReport:
    manifest = manifest or load_trade1_universe()
    selected = entries if entries is not None else manifest.selected_entries()
    market_client = client or BinanceMarketDataClient(market_name="futures")
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    output = progress or (lambda _message: None)
    for entry in selected:
        try:
            frames[entry.perpetual_symbol] = update_market_cache(
                scalp_cache_path(settings.scalp_data_dir, entry.perpetual_symbol),
                entry.perpetual_symbol,
                SCALP_INTERVAL,
                days=settings.scalp_cache_days,
                client=market_client,
                now=now,
                warn=output,
            )
        except (MarketDataError, OSError, TypeError, ValueError) as error:
            errors.append(f"{entry.spot_symbol}: {error}")
    return _scan_frames(settings, manifest, selected, frames, errors=errors, now=now)


def scan_cached_scalp_universe(
    settings: Settings,
    *,
    manifest: UniverseManifest | None = None,
    entries: tuple[UniverseEntry, ...] | None = None,
    now: datetime | None = None,
) -> ScalpScanReport:
    manifest = manifest or load_trade1_universe()
    selected = entries if entries is not None else manifest.selected_entries()
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for entry in selected:
        try:
            frames[entry.perpetual_symbol] = load_cache(
                scalp_cache_path(settings.scalp_data_dir, entry.perpetual_symbol)
            )
        except (MarketDataError, OSError) as error:
            errors.append(f"{entry.spot_symbol}: {error}")
    return _scan_frames(settings, manifest, selected, frames, errors=errors, now=now)


def format_scalp_observation_digest(
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    top_k: int,
) -> str:
    shown = report.top(top_k)
    if not shown:
        raise ValueError("Scalp gozlem raporu icin kurulum yok")
    stamp = local_text(report.evaluated_at_ms, with_seconds=False)
    lines = [
        f"🧪 SCALP | 5m | {stamp}",
        f"{report.fresh}/{report.attempted} taze • {len(report.observations)} kurulum "
        f"• ilk {len(shown)} • maliyet {manifest.scalp_round_trip_cost_bps:g} bps",
        "",
    ]
    for index, item in enumerate(shown, start=1):
        mapping = (
            f"→{item.perpetual_symbol} "
            if item.perpetual_symbol != item.spot_symbol
            else ""
        )
        detail = ", ".join(item.details)
        lines.append(
            f"{index}. {item.spot_symbol} {mapping}↑ {item.family} "
            f"({FAMILY_LABELS[item.family]}) • {item.score:.2f} • "
            f"${item.price:,.8g} • {detail}"
        )
    lines.extend(
        [
            "",
            "ISLEM ADAYI DEGILDIR • puan olasılık/getiri değildir • ileri test: 15/30/60 dk",
        ]
    )
    message = "\n".join(lines)
    if len(message) > 4096:
        raise ValueError("Scalp Telegram mesaji 4096 karakteri asti")
    return message


def deliver_scalp_observations(
    settings: Settings,
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    notifier: TelegramNotifier | None = None,
) -> TelegramDelivery | None:
    if report.coverage < settings.scalp_minimum_coverage:
        raise RuntimeError(
            f"Scalp evren kapsami %{report.coverage * 100:.1f}; "
            f"en az %{settings.scalp_minimum_coverage * 100:.1f} olmali"
        )
    shown = report.top(settings.scalp_top_k)
    if not shown:
        return None
    newest_close = max(item.bar_close_time_ms for item in shown)
    bucket = newest_close // SCALP_STEP_MS
    signal_id = digest_signal_id(f"scalp-observation|{manifest.version}", bucket)
    return (notifier or TelegramNotifier()).deliver_once(
        signal_id=signal_id,
        text=format_scalp_observation_digest(
            report,
            manifest=manifest,
            top_k=settings.scalp_top_k,
        ),
        state_dir=settings.telegram_state_dir / "scalp",
    )


def record_scalp_observations(
    state_dir: Path,
    observations: Iterable[ScalpObservation],
    *,
    manifest: UniverseManifest,
) -> int:
    directory = _pending_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    added = 0
    for item in observations:
        path = directory / f"{item.signal_id}.json"
        payload = {
            "schema": SCALP_PENDING_SCHEMA,
            "signal_id": item.signal_id,
            "universe_version": item.universe_version,
            "spot_symbol": item.spot_symbol,
            "perpetual_symbol": item.perpetual_symbol,
            "universe_group": item.universe_group,
            "family": item.family,
            "score": item.score,
            "source_price": item.price,
            "bar_open_time_ms": item.bar_open_time_ms,
            "bar_close_time_ms": item.bar_close_time_ms,
            "horizons_minutes": list(manifest.scalp_horizons_minutes),
            "round_trip_cost_bps": manifest.scalp_round_trip_cost_bps,
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
            added += 1
        except FileExistsError:
            continue
    return added


def settle_scalp_observations(
    state_dir: Path,
    data_dir: Path,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    directory = _pending_dir(state_dir)
    if not directory.exists():
        return []
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    grace_ms = SCALP_SETTLEMENT_GRACE_DAYS * 24 * 60 * 60 * 1000
    frames: dict[str, pd.DataFrame] = {}
    settled: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_pending(path)
        if record is None:
            path.unlink(missing_ok=True)
            continue
        horizons = tuple(int(value) for value in record["horizons_minutes"])
        deadline_ms = int(record["bar_close_time_ms"]) + max(horizons) * 60_000
        if current_ms < deadline_ms:
            continue
        symbol = str(record["perpetual_symbol"])
        if symbol not in frames:
            try:
                frames[symbol] = load_cache(scalp_cache_path(data_dir, symbol))
            except (MarketDataError, OSError, ValueError):
                frames[symbol] = pd.DataFrame()
        rows = _time_exit_outcomes(record, frames[symbol])
        if rows is None:
            if current_ms - deadline_ms > grace_ms:
                path.unlink(missing_ok=True)
            continue
        settled.extend(rows)
        path.unlink(missing_ok=True)
    if settled:
        _append_scalp_ledger(state_dir, settled)
    return settled


def load_scalp_ledger(state_dir: Path, *, limit: int = 20_000) -> list[dict[str, Any]]:
    path = _ledger_path(state_dir)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCALP_LEDGER_SCHEMA:
            rows.append(payload)
    return rows


def scalp_scorecard(
    rows: Iterable[dict[str, Any]],
    *,
    days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    if days < 1:
        raise ValueError("Scalp karne gun sayisi pozitif olmali")
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    cutoff = current_ms - days * 86_400_000
    recent = [row for row in rows if int(row.get("exit_time_ms", 0)) >= cutoff]
    keys = sorted({(str(row["family"]), int(row["horizon_minutes"])) for row in recent})
    return {
        "days": days,
        "observationCount": len({row.get("signal_id") for row in recent}),
        "outcomeCount": len(recent),
        "byFamilyHorizon": {
            f"{family}_{horizon}": _aggregate_scalp(
                [
                    row
                    for row in recent
                    if row.get("family") == family
                    and int(row.get("horizon_minutes", 0)) == horizon
                ]
            )
            for family, horizon in keys
        },
    }


def format_scalp_scorecard(card: dict[str, Any]) -> str:
    lines = [
        f"🧪 SCALP ILERI TEST KARNESI — son {card['days']} gun",
        "",
        f"Olgun gozlem: {card['observationCount']}",
    ]
    if not card["outcomeCount"]:
        lines.append("Henuz 15/30/60 dakika sonucu olusan gozlem yok.")
    else:
        for key, stats in card["byFamilyHorizon"].items():
            family, horizon = key.split("_", 1)
            lines.append(
                f"• {family} {horizon}dk: n={stats['count']}, "
                f"net ort {stats['meanNetBps']:+.2f} bps, "
                f"medyan {stats['medianNetBps']:+.2f}, kazanma %{stats['winRate'] * 100:.1f}"
            )
    lines.extend(
        [
            "",
            "Bu ileri test otomatik terfi kapisi degildir; gozlemler halen islem adayi degildir.",
        ]
    )
    return "\n".join(lines)


def _scan_frames(
    settings: Settings,
    manifest: UniverseManifest,
    entries: tuple[UniverseEntry, ...],
    frames: dict[str, pd.DataFrame],
    *,
    errors: list[str],
    now: datetime | None,
) -> ScalpScanReport:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current_ms = int(current.timestamp() * 1000)
    maximum_age_ms = settings.scalp_maximum_bar_age_minutes * 60_000
    observations: list[ScalpObservation] = []
    fresh_close_times: list[int] = []
    fresh = 0
    stale = 0
    for entry in entries:
        frame = frames.get(entry.perpetual_symbol)
        if frame is None or frame.empty:
            continue
        latest_close = int(frame["close_time_ms"].iloc[-1])
        if current_ms - latest_close > maximum_age_ms or latest_close > current_ms:
            stale += 1
            continue
        fresh += 1
        fresh_close_times.append(latest_close)
        observations.extend(
            scan_scalp_frame(entry, frame, universe_version=manifest.version)
        )
    if fresh_close_times:
        newest_close = max(fresh_close_times)
        # A lagging API/cache may still be "fresh" by the age limit.  Do not
        # repeat its previous-bar setup inside the newest bar's digest.
        observations = [
            item for item in observations if item.bar_close_time_ms == newest_close
        ]
    return ScalpScanReport(
        universe_version=manifest.version,
        attempted=len(entries),
        fresh=fresh,
        stale=stale,
        errors=tuple(errors),
        observations=tuple(observations),
        evaluated_at_ms=current_ms,
    )


def _observation(
    entry: UniverseEntry,
    universe_version: str,
    *,
    family: str,
    score: float,
    price: float,
    bar_open_ms: int,
    bar_close_ms: int,
    details: tuple[str, ...],
) -> ScalpObservation:
    return ScalpObservation(
        universe_version=universe_version,
        spot_symbol=entry.spot_symbol,
        perpetual_symbol=entry.perpetual_symbol,
        universe_group=entry.group,
        family=family,
        score=float(score),
        price=price,
        bar_open_time_ms=bar_open_ms,
        bar_close_time_ms=bar_close_ms,
        details=details,
    )


def _edge_is_new(condition: pd.Series, latest: int, *, cooldown_bars: int = 12) -> bool:
    """Match trade1's rising-edge plus one-hour greedy cooldown on 5m bars."""
    if latest < 1 or not bool(condition.iloc[latest]) or bool(condition.iloc[latest - 1]):
        return False
    rising = condition & ~condition.shift(1, fill_value=False)
    start = max(0, latest - cooldown_bars + 1)
    return not bool(rising.iloc[start:latest].any())


def _time_exit_outcomes(
    record: dict[str, Any], frame: pd.DataFrame
) -> list[dict[str, Any]] | None:
    if frame.empty:
        return None
    matches = frame.index[frame["open_time_ms"] == int(record["bar_open_time_ms"])].tolist()
    if not matches:
        return None
    event_index = int(matches[-1])
    horizons = tuple(int(value) for value in record["horizons_minutes"])
    required_index = event_index + max(horizons) // 5
    if event_index + 1 >= len(frame) or required_index >= len(frame):
        return None
    entry_price = float(frame.iloc[event_index + 1]["open"])
    cost = float(record["round_trip_cost_bps"])
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        exit_row = frame.iloc[event_index + horizon // 5]
        exit_price = float(exit_row["close"])
        gross_bps = math.log(exit_price / entry_price) * 10_000.0
        results.append(
            {
                "schema": SCALP_LEDGER_SCHEMA,
                "signal_id": record["signal_id"],
                "universe_version": record["universe_version"],
                "spot_symbol": record["spot_symbol"],
                "perpetual_symbol": record["perpetual_symbol"],
                "universe_group": record["universe_group"],
                "family": record["family"],
                "score": float(record["score"]),
                "bar_close_time_ms": int(record["bar_close_time_ms"]),
                "entry_price": entry_price,
                "horizon_minutes": horizon,
                "exit_price": exit_price,
                "exit_time_ms": int(exit_row["close_time_ms"]),
                "gross_bps": gross_bps,
                "net_bps": gross_bps - cost,
                "round_trip_cost_bps": cost,
            }
        )
    return results


def _read_pending(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    required = (
        "signal_id",
        "universe_version",
        "spot_symbol",
        "perpetual_symbol",
        "universe_group",
        "family",
        "score",
        "bar_open_time_ms",
        "bar_close_time_ms",
        "horizons_minutes",
        "round_trip_cost_bps",
    )
    if not isinstance(payload, dict) or payload.get("schema") != SCALP_PENDING_SCHEMA:
        return None
    if any(key not in payload for key in required):
        return None
    try:
        validate_market_symbol(str(payload["perpetual_symbol"]))
        int(payload["bar_open_time_ms"])
        int(payload["bar_close_time_ms"])
        float(payload["score"])
        float(payload["round_trip_cost_bps"])
        horizons = tuple(int(value) for value in payload["horizons_minutes"])
    except (TypeError, ValueError):
        return None
    if horizons != (15, 30, 60) or payload.get("family") not in FAMILY_LABELS:
        return None
    return payload


def _pending_dir(state_dir: Path) -> Path:
    return state_dir / "pending"


def _ledger_path(state_dir: Path) -> Path:
    return state_dir / "ledger.jsonl"


def _append_scalp_ledger(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _ledger_path(state_dir).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _aggregate_scalp(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["net_bps"]) for row in rows]
    if not values:
        return {"count": 0, "meanNetBps": 0.0, "medianNetBps": 0.0, "winRate": 0.0}
    return {
        "count": len(values),
        "meanNetBps": sum(values) / len(values),
        "medianNetBps": median(values),
        "winRate": sum(value > 0 for value in values) / len(values),
    }


__all__ = [
    "FAMILY_EVIDENCE",
    "FAMILY_LABELS",
    "SCALP_INTERVAL",
    "ScalpObservation",
    "ScalpScanReport",
    "deliver_scalp_observations",
    "format_scalp_observation_digest",
    "format_scalp_scorecard",
    "load_scalp_ledger",
    "record_scalp_observations",
    "refresh_and_scan_scalp_universe",
    "scalp_cache_path",
    "scalp_scorecard",
    "scan_cached_scalp_universe",
    "scan_scalp_frame",
    "settle_scalp_observations",
]
