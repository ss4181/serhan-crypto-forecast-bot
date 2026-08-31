"""Research-only broad-universe 5m scalp observer for trade3.

The original detector mirrors the preregistered F1/F2/F3 evidence imported as
a read-only snapshot from trade1.  B1/B2/B3 are new trade3 hypotheses and only
run under the causal market-wide bull label.  Every event remains an
observation, never a trade candidate, and is scored at fixed 15/30/60 minute
time exits without changing the question after seeing the answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings, local_text, validate_market_symbol
from .data import (
    BinanceMarketDataClient,
    FuturesMarketSnapshot,
    MarketDataError,
    load_cache,
    update_market_cache,
)
from .telegram import TelegramDelivery, TelegramNotifier, digest_signal_id
from .universe import UniverseEntry, UniverseManifest, load_trade1_universe

SCALP_INTERVAL = "5m"
SCALP_STEP_MS = 5 * 60 * 1000
SCALP_MINIMUM_BARS = 290
SCALP_PENDING_SCHEMA = "scalp-observation-pending-v1"
SCALP_LEDGER_SCHEMA = "scalp-observation-outcome-v1"
SCALP_SETTLEMENT_GRACE_DAYS = 2
SCALP_BACKTEST_HORIZONS = (15, 30, 60)
FAMILY_LABELS = {
    "F1": "hacim momentumu",
    "F2": "kaskad tepki",
    "F3": "kirilim devami",
    "B1": "boga 24s kirilimi",
    "B2": "boga geri-cekilme donusu",
    "B3": "goreli guc gecisi",
}
FAMILY_EVIDENCE = {
    "F1": "30-coin tarihsel testte maliyet sonrasi kapiyi gecemedi",
    "F2": "30-coin tarihsel testte net ortalama negatifti",
    "F3": "30-coin tarihsel testte net medyan negatifti",
    "B1": "yalniz boga rejiminde ileri test edilen yeni hipotez",
    "B2": "yalniz boga rejiminde ileri test edilen yeni hipotez",
    "B3": "yalniz boga rejiminde ileri test edilen yeni hipotez",
}


@dataclass(frozen=True, slots=True)
class BullRegime:
    state: str
    score: float
    breadth: float
    trend_fraction: float
    persistent_up: bool
    eligible_markets: int

    @property
    def label(self) -> str:
        return {"BULL": "BOGA", "TRANSITION": "GECIS", "OFF": "KAPALI"}.get(
            self.state, "BILINMIYOR"
        )


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
    regime_state: str = "UNKNOWN"
    regime_score: float = 0.0
    breadth: float = 0.0
    spread_bps: float | None = None
    funding_rate_bps: float | None = None
    estimated_cost_bps: float = 12.0
    execution_eligible: bool = False
    alert_tier: str = "RADAR"

    @property
    def signal_id(self) -> str:
        material = f"{self.universe_version}|{self.perpetual_symbol}|{self.family}|{self.bar_close_time_ms}"
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
    regime: BullRegime | None = None
    quoted: int = 0

    @property
    def coverage(self) -> float:
        return self.fresh / self.attempted if self.attempted else 0.0

    def top(self, limit: int) -> tuple[ScalpObservation, ...]:
        if limit < 1:
            raise ValueError("Scalp top-K en az 1 olmali")
        return tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.alert_tier != "KURULUM",
                    not item.execution_eligible,
                    -item.score,
                    item.spot_symbol,
                    item.family,
                ),
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
    regime: BullRegime | None = None,
    snapshot: FuturesMarketSnapshot | None = None,
    settings: Settings | None = None,
    historical_cost_bps: float = 12.0,
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
    high_24h = high.rolling(288, min_periods=288).max().shift(1)
    ema_48h = close.ewm(span=576, min_periods=288, adjust=False).mean()

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
                regime=regime,
                snapshot=snapshot,
                settings=settings,
                historical_cost_bps=historical_cost_bps,
            )
        )

    ret_now = float(return_30m.iloc[latest])
    sigma30_now = float(sigma5.iloc[latest]) * math.sqrt(6.0)
    sigma30 = sigma5 * math.sqrt(6.0)
    f2_condition = (
        return_30m.le(-3.0 * sigma30) & sigma30.gt(0.0) & volume_z.ge(2.0)
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
                regime=regime,
                snapshot=snapshot,
                settings=settings,
                historical_cost_bps=historical_cost_bps,
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
                regime=regime,
                snapshot=snapshot,
                settings=settings,
                historical_cost_bps=historical_cost_bps,
            )
        )

    if regime is not None and regime.state == "BULL":
        high_24h_now = float(high_24h.iloc[latest])
        b1_condition = (close.gt(high_24h) & volume_z.ge(1.0)).fillna(False)
        if _edge_is_new(b1_condition, latest) and math.isfinite(high_24h_now):
            breakout_bps = math.log(price / high_24h_now) * 10_000.0
            observations.append(
                _observation(
                    entry,
                    universe_version,
                    family="B1",
                    score=1.0 + max(z_now, 0.0) / 3.0 + min(breakout_bps / 25.0, 1.0),
                    price=price,
                    bar_open_ms=bar_open_ms,
                    bar_close_ms=bar_close_ms,
                    details=(
                        f"24s zirvesinin {breakout_bps:+.1f} bps ustu",
                        f"hacim z={z_now:+.2f}",
                    ),
                    regime=regime,
                    snapshot=snapshot,
                    settings=settings,
                    historical_cost_bps=historical_cost_bps,
                )
            )

        previous_pullback = (
            return_30m.shift(1).le(-0.75 * sigma30.shift(1))
            & return_30m.shift(1).ge(-6.0 * sigma30.shift(1))
            & sigma30.shift(1).gt(0.0)
        )
        b2_condition = (
            previous_pullback
            & close.gt(open_)
            & close.gt(high.shift(1))
            & close.gt(ema_48h)
            & volume_z.ge(0.5)
        ).fillna(False)
        if _edge_is_new(b2_condition, latest):
            recovery_bps = math.log(price / float(close.iloc[latest - 1])) * 10_000.0
            observations.append(
                _observation(
                    entry,
                    universe_version,
                    family="B2",
                    score=1.0
                    + min(max(recovery_bps, 0.0) / 25.0, 1.0)
                    + max(z_now, 0.0) / 4.0,
                    price=price,
                    bar_open_ms=bar_open_ms,
                    bar_close_ms=bar_close_ms,
                    details=(
                        f"geri donus {recovery_bps:+.1f} bps",
                        "48s trend ustunde",
                    ),
                    regime=regime,
                    snapshot=snapshot,
                    settings=settings,
                    historical_cost_bps=historical_cost_bps,
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
    snapshots: dict[str, FuturesMarketSnapshot] = {}
    try:
        snapshots = market_client.fetch_futures_market_snapshots()
    except (MarketDataError, OSError, TypeError, ValueError) as error:
        errors.append(f"anlik kotasyon: {error}")
    return _scan_frames(
        settings,
        manifest,
        selected,
        frames,
        errors=errors,
        now=now,
        snapshots=snapshots,
    )


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
    return _scan_frames(
        settings,
        manifest,
        selected,
        frames,
        errors=errors,
        now=now,
        snapshots={},
    )


def evaluate_bull_regime(
    frames: dict[str, pd.DataFrame],
    major_frames: dict[str, pd.DataFrame],
    *,
    breadth_threshold: float = 0.60,
) -> BullRegime:
    """Causal market-wide bull label for research stratification.

    It combines a 48-hour broad-universe breadth reading with persistent
    four-week BTC/ETH direction and a slow 50-day trend.  It never suppresses
    the original F1/F2/F3 observations; only new B-family hypotheses depend on
    it.
    """
    breadth_votes: list[bool] = []
    for frame in frames.values():
        if len(frame) < 288:
            continue
        close = frame["close"].astype("float64")
        ema = close.ewm(span=576, min_periods=288, adjust=False).mean()
        if math.isfinite(float(ema.iloc[-1])):
            breadth_votes.append(float(close.iloc[-1]) > float(ema.iloc[-1]))
    breadth = sum(breadth_votes) / len(breadth_votes) if breadth_votes else 0.0

    trend_votes: list[bool] = []
    current_4w: list[float] = []
    prior_4w: list[float] = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frame = major_frames.get(symbol)
        if frame is None or len(frame) < 1369:
            continue
        close = frame["close"].astype("float64")
        ema50d = close.ewm(span=1200, min_periods=1200, adjust=False).mean()
        latest_ema = float(ema50d.iloc[-1])
        week_ago_ema = float(ema50d.iloc[-169])
        if not (math.isfinite(latest_ema) and math.isfinite(week_ago_ema)):
            continue
        trend_votes.extend(
            (
                float(close.iloc[-1]) > latest_ema,
                latest_ema > week_ago_ema,
            )
        )
        current_4w.append(math.log(float(close.iloc[-1]) / float(close.iloc[-673])))
        prior_4w.append(math.log(float(close.iloc[-169]) / float(close.iloc[-841])))

    if len(trend_votes) < 4 or not breadth_votes:
        return BullRegime("UNKNOWN", 0.0, breadth, 0.0, False, len(breadth_votes))
    trend_fraction = sum(trend_votes) / len(trend_votes)
    persistent_up = (
        sum(current_4w) / len(current_4w) > 0.0 and sum(prior_4w) / len(prior_4w) > 0.0
    )
    breadth_component = min(breadth / breadth_threshold, 1.0)
    score = (
        0.40 * float(persistent_up) + 0.30 * trend_fraction + 0.30 * breadth_component
    )
    if persistent_up and trend_fraction >= 0.75 and breadth >= breadth_threshold:
        state = "BULL"
    elif score >= 0.55:
        state = "TRANSITION"
    else:
        state = "OFF"
    return BullRegime(
        state,
        float(score),
        float(breadth),
        float(trend_fraction),
        persistent_up,
        len(breadth_votes),
    )


def format_scalp_observation_digest(
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    top_k: int,
    ledger: Iterable[dict[str, Any]] = (),
) -> str:
    shown = report.top(top_k)
    if not shown:
        raise ValueError("Scalp gozlem raporu icin kurulum yok")
    ledger_rows = tuple(ledger)
    stamp = local_text(report.evaluated_at_ms, with_seconds=False)
    regime = report.regime or BullRegime("UNKNOWN", 0.0, 0.0, 0.0, False, 0)
    lines = [
        f"🧪 SCALP | 5m | {stamp}",
        (
            f"Rejim {regime.label} {regime.score:.2f} • "
            f"genislik %{regime.breadth * 100:.0f} • "
            f"kotasyon {report.quoted}/{report.attempted}"
        ),
        (
            f"{report.fresh}/{report.attempted} taze • {len(report.observations)} radar • ilk {len(shown)}"
        ),
        "",
    ]
    grouped: dict[str, list[ScalpObservation]] = {}
    for item in shown:
        grouped.setdefault(item.spot_symbol, []).append(item)
    for index, items in enumerate(grouped.values(), start=1):
        item = items[0]
        mapping = (
            f"→{item.perpetual_symbol} "
            if item.perpetual_symbol != item.spot_symbol
            else ""
        )
        families = "+".join(value.family for value in items)
        tier = (
            "KURULUM"
            if any(value.alert_tier == "KURULUM" for value in items)
            else "RADAR"
        )
        spread = f"sp {item.spread_bps:.1f}" if item.spread_bps is not None else "sp ?"
        cost = item.estimated_cost_bps or manifest.scalp_round_trip_cost_bps
        detail = ", ".join(item.details)
        lines.append(
            f"{index}. {tier} {item.spot_symbol} {mapping}{families} • "
            f"skor {item.score:.2f} • maliyet~{cost:.1f}bps/{spread} • {detail}"
        )
        for family_item in items:
            lines.append(
                "    "
                + _format_scalp_backtest(
                    family_item,
                    ledger_rows,
                )
            )
    lines.extend(
        [
            "",
            "ISLEM ADAYI DEGILDIR • RADAR tek aile, KURULUM coklu teyit • ileri test 15/30/60dk",
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
    ledger = load_scalp_ledger(settings.scalp_state_dir)
    newest_close = max(item.bar_close_time_ms for item in shown)
    bucket = newest_close // SCALP_STEP_MS
    signal_id = digest_signal_id(f"scalp-observation|{manifest.version}", bucket)
    return (notifier or TelegramNotifier()).deliver_once(
        signal_id=signal_id,
        text=format_scalp_observation_digest(
            report,
            manifest=manifest,
            top_k=settings.scalp_top_k,
            ledger=ledger,
        ),
        state_dir=settings.telegram_state_dir / "scalp",
    )


def scalp_forecast_stats(
    item: ScalpObservation,
    rows: Iterable[dict[str, Any]],
) -> dict[int, tuple[int, float, float, float]]:
    """Return empirical up probability and price movement by exit horizon.

    The ledger contains only settled observations, so this is a forward-test
    estimate rather than a model prediction.  Prefer the same symbol and
    regime when at least ten observations exist; otherwise use the broader
    family/regime sample and expose the sample count in the alert.

    Each value is ``(count, probability_up, median_gross_bps, median_net_bps)``.
    Missing or malformed rows are ignored rather than taking down Telegram.
    """
    all_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("family", "")) == item.family
        and _scalp_row_has_numbers(row)
    ]
    if not all_rows:
        return {}

    same_symbol_regime = [
        row
        for row in all_rows
        if str(row.get("perpetual_symbol", "")) == item.perpetual_symbol
        and str(row.get("regime_state", "UNKNOWN")) == item.regime_state
    ]
    same_regime = [
        row
        for row in all_rows
        if str(row.get("regime_state", "UNKNOWN")) == item.regime_state
    ]
    if len(same_symbol_regime) >= 10:
        candidates = same_symbol_regime
    elif len(same_regime) >= 10:
        candidates = same_regime
    else:
        candidates = all_rows

    result: dict[int, tuple[int, float, float, float]] = {}
    for horizon in SCALP_BACKTEST_HORIZONS:
        selected = [
            row
            for row in candidates
            if int(row.get("horizon_minutes", 0)) == horizon
        ]
        if not selected:
            continue
        gross = [float(row["gross_bps"]) for row in selected]
        net: list[float] = []
        for row, gross_value in zip(selected, gross):
            try:
                net_value = float(row.get("net_bps"))
            except (TypeError, ValueError):
                try:
                    net_value = gross_value - float(
                        row.get("round_trip_cost_bps", 0.0)
                    )
                except (TypeError, ValueError):
                    net_value = gross_value
            if not math.isfinite(net_value):
                net_value = gross_value
            net.append(net_value)
        result[horizon] = (
            len(gross),
            sum(value > 0.0 for value in gross) / len(gross),
            float(median(gross)),
            float(median(net)),
        )
    return result


def _scalp_row_has_numbers(row: dict[str, Any]) -> bool:
    try:
        gross = float(row["gross_bps"])
        horizon = int(row["horizon_minutes"])
    except (KeyError, TypeError, ValueError):
        return False
    return horizon in SCALP_BACKTEST_HORIZONS and math.isfinite(gross)


def _format_scalp_backtest(
    item: ScalpObservation,
    rows: Iterable[dict[str, Any]],
) -> str:
    stats = scalp_forecast_stats(item, rows)
    if not stats:
        return f"{item.family} BT: henuz yerlesmis ileri-test sonucu yok"
    up = "/".join(
        f"%{stats[horizon][1] * 100:.0f}" if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    down = "/".join(
        f"%{(1.0 - stats[horizon][1]) * 100:.0f}" if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    gross = "/".join(
        f"{stats[horizon][2]:+.1f}" if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    gross_pct = "/".join(
        f"{stats[horizon][2] / 100.0:+.2f}%" if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    net = "/".join(
        f"{stats[horizon][3]:+.1f}" if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    counts = "/".join(
        str(stats[horizon][0]) if horizon in stats else "-"
        for horizon in SCALP_BACKTEST_HORIZONS
    )
    return (
        f"{item.family} BT 15/30/60dk: yukari {up} | asagi {down} • "
        f"med hareket {gross}bps ({gross_pct}) • net {net}bps • n={counts}"
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
            "round_trip_cost_bps": max(
                manifest.scalp_round_trip_cost_bps, item.estimated_cost_bps
            ),
            "regime_state": item.regime_state,
            "regime_score": item.regime_score,
            "breadth": item.breadth,
            "spread_bps": item.spread_bps,
            "funding_rate_bps": item.funding_rate_bps,
            "execution_eligible": item.execution_eligible,
            "alert_tier": item.alert_tier,
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        payload, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
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
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
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
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
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
        "byRegime": {
            regime: _aggregate_scalp(
                [
                    row
                    for row in recent
                    if str(row.get("regime_state", "UNKNOWN")) == regime
                ]
            )
            for regime in sorted(
                {str(row.get("regime_state", "UNKNOWN")) for row in recent}
            )
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
        for regime, stats in card.get("byRegime", {}).items():
            lines.append(
                f"• Rejim {regime}: n={stats['count']}, "
                f"net ort {stats['meanNetBps']:+.2f} bps, "
                f"kazanma %{stats['winRate'] * 100:.1f}"
            )
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
    snapshots: dict[str, FuturesMarketSnapshot],
) -> ScalpScanReport:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    current_ms = int(current.timestamp() * 1000)
    maximum_age_ms = settings.scalp_maximum_bar_age_minutes * 60_000
    observations: list[ScalpObservation] = []
    fresh_close_times: list[int] = []
    fresh = 0
    stale = 0
    eligible_frames: dict[str, pd.DataFrame] = {}
    major_frames = _load_major_frames(settings)
    provisional_fresh: dict[str, pd.DataFrame] = {}
    for entry in entries:
        frame = frames.get(entry.perpetual_symbol)
        if frame is None or frame.empty:
            continue
        latest_close = int(frame["close_time_ms"].iloc[-1])
        if current_ms - latest_close <= maximum_age_ms and latest_close <= current_ms:
            provisional_fresh[entry.perpetual_symbol] = frame
    regime = evaluate_bull_regime(
        provisional_fresh,
        major_frames,
        breadth_threshold=settings.scalp_bull_breadth_threshold,
    )
    for entry in entries:
        frame = frames.get(entry.perpetual_symbol)
        if frame is None or frame.empty:
            continue
        latest_close = int(frame["close_time_ms"].iloc[-1])
        if current_ms - latest_close > maximum_age_ms or latest_close > current_ms:
            stale += 1
            continue
        fresh += 1
        eligible_frames[entry.perpetual_symbol] = frame
        fresh_close_times.append(latest_close)
        observations.extend(
            scan_scalp_frame(
                entry,
                frame,
                universe_version=manifest.version,
                regime=regime,
                snapshot=snapshots.get(entry.perpetual_symbol),
                settings=settings,
                historical_cost_bps=manifest.scalp_round_trip_cost_bps,
            )
        )
    observations.extend(
        _relative_strength_observations(
            entries,
            eligible_frames,
            regime=regime,
            snapshots=snapshots,
            settings=settings,
            universe_version=manifest.version,
            historical_cost_bps=manifest.scalp_round_trip_cost_bps,
        )
    )
    if fresh_close_times:
        newest_close = max(fresh_close_times)
        # A lagging API/cache may still be "fresh" by the age limit.  Do not
        # repeat its previous-bar setup inside the newest bar's digest.
        observations = [
            item for item in observations if item.bar_close_time_ms == newest_close
        ]
    observations = _assign_alert_tiers(observations, regime)
    return ScalpScanReport(
        universe_version=manifest.version,
        attempted=len(entries),
        fresh=fresh,
        stale=stale,
        errors=tuple(errors),
        observations=tuple(observations),
        evaluated_at_ms=current_ms,
        regime=regime,
        quoted=sum(entry.perpetual_symbol in snapshots for entry in entries),
    )


def _assign_alert_tiers(
    observations: Iterable[ScalpObservation], regime: BullRegime
) -> list[ScalpObservation]:
    items = list(observations)
    families_by_symbol: dict[str, set[str]] = {}
    for item in items:
        families_by_symbol.setdefault(item.perpetual_symbol, set()).add(item.family)
    return [
        replace(item, alert_tier="KURULUM")
        if (
            regime.state == "BULL"
            and item.execution_eligible
            and len(families_by_symbol[item.perpetual_symbol]) >= 2
        )
        else item
        for item in items
    ]


def _load_major_frames(settings: Settings) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = settings.data_dir / f"{symbol}_1h_futures.csv"
        try:
            frames[symbol] = load_cache(path)
        except (MarketDataError, OSError):
            continue
    return frames


def _relative_strength_observations(
    entries: tuple[UniverseEntry, ...],
    frames: dict[str, pd.DataFrame],
    *,
    regime: BullRegime,
    snapshots: dict[str, FuturesMarketSnapshot],
    settings: Settings,
    universe_version: str,
    historical_cost_bps: float,
) -> list[ScalpObservation]:
    if regime.state != "BULL":
        return []
    current_returns: dict[str, float] = {}
    prior_returns: dict[str, float] = {}
    for symbol, frame in frames.items():
        if len(frame) < 290:
            continue
        close = frame["close"].astype("float64")
        current_returns[symbol] = math.log(
            float(close.iloc[-1]) / float(close.iloc[-289])
        )
        prior_returns[symbol] = math.log(
            float(close.iloc[-2]) / float(close.iloc[-290])
        )
    if len(current_returns) < 10:
        return []
    current_cut = float(np.quantile(list(current_returns.values()), 0.90))
    prior_cut = float(np.quantile(list(prior_returns.values()), 0.90))
    by_symbol = {entry.perpetual_symbol: entry for entry in entries}
    observations: list[ScalpObservation] = []
    for symbol, day_return in current_returns.items():
        if day_return < current_cut or prior_returns[symbol] >= prior_cut:
            continue
        frame = frames[symbol]
        close = frame["close"].astype("float64")
        if math.log(float(close.iloc[-1]) / float(close.iloc[-7])) <= 0.0:
            continue
        entry = by_symbol.get(symbol)
        if entry is None:
            continue
        excess_bps = (day_return - current_cut) * 10_000.0
        observations.append(
            _observation(
                entry,
                universe_version,
                family="B3",
                score=1.0 + min(max(excess_bps, 0.0) / 50.0, 2.0),
                price=float(close.iloc[-1]),
                bar_open_ms=int(frame["open_time_ms"].iloc[-1]),
                bar_close_ms=int(frame["close_time_ms"].iloc[-1]),
                details=(
                    f"24s getiri %{day_return * 100:+.2f}",
                    "ilk %10 goreli guce yeni giris",
                ),
                regime=regime,
                snapshot=snapshots.get(symbol),
                settings=settings,
                historical_cost_bps=historical_cost_bps,
            )
        )
    return observations


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
    regime: BullRegime | None = None,
    snapshot: FuturesMarketSnapshot | None = None,
    settings: Settings | None = None,
    historical_cost_bps: float = 12.0,
) -> ScalpObservation:
    active_regime = regime or BullRegime("UNKNOWN", 0.0, 0.0, 0.0, False, 0)
    spread_bps = snapshot.spread_bps if snapshot is not None else None
    if snapshot is None or settings is None:
        estimated_cost_bps = historical_cost_bps
        execution_eligible = False
    else:
        estimated_cost_bps = max(
            historical_cost_bps,
            2.0 * settings.scalp_taker_fee_bps
            + snapshot.spread_bps
            + 2.0 * settings.scalp_slippage_bps_per_side,
        )
        execution_eligible = snapshot.spread_bps <= settings.scalp_maximum_spread_bps
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
        regime_state=active_regime.state,
        regime_score=active_regime.score,
        breadth=active_regime.breadth,
        spread_bps=spread_bps,
        funding_rate_bps=(snapshot.funding_rate_bps if snapshot is not None else None),
        estimated_cost_bps=estimated_cost_bps,
        execution_eligible=execution_eligible,
    )


def _edge_is_new(condition: pd.Series, latest: int, *, cooldown_bars: int = 12) -> bool:
    """Match trade1's rising-edge plus one-hour greedy cooldown on 5m bars."""
    if (
        latest < 1
        or not bool(condition.iloc[latest])
        or bool(condition.iloc[latest - 1])
    ):
        return False
    rising = condition & ~condition.shift(1, fill_value=False)
    start = max(0, latest - cooldown_bars + 1)
    return not bool(rising.iloc[start:latest].any())


def _time_exit_outcomes(
    record: dict[str, Any], frame: pd.DataFrame
) -> list[dict[str, Any]] | None:
    if frame.empty:
        return None
    matches = frame.index[
        frame["open_time_ms"] == int(record["bar_open_time_ms"])
    ].tolist()
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
                "regime_state": str(record.get("regime_state", "UNKNOWN")),
                "regime_score": float(record.get("regime_score", 0.0)),
                "breadth": float(record.get("breadth", 0.0)),
                "spread_bps": record.get("spread_bps"),
                "funding_rate_bps": record.get("funding_rate_bps"),
                "execution_eligible": bool(record.get("execution_eligible", False)),
                "alert_tier": str(record.get("alert_tier", "RADAR")),
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
    "BullRegime",
    "FAMILY_EVIDENCE",
    "FAMILY_LABELS",
    "SCALP_INTERVAL",
    "ScalpObservation",
    "ScalpScanReport",
    "deliver_scalp_observations",
    "evaluate_bull_regime",
    "format_scalp_observation_digest",
    "format_scalp_scorecard",
    "load_scalp_ledger",
    "record_scalp_observations",
    "refresh_and_scan_scalp_universe",
    "scalp_cache_path",
    "scalp_forecast_stats",
    "scalp_scorecard",
    "scan_cached_scalp_universe",
    "scan_scalp_frame",
    "settle_scalp_observations",
]
