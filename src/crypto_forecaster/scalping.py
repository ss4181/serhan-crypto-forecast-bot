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
from .telegram import (
    TelegramDelivery,
    TelegramNotifier,
    digest_signal_id,
    telegram_channel_keyboard,
)
from .universe import UniverseEntry, UniverseManifest, load_trade1_universe

SCALP_INTERVAL = "5m"
SCALP_STEP_MS = 5 * 60 * 1000
SCALP_MINIMUM_BARS = 290
SCALP_PENDING_SCHEMA = "scalp-observation-pending-v1"
SCALP_LEDGER_SCHEMA = "scalp-observation-outcome-v1"
SCALP_TARGET_PENDING_SCHEMA = "scalp-target-pending-v1"
SCALP_TARGET_LEDGER_SCHEMA = "scalp-target-outcome-v1"
SCALP_BRACKET_PENDING_SCHEMA = "scalp-bracket-pending-v1"
SCALP_BRACKET_LEDGER_SCHEMA = "scalp-bracket-outcome-v1"
SCALP_TARGET_TOUCH_PERCENTS = (2.0, 3.0)
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
STRATEGY_LABELS = {
    "BULL_CONTINUATION_LONG": "Boğa devamı LONG",
    "BULL_EXHAUSTION_SHORT": "Aşırı yükseliş geri çekilmesi SHORT",
    "FLUSH_RECOVERY_LONG": "Sert düşüş sonrası toparlanma LONG",
    "DOWNSIDE_CONTINUATION_SHORT": "Düşüş devamı SHORT",
    "DIRECTIONAL_LONG": "Yönsel momentum LONG",
    "DIRECTIONAL_SHORT": "Yönsel momentum SHORT",
}
STRATEGY_HORIZONS = {
    "BULL_CONTINUATION_LONG": 60,
    "BULL_EXHAUSTION_SHORT": 30,
    "FLUSH_RECOVERY_LONG": 60,
    "DOWNSIDE_CONTINUATION_SHORT": 30,
    "DIRECTIONAL_LONG": 60,
    "DIRECTIONAL_SHORT": 60,
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
class _ScalpMarketContext:
    """Closed-candle context shown alongside a scalp observation."""

    return_24h_pct: float | None = None
    rank_24h: int | None = None
    universe_size: int | None = None
    volume_1h_ratio: float | None = None
    taker_buy_ratio_1h: float | None = None
    volatility_bps: float | None = None


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
    return_24h_pct: float | None = None
    rank_24h: int | None = None
    universe_size: int | None = None
    volume_1h_ratio: float | None = None
    mark_price: float | None = None
    taker_buy_ratio_1h: float | None = None
    volatility_bps: float | None = None

    @property
    def signal_id(self) -> str:
        material = f"{self.universe_version}|{self.perpetual_symbol}|{self.family}|{self.bar_close_time_ms}"
        return sha256(material.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScalpSetupAssessment:
    """Empirical, direction-aware quality for one multi-family setup."""

    strategy_code: str
    strategy_label: str
    direction: str
    horizon_minutes: int | None
    success_probability: float | None
    expected_net_bps: float | None
    sample_count: int
    independent_days: int
    quality_percentile: float | None
    confidence: str

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
    context: _ScalpMarketContext | None = None,
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
                context=context,
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
                context=context,
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
                context=context,
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
                    context=context,
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
                    context=context,
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
    horizon_label = "/".join(str(value) for value in SCALP_BACKTEST_HORIZONS)
    lines = [
        f"🧪 SCALP GÖZLEMİ | 5m | {stamp}",
        (
            f"🧭 Rejim: {regime.label} {regime.score:.2f} • "
            f"Genişlik: %{regime.breadth * 100:.0f} • "
            f"Kotasyon: {report.quoted}/{report.attempted}"
        ),
        (
            f"📡 Taze: {report.fresh}/{report.attempted} • "
            f"Gözlem: {len(report.observations)} • İlk: {len(shown)}"
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
        spread = f"{item.spread_bps:.1f} bps" if item.spread_bps is not None else "veri yok"
        cost = item.estimated_cost_bps or manifest.scalp_round_trip_cost_bps
        detail = ", ".join(item.details)
        tier_icon = "✅" if tier == "KURULUM" else "🔔"
        direction, horizon_directions = scalp_setup_direction(items, ledger_rows)
        assessment = scalp_setup_assessment(items, ledger_rows)
        lines.extend(
            [
                f"{tier_icon} {index}. {tier} | {item.spot_symbol} {mapping}".rstrip(),
                "   🧪 Araştırma • Güven: GÖZLEM",
                f"   💰 Sinyal fiyati: {_format_signal_price(item.price)}",
                f"   ⏱ Beklenen ufuk: {horizon_label} dk",
                "   🏦 Piyasa: Binance USD-M perp",
                f"   🧠 Strateji: {assessment.strategy_label}",
                f"   🧩 Aile: {families} | Ham güç: {item.score:.2f}",
                f"   💸 Maliyet: ~{cost:.1f} bps | Spread: {spread}",
                f"   💡 Tetikleyici: {detail}",
            ]
        )
        lines.append(
            f"   🧭 Yön özeti (yerleşmiş BT): {direction} | "
            f"15/30/60dk: {'/'.join(horizon_directions)}"
        )
        if assessment.success_probability is not None:
            quality = (
                f"%{assessment.quality_percentile * 100:.0f}"
                if assessment.quality_percentile is not None
                else "veri yok"
            )
            lines.extend(
                [
                    f"   🎯 Net başarı olasılığı: %{assessment.success_probability * 100:.1f}",
                    f"   💹 Beklenen net: {assessment.expected_net_bps / 100:+.2f}% "
                    f"| seçilen ufuk: {assessment.horizon_minutes} dk",
                    f"   🛡 Güven: {assessment.confidence} | "
                    f"n={assessment.sample_count}, bağımsız gün={assessment.independent_days}",
                    f"   📏 Aile içi güç yüzdeliği: {quality}",
                ]
            )
        if direction in {"YUKARI", "AŞAĞI"}:
            target_bps, stop_bps = _dynamic_bracket_bps(items)
            target_multiplier = 1.0 + target_bps / 10_000.0 if direction == "YUKARI" else 1.0 - target_bps / 10_000.0
            stop_multiplier = 1.0 - stop_bps / 10_000.0 if direction == "YUKARI" else 1.0 + stop_bps / 10_000.0
            lines.append(
                f"   🎚 Tahmini parantez (sinyal fiyatından): hedef {_format_signal_price(item.price * target_multiplier)} "
                f"(+{target_bps / 100:.2f}%) | stop {_format_signal_price(item.price * stop_multiplier)} "
                f"(-{stop_bps / 100:.2f}%)"
            )
        if item.mark_price is not None:
            lines.append(f"   📍 Güncel mark: {_format_signal_price(item.mark_price)}")
        if item.return_24h_pct is not None:
            rank = (
                f"{item.rank_24h}/{item.universe_size}"
                if item.rank_24h is not None and item.universe_size
                else "veri yok"
            )
            lines.append(
                f"   📈 24s kapalı mum getirisi: {item.return_24h_pct:+.2f}% | "
                f"Yükselen sırası: {rank}"
            )
        if item.volume_1h_ratio is not None:
            lines.append(
                f"   📊 1s hacim / önceki 24s medyanı: {item.volume_1h_ratio:.2f}x"
            )
        if item.funding_rate_bps is not None:
            lines.append(f"   🧾 Funding: {item.funding_rate_bps:+.2f} bps")
        if item.taker_buy_ratio_1h is not None:
            lines.append(f"   🔄 Son 1s aktif alıcı payı: %{item.taker_buy_ratio_1h * 100:.1f}")
        for family_item in items:
            lines.extend(
                "   " + line
                for line in _format_scalp_backtest(
                    family_item,
                    ledger_rows,
                ).splitlines()
            )
    lines.extend(
        [
            "",
            "ISLEM ADAYI DEGILDIR • Başarı ve beklenen net, yerleşmiş ileri gözlemlerden hesaplanan araştırma verisidir.",
        ]
    )
    message = "\n".join(lines)
    if len(message) > 4096:
        raise ValueError("Scalp Telegram mesaji 4096 karakteri asti")
    return message


def filter_scalp_notification_report(
    report: ScalpScanReport,
    *,
    minimum_score: float,
    ledger: Iterable[dict[str, Any]] = (),
    minimum_quality_percentile: float | None = None,
    minimum_direction_probability: float | None = None,
    minimum_expected_net_bps: float | None = None,
    minimum_calibration_samples: int = 0,
) -> ScalpScanReport:
    """Keep only exact-direction, high-score multi-family setups for Telegram.

    The original report is deliberately left untouched for shadow scoring and
    the GitHub dashboard.  This filtered copy controls only what is sent to the
    channel, so muted candidates remain measurable.
    """
    try:
        threshold = float(minimum_score)
    except (TypeError, ValueError):
        raise ValueError("Scalp bildirim skoru sayi olmali") from None
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Scalp bildirim skoru negatif veya sonsuz olamaz")
    ledger_rows = tuple(ledger)
    grouped: dict[str, list[ScalpObservation]] = {}
    for item in report.observations:
        grouped.setdefault(item.perpetual_symbol, []).append(item)
    eligible_symbols: set[str] = set()
    for symbol, items in grouped.items():
        if (
            len({item.family for item in items}) < 2
            or not any(item.alert_tier == "KURULUM" for item in items)
        ):
            continue
        assessment = scalp_setup_assessment(items, ledger_rows)
        if assessment.direction not in {"YUKARI", "AŞAĞI"}:
            continue
        calibrated = assessment.sample_count >= max(int(minimum_calibration_samples), 0)
        if calibrated and minimum_calibration_samples > 0:
            if (
                assessment.success_probability is None
                or assessment.expected_net_bps is None
                or assessment.success_probability
                < float(minimum_direction_probability or 0.0)
                or assessment.expected_net_bps < float(minimum_expected_net_bps or 0.0)
            ):
                continue
            if (
                minimum_quality_percentile is not None
                and assessment.quality_percentile is None
                and max(item.score for item in items) < threshold
            ):
                continue
            if (
                minimum_quality_percentile is not None
                and assessment.quality_percentile is not None
                and assessment.quality_percentile < float(minimum_quality_percentile)
            ):
                continue
        elif max(item.score for item in items) < threshold:
            # Until a family/regime has a minimally useful forward sample, keep
            # the old detector-strength gate as an explicitly temporary fallback.
            continue
        eligible_symbols.add(symbol)
    return replace(
        report,
        observations=tuple(
            item for item in report.observations if item.perpetual_symbol in eligible_symbols
        ),
    )


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
    ledger = load_scalp_ledger(settings.scalp_state_dir)
    filtered_report = filter_scalp_notification_report(
        report,
        minimum_score=settings.scalp_minimum_alert_score,
        ledger=ledger,
        minimum_quality_percentile=settings.scalp_minimum_quality_percentile,
        minimum_direction_probability=settings.scalp_minimum_direction_probability,
        minimum_expected_net_bps=settings.scalp_minimum_expected_net_bps,
        minimum_calibration_samples=settings.scalp_minimum_calibration_samples,
    )
    shown = filtered_report.top(settings.scalp_top_k) if filtered_report.observations else ()
    if not shown:
        return None
    newest_close = max(item.bar_close_time_ms for item in shown)
    bucket = newest_close // SCALP_STEP_MS
    signal_id = digest_signal_id(f"scalp-observation|{manifest.version}", bucket)
    return (
        notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)
    ).deliver_once(
        signal_id=signal_id,
        text=format_scalp_observation_digest(
            filtered_report,
            manifest=manifest,
            top_k=settings.scalp_top_k,
            ledger=ledger,
        ),
        state_dir=settings.telegram_state_dir / "scalp",
        reply_markup=telegram_channel_keyboard(),
    )


def record_scalp_target_setups(
    state_dir: Path,
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    top_k: int,
    ledger: Iterable[dict[str, Any]] = (),
    notification_sent: bool = False,
    milestone_horizon_hours: int = 24,
) -> int:
    """Park directional setups for shadow +/−2% and +/−3% milestones.

    A digest can contain several families for one market.  The target watcher is
    therefore keyed by the displayed market setup, not by an individual family.
    Only a multi-family ``KURULUM`` with an exact BT direction is watched; radar
    and ``KARIŞIK``/``AĞIRLIKLI`` summaries remain research-only.  ``notification_sent``
    marks whether the setup passed the Telegram filter; muted setups are still
    settled into the shadow target ledger.
    """
    shown = report.top(top_k)
    ledger_rows = tuple(ledger)
    grouped: dict[str, list[ScalpObservation]] = {}
    for item in shown:
        grouped.setdefault(item.perpetual_symbol, []).append(item)
    directory = _target_pending_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    added = 0
    for items in grouped.values():
        if (
            not any(item.alert_tier == "KURULUM" for item in items)
            or len({item.family for item in items}) < 2
        ):
            continue
        direction, horizon_directions = scalp_setup_direction(items, ledger_rows)
        if direction not in {"YUKARI", "AŞAĞI"}:
            continue
        aggregate = scalp_setup_forecast_stats(items, ledger_rows)
        assessment = scalp_setup_assessment(items, ledger_rows)
        source = items[0]
        setup_id = _scalp_setup_id(source, manifest.version)
        path = directory / f"{setup_id}.json"
        payload = {
            "schema": SCALP_TARGET_PENDING_SCHEMA,
            "setup_id": setup_id,
            "signal_id": setup_id,
            "universe_version": manifest.version,
            "spot_symbol": source.spot_symbol,
            "perpetual_symbol": source.perpetual_symbol,
            "universe_group": source.universe_group,
            "families": [item.family for item in items],
            "direction": direction,
            "horizon_directions": list(horizon_directions),
            "source_price": float(source.price),
            "bar_close_time_ms": int(source.bar_close_time_ms),
            "horizons_minutes": list(manifest.scalp_horizons_minutes),
            "horizon_ms": max(int(milestone_horizon_hours), 1) * 3_600_000,
            "strategy_code": assessment.strategy_code,
            "strategy_label": assessment.strategy_label,
            "assessment_horizon_minutes": assessment.horizon_minutes,
            "success_probability": assessment.success_probability,
            "expected_net_bps": assessment.expected_net_bps,
            "calibration_sample_count": assessment.sample_count,
            "independent_days": assessment.independent_days,
            "quality_percentile": assessment.quality_percentile,
            "confidence": assessment.confidence,
            "tier": "KURULUM",
            "score": float(max(item.score for item in items)),
            "probability_up": {
                str(horizon): float(aggregate[horizon][1])
                for horizon in SCALP_BACKTEST_HORIZONS
                if horizon in aggregate
            },
            "probability_down": {
                str(horizon): float(1.0 - aggregate[horizon][1])
                for horizon in SCALP_BACKTEST_HORIZONS
                if horizon in aggregate
            },
            "sample_count": {
                str(horizon): int(aggregate[horizon][0])
                for horizon in SCALP_BACKTEST_HORIZONS
                if horizon in aggregate
            },
            "delivered_percents": [],
            "outcome_recorded_percents": [],
            "notification_sent": bool(notification_sent),
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
            added += 1
        except FileExistsError:
            if notification_sent:
                existing = _read_scalp_target_record(path)
                if existing is not None and not bool(existing.get("notification_sent", False)):
                    existing["notification_sent"] = True
                    path.write_text(
                        json.dumps(
                            existing,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            continue
    return added


def pending_scalp_target_touches(
    state_dir: Path,
    data_dir: Path,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Find newly touched directional scalp +/−2% and +/−3% levels."""
    directory = _target_pending_dir(state_dir)
    if not directory.exists():
        return []
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    grace_ms = SCALP_SETTLEMENT_GRACE_DAYS * 24 * 60 * 60 * 1000
    frames: dict[str, pd.DataFrame] = {}
    events: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_scalp_target_record(path)
        if record is None:
            path.unlink(missing_ok=True)
            continue
        source_ms = int(record["bar_close_time_ms"])
        deadline_ms = source_ms + int(record["horizon_ms"])
        if current_ms - deadline_ms > grace_ms:
            path.unlink(missing_ok=True)
            continue
        symbol = str(record["perpetual_symbol"])
        if symbol not in frames:
            try:
                frames[symbol] = load_cache(scalp_cache_path(data_dir, symbol))
            except (MarketDataError, OSError, ValueError):
                frames[symbol] = pd.DataFrame()
        until_ms = min(current_ms, deadline_ms)
        window = _scalp_target_window(frames[symbol], source_ms, until_ms)
        touched = _scalp_target_touches(record, window)
        delivered = {float(value) for value in record.get("delivered_percents", [])}
        if not bool(record.get("notification_sent", True)):
            continue
        for percent in SCALP_TARGET_TOUCH_PERCENTS:
            if percent in delivered or percent not in touched:
                continue
            event = dict(record)
            event.update(touched[percent])
            event["target_percent"] = percent
            events.append(event)
    return events


def settle_scalp_target_outcomes(
    state_dir: Path,
    data_dir: Path,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Persist target hits/misses for both notified and muted setups."""
    directory = _target_pending_dir(state_dir)
    if not directory.exists():
        return []
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    grace_ms = SCALP_SETTLEMENT_GRACE_DAYS * 24 * 60 * 60 * 1000
    frames: dict[str, pd.DataFrame] = {}
    settled: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_scalp_target_record(path)
        if record is None:
            path.unlink(missing_ok=True)
            continue
        source_ms = int(record["bar_close_time_ms"])
        deadline_ms = source_ms + int(record["horizon_ms"])
        if current_ms < deadline_ms:
            continue
        symbol = str(record["perpetual_symbol"])
        if symbol not in frames:
            try:
                frames[symbol] = load_cache(scalp_cache_path(data_dir, symbol))
            except (MarketDataError, OSError, ValueError):
                frames[symbol] = pd.DataFrame()
        window = _scalp_target_window(frames[symbol], source_ms, deadline_ms)
        if window.empty:
            if current_ms - deadline_ms > grace_ms:
                path.unlink(missing_ok=True)
            continue
        touched = _scalp_target_touches(record, window)
        recorded = {float(value) for value in record.get("outcome_recorded_percents", [])}
        hit_levels = set(touched)
        for percent in SCALP_TARGET_TOUCH_PERCENTS:
            if percent in recorded:
                continue
            target_price = float(record["source_price"]) * (
                1.0 + percent / 100.0
                if str(record["direction"]) == "YUKARI"
                else 1.0 - percent / 100.0
            )
            hit = percent in touched
            outcome = {
                "schema": SCALP_TARGET_LEDGER_SCHEMA,
                "setup_id": record["setup_id"],
                "universe_version": record.get("universe_version", ""),
                "spot_symbol": record["spot_symbol"],
                "perpetual_symbol": record["perpetual_symbol"],
                "direction": record["direction"],
                "families": record.get("families", []),
                "score": float(record.get("score", 0.0)),
                "notification_sent": bool(record.get("notification_sent", True)),
                "bar_close_time_ms": source_ms,
                "source_price": float(record["source_price"]),
                "horizon_ms": int(record["horizon_ms"]),
                "target_percent": percent,
                "target_price": target_price,
                "hit": hit,
                "touch_price": touched[percent]["touch_price"] if hit else None,
                "touch_close_time_ms": touched[percent]["touch_close_time_ms"] if hit else None,
                "settled_at_ms": current_ms,
                "probability_up": record.get("probability_up", {}),
                "probability_down": record.get("probability_down", {}),
                "strategy_code": record.get("strategy_code", ""),
                "strategy_label": record.get("strategy_label", ""),
                "success_probability": record.get("success_probability"),
                "expected_net_bps": record.get("expected_net_bps"),
                "calibration_sample_count": record.get("calibration_sample_count", 0),
                "independent_days": record.get("independent_days", 0),
                "quality_percentile": record.get("quality_percentile"),
                "confidence": record.get("confidence", "VERİ YOK"),
            }
            settled.append(outcome)
            recorded.add(percent)
        record["outcome_recorded_percents"] = sorted(recorded)
        delivered = {float(value) for value in record.get("delivered_percents", [])}
        if all(level in recorded for level in SCALP_TARGET_TOUCH_PERCENTS) and (
            not bool(record.get("notification_sent", True))
            or hit_levels.issubset(delivered)
        ):
            path.unlink(missing_ok=True)
        else:
            path.write_text(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
    if settled:
        _append_scalp_target_ledger(state_dir, settled)
    return settled


def record_scalp_bracket_setups(
    state_dir: Path,
    report: ScalpScanReport,
    *,
    manifest: UniverseManifest,
    top_k: int,
    ledger: Iterable[dict[str, Any]] = (),
    notification_sent: bool = False,
    horizon_minutes: int = 60,
) -> int:
    """Park exact-direction setups for a realistic first-touch TP/SL test."""
    shown = report.top(top_k)
    ledger_rows = tuple(ledger)
    grouped: dict[str, list[ScalpObservation]] = {}
    for item in shown:
        grouped.setdefault(item.perpetual_symbol, []).append(item)
    directory = _bracket_pending_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    added = 0
    for items in grouped.values():
        if (
            not any(item.alert_tier == "KURULUM" for item in items)
            or len({item.family for item in items}) < 2
        ):
            continue
        assessment = scalp_setup_assessment(items, ledger_rows)
        if assessment.direction not in {"YUKARI", "AŞAĞI"}:
            continue
        source = items[0]
        setup_id = _scalp_setup_id(source, manifest.version)
        path = directory / f"{setup_id}.json"
        target_bps, stop_bps = _dynamic_bracket_bps(items)
        payload = {
            "schema": SCALP_BRACKET_PENDING_SCHEMA,
            "setup_id": setup_id,
            "universe_version": manifest.version,
            "spot_symbol": source.spot_symbol,
            "perpetual_symbol": source.perpetual_symbol,
            "universe_group": source.universe_group,
            "families": [item.family for item in items],
            "direction": assessment.direction,
            "strategy_code": assessment.strategy_code,
            "strategy_label": assessment.strategy_label,
            "source_price": float(source.price),
            "bar_close_time_ms": int(source.bar_close_time_ms),
            "horizon_minutes": max(int(horizon_minutes), 5),
            "target_bps": target_bps,
            "stop_bps": stop_bps,
            "round_trip_cost_bps": max(
                float(item.estimated_cost_bps) for item in items
            ),
            "raw_score": max(float(item.score) for item in items),
            "quality_percentile": assessment.quality_percentile,
            "success_probability": assessment.success_probability,
            "expected_net_bps": assessment.expected_net_bps,
            "calibration_sample_count": assessment.sample_count,
            "independent_days": assessment.independent_days,
            "confidence": assessment.confidence,
            "notification_sent": bool(notification_sent),
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
            added += 1
        except FileExistsError:
            if notification_sent:
                existing = _read_scalp_bracket_record(path)
                if existing is not None and not bool(existing.get("notification_sent", False)):
                    existing["notification_sent"] = True
                    path.write_text(
                        json.dumps(existing, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        + "\n",
                        encoding="utf-8",
                    )
    return added


def settle_scalp_bracket_outcomes(
    state_dir: Path,
    data_dir: Path,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Settle at the first target/stop touch, conservatively resolving ties."""
    directory = _bracket_pending_dir(state_dir)
    if not directory.exists():
        return []
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    grace_ms = SCALP_SETTLEMENT_GRACE_DAYS * 24 * 60 * 60 * 1000
    frames: dict[str, pd.DataFrame] = {}
    settled: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = _read_scalp_bracket_record(path)
        if record is None:
            path.unlink(missing_ok=True)
            continue
        source_ms = int(record["bar_close_time_ms"])
        deadline_ms = source_ms + int(record["horizon_minutes"]) * 60_000
        symbol = str(record["perpetual_symbol"])
        if symbol not in frames:
            try:
                frames[symbol] = load_cache(scalp_cache_path(data_dir, symbol))
            except (MarketDataError, OSError, ValueError):
                frames[symbol] = pd.DataFrame()
        window = _scalp_bracket_window(
            frames[symbol], source_ms, min(current_ms, deadline_ms)
        )
        if window.empty:
            if current_ms - deadline_ms > grace_ms:
                path.unlink(missing_ok=True)
            continue
        outcome = _first_touch_bracket(record, window, deadline_reached=current_ms >= deadline_ms)
        if outcome is None:
            continue
        settled.append(outcome)
        path.unlink(missing_ok=True)
    if settled:
        _append_scalp_bracket_ledger(state_dir, settled)
    return settled


def load_scalp_bracket_ledger(
    state_dir: Path, *, limit: int = 20_000
) -> list[dict[str, Any]]:
    path = _bracket_ledger_path(state_dir)
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
        if isinstance(payload, dict) and payload.get("schema") == SCALP_BRACKET_LEDGER_SCHEMA:
            rows.append(payload)
    return rows


def load_pending_scalp_brackets(
    state_dir: Path, *, limit: int = 20_000
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("Scalp parantez limiti pozitif olmali")
    directory = _bracket_pending_dir(state_dir)
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"))[-limit:]:
        payload = _read_scalp_bracket_record(path)
        if payload is not None:
            rows.append(payload)
    return rows


def format_scalp_bracket_result(event: dict[str, Any]) -> str:
    """Explain a realised volatility-aware scalp target in stacked form."""
    direction = str(event.get("direction", ""))
    return "\n".join(
        [
            f"✅ SCALP HEDEFİ | {event['spot_symbol']} | {direction}",
            f"🧠 Strateji: {event.get('strategy_label', '-')}",
            f"📍 Gerçekçi giriş: {_format_signal_price(float(event['entry_price']))}",
            f"🎯 Hedef: {_format_signal_price(float(event['target_price']))} "
            f"(+{float(event['target_bps']) / 100:.2f}%)",
            f"🛑 Stop: {_format_signal_price(float(event['stop_price']))} "
            f"(-{float(event['stop_bps']) / 100:.2f}%)",
            f"💹 Net sonuç: {float(event['net_bps']) / 100:+.2f}% "
            f"| maliyet dahil",
            f"📈 En iyi / en kötü hareket: +{float(event['mfe_bps']) / 100:.2f}% "
            f"/ -{float(event['mae_bps']) / 100:.2f}%",
            f"🕒 Süre: {float(event['elapsed_minutes']):.0f} dk",
            "ℹ️ Araştırma sonucu; emir veya kazanç garantisi değildir.",
        ]
    )


def deliver_scalp_bracket_wins(
    settings: Settings,
    *,
    notifier: TelegramNotifier | None = None,
) -> list[tuple[dict[str, Any], TelegramDelivery]]:
    """Deliver notified bracket wins idempotently; shadow outcomes stay silent."""
    sender = notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)
    deliveries: list[tuple[dict[str, Any], TelegramDelivery]] = []
    for event in load_scalp_bracket_ledger(settings.scalp_state_dir):
        if event.get("resolution") != "TARGET" or not bool(event.get("notification_sent", False)):
            continue
        signal_id = digest_signal_id(f"scalp-bracket|{event['setup_id']}", 0)
        delivery = sender.deliver_once(
            signal_id=signal_id,
            text=format_scalp_bracket_result(event),
            state_dir=settings.telegram_state_dir / "scalp-brackets",
            reply_markup=telegram_channel_keyboard(),
        )
        deliveries.append((event, delivery))
    return deliveries


def load_scalp_target_ledger(state_dir: Path, *, limit: int = 20_000) -> list[dict[str, Any]]:
    path = _target_ledger_path(state_dir)
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
        if isinstance(payload, dict) and payload.get("schema") == SCALP_TARGET_LEDGER_SCHEMA:
            rows.append(payload)
    return rows


def load_pending_scalp_targets(
    state_dir: Path, *, limit: int = 20_000
) -> list[dict[str, Any]]:
    """Load valid open target setups without exposing mutable state paths."""
    if limit < 1:
        raise ValueError("Scalp hedef limiti pozitif olmali")
    directory = _target_pending_dir(state_dir)
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"))[-limit:]:
        payload = _read_scalp_target_record(path)
        if payload is not None:
            rows.append(payload)
    return rows


def mark_scalp_target_touch_delivered(
    state_dir: Path, setup_id: str, target_percent: float
) -> None:
    """Persist one successful scalp target notification, idempotently."""
    try:
        percent = float(target_percent)
    except (TypeError, ValueError):
        return
    if percent not in SCALP_TARGET_TOUCH_PERCENTS:
        return
    path = _target_pending_dir(state_dir) / f"{setup_id}.json"
    record = _read_scalp_target_record(path)
    if record is None:
        return
    delivered = {float(value) for value in record.get("delivered_percents", [])}
    delivered.add(percent)
    record["delivered_percents"] = sorted(delivered)
    recorded = {
        float(value) for value in record.get("outcome_recorded_percents", [])
    }
    if all(level in recorded for level in SCALP_TARGET_TOUCH_PERCENTS):
        path.unlink(missing_ok=True)
        return
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def format_scalp_target_touch(event: dict[str, Any]) -> str:
    """Format one compact scalp target touch with its BT context."""
    direction = str(event.get("direction", ""))
    sign = "+" if direction == "YUKARI" else "-"
    percent = float(event["target_percent"])
    horizons = tuple(int(value) for value in event.get("horizons_minutes", SCALP_BACKTEST_HORIZONS))
    up = _format_probability_map(event.get("probability_up", {}), horizons)
    down = _format_probability_map(event.get("probability_down", {}), horizons)
    horizon_directions = "/".join(str(value) for value in event.get("horizon_directions", []))
    families = "+".join(str(value) for value in event.get("families", [])) or "-"
    source_ms = int(event["bar_close_time_ms"])
    touch_ms = int(event["touch_close_time_ms"])
    milestone_hours = int(event.get("horizon_ms", 0)) / 3_600_000.0
    assessment_lines: list[str] = []
    if event.get("strategy_label"):
        assessment_lines.append(f"🧠 Strateji: {event['strategy_label']}")
    if event.get("success_probability") is not None:
        assessment_lines.append(
            f"📐 Sinyaldeki net başarı tahmini: %{float(event['success_probability']) * 100:.1f} "
            f"| güven {event.get('confidence', '-')}"
        )
    return "\n".join(
        [
            f"🎯 MOMENTUM KİLOMETRE TAŞI | {event['spot_symbol']}",
            f"🧭 Yön özeti: {direction} | 15/30/60dk: {horizon_directions or '-'}",
            *assessment_lines,
            f"📍 Sinyal fiyatı: {_format_signal_price(float(event['source_price']))}",
            f"✅ Hedef kademe: {sign}{percent:g}%",
            f"🎯 Hedef fiyatı: {_format_signal_price(float(event['target_price']))}",
            f"💹 Mumda görülen: {_format_signal_price(float(event['touch_price']))}",
            f"⏱ Kilometre taşı izleme ufku: {milestone_hours:g} saat",
            f"📊 BT yukarı olasılığı (15/30/60dk): {up}",
            f"📉 BT aşağı olasılığı (15/30/60dk): {down}",
            f"🧩 Aile: {families} | örneklem ağırlıklı geçmiş sentez",
            f"🕒 Sinyal zamanı: {local_text(source_ms, with_seconds=False)}",
            f"🕒 Dokunma zamanı: {local_text(touch_ms, with_seconds=False)}",
            "ℹ️ Araştırma bildirimi; emir veya kazanç garantisi değildir.",
        ]
    )


def deliver_scalp_target_touches(
    settings: Settings,
    *,
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> list[tuple[dict[str, Any], TelegramDelivery]]:
    """Deliver each scalp setup's +/−2% and +/−3% touch once."""
    events = pending_scalp_target_touches(settings.scalp_state_dir, settings.scalp_data_dir, now=now)
    if not events:
        return []
    sender = notifier or TelegramNotifier(state_dir=settings.telegram_state_dir)
    deliveries: list[tuple[dict[str, Any], TelegramDelivery]] = []
    for event in events:
        percent = float(event["target_percent"])
        signal_id = digest_signal_id(
            f"scalp-target-touch|{event['setup_id']}|{percent:g}", 0
        )
        delivery = sender.deliver_once(
            signal_id=signal_id,
            text=format_scalp_target_touch(event),
            state_dir=settings.telegram_state_dir / "scalp-targets",
            reply_markup=telegram_channel_keyboard(),
        )
        deliveries.append((event, delivery))
        if delivery.status in {"SENT", "DEDUPLICATED"}:
            mark_scalp_target_touch_delivered(
                settings.scalp_state_dir, str(event["setup_id"]), percent
            )
    return deliveries


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

    candidates = _scalp_candidate_rows(item, all_rows)

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


def scalp_setup_direction(
    items: Iterable[ScalpObservation],
    rows: Iterable[dict[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """Summarise a setup's empirical direction without inventing a model.

    Family forward-test samples are weighted by their settled count.  A horizon
    is called directional only when its weighted up probability and weighted
    median net move agree; otherwise it remains explicitly mixed.
    """
    item_rows = tuple(rows)
    item_tuple = tuple(items)
    family_stats = [scalp_forecast_stats(item, item_rows) for item in item_tuple]
    horizon_labels: list[str] = []
    family_disagreement = False
    for horizon in SCALP_BACKTEST_HORIZONS:
        selected = [stats[horizon] for stats in family_stats if horizon in stats]
        if not selected:
            horizon_labels.append("VERI YOK")
            continue
        family_labels = [_classify_bt_direction(stats) for stats in selected]
        family_disagreement = family_disagreement or len(set(family_labels)) > 1
        total = sum(max(int(stats[0]), 0) for stats in selected)
        if total <= 0:
            horizon_labels.append("VERI YOK")
            continue
        weighted_up = sum(int(stats[0]) * float(stats[1]) for stats in selected) / total
        weighted_net = sum(int(stats[0]) * float(stats[3]) for stats in selected) / total
        if weighted_up >= 0.55 and weighted_net > 0.0:
            horizon_labels.append("YUKARI")
        elif weighted_up <= 0.45 and weighted_net < 0.0:
            horizon_labels.append("AŞAĞI")
        else:
            horizon_labels.append("KARIŞIK")
    valid = [label for label in horizon_labels if label != "VERI YOK"]
    if not valid:
        return "VERI YOK", tuple(horizon_labels)
    if all(label == valid[0] for label in valid):
        overall = f"{valid[0]} AĞIRLIKLI" if family_disagreement else valid[0]
        return overall, tuple(horizon_labels)
    up = valid.count("YUKARI")
    down = valid.count("AŞAĞI")
    if down > up and down >= 2:
        return "AŞAĞI AĞIRLIKLI", tuple(horizon_labels)
    if up > down and up >= 2:
        return "YUKARI AĞIRLIKLI", tuple(horizon_labels)
    return "KARIŞIK", tuple(horizon_labels)


def scalp_setup_forecast_stats(
    items: Iterable[ScalpObservation],
    rows: Iterable[dict[str, Any]],
) -> dict[int, tuple[int, float, float, float]]:
    """Return sample-weighted BT stats for a displayed multi-family setup."""
    item_rows = tuple(rows)
    family_stats = [scalp_forecast_stats(item, item_rows) for item in items]
    result: dict[int, tuple[int, float, float, float]] = {}
    for horizon in SCALP_BACKTEST_HORIZONS:
        selected = [stats[horizon] for stats in family_stats if horizon in stats]
        total = sum(max(int(stats[0]), 0) for stats in selected)
        if total <= 0:
            continue
        result[horizon] = (
            total,
            sum(int(stats[0]) * float(stats[1]) for stats in selected) / total,
            sum(int(stats[0]) * float(stats[2]) for stats in selected) / total,
            sum(int(stats[0]) * float(stats[3]) for stats in selected) / total,
        )
    return result


def scalp_setup_assessment(
    items: Iterable[ScalpObservation],
    rows: Iterable[dict[str, Any]],
) -> ScalpSetupAssessment:
    """Turn settled family observations into one comparable setup assessment.

    Detector strength remains useful inside its own family, but it is not mixed
    as though B1=2.5 and F1=2.5 meant the same thing.  Instead, the assessment
    uses each family's score percentile plus direction-aware net outcomes.
    """
    item_tuple = tuple(items)
    row_tuple = tuple(row for row in rows if isinstance(row, dict))
    direction, _ = scalp_setup_direction(item_tuple, row_tuple)
    strategy_code = _strategy_code(item_tuple, direction)
    strategy_label = STRATEGY_LABELS.get(strategy_code, strategy_code)
    if direction not in {"YUKARI", "AŞAĞI"}:
        return ScalpSetupAssessment(
            strategy_code,
            strategy_label,
            direction,
            None,
            None,
            None,
            0,
            0,
            _setup_quality_percentile(item_tuple, row_tuple),
            "VERİ YOK",
        )

    by_horizon: dict[int, tuple[float, float, int, int]] = {}
    for horizon in SCALP_BACKTEST_HORIZONS:
        directed_net: list[float] = []
        days: set[str] = set()
        for item in item_tuple:
            family_rows = [
                row
                for row in row_tuple
                if str(row.get("family", "")) == item.family
                and _scalp_row_has_numbers(row)
            ]
            for row in _scalp_candidate_rows(item, family_rows):
                if int(row.get("horizon_minutes", 0)) != horizon:
                    continue
                gross = float(row["gross_bps"])
                try:
                    cost = float(row.get("round_trip_cost_bps", item.estimated_cost_bps))
                except (TypeError, ValueError):
                    cost = float(item.estimated_cost_bps)
                net = gross - cost if direction == "YUKARI" else -gross - cost
                if math.isfinite(net):
                    directed_net.append(net)
                try:
                    exit_ms = int(row.get("exit_time_ms", 0))
                    if exit_ms > 0:
                        days.add(datetime.fromtimestamp(exit_ms / 1000, tz=UTC).date().isoformat())
                except (OSError, OverflowError, TypeError, ValueError):
                    pass
        if directed_net:
            wins = sum(value > 0.0 for value in directed_net)
            # A weak beta prior avoids displaying 0%/100% from tiny samples.
            probability = (wins + 1.0) / (len(directed_net) + 2.0)
            by_horizon[horizon] = (
                probability,
                sum(directed_net) / len(directed_net),
                len(directed_net),
                len(days),
            )
    percentile = _setup_quality_percentile(item_tuple, row_tuple)
    if not by_horizon:
        return ScalpSetupAssessment(
            strategy_code,
            strategy_label,
            direction,
            None,
            None,
            None,
            0,
            0,
            percentile,
            "VERİ YOK",
        )
    # Horizon is fixed by playbook before reading its realised return.  Picking
    # whichever horizon happened to look best would introduce selection bias.
    horizon = STRATEGY_HORIZONS[strategy_code]
    selected = by_horizon.get(horizon)
    if selected is None:
        return ScalpSetupAssessment(
            strategy_code,
            strategy_label,
            direction,
            horizon,
            None,
            None,
            0,
            0,
            percentile,
            "VERİ YOK",
        )
    probability, expected_net, sample_count, independent_days = selected
    confidence = _assessment_confidence(sample_count, independent_days)
    return ScalpSetupAssessment(
        strategy_code,
        strategy_label,
        direction,
        horizon,
        probability,
        expected_net,
        sample_count,
        independent_days,
        percentile,
        confidence,
    )


def _scalp_candidate_rows(
    item: ScalpObservation, rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    all_rows = list(rows)
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
        return same_symbol_regime
    if len(same_regime) >= 10:
        return same_regime
    return all_rows


def _setup_quality_percentile(
    items: Iterable[ScalpObservation], rows: Iterable[dict[str, Any]]
) -> float | None:
    row_tuple = tuple(rows)
    percentiles: list[float] = []
    for item in items:
        scores: list[float] = []
        for row in row_tuple:
            if str(row.get("family", "")) != item.family:
                continue
            if str(row.get("regime_state", "UNKNOWN")) != item.regime_state:
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores.append(score)
        if len(scores) >= 10:
            percentiles.append(sum(value <= item.score for value in scores) / len(scores))
    return sum(percentiles) / len(percentiles) if percentiles else None


def _strategy_code(items: Iterable[ScalpObservation], direction: str) -> str:
    item_tuple = tuple(items)
    families = {item.family for item in item_tuple}
    bull = any(item.regime_state == "BULL" for item in item_tuple)
    if direction == "YUKARI" and "F2" in families:
        return "FLUSH_RECOVERY_LONG"
    if direction == "YUKARI" and bull:
        return "BULL_CONTINUATION_LONG"
    if direction == "AŞAĞI" and bull:
        return "BULL_EXHAUSTION_SHORT"
    if direction == "AŞAĞI" and "F2" in families:
        return "DOWNSIDE_CONTINUATION_SHORT"
    if direction == "YUKARI":
        return "DIRECTIONAL_LONG"
    return "DIRECTIONAL_SHORT"


def _assessment_confidence(sample_count: int, independent_days: int) -> str:
    if sample_count < 30 or independent_days < 3:
        return "DÜŞÜK"
    if sample_count < 100 or independent_days < 10:
        return "ORTA"
    return "YÜKSEK"


def _classify_bt_direction(stats: tuple[int, float, float, float]) -> str:
    probability_up = float(stats[1])
    median_net = float(stats[3])
    if probability_up >= 0.55 and median_net > 0.0:
        return "YUKARI"
    if probability_up <= 0.45 and median_net < 0.0:
        return "AŞAĞI"
    return "KARIŞIK"


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
    return "\n".join(
        [
            f"{item.family} BT 15/30/60dk (yerlesmis ileri-test):",
            f"  Yukari olasiligi: {up} | Asagi olasiligi: {down}",
            f"  Medyan hareket: {gross} bps ({gross_pct})",
            f"  Medyan net hareket: {net} bps | Ornek sayisi n: {counts}",
        ]
    )


def _format_signal_price(price: float) -> str:
    """Keep tiny altcoin prices readable without hiding precision."""
    text = f"${price:,.8f}".rstrip("0").rstrip(".")
    return text if text != "$" else "$0"


def _dynamic_bracket_bps(
    items: Iterable[ScalpObservation],
) -> tuple[float, float]:
    """Return a predeclared volatility/cost-aware research target and stop."""
    item_tuple = tuple(items)
    ranges = [
        float(item.volatility_bps)
        for item in item_tuple
        if item.volatility_bps is not None
        and math.isfinite(float(item.volatility_bps))
        and float(item.volatility_bps) > 0.0
    ]
    volatility = float(median(ranges)) if ranges else 25.0
    cost = max((float(item.estimated_cost_bps) for item in item_tuple), default=12.0)
    # Four cost units keeps the target from being swallowed by execution;
    # range multiples make the bracket expand for volatile contracts.  Caps
    # prevent one abnormal candle from creating a non-scalp target.
    target_bps = min(max(2.0 * volatility, 4.0 * cost, 30.0), 150.0)
    stop_bps = min(max(1.25 * volatility, 2.5 * cost, 25.0), 100.0)
    return float(target_bps), float(stop_bps)


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
            "return_24h_pct": item.return_24h_pct,
            "rank_24h": item.rank_24h,
            "universe_size": item.universe_size,
            "volume_1h_ratio": item.volume_1h_ratio,
            "mark_price": item.mark_price,
            "taker_buy_ratio_1h": item.taker_buy_ratio_1h,
            "volatility_bps": item.volatility_bps,
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
    bracket_rows: Iterable[dict[str, Any]] = (),
    days: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    if days < 1:
        raise ValueError("Scalp karne gun sayisi pozitif olmali")
    current_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    cutoff = current_ms - days * 86_400_000
    recent = [row for row in rows if int(row.get("exit_time_ms", 0)) >= cutoff]
    keys = sorted({(str(row["family"]), int(row["horizon_minutes"])) for row in recent})
    recent_brackets = [
        row
        for row in bracket_rows
        if int(row.get("exit_time_ms", 0)) >= cutoff
    ]
    strategies = sorted(
        {str(row.get("strategy_label", "Bilinmeyen")) for row in recent_brackets}
    )
    notified_brackets = [
        row for row in recent_brackets if bool(row.get("notification_sent", False))
    ]
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
        "bracketCount": len(recent_brackets),
        "bracketWinRate": (
            sum(row.get("resolution") == "TARGET" for row in recent_brackets)
            / len(recent_brackets)
            if recent_brackets
            else 0.0
        ),
        "bracketMeanNetBps": (
            sum(float(row.get("net_bps", 0.0)) for row in recent_brackets)
            / len(recent_brackets)
            if recent_brackets
            else 0.0
        ),
        "bracketNotified": _aggregate_brackets(notified_brackets),
        "bracketShadow": _aggregate_brackets(
            [row for row in recent_brackets if not bool(row.get("notification_sent", False))]
        ),
        "byStrategy": {
            strategy: _aggregate_brackets(
                [
                    row
                    for row in recent_brackets
                    if str(row.get("strategy_label", "Bilinmeyen")) == strategy
                ]
            )
            for strategy in strategies
        },
    }


def format_scalp_scorecard(card: dict[str, Any]) -> str:
    lines = [
        f"🧪 SCALP ILERI TEST KARNESI — son {card['days']} gun",
        "",
        "📌 OZET",
        f"Olgun gozlem: {card['observationCount']}",
        f"Sonuc sayisi: {card['outcomeCount']}",
    ]
    if not card["outcomeCount"]:
        lines.append("ℹ️ Henuz 15/30/60 dakika sonucu olusan gozlem yok.")
    else:
        lines.append("")
        lines.append("🧭 REJIM BAZINDA")
        for regime, stats in card.get("byRegime", {}).items():
            lines.append(
                f"• Rejim {regime}: n={stats['count']}, "
                f"net ort {stats['meanNetBps']:+.2f} bps, "
                f"kazanma %{stats['winRate'] * 100:.1f}"
            )
        lines.append("")
        lines.append("🧩 AILE + UFUK BAZINDA")
        for key, stats in card["byFamilyHorizon"].items():
            family, horizon = key.split("_", 1)
            lines.append(
                f"• {family} {horizon}dk: n={stats['count']}, "
                f"net ort {stats['meanNetBps']:+.2f} bps, "
                f"medyan {stats['medianNetBps']:+.2f}, kazanma %{stats['winRate'] * 100:.1f}"
            )
    lines.append("")
    lines.append("🎚 GERÇEKÇİ İLK-DOKUNUŞ TP/SL")
    if not card.get("bracketCount", 0):
        lines.append("• Henüz olgun dinamik hedef/stop sonucu yok.")
    else:
        lines.append(
            f"• Toplam: n={card['bracketCount']}, hedef önce %{card['bracketWinRate'] * 100:.1f}, "
            f"net ort {card['bracketMeanNetBps']:+.2f} bps"
        )
        notified = card.get("bracketNotified", {})
        shadow = card.get("bracketShadow", {})
        lines.append(
            f"• Telegram: n={notified.get('count', 0)}, "
            f"hedef önce %{notified.get('winRate', 0.0) * 100:.1f}, "
            f"net ort {notified.get('meanNetBps', 0.0):+.2f} bps"
        )
        lines.append(
            f"• Sessiz karşılaştırma: n={shadow.get('count', 0)}, "
            f"hedef önce %{shadow.get('winRate', 0.0) * 100:.1f}, "
            f"net ort {shadow.get('meanNetBps', 0.0):+.2f} bps"
        )
        for strategy, stats in card.get("byStrategy", {}).items():
            lines.append(
                f"• {strategy}: n={stats['count']}, hedef önce %{stats['winRate'] * 100:.1f}, "
                f"net ort {stats['meanNetBps']:+.2f} bps, "
                f"stop {stats['stops']} / süre {stats['timeouts']}"
            )
    lines.extend(
        [
            "",
            "⚠️ Bu ileri test otomatik terfi kapisi degildir; gozlemler halen islem adayi degildir.",
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

    contexts = _rank_market_contexts(eligible_frames)
    for entry in entries:
        frame = eligible_frames.get(entry.perpetual_symbol)
        if frame is None:
            continue
        observations.extend(
            scan_scalp_frame(
                entry,
                frame,
                universe_version=manifest.version,
                regime=regime,
                snapshot=snapshots.get(entry.perpetual_symbol),
                settings=settings,
                historical_cost_bps=manifest.scalp_round_trip_cost_bps,
                context=contexts.get(entry.perpetual_symbol),
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
            contexts=contexts,
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


def _closed_market_context(frame: pd.DataFrame) -> _ScalpMarketContext:
    """Calculate compact, causal market context from closed 5m candles."""
    if len(frame) < 289:
        return _ScalpMarketContext()
    close = frame["close"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    volume = frame["volume"].astype("float64")
    latest = len(frame) - 1
    try:
        price = float(close.iloc[latest])
        prior = float(close.iloc[latest - 288])
    except (IndexError, TypeError, ValueError):
        return _ScalpMarketContext()
    if not (math.isfinite(price) and math.isfinite(prior) and prior > 0.0):
        return _ScalpMarketContext()
    return_24h_pct = (price / prior - 1.0) * 100.0
    volume_1h_ratio: float | None = None
    taker_buy_ratio_1h: float | None = None
    # A complete current hour plus 24 preceding, non-overlapping hourly
    # windows require 300 closed 5m bars.  If unavailable, omit the field.
    if len(frame) >= 300:
        current_1h = float(volume.iloc[-12:].sum())
        prior_hours = []
        for offset in range(1, 25):
            end = len(frame) - 12 * offset
            start = end - 12
            prior_hours.append(float(volume.iloc[start:end].sum()))
        baseline = float(median(prior_hours)) if prior_hours else 0.0
        if math.isfinite(current_1h) and math.isfinite(baseline) and baseline > 0.0:
            volume_1h_ratio = current_1h / baseline
    if "taker_buy_base" in frame.columns:
        taker_buy = frame["taker_buy_base"].astype("float64").iloc[-12:].sum()
        recent_volume = volume.iloc[-12:].sum()
        if math.isfinite(float(taker_buy)) and recent_volume > 0.0:
            taker_buy_ratio_1h = min(max(float(taker_buy / recent_volume), 0.0), 1.0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    recent_range_bps = (true_range / close.replace(0.0, np.nan) * 10_000.0).iloc[-24:]
    finite_ranges = [float(value) for value in recent_range_bps if math.isfinite(float(value))]
    volatility_bps = float(median(finite_ranges)) if finite_ranges else None
    return _ScalpMarketContext(
        return_24h_pct=return_24h_pct,
        volume_1h_ratio=volume_1h_ratio,
        taker_buy_ratio_1h=taker_buy_ratio_1h,
        volatility_bps=volatility_bps,
    )


def _rank_market_contexts(
    frames: dict[str, pd.DataFrame],
) -> dict[str, _ScalpMarketContext]:
    contexts = {symbol: _closed_market_context(frame) for symbol, frame in frames.items()}
    ranked = sorted(
        (
            (symbol, context.return_24h_pct)
            for symbol, context in contexts.items()
            if context.return_24h_pct is not None and math.isfinite(context.return_24h_pct)
        ),
        key=lambda item: (-float(item[1]), item[0]),
    )
    universe_size = len(ranked)
    for rank, (symbol, _return) in enumerate(ranked, start=1):
        contexts[symbol] = replace(
            contexts[symbol], rank_24h=rank, universe_size=universe_size
        )
    return contexts


def _relative_strength_observations(
    entries: tuple[UniverseEntry, ...],
    frames: dict[str, pd.DataFrame],
    *,
    regime: BullRegime,
    snapshots: dict[str, FuturesMarketSnapshot],
    settings: Settings,
    universe_version: str,
    historical_cost_bps: float,
    contexts: dict[str, _ScalpMarketContext] | None = None,
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
                context=(contexts or {}).get(symbol),
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
    context: _ScalpMarketContext | None = None,
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
        return_24h_pct=(context.return_24h_pct if context is not None else None),
        rank_24h=(context.rank_24h if context is not None else None),
        universe_size=(context.universe_size if context is not None else None),
        volume_1h_ratio=(context.volume_1h_ratio if context is not None else None),
        mark_price=(snapshot.mark_price if snapshot is not None else None),
        taker_buy_ratio_1h=(
            context.taker_buy_ratio_1h if context is not None else None
        ),
        volatility_bps=(context.volatility_bps if context is not None else None),
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
                "return_24h_pct": record.get("return_24h_pct"),
                "rank_24h": record.get("rank_24h"),
                "universe_size": record.get("universe_size"),
                "volume_1h_ratio": record.get("volume_1h_ratio"),
                "mark_price": record.get("mark_price"),
                "taker_buy_ratio_1h": record.get("taker_buy_ratio_1h"),
                "volatility_bps": record.get("volatility_bps"),
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


def _target_pending_dir(state_dir: Path) -> Path:
    return state_dir / "target_pending"


def _bracket_pending_dir(state_dir: Path) -> Path:
    return state_dir / "bracket_pending"


def _scalp_setup_id(item: ScalpObservation, universe_version: str) -> str:
    material = f"{universe_version}|{item.perpetual_symbol}|setup|{item.bar_close_time_ms}"
    return sha256(material.encode("ascii")).hexdigest()


def _scalp_target_window(
    frame: pd.DataFrame, after_ms: int, until_ms: int
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"close_time_ms", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    selected = frame[
        (frame["close_time_ms"] > after_ms) & (frame["close_time_ms"] <= until_ms)
    ]
    return selected.loc[:, ["close_time_ms", "high", "low", "close"]]


def _scalp_bracket_window(
    frame: pd.DataFrame, after_ms: int, until_ms: int
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"close_time_ms", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    selected = frame[
        (frame["close_time_ms"] > after_ms) & (frame["close_time_ms"] <= until_ms)
    ]
    return selected.loc[:, ["close_time_ms", "open", "high", "low", "close"]]


def _first_touch_bracket(
    record: dict[str, Any],
    window: pd.DataFrame,
    *,
    deadline_reached: bool,
) -> dict[str, Any] | None:
    if window.empty:
        return None
    entry_price = float(window.iloc[0]["open"])
    direction = str(record["direction"])
    long_side = direction == "YUKARI"
    target_bps = float(record["target_bps"])
    stop_bps = float(record["stop_bps"])
    target_price = entry_price * (
        1.0 + target_bps / 10_000.0 if long_side else 1.0 - target_bps / 10_000.0
    )
    stop_price = entry_price * (
        1.0 - stop_bps / 10_000.0 if long_side else 1.0 + stop_bps / 10_000.0
    )
    resolution: str | None = None
    ambiguous = False
    exit_price: float | None = None
    exit_time_ms: int | None = None
    exit_position = len(window) - 1
    for position, (close_time, _open, high, low, _close) in enumerate(
        window.itertuples(index=False)
    ):
        high_value = float(high)
        low_value = float(low)
        target_hit = high_value >= target_price if long_side else low_value <= target_price
        stop_hit = low_value <= stop_price if long_side else high_value >= stop_price
        if target_hit and stop_hit:
            # Five-minute bars do not reveal intrabar ordering.  Counting the
            # tie as STOP avoids flattering the strategy with unknowable wins.
            resolution = "STOP"
            ambiguous = True
            exit_price = stop_price
        elif target_hit:
            resolution = "TARGET"
            exit_price = target_price
        elif stop_hit:
            resolution = "STOP"
            exit_price = stop_price
        if resolution is not None:
            exit_time_ms = int(close_time)
            exit_position = position
            break
    if resolution is None:
        if not deadline_reached:
            return None
        resolution = "TIME_EXIT"
        exit_price = float(window.iloc[-1]["close"])
        exit_time_ms = int(window.iloc[-1]["close_time_ms"])
    assert exit_price is not None and exit_time_ms is not None
    gross_bps = (
        math.log(exit_price / entry_price) * 10_000.0
        if long_side
        else math.log(entry_price / exit_price) * 10_000.0
    )
    realised_window = window.iloc[: exit_position + 1]
    highs = realised_window["high"].astype("float64")
    lows = realised_window["low"].astype("float64")
    if long_side:
        mfe_bps = max(math.log(float(highs.max()) / entry_price) * 10_000.0, 0.0)
        mae_bps = max(math.log(entry_price / float(lows.min())) * 10_000.0, 0.0)
    else:
        mfe_bps = max(math.log(entry_price / float(lows.min())) * 10_000.0, 0.0)
        mae_bps = max(math.log(float(highs.max()) / entry_price) * 10_000.0, 0.0)
    cost = float(record["round_trip_cost_bps"])
    return {
        "schema": SCALP_BRACKET_LEDGER_SCHEMA,
        **{key: value for key, value in record.items() if key != "schema"},
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "exit_price": exit_price,
        "exit_time_ms": exit_time_ms,
        "resolution": resolution,
        "success": resolution == "TARGET",
        "ambiguous_same_bar": ambiguous,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - cost,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "elapsed_minutes": max(
            0.0, (exit_time_ms - int(record["bar_close_time_ms"])) / 60_000.0
        ),
    }


def _scalp_target_touches(
    record: dict[str, Any], window: pd.DataFrame
) -> dict[float, dict[str, Any]]:
    if window.empty:
        return {}
    source_price = float(record["source_price"])
    long_side = str(record["direction"]) == "YUKARI"
    result: dict[float, dict[str, Any]] = {}
    for close_time, high, low, close in window.itertuples(index=False):
        extreme = float(high) if long_side else float(low)
        for percent in SCALP_TARGET_TOUCH_PERCENTS:
            if percent in result:
                continue
            multiplier = 1.0 + percent / 100.0 if long_side else 1.0 - percent / 100.0
            target_price = source_price * multiplier
            reached = extreme >= target_price if long_side else extreme <= target_price
            if reached:
                result[percent] = {
                    "target_price": target_price,
                    "touch_price": extreme,
                    "touch_close_time_ms": int(close_time),
                    "touch_close": float(close),
                }
    return result


def _read_scalp_target_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    required = (
        "setup_id",
        "spot_symbol",
        "perpetual_symbol",
        "direction",
        "source_price",
        "bar_close_time_ms",
        "horizon_ms",
    )
    if not isinstance(payload, dict) or payload.get("schema") != SCALP_TARGET_PENDING_SCHEMA:
        return None
    if any(key not in payload for key in required):
        return None
    try:
        source_price = float(payload["source_price"])
        source_ms = int(payload["bar_close_time_ms"])
        horizon_ms = int(payload["horizon_ms"])
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(source_price)
        or source_price <= 0.0
        or source_ms < 0
        or horizon_ms <= 0
        or str(payload["direction"]) not in {"YUKARI", "AŞAĞI"}
    ):
        return None
    delivered = payload.get("delivered_percents", [])
    if not isinstance(delivered, list):
        return None
    try:
        payload["delivered_percents"] = sorted(
            {
                float(value)
                for value in delivered
                if float(value) in SCALP_TARGET_TOUCH_PERCENTS
            }
        )
    except (TypeError, ValueError):
        return None
    recorded = payload.get("outcome_recorded_percents", [])
    if not isinstance(recorded, list):
        return None
    try:
        payload["outcome_recorded_percents"] = sorted(
            {
                float(value)
                for value in recorded
                if float(value) in SCALP_TARGET_TOUCH_PERCENTS
            }
        )
    except (TypeError, ValueError):
        return None
    return payload


def _read_scalp_bracket_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    required = (
        "setup_id",
        "spot_symbol",
        "perpetual_symbol",
        "direction",
        "source_price",
        "bar_close_time_ms",
        "horizon_minutes",
        "target_bps",
        "stop_bps",
        "round_trip_cost_bps",
    )
    if not isinstance(payload, dict) or payload.get("schema") != SCALP_BRACKET_PENDING_SCHEMA:
        return None
    if any(key not in payload for key in required):
        return None
    try:
        source_price = float(payload["source_price"])
        source_ms = int(payload["bar_close_time_ms"])
        horizon = int(payload["horizon_minutes"])
        target = float(payload["target_bps"])
        stop = float(payload["stop_bps"])
        cost = float(payload["round_trip_cost_bps"])
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (source_price, target, stop, cost))
        or source_price <= 0.0
        or source_ms < 0
        or horizon <= 0
        or target <= 0.0
        or stop <= 0.0
        or cost < 0.0
        or str(payload["direction"]) not in {"YUKARI", "AŞAĞI"}
    ):
        return None
    return payload


def _format_probability_map(value: Any, horizons: tuple[int, ...]) -> str:
    mapping = value if isinstance(value, dict) else {}
    return "/".join(
        f"%{float(mapping[str(horizon)]) * 100:.0f}"
        if str(horizon) in mapping
        else "-"
        for horizon in horizons
    )


def _ledger_path(state_dir: Path) -> Path:
    return state_dir / "ledger.jsonl"


def _target_ledger_path(state_dir: Path) -> Path:
    return state_dir / "target_ledger.jsonl"


def _bracket_ledger_path(state_dir: Path) -> Path:
    return state_dir / "bracket_ledger.jsonl"


def _append_scalp_ledger(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _ledger_path(state_dir).open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_scalp_target_ledger(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _target_ledger_path(state_dir).open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_scalp_bracket_ledger(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _bracket_ledger_path(state_dir).open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
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


def _aggregate_brackets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "winRate": 0.0,
            "meanNetBps": 0.0,
            "stops": 0,
            "timeouts": 0,
        }
    return {
        "count": len(rows),
        "winRate": sum(row.get("resolution") == "TARGET" for row in rows) / len(rows),
        "meanNetBps": sum(float(row.get("net_bps", 0.0)) for row in rows) / len(rows),
        "stops": sum(row.get("resolution") == "STOP" for row in rows),
        "timeouts": sum(row.get("resolution") == "TIME_EXIT" for row in rows),
    }


__all__ = [
    "FAMILY_EVIDENCE",
    "FAMILY_LABELS",
    "SCALP_INTERVAL",
    "SCALP_BRACKET_LEDGER_SCHEMA",
    "SCALP_TARGET_LEDGER_SCHEMA",
    "SCALP_TARGET_TOUCH_PERCENTS",
    "BullRegime",
    "ScalpObservation",
    "ScalpScanReport",
    "ScalpSetupAssessment",
    "deliver_scalp_bracket_wins",
    "deliver_scalp_observations",
    "deliver_scalp_target_touches",
    "evaluate_bull_regime",
    "filter_scalp_notification_report",
    "format_scalp_observation_digest",
    "format_scalp_bracket_result",
    "format_scalp_scorecard",
    "format_scalp_target_touch",
    "load_pending_scalp_targets",
    "load_pending_scalp_brackets",
    "load_scalp_bracket_ledger",
    "load_scalp_ledger",
    "load_scalp_target_ledger",
    "mark_scalp_target_touch_delivered",
    "pending_scalp_target_touches",
    "record_scalp_observations",
    "record_scalp_bracket_setups",
    "record_scalp_target_setups",
    "refresh_and_scan_scalp_universe",
    "scalp_cache_path",
    "scalp_forecast_stats",
    "scalp_scorecard",
    "scalp_setup_direction",
    "scalp_setup_assessment",
    "scalp_setup_forecast_stats",
    "scan_cached_scalp_universe",
    "scan_scalp_frame",
    "settle_scalp_observations",
    "settle_scalp_bracket_outcomes",
    "settle_scalp_target_outcomes",
]
