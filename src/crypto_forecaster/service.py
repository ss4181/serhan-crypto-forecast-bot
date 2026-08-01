from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .config import (
    INTERVAL_LABELS,
    INTERVAL_MILLISECONDS,
    INTERVALS,
    SYMBOLS,
    Settings,
    cache_path,
    model_path,
)
from .data import BinanceMarketDataClient, load_cache, update_cache
from .features import FEATURE_LABELS_TR, FEATURE_NAMES, latest_feature_vector
from .model import BacktestMetrics, ModelBundle, load_bundle, select_scenario
from .research import research_all
from .telegram import TelegramDelivery, TelegramNotifier


@dataclass(frozen=True, slots=True)
class IndicatorContribution:
    name: str
    display_value: str
    direction_effect: str
    strength: float


@dataclass(frozen=True, slots=True)
class Prediction:
    symbol: str
    interval: str
    source_open_time_ms: int
    source_close_time_ms: int
    source_price: float
    atr: float
    probability_up: float
    probability_down: float
    direction: str
    confidence: float
    target_up_price: float
    target_down_price: float
    target_up_touch_probability: float
    target_down_touch_probability: float
    close_range_low: float
    close_range_median: float
    close_range_high: float
    scenario_count: int
    indicators: tuple[IndicatorContribution, ...]
    backtest: BacktestMetrics
    eligible: bool
    ineligible_reasons: tuple[str, ...]

    @property
    def signal_id(self) -> str:
        payload = (
            f"{self.symbol}|{self.interval}|{self.source_close_time_ms}|{self.direction}"
        ).encode("ascii")
        return sha256(payload).hexdigest()


def make_prediction(
    settings: Settings,
    symbol: str,
    interval: str,
    *,
    now: datetime | None = None,
) -> Prediction:
    bars = load_cache(cache_path(settings.data_dir, symbol, interval))
    bundle = load_bundle(model_path(settings.model_dir, symbol, interval))
    if bundle.symbol != symbol or bundle.interval != interval:
        raise ValueError("Model sembol/zaman dilimi uyusmuyor")
    row, vector = latest_feature_vector(bars)
    probability_up = bundle.predict_up_probability(vector)
    probability_down = 1.0 - probability_up
    direction = "YUKARI" if probability_up >= 0.5 else "ASAGI"
    confidence = max(probability_up, probability_down)
    scenario = select_scenario(
        bundle.scenarios,
        probability_up,
        float(row["atr_pct"]),
        minimum_count=settings.scenario_minimum_count,
    )
    price = float(row["close"])
    atr = float(row["atr"])
    target_multiple = float(bundle.scenarios["target_atr_multiple"])
    close_low = price * math.exp(float(scenario["close_return_atr_p10"]) * atr / price)
    close_median = price * math.exp(float(scenario["close_return_atr_p50"]) * atr / price)
    close_high = price * math.exp(float(scenario["close_return_atr_p90"]) * atr / price)
    indicators = _indicator_contributions(bundle, vector, direction)
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    latest_close_ms = int(row["close_time_ms"])
    reasons: list[str] = []
    if not bundle.backtest.passed_research_gate:
        reasons.append("model walk-forward arastirma kapisini gecemedi")
    if confidence < bundle.backtest.signal_threshold:
        reasons.append(
            f"yon olasiligi %{bundle.backtest.signal_threshold * 100:.0f} esiginin altinda"
        )
    if current_ms - latest_close_ms > 3 * INTERVAL_MILLISECONDS[interval]:
        reasons.append("son kapali mum bayat")
    maximum_model_age_ms = settings.maximum_model_age_days * 24 * 60 * 60 * 1000
    if latest_close_ms - bundle.training_last_close_ms > maximum_model_age_ms:
        reasons.append("model yeniden arastirilacak kadar eski")
    return Prediction(
        symbol=symbol,
        interval=interval,
        source_open_time_ms=int(row["open_time_ms"]),
        source_close_time_ms=latest_close_ms,
        source_price=price,
        atr=atr,
        probability_up=probability_up,
        probability_down=probability_down,
        direction=direction,
        confidence=confidence,
        target_up_price=price + target_multiple * atr,
        target_down_price=max(0.0, price - target_multiple * atr),
        target_up_touch_probability=float(scenario["touch_up_half_atr_probability"]),
        target_down_touch_probability=float(scenario["touch_down_half_atr_probability"]),
        close_range_low=close_low,
        close_range_median=close_median,
        close_range_high=close_high,
        scenario_count=int(scenario["count"]),
        indicators=indicators,
        backtest=bundle.backtest,
        eligible=not reasons,
        ineligible_reasons=tuple(reasons),
    )


def format_prediction(prediction: Prediction) -> str:
    icon = "🟢" if prediction.direction == "YUKARI" else "🔴"
    source_close = _utc_text(prediction.source_close_time_ms)
    target_close_ms = prediction.source_close_time_ms + INTERVAL_MILLISECONDS[prediction.interval]
    target_close = _utc_text(target_close_ms)
    metrics = prediction.backtest
    indicator_lines = [
        f"• {item.name}: {item.display_value} — {item.direction_effect}"
        for item in prediction.indicators
    ]
    status = (
        "BILDIRIM UYGUN"
        if prediction.eligible
        else "BILDIRIM YOK: " + "; ".join(prediction.ineligible_reasons)
    )
    message = "\n".join(
        [
            f"{icon} {prediction.symbol} | {INTERVAL_LABELS[prediction.interval]} | {prediction.direction}",
            "",
            f"Sinyal zamani: {source_close}",
            f"Tahmin edilen kapanis: {target_close}",
            f"Referans fiyat (son kapali mum): ${prediction.source_price:,.2f}",
            f"Yon olasiligi: YUKARI %{prediction.probability_up * 100:.1f} | ASAGI %{prediction.probability_down * 100:.1f}",
            f"Durum: {status}",
            "",
            "FIYAT SENARYOLARI (benzer kalibre edilmis gecmis durumlar)",
            f"• ${prediction.target_up_price:,.2f} (+0.5 ATR) gorulme: %{prediction.target_up_touch_probability * 100:.1f}",
            f"• ${prediction.target_down_price:,.2f} (-0.5 ATR) gorulme: %{prediction.target_down_touch_probability * 100:.1f}",
            f"• Kapanis icin %80 aralik: ${prediction.close_range_low:,.2f} – ${prediction.close_range_high:,.2f}",
            f"• Senaryo medyan kapanisi: ${prediction.close_range_median:,.2f} (benzer n={prediction.scenario_count})",
            "",
            "WALK-FORWARD BACKTEST (tamamen OOS)",
            f"• Yuksek guven sinyali: %{metrics.signal_accuracy * 100:.1f} dogru "
            f"(%95 GA %{metrics.signal_ci95_low * 100:.1f}–%{metrics.signal_ci95_high * 100:.1f}, n={metrics.signal_count})",
            f"• 6 model icin aile-duzeltilmis %95 GA: %{metrics.signal_familywise_ci95_low * 100:.1f}–%{metrics.signal_familywise_ci95_high * 100:.1f}",
            f"• Tum mum yon dogrulugu: %{metrics.accuracy * 100:.1f} | taban: %{metrics.baseline_accuracy * 100:.1f}",
            f"• Sinyal kapsami: %{metrics.signal_coverage * 100:.1f} | Brier: {metrics.brier_score:.4f} | ECE: %{metrics.expected_calibration_error * 100:.1f}",
            "",
            "SINYALI EN COK ETKILEYEN BELIRTECLER",
            *indicator_lines,
            "",
            "Yalnizca arastirma bildirimidir; yatirim tavsiyesi veya emir degildir. Olasiliklar garanti degildir.",
        ]
    )
    if len(message) > 4096:
        raise ValueError("Telegram mesaji 4096 karakteri asti")
    return message


def evaluate_all(settings: Settings) -> list[Prediction]:
    return [
        make_prediction(settings, symbol, interval)
        for symbol in SYMBOLS
        for interval in INTERVALS
    ]


def dashboard_snapshot(predictions: list[Prediction]) -> dict[str, object]:
    if not predictions:
        raise ValueError("Panel ozeti icin tahmin yok")
    observed_ms = max(item.source_close_time_ms for item in predictions)
    observed_at = datetime.fromtimestamp(observed_ms / 1000, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    prediction_rows: list[dict[str, object]] = []
    verified_rows: list[dict[str, object]] = []
    for item in predictions:
        row: dict[str, object] = {
            "symbol": item.symbol,
            "interval": item.interval,
            "intervalLabel": INTERVAL_LABELS[item.interval],
            "direction": "YUKARI" if item.direction == "YUKARI" else "AŞAĞI",
            "confidence": item.confidence,
            "sourcePrice": item.source_price,
            "eligible": item.eligible,
        }
        if item.eligible:
            row["signalId"] = item.signal_id
        prediction_rows.append(row)
        if item.backtest.passed_research_gate:
            verified_rows.append(
                {
                    "symbol": item.symbol,
                    "interval": item.interval,
                    "intervalLabel": INTERVAL_LABELS[item.interval],
                    "signalAccuracy": item.backtest.signal_accuracy,
                    "signalCount": item.backtest.signal_count,
                    "familyCiLow": item.backtest.signal_familywise_ci95_low,
                    "familyCiHigh": item.backtest.signal_familywise_ci95_high,
                    "coverage": item.backtest.signal_coverage,
                }
            )
    return {
        "schema": "serhan-lab-snapshot-v1",
        "observedAtUtc": observed_at,
        "predictions": prediction_rows,
        "verifiedModels": verified_rows,
    }


def deliver_eligible(
    settings: Settings,
    predictions: list[Prediction],
    *,
    notifier: TelegramNotifier | None = None,
) -> list[tuple[Prediction, TelegramDelivery]]:
    eligible_predictions = [prediction for prediction in predictions if prediction.eligible]
    if not eligible_predictions:
        return []
    client = notifier or TelegramNotifier()
    deliveries: list[tuple[Prediction, TelegramDelivery]] = []
    for prediction in eligible_predictions:
        delivery = client.deliver_once(
            signal_id=prediction.signal_id,
            text=format_prediction(prediction),
            state_dir=settings.telegram_state_dir,
        )
        deliveries.append((prediction, delivery))
    return deliveries


def serve_forever(
    settings: Settings,
    *,
    days: int,
    poll_seconds: int,
    progress: Callable[[str], None] = print,
) -> None:
    if poll_seconds < 20:
        raise ValueError("Tarama araligi en az 20 saniye olmali")
    client = BinanceMarketDataClient()
    last_research_day: str | None = None
    while True:
        now = datetime.now(timezone.utc)
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                update_cache(
                    settings.data_dir,
                    symbol,
                    interval,
                    days=days,
                    client=client,
                    now=now,
                )
        today = now.date().isoformat()
        models_missing = any(
            not model_path(settings.model_dir, symbol, interval).exists()
            for symbol in SYMBOLS
            for interval in INTERVALS
        )
        if models_missing or last_research_day != today:
            progress("Gunluk walk-forward arastirma ve model yenileme basladi")
            research_all(settings, progress=progress)
            last_research_day = today
        predictions = evaluate_all(settings)
        deliveries = deliver_eligible(settings, predictions)
        for prediction, delivery in deliveries:
            progress(
                f"{prediction.symbol} {prediction.interval} {prediction.direction}: {delivery.status}"
            )
        time.sleep(poll_seconds)


def _indicator_contributions(
    bundle: ModelBundle, vector: np.ndarray, direction: str
) -> tuple[IndicatorContribution, ...]:
    standardized = bundle.standardizer.transform(vector.reshape(1, -1))[0]
    contributions = standardized * bundle.model.coefficients
    direction_sign = 1.0 if direction == "YUKARI" else -1.0
    aligned = direction_sign * contributions
    indices = np.argsort(-np.abs(aligned))[:4]
    result: list[IndicatorContribution] = []
    for index in indices:
        name = FEATURE_NAMES[int(index)]
        supports = aligned[int(index)] >= 0
        result.append(
            IndicatorContribution(
                name=FEATURE_LABELS_TR[name],
                display_value=_display_feature_value(name, float(vector[int(index)])),
                direction_effect=(
                    f"{direction.lower()} yonunu destekliyor"
                    if supports
                    else f"{direction.lower()} yonunu zayiflatiyor"
                ),
                strength=float(abs(aligned[int(index)])),
            )
        )
    return tuple(result)


def _display_feature_value(name: str, value: float) -> str:
    if name == "rsi_14_centered":
        return f"{value * 25 + 50:.1f}"
    if name == "atr_pct_14_log":
        return f"%{math.exp(value) * 100:.2f}"
    if name == "volume_z_20":
        return f"{value:+.2f} z"
    if name == "bollinger_z_20":
        return f"{value:+.2f} sigma"
    if name == "ema_8_21_atr":
        return f"{value:+.2f} ATR"
    if name == "candle_pressure":
        return f"{value:+.2f}"
    return f"{value:+.2f} ATR"


def _utc_text(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


__all__ = [
    "Prediction",
    "deliver_eligible",
    "dashboard_snapshot",
    "evaluate_all",
    "format_prediction",
    "make_prediction",
    "serve_forever",
]
