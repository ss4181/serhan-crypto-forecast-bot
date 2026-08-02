from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
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
from .commands import CommandOutcome, poll_and_answer
from .data import BinanceMarketDataClient, load_cache, update_cache
from .features import FEATURE_LABELS_TR, FEATURE_NAMES, latest_feature_vector
from .hub import hub_configured, post_snapshot, write_snapshot
from .model import BacktestMetrics, ModelBundle, load_bundle, select_scenario
from .openinterest import OpenInterestError, update_open_interest
from .outcomes import (
    format_scorecard,
    load_ledger,
    record_delivery,
    scorecard,
    settle_pending,
)
from .research import research_all
from .telegram import TelegramDelivery, TelegramNotifier, digest_signal_id, is_primary


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
    target_close_time_ms: int
    evaluated_at_ms: int
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
    touch_both_probability: float
    touch_neither_probability: float
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

    @property
    def tier(self) -> str:
        """ISLEM only when the model's measured edge survives trading costs."""
        return "ISLEM" if self.eligible else "GOZLEM"


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
    interval_ms = INTERVAL_MILLISECONDS[interval]
    target_close_ms = latest_close_ms + interval_ms
    remaining_ms = target_close_ms - current_ms
    reasons: list[str] = []
    if not bundle.backtest.passed_research_gate:
        reasons.append("model walk-forward arastirma kapisini gecemedi")
    if confidence < bundle.backtest.signal_threshold:
        reasons.append(
            f"yon olasiligi %{bundle.backtest.signal_threshold * 100:.0f} esiginin altinda"
        )
    # A forecast about a candle that is nearly over cannot be acted on, and one
    # about a candle that already closed is not a forecast at all.
    if remaining_ms < settings.minimum_remaining_fraction * interval_ms:
        if remaining_ms <= 0:
            reasons.append("hedef mum zaten kapandi")
        else:
            reasons.append(
                f"hedef mumun yalnizca {remaining_ms / 60000:.1f} dakikasi kaldi; "
                f"en az %{settings.minimum_remaining_fraction * 100:.0f} kalmali"
            )
    maximum_model_age_ms = settings.maximum_model_age_days * 24 * 60 * 60 * 1000
    if latest_close_ms - bundle.training_last_close_ms > maximum_model_age_ms:
        reasons.append("model yeniden arastirilacak kadar eski")
    return Prediction(
        symbol=symbol,
        interval=interval,
        source_open_time_ms=int(row["open_time_ms"]),
        source_close_time_ms=latest_close_ms,
        target_close_time_ms=target_close_ms,
        evaluated_at_ms=current_ms,
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
        touch_both_probability=float(scenario.get("touch_both_probability", 0.0)),
        touch_neither_probability=float(scenario.get("touch_neither_probability", 0.0)),
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
    target_close = _utc_text(prediction.target_close_time_ms)
    remaining_minutes = max(
        0.0, (prediction.target_close_time_ms - prediction.evaluated_at_ms) / 60000
    )
    metrics = prediction.backtest
    indicator_lines = [
        f"• {item.name}: {item.display_value} — {item.direction_effect}"
        for item in prediction.indicators
    ]
    if prediction.eligible:
        header = f"{icon} ISLEM ADAYI"
        status = "Maliyet sonrasi olculmus pozitif beklentisi olan tek tier."
    else:
        header = "🔎 GOZLEM"
        status = "ISLEM ADAYI DEGIL: " + "; ".join(prediction.ineligible_reasons)
    message = "\n".join(
        [
            f"{header} | {prediction.symbol} | {INTERVAL_LABELS[prediction.interval]} | {prediction.direction}",
            "",
            f"Sinyal zamani: {source_close}",
            f"Tahmin edilen kapanis: {target_close} ({remaining_minutes:.1f} dk kaldi)",
            f"Referans fiyat (son kapali mum): ${prediction.source_price:,.2f}",
            f"Yon olasiligi: YUKARI %{prediction.probability_up * 100:.1f} | ASAGI %{prediction.probability_down * 100:.1f}",
            f"Durum: {status}",
            "",
            "HEDEF TANIMI (uclu bariyer)",
            f"• Yon = fiyatin once hangi tarafa ±{metrics.barrier_bps_median:.0f} bps "
            f"hareket ettigi; en fazla {metrics.barrier_horizon_candles} mum beklenir",
            f"• Bu hedef {metrics.round_trip_cost_bps:.1f} bps gidis-donus maliyetinin "
            "en az iki kati secilir, yani kazanan islem masrafini fazlasiyla karsilar",
            f"• Gecmiste sinyallerin %{metrics.resolved_fraction * 100:.0f}'i bariyere ulasti; "
            f"kalani sure dolunca piyasadan kapandi",
            "",
            "MALIYET SONRASI BEKLENTI (bu tahminin tek gecerli olcusu)",
            f"• Olculen net beklenti: {metrics.net_edge_bps:+.2f} bps/sinyal "
            f"({metrics.round_trip_cost_bps:.1f} bps gidis-donus maliyeti dusulmus)",
            f"• Gun bloklu %95 aralik: {metrics.net_edge_ci95_low:+.2f} – {metrics.net_edge_ci95_high:+.2f} bps",
            f"• Ortalama kazanc {metrics.average_win_bps:+.1f} bps / ortalama kayip "
            f"{metrics.average_loss_bps:+.1f} bps",
            "",
            "FIYAT SENARYOLARI (benzer kalibre edilmis gecmis durumlar)",
            f"• ${prediction.target_up_price:,.2f} (+0.5 ATR) gorulme: %{prediction.target_up_touch_probability * 100:.1f}",
            f"• ${prediction.target_down_price:,.2f} (-0.5 ATR) gorulme: %{prediction.target_down_touch_probability * 100:.1f}",
            f"• Ikisi de ayni mumda gorulur: %{prediction.touch_both_probability * 100:.1f} — "
            "hangisinin once geldigi mum verisinden bilinemez, bu bir hedef/stop cifti degildir",
            f"• Kapanis icin %80 aralik: ${prediction.close_range_low:,.2f} – ${prediction.close_range_high:,.2f}",
            f"• Senaryo medyan kapanisi: ${prediction.close_range_median:,.2f} (benzer n={prediction.scenario_count})",
            "",
            "WALK-FORWARD BACKTEST (tamamen OOS)",
            f"• Yuksek guven yon isabeti: %{metrics.signal_accuracy * 100:.1f} "
            f"(n={metrics.signal_count}, {metrics.signal_days} ayri gun)",
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


def format_observation_digest(
    predictions: list[Prediction], *, now: datetime | None = None
) -> str:
    """One message covering every model, including the ones that cannot trade.

    Keeps all six models visible in Telegram without dressing a
    negative-expectancy forecast up as something actionable.
    """
    if not predictions:
        raise ValueError("Gozlem raporu icin tahmin yok")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    tradeable = [item for item in predictions if item.eligible]
    lines = [
        f"🔎 GOZLEM RAPORU | {len(predictions)} model | {stamp}",
        "",
        "Her modelin o anki durumu. ISLEM ADAYI olmayanlar da burada gorunur ki",
        "hicbir modelin sessiz kalmadigi dogrulanabilsin.",
        "",
        f"Islem adayi: {len(tradeable)} / {len(predictions)}",
        "",
    ]
    for item in predictions:
        metrics = item.backtest
        mark = "🟢" if item.eligible else "▫️"
        lines.append(
            f"{mark} {item.symbol} {INTERVAL_LABELS[item.interval]} — {item.direction} "
            f"%{item.confidence * 100:.1f} | ${item.source_price:,.2f}"
        )
        lines.append(
            f"    net beklenti {metrics.net_edge_bps:+.2f} bps "
            f"({metrics.net_edge_ci95_low:+.1f} / {metrics.net_edge_ci95_high:+.1f}), "
            f"isabet %{metrics.signal_accuracy * 100:.1f} (n={metrics.signal_count})"
        )
        if not item.eligible:
            lines.append(f"    engel: {_first_reason(item.ineligible_reasons)}")
    lines.extend(
        [
            "",
            "Yalnizca arastirma bildirimidir; yatirim tavsiyesi veya emir degildir.",
        ]
    )
    message = "\n".join(lines)
    if len(message) > 4096:
        raise ValueError("Telegram mesaji 4096 karakteri asti")
    return message


def _first_reason(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "-"
    reason = reasons[0]
    return reason if len(reason) <= 150 else reason[:147] + "..."


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
        metrics = item.backtest
        row: dict[str, object] = {
            "symbol": item.symbol,
            "interval": item.interval,
            "intervalLabel": INTERVAL_LABELS[item.interval],
            "direction": "YUKARI" if item.direction == "YUKARI" else "AŞAĞI",
            "tier": item.tier,
            "confidence": item.confidence,
            "probabilityUp": item.probability_up,
            "probabilityDown": item.probability_down,
            "sourcePrice": item.source_price,
            "sourceCloseAtUtc": _iso_utc(item.source_close_time_ms),
            "targetCloseAtUtc": _iso_utc(
                item.source_close_time_ms + INTERVAL_MILLISECONDS[item.interval]
            ),
            "targetUpPrice": item.target_up_price,
            "targetDownPrice": item.target_down_price,
            "targetUpTouchProbability": item.target_up_touch_probability,
            "targetDownTouchProbability": item.target_down_touch_probability,
            "touchBothProbability": item.touch_both_probability,
            "touchNeitherProbability": item.touch_neither_probability,
            "closeRangeLow": item.close_range_low,
            "closeRangeMedian": item.close_range_median,
            "closeRangeHigh": item.close_range_high,
            "scenarioCount": item.scenario_count,
            "eligible": item.eligible,
            "ineligibleReasons": list(item.ineligible_reasons),
            "indicators": [
                {
                    "name": indicator.name,
                    "displayValue": indicator.display_value,
                    "directionEffect": indicator.direction_effect,
                    "strength": indicator.strength,
                }
                for indicator in item.indicators
            ],
            "backtest": {
                "passedGate": metrics.passed_research_gate,
                "gateReasons": list(metrics.gate_reasons),
                "folds": metrics.folds,
                "sampleCount": metrics.sample_count,
                "accuracy": metrics.accuracy,
                "baselineAccuracy": metrics.baseline_accuracy,
                "brierScore": metrics.brier_score,
                "expectedCalibrationError": metrics.expected_calibration_error,
                "signalThreshold": metrics.signal_threshold,
                "signalCount": metrics.signal_count,
                "signalCoverage": metrics.signal_coverage,
                "signalAccuracy": metrics.signal_accuracy,
                "signalCiLow": metrics.signal_ci95_low,
                "signalCiHigh": metrics.signal_ci95_high,
                "familyCiLow": metrics.signal_familywise_ci95_low,
                "familyCiHigh": metrics.signal_familywise_ci95_high,
                "roundTripCostBps": metrics.round_trip_cost_bps,
                "grossEdgeBps": metrics.gross_edge_bps,
                "netEdgeBps": metrics.net_edge_bps,
                "netEdgeCiLow": metrics.net_edge_ci95_low,
                "netEdgeCiHigh": metrics.net_edge_ci95_high,
                "averageWinBps": metrics.average_win_bps,
                "averageLossBps": metrics.average_loss_bps,
                "signalDays": metrics.signal_days,
            },
        }
        if item.eligible:
            row["signalId"] = item.signal_id
        prediction_rows.append(row)
        if metrics.passed_research_gate:
            verified_rows.append(
                {
                    "symbol": item.symbol,
                    "interval": item.interval,
                    "intervalLabel": INTERVAL_LABELS[item.interval],
                    "signalAccuracy": metrics.signal_accuracy,
                    "signalCount": metrics.signal_count,
                    "familyCiLow": metrics.signal_familywise_ci95_low,
                    "familyCiHigh": metrics.signal_familywise_ci95_high,
                    "coverage": metrics.signal_coverage,
                }
            )
    return {
        "schema": "serhan-lab-snapshot-v2",
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
        if delivery.status == "SENT":
            _record(settings, prediction)
        deliveries.append((prediction, delivery))
    return deliveries


def deliver_observation_digest(
    settings: Settings,
    predictions: list[Prediction],
    *,
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> TelegramDelivery | None:
    """Publish every model on a fixed cadence, deduplicated per time bucket."""
    if not predictions or settings.observation_digest_hours <= 0:
        return None
    current = now or datetime.now(timezone.utc)
    bucket_ms = settings.observation_digest_hours * 60 * 60 * 1000
    signal_id = digest_signal_id(
        "observation-digest", int(current.timestamp() * 1000) // bucket_ms
    )
    client = notifier or TelegramNotifier()
    delivery = client.deliver_once(
        signal_id=signal_id,
        text=format_observation_digest(predictions, now=current),
        state_dir=settings.telegram_state_dir,
    )
    if delivery.status == "SENT":
        for prediction in predictions:
            _record(settings, prediction)
    return delivery


def deliver_scorecard(
    settings: Settings,
    *,
    days: int = 30,
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> TelegramDelivery | None:
    current = now or datetime.now(timezone.utc)
    card = scorecard(load_ledger(settings.outcome_state_dir), days=days, now=current)
    signal_id = digest_signal_id(
        "scorecard", int(current.timestamp() * 1000) // (24 * 60 * 60 * 1000)
    )
    client = notifier or TelegramNotifier()
    return client.deliver_once(
        signal_id=signal_id,
        text=format_scorecard(card),
        state_dir=settings.telegram_state_dir,
    )


def answer_commands(
    settings: Settings,
    predictions: list[Prediction],
    *,
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> CommandOutcome:
    """Reply to whoever asked the bot a question since the last run."""
    current = now or datetime.now(timezone.utc)
    return poll_and_answer(
        settings,
        status_text=lambda: format_observation_digest(predictions, now=current),
        performance_text=lambda days: format_scorecard(
            scorecard(load_ledger(settings.outcome_state_dir), days=days, now=current)
        ),
        notifier=notifier,
        now=current,
    )


def _record(settings: Settings, prediction: Prediction) -> None:
    record_delivery(
        settings.outcome_state_dir,
        signal_id=prediction.signal_id,
        symbol=prediction.symbol,
        interval=prediction.interval,
        tier=prediction.tier,
        direction=prediction.direction,
        probability=prediction.confidence,
        source_price=prediction.source_price,
        source_close_time_ms=prediction.source_close_time_ms,
        target_close_time_ms=prediction.target_close_time_ms,
        delivered_at_ms=prediction.evaluated_at_ms,
    )


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
    consecutive_failures = 0
    while True:
        try:
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
            settle_pending(
                settings.outcome_state_dir,
                settings.data_dir,
                round_trip_cost_bps=settings.round_trip_cost_bps,
                now=now,
            )
            record_open_interest(settings, now=now)
            today = now.date().isoformat()
            if models_need_research(settings) or last_research_day != today:
                progress("Gunluk walk-forward arastirma ve model yenileme basladi")
                research_all(settings, progress=progress)
                last_research_day = today
            predictions = evaluate_all(settings)
            if is_primary():
                deliveries = deliver_eligible(settings, predictions)
                for prediction, delivery in deliveries:
                    progress(
                        f"{prediction.symbol} {prediction.interval} {prediction.direction}: "
                        f"{delivery.status}{_detail_suffix(delivery)}"
                    )
                digest = deliver_observation_digest(settings, predictions, now=now)
                if digest is not None and digest.status != "DEDUPLICATED":
                    progress(f"Gozlem raporu: {digest.status}{_detail_suffix(digest)}")
                answers = answer_commands(settings, predictions, now=now)
                if answers.received or answers.failed:
                    line = (
                        f"Komut: {answers.received} guncelleme, {answers.answered} yanit, "
                        f"{answers.refused} yetkisiz"
                    )
                    if answers.failed:
                        line += f", {answers.failed} HATA ({answers.detail})"
                    progress(line)
            snapshot = dashboard_snapshot(predictions)
            write_snapshot(settings.report_dir, snapshot)
            if hub_configured():
                post_snapshot(snapshot)
            consecutive_failures = 0
        except KeyboardInterrupt:
            raise
        except Exception as error:  # a 24/7 loop must outlive a bad response
            consecutive_failures += 1
            backoff = min(poll_seconds * 2**consecutive_failures, 900)
            progress(f"Dongu hatasi ({consecutive_failures}): {error}; {backoff} sn bekleniyor")
            time.sleep(backoff)
            continue
        time.sleep(poll_seconds)


def record_open_interest(settings: Settings, *, now: datetime | None = None) -> int:
    """Keep the forward record growing.  A failure here must never stop a cycle."""
    added = 0
    for symbol in SYMBOLS:
        try:
            added += update_open_interest(settings.data_dir, symbol, now=now)
        except (OpenInterestError, OSError):
            continue
    return added


def models_need_research(settings: Settings) -> bool:
    """True when any of the six bundles is missing or no longer loadable."""
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            path = model_path(settings.model_dir, symbol, interval)
            if not path.exists():
                return True
            try:
                load_bundle(path)
            except ValueError:
                return True
    return False


def _detail_suffix(delivery: TelegramDelivery) -> str:
    return f" ({delivery.detail})" if delivery.detail else ""


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


def _iso_utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


__all__ = [
    "Prediction",
    "answer_commands",
    "dashboard_snapshot",
    "deliver_eligible",
    "deliver_observation_digest",
    "deliver_scorecard",
    "evaluate_all",
    "format_observation_digest",
    "format_prediction",
    "make_prediction",
    "models_need_research",
    "record_open_interest",
    "serve_forever",
]
