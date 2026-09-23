"""On-demand, coin-specific and chronologically tested Telegram forecasts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings, local_text, validate_market_symbol
from .data import (
    CSV_COLUMNS,
    BinanceMarketDataClient,
    MarketDataError,
    load_cache,
)
from .features import build_supervised_dataset, latest_feature_vector
from .model import Standardizer, fit_logistic, fit_platt
from .persistence import atomic_write_json, atomic_write_text
from .scalping import scalp_cache_path

HORIZONS_MINUTES = (15, 60, 240, 1_440)
HORIZON_LABELS = {15: "15 dk", 60: "1 saat", 240: "4 saat", 1_440: "1 gün"}
BAR_MINUTES = 5
MODEL_SCHEMA = "trade3-coin-query-model-v1"
SYMBOL_INPUT = re.compile(r"[A-Za-z0-9]{1,16}(?:USDT)?\Z")
MINIMUM_TRAIN_ROWS = 500
MINIMUM_TEST_ROWS = 120


@dataclass(frozen=True, slots=True)
class CoinHorizonForecast:
    horizon_minutes: int
    probability_up: float
    probability_down: float
    test_accuracy: float
    accuracy_ci95: tuple[float, float]
    test_count: int
    test_brier: float
    baseline_brier: float
    mean_net_bps: float


@dataclass(frozen=True, slots=True)
class CoinForecast:
    symbol: str
    mark_price: float
    quote_volume_24h_usdt: float | None
    spread_bps: float
    funding_rate_bps: float | None
    source_close_time_ms: int
    history_days: float
    history_rows: int
    round_trip_cost_bps: float
    forecasts: tuple[CoinHorizonForecast, ...]


def normalize_coin_query(value: str) -> str:
    """Accept an asset code or a USDT-M symbol; never accept path syntax."""
    candidate = value.strip()
    if not SYMBOL_INPUT.fullmatch(candidate):
        raise ValueError("Sembol biçimi geçersiz. Örnek: ALLO veya ALLOUSDT")
    candidate = candidate.upper()
    if not candidate.endswith("USDT"):
        candidate += "USDT"
    return validate_market_symbol(candidate)


def forecast_coin(
    settings: Settings,
    value: str,
    *,
    now: datetime | None = None,
    client: BinanceMarketDataClient | None = None,
) -> CoinForecast:
    """Fetch one USD-M market and produce four per-coin calibrated forecasts.

    Every probability is fitted from that coin's own 5m history. The fixed
    BTC/ETH bundles are never loaded or applied to this query.
    """
    symbol = normalize_coin_query(value)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current = current.astimezone(UTC)
    current_ms = int(current.timestamp() * 1000)
    market_client = client or BinanceMarketDataClient(market_name="futures")
    try:
        snapshot = market_client.fetch_futures_market_snapshots().get(symbol)
    except (MarketDataError, OSError, ValueError) as error:
        raise ValueError(
            f"Binance USD-M piyasa verisi alınamadı ({type(error).__name__})."
        ) from None
    if snapshot is None:
        raise ValueError(f"{symbol} Binance USD-M perpetual piyasasında bulunamadı.")

    bars = _refresh_query_history(settings, symbol, market_client, current_ms)
    if len(bars) < 1_900:
        days = (
            (int(bars["close_time_ms"].iloc[-1]) - int(bars["open_time_ms"].iloc[0]))
            / 86_400_000
            if not bars.empty
            else 0
        )
        raise ValueError(
            f"{symbol} için yeterli tarihçe yok ({len(bars)} adet 5 dk mum, yaklaşık {days:.1f} gün). "
            "Dört ufuk için en az yaklaşık 7 günlük kesintisiz veri gerekiyor."
        )
    source_close_ms = int(bars["close_time_ms"].iloc[-1])
    effective_cost = max(
        settings.round_trip_cost_bps,
        2 * settings.scalp_taker_fee_bps
        + snapshot.spread_bps
        + 2 * settings.scalp_slippage_bps_per_side,
    )
    model_path = _query_model_path(settings, symbol)
    cached = _read_model_cache(
        model_path,
        symbol,
        current_ms,
        source_close_ms,
        settings,
        history_days=settings.coin_query_history_days,
        barrier_bps=settings.barrier_target_bps,
    )
    if cached is None:
        cached = _fit_coin_models(
            bars,
            symbol=symbol,
            cost_bps=effective_cost,
            generated_at_ms=current_ms,
            barrier_bps=settings.barrier_target_bps,
            history_days=settings.coin_query_history_days,
        )
        atomic_write_json(model_path, cached)
    _, current_vector = latest_feature_vector(bars)
    forecasts = tuple(
        _forecast_from_cached(
            cached["horizons"][str(horizon)],
            current_vector,
            horizon,
            cost_adjustment_bps=effective_cost - float(cached["round_trip_cost_bps"]),
        )
        for horizon in HORIZONS_MINUTES
    )
    history_days = max(
        0.0, (source_close_ms - int(bars["open_time_ms"].iloc[0])) / 86_400_000
    )
    return CoinForecast(
        symbol=symbol,
        mark_price=snapshot.mark_price or float(bars["close"].iloc[-1]),
        quote_volume_24h_usdt=snapshot.quote_volume_24h_usdt,
        spread_bps=snapshot.spread_bps,
        funding_rate_bps=snapshot.funding_rate_bps,
        source_close_time_ms=source_close_ms,
        history_days=history_days,
        history_rows=len(bars),
        round_trip_cost_bps=effective_cost,
        forecasts=forecasts,
    )


def format_coin_forecast(forecast: CoinForecast) -> str:
    volume = (
        f"${forecast.quote_volume_24h_usdt / 1_000_000:.1f}M"
        if forecast.quote_volume_24h_usdt is not None
        else "ölçülemedi"
    )
    funding = (
        f"{forecast.funding_rate_bps:+.2f} bps"
        if forecast.funding_rate_bps is not None
        else "ölçülemedi"
    )
    lines = [
        f"🧭 {forecast.symbol} • Binance USD-M perpetual",
        f"Anlık mark: ${forecast.mark_price:,.8g} • 24s hacim: {volume}",
        f"Spread: {forecast.spread_bps:.2f} bps • Funding: {funding}",
        (
            f"Veri: {forecast.history_days:.0f} gün / {forecast.history_rows:,} adet 5dk mum • "
            f"son kapanış {local_text(forecast.source_close_time_ms, with_seconds=False)}"
        ),
        "",
    ]
    for row in forecast.forecasts:
        skill = (
            "model taban çizgisini geçti"
            if row.test_brier < row.baseline_brier
            else "model taban çizgisini geçemedi"
        )
        message = "\n".join(
            (
                f"{HORIZON_LABELS[row.horizon_minutes]}: ↑ %{row.probability_up * 100:.1f} / ↓ %{row.probability_down * 100:.1f}",
                f"  İleri-test yön isabeti: %{row.test_accuracy * 100:.1f} "
                + f"(n={row.test_count}, %95 GA %{row.accuracy_ci95[0] * 100:.0f}–"
                + f"%{row.accuracy_ci95[1] * 100:.0f}) • ort. net {row.mean_net_bps:+.1f} bps",
                f"  Kalibrasyon: {skill}",
            )
        )
        lines.append(message)
    lines.extend(
        [
            "",
            f"İleri-test maliyet varsayımı: yaklaşık {forecast.round_trip_cost_bps:.1f} bps gidiş-dönüş.",
            "Olasılıklar coin-özel 5dk verisiyle eğitilmiş, kronolojik ayrılmış model tahminidir; "
            + "garanti veya al/sat talimatı değildir. Geçmiş test sonucu geleceği garanti etmez.",
        ]
    )
    return "\n".join(lines)


def _refresh_query_history(
    settings: Settings, symbol: str, client: BinanceMarketDataClient, current_ms: int
) -> pd.DataFrame:
    path = scalp_cache_path(settings.scalp_data_dir / "coin-query", symbol)
    meta_path = path.with_suffix(".meta.json")
    step_ms = BAR_MINUTES * 60_000
    end_ms = current_ms // step_ms * step_ms
    start_ms = end_ms - settings.coin_query_history_days * 86_400_000
    try:
        old = load_cache(path) if path.exists() else pd.DataFrame(columns=CSV_COLUMNS)
    except MarketDataError:
        old = pd.DataFrame(columns=CSV_COLUMNS)
    covered_start_ms: int | None = None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") == "coin-query-history-v1"
            and metadata.get("symbol") == symbol
        ):
            covered_start_ms = int(metadata["requested_start_ms"])
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        pass
    fetched_frames: list[pd.DataFrame] = []
    if old.empty:
        fetched_frames.append(
            client.fetch_market_klines(symbol, "5m", start_ms=start_ms, end_ms=end_ms)
        )
    else:
        first_open = int(old["open_time_ms"].iloc[0])
        if first_open > start_ms and (
            covered_start_ms is None or covered_start_ms > start_ms
        ):
            fetched_frames.append(
                client.fetch_market_klines(
                    symbol, "5m", start_ms=start_ms, end_ms=first_open
                )
            )
        last_close = int(old["close_time_ms"].iloc[-1])
        if last_close < end_ms - 1:
            fetched_frames.append(
                client.fetch_market_klines(
                    symbol, "5m", start_ms=last_close + 1, end_ms=end_ms
                )
            )
        if not fetched_frames:
            return _trim_and_check(old, start_ms, symbol)
    combined = pd.concat([old, *fetched_frames], ignore_index=True)
    combined = (
        combined.loc[:, CSV_COLUMNS]
        .sort_values("open_time_ms")
        .drop_duplicates("open_time_ms", keep="last")
    )
    combined = _trim_and_check(combined, start_ms, symbol)
    atomic_write_text(
        path, combined.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    )
    if (
        old.empty
        or any(
            int(frame["open_time_ms"].min()) <= start_ms
            for frame in fetched_frames
            if not frame.empty
        )
        or (covered_start_ms is None and fetched_frames)
    ):
        covered_start_ms = (
            start_ms if covered_start_ms is None else min(covered_start_ms, start_ms)
        )
    if covered_start_ms is not None:
        atomic_write_json(
            meta_path,
            {
                "schema": "coin-query-history-v1",
                "symbol": symbol,
                "requested_start_ms": covered_start_ms,
            },
        )
    return combined


def _trim_and_check(frame: pd.DataFrame, start_ms: int, symbol: str) -> pd.DataFrame:
    result = frame.loc[frame["open_time_ms"] >= start_ms, CSV_COLUMNS].copy()
    if result.empty:
        raise MarketDataError(f"{symbol} için kapanmış mum bulunamadı")
    opens = result["open_time_ms"].to_numpy(dtype=np.int64)
    if opens.size > 1 and not np.all(np.diff(opens) == BAR_MINUTES * 60_000):
        raise MarketDataError(
            f"{symbol} geçmişinde 5 dk mum boşluğu var; eksik veriden olasılık üretmedim"
        )
    return result.reset_index(drop=True)


def _fit_coin_models(
    bars: pd.DataFrame,
    *,
    symbol: str,
    cost_bps: float,
    generated_at_ms: int,
    barrier_bps: float,
    history_days: int = 180,
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    _, current_vector = latest_feature_vector(bars)
    for horizon_minutes in HORIZONS_MINUTES:
        candles = horizon_minutes // BAR_MINUTES
        dataset = build_supervised_dataset(
            bars,
            barrier_horizon_candles=candles,
            barrier_atr_multiple=1.0,
            minimum_barrier_bps=barrier_bps,
        )
        n = len(dataset)
        test_count = max(MINIMUM_TEST_ROWS, int(n * 0.15))
        calibration_count = max(MINIMUM_TEST_ROWS, int(n * 0.15))
        test_start = n - test_count
        calibration_end = test_start - candles
        calibration_start = calibration_end - calibration_count
        train_end = calibration_start - candles
        if train_end < MINIMUM_TRAIN_ROWS or calibration_start < 0 or test_start >= n:
            raise ValueError(
                f"{HORIZON_LABELS[horizon_minutes]} için yeterli ileri-test örneği üretilemedi"
            )

        validation_scale = Standardizer.fit(dataset.x[:train_end])
        validation_model = fit_logistic(
            validation_scale.transform(dataset.x[:train_end]), dataset.y[:train_end]
        )
        calibration_logits = validation_model.logits(
            validation_scale.transform(dataset.x[calibration_start:calibration_end])
        )
        validation_calibrator = fit_platt(
            calibration_logits, dataset.y[calibration_start:calibration_end]
        )
        all_test_indices = np.arange(test_start, n, dtype=np.int64)
        all_test_probabilities = validation_calibrator.predict(
            validation_model.logits(validation_scale.transform(dataset.x[test_start:n]))
        )
        # Evaluate reporting metrics on disjoint outcome windows. The full
        # five-minute stream would overlap 24h labels hundreds of times and
        # make n and its confidence interval look much stronger than they are.
        non_overlapping = np.arange(0, len(all_test_indices), candles, dtype=np.int64)
        test_indices = all_test_indices[non_overlapping]
        test_probabilities = np.asarray(all_test_probabilities)[non_overlapping]
        test_labels = dataset.y[test_indices]
        guesses = (test_probabilities >= 0.5).astype(np.float64)
        correct = int(np.sum(guesses == test_labels))
        accuracy = correct / len(test_labels)
        low, high = _wilson_interval(correct, len(test_labels))
        baseline_probability = float(np.mean(dataset.y[test_start:n]))
        brier = float(np.mean((test_probabilities - test_labels) ** 2))
        baseline_brier = float(np.mean((baseline_probability - test_labels) ** 2))
        sides = np.where(test_probabilities >= 0.5, 1.0, -1.0)
        mean_net = float(
            np.mean(dataset.outcome_bps(sides, cost_bps, indices=test_indices))
        )

        # Refit an operational model on all matured history, leaving the final
        # embargo before the calibration slice to prevent look-ahead leakage.
        live_calibration_start = n - calibration_count
        live_train_end = live_calibration_start - candles
        if live_train_end < MINIMUM_TRAIN_ROWS:
            raise ValueError(
                f"{HORIZON_LABELS[horizon_minutes]} canlı model için veri yetersiz"
            )
        scale = Standardizer.fit(dataset.x[:live_train_end])
        model = fit_logistic(
            scale.transform(dataset.x[:live_train_end]), dataset.y[:live_train_end]
        )
        live_logits = model.logits(scale.transform(dataset.x[live_calibration_start:n]))
        calibrator = fit_platt(live_logits, dataset.y[live_calibration_start:n])
        horizons[str(horizon_minutes)] = {
            "horizon_minutes": horizon_minutes,
            "training_last_close_ms": int(dataset.close_time_ms[live_train_end - 1]),
            "probability_up": float(
                np.asarray(
                    calibrator.predict(model.logits(scale.transform(current_vector)))
                ).reshape(-1)[0]
            ),
            "test_accuracy": accuracy,
            "accuracy_ci95": [low, high],
            "test_count": len(test_labels),
            "test_brier": brier,
            "baseline_brier": baseline_brier,
            "mean_net_bps": mean_net,
            "standardizer_mean": scale.mean.tolist(),
            "standardizer_scale": scale.scale.tolist(),
            "coefficients": model.coefficients.tolist(),
            "intercept": model.intercept,
            "calibrator_slope": calibrator.slope,
            "calibrator_intercept": calibrator.intercept,
        }
    return {
        "schema": MODEL_SCHEMA,
        "symbol": symbol,
        "generated_at_ms": generated_at_ms,
        "training_source_close_ms": int(bars["close_time_ms"].iloc[-1]),
        "history_days": history_days,
        "barrier_bps": barrier_bps,
        "round_trip_cost_bps": cost_bps,
        "horizons": horizons,
    }


def _read_model_cache(
    path: Path,
    symbol: str,
    current_ms: int,
    source_close_ms: int,
    settings: Settings,
    *,
    history_days: int,
    barrier_bps: float,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = int(payload["generated_at_ms"])
        trained_close = int(payload["training_source_close_ms"])
        horizons = payload["horizons"]
        if (
            payload.get("schema") != MODEL_SCHEMA
            or payload.get("symbol") != symbol
            or int(payload.get("history_days", -1)) != history_days
            or float(payload.get("barrier_bps", -1)) != barrier_bps
            or current_ms - generated
            > settings.coin_query_model_refresh_days * 86_400_000
            or source_close_ms - trained_close
            > settings.coin_query_model_refresh_days * 86_400_000
            or not isinstance(horizons, dict)
            or any(str(value) not in horizons for value in HORIZONS_MINUTES)
        ):
            return None
        for value in horizons.values():
            _validate_horizon_model(value)
        return payload
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def _forecast_from_cached(
    row: dict[str, Any],
    vector: np.ndarray,
    horizon: int,
    *,
    cost_adjustment_bps: float = 0.0,
) -> CoinHorizonForecast:
    _validate_horizon_model(row)
    mean = np.asarray(row["standardizer_mean"], dtype=np.float64)
    scale = np.asarray(row["standardizer_scale"], dtype=np.float64)
    coefficients = np.asarray(row["coefficients"], dtype=np.float64)
    standardized = np.clip((vector - mean) / scale, -8.0, 8.0)
    logit = float(row["intercept"] + standardized @ coefficients)
    raw = row["calibrator_slope"] * logit + row["calibrator_intercept"]
    probability_up = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, raw))))
    return CoinHorizonForecast(
        horizon_minutes=horizon,
        probability_up=probability_up,
        probability_down=1.0 - probability_up,
        test_accuracy=float(row["test_accuracy"]),
        accuracy_ci95=(float(row["accuracy_ci95"][0]), float(row["accuracy_ci95"][1])),
        test_count=int(row["test_count"]),
        test_brier=float(row["test_brier"]),
        baseline_brier=float(row["baseline_brier"]),
        mean_net_bps=float(row["mean_net_bps"]) - cost_adjustment_bps,
    )


def _validate_horizon_model(row: Any) -> None:
    if not isinstance(row, dict):
        raise TypeError("model row")
    for name, size in (
        ("standardizer_mean", 9),
        ("standardizer_scale", 9),
        ("coefficients", 9),
    ):
        values = np.asarray(row[name], dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("model vector")
        if name == "standardizer_scale" and np.any(values <= 0):
            raise ValueError("model scale")
    for name in (
        "intercept",
        "calibrator_slope",
        "calibrator_intercept",
        "test_accuracy",
        "test_brier",
        "baseline_brier",
        "mean_net_bps",
    ):
        if not math.isfinite(float(row[name])):
            raise ValueError("model number")
    interval = row["accuracy_ci95"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(math.isfinite(float(x)) for x in interval)
    ):
        raise ValueError("model interval")


def _query_model_path(settings: Settings, symbol: str) -> Path:
    safe_symbol = validate_market_symbol(symbol)
    return settings.scalp_state_dir / "coin-query-models" / f"{safe_symbol}.json"


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    p = successes / trials
    z = 1.959963984540054
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    margin = (
        z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    )
    return max(0.0, center - margin), min(1.0, center + margin)
