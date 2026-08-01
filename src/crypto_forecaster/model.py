from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, SupervisedDataset


SCHEMA_VERSION = "btc-eth-probability-model-v1"
PROBABILITY_EDGES = (0.0, 0.35, 0.45, 0.55, 0.65, 1.0000001)


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.scale, -8.0, 8.0)


@dataclass(frozen=True, slots=True)
class LogisticModel:
    intercept: float
    coefficients: np.ndarray

    def logits(self, standardized_x: np.ndarray) -> np.ndarray:
        return self.intercept + standardized_x @ self.coefficients


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: float
    intercept: float

    def predict(self, raw_logits: np.ndarray | float) -> np.ndarray:
        return sigmoid(self.slope * np.asarray(raw_logits) + self.intercept)


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    folds: int
    sample_count: int
    accuracy: float
    baseline_accuracy: float
    brier_score: float
    baseline_brier_score: float
    expected_calibration_error: float
    signal_threshold: float
    signal_count: int
    signal_coverage: float
    signal_accuracy: float
    signal_ci95_low: float
    signal_ci95_high: float
    signal_familywise_ci95_low: float
    signal_familywise_ci95_high: float
    passed_research_gate: bool
    gate_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_reasons"] = list(self.gate_reasons)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BacktestMetrics":
        return cls(
            folds=int(payload["folds"]),
            sample_count=int(payload["sample_count"]),
            accuracy=float(payload["accuracy"]),
            baseline_accuracy=float(payload["baseline_accuracy"]),
            brier_score=float(payload["brier_score"]),
            baseline_brier_score=float(payload["baseline_brier_score"]),
            expected_calibration_error=float(payload["expected_calibration_error"]),
            signal_threshold=float(payload["signal_threshold"]),
            signal_count=int(payload["signal_count"]),
            signal_coverage=float(payload["signal_coverage"]),
            signal_accuracy=float(payload["signal_accuracy"]),
            signal_ci95_low=float(payload["signal_ci95_low"]),
            signal_ci95_high=float(payload["signal_ci95_high"]),
            signal_familywise_ci95_low=float(payload["signal_familywise_ci95_low"]),
            signal_familywise_ci95_high=float(payload["signal_familywise_ci95_high"]),
            passed_research_gate=bool(payload["passed_research_gate"]),
            gate_reasons=tuple(str(item) for item in payload["gate_reasons"]),
        )


@dataclass(frozen=True, slots=True)
class ModelBundle:
    symbol: str
    interval: str
    trained_at_utc: str
    training_first_open_ms: int
    training_last_close_ms: int
    feature_names: tuple[str, ...]
    standardizer: Standardizer
    model: LogisticModel
    calibrator: PlattCalibrator
    scenarios: dict[str, Any]
    backtest: BacktestMetrics

    def predict_up_probability(self, x: np.ndarray) -> float:
        matrix = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("Model belirtec sayisi uyusmuyor")
        raw_logit = self.model.logits(self.standardizer.transform(matrix))[0]
        return float(self.calibrator.predict(raw_logit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "symbol": self.symbol,
            "interval": self.interval,
            "trained_at_utc": self.trained_at_utc,
            "training_first_open_ms": self.training_first_open_ms,
            "training_last_close_ms": self.training_last_close_ms,
            "feature_names": list(self.feature_names),
            "standardizer": {
                "mean": self.standardizer.mean.tolist(),
                "scale": self.standardizer.scale.tolist(),
            },
            "model": {
                "intercept": self.model.intercept,
                "coefficients": self.model.coefficients.tolist(),
            },
            "calibrator": {
                "slope": self.calibrator.slope,
                "intercept": self.calibrator.intercept,
            },
            "scenarios": self.scenarios,
            "backtest": self.backtest.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelBundle":
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError("Desteklenmeyen model semasi")
        names = tuple(str(item) for item in payload["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError("Model belirtec semasi degismis")
        scaler = payload["standardizer"]
        model = payload["model"]
        calibrator = payload["calibrator"]
        bundle = cls(
            symbol=str(payload["symbol"]),
            interval=str(payload["interval"]),
            trained_at_utc=str(payload["trained_at_utc"]),
            training_first_open_ms=int(payload["training_first_open_ms"]),
            training_last_close_ms=int(payload["training_last_close_ms"]),
            feature_names=names,
            standardizer=Standardizer(
                mean=np.asarray(scaler["mean"], dtype=np.float64),
                scale=np.asarray(scaler["scale"], dtype=np.float64),
            ),
            model=LogisticModel(
                intercept=float(model["intercept"]),
                coefficients=np.asarray(model["coefficients"], dtype=np.float64),
            ),
            calibrator=PlattCalibrator(
                slope=float(calibrator["slope"]),
                intercept=float(calibrator["intercept"]),
            ),
            scenarios=dict(payload["scenarios"]),
            backtest=BacktestMetrics.from_dict(dict(payload["backtest"])),
        )
        _validate_bundle(bundle)
        return bundle


def fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> LogisticModel:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or y.size < 2:
        raise ValueError("Gecersiz lojistik regresyon verisi")
    positive_rate = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
    parameters = np.zeros(x.shape[1] + 1, dtype=np.float64)
    parameters[0] = math.log(positive_rate / (1 - positive_rate))
    design = np.column_stack([np.ones(y.size, dtype=np.float64), x])
    penalty = np.eye(parameters.size, dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(60):
        probabilities = sigmoid(design @ parameters)
        weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
        gradient = design.T @ (probabilities - y) + penalty @ parameters
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        parameters -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    if not np.isfinite(parameters).all():
        raise ValueError("Lojistik regresyon yakinsamadi")
    return LogisticModel(intercept=float(parameters[0]), coefficients=parameters[1:])


def fit_platt(raw_logits: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    values = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if values.size != labels.size or values.size < 10:
        raise ValueError("Kalibrasyon icin yetersiz veri")
    if np.unique(labels).size < 2 or float(np.std(values)) < 1e-9:
        rate = float(np.clip(np.mean(labels), 1e-4, 1 - 1e-4))
        return PlattCalibrator(slope=0.0, intercept=math.log(rate / (1 - rate)))
    design = np.column_stack([values, np.ones(values.size, dtype=np.float64)])
    parameters = np.array([1.0, 0.0], dtype=np.float64)
    penalty = np.diag([0.01, 0.0])
    for _ in range(60):
        probabilities = sigmoid(design @ parameters)
        weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
        gradient = design.T @ (probabilities - labels) + penalty @ parameters
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        parameters -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    # Negative slopes invert the classifier and are unstable evidence of regime change.
    if not np.isfinite(parameters).all() or parameters[0] < 0:
        rate = float(np.clip(np.mean(labels), 1e-4, 1 - 1e-4))
        return PlattCalibrator(slope=0.0, intercept=math.log(rate / (1 - rate)))
    return PlattCalibrator(slope=float(parameters[0]), intercept=float(parameters[1]))


def walk_forward_backtest(
    dataset: SupervisedDataset,
    *,
    signal_threshold: float,
    minimum_signal_count: int,
    minimum_signal_accuracy: float,
    maximum_ece: float,
) -> BacktestMetrics:
    n = len(dataset)
    if n < 800:
        raise ValueError("Walk-forward backtest icin en az 800 ornek gerekli")
    train_end = max(500, int(n * 0.50))
    calibration_size = max(120, int(n * 0.10))
    test_size = max(120, int(n * 0.10))
    probabilities_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    baseline_probabilities_parts: list[np.ndarray] = []
    fold_count = 0
    while True:
        calibration_start = train_end + 1  # one-label embargo
        calibration_end = calibration_start + calibration_size
        test_start = calibration_end + 1  # one-label embargo
        if test_start >= n:
            break
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 60:
            break
        scaler = Standardizer.fit(dataset.x[:train_end])
        model = fit_logistic(scaler.transform(dataset.x[:train_end]), dataset.y[:train_end])
        calibration_logits = model.logits(
            scaler.transform(dataset.x[calibration_start:calibration_end])
        )
        calibrator = fit_platt(
            calibration_logits,
            dataset.y[calibration_start:calibration_end],
        )
        test_logits = model.logits(scaler.transform(dataset.x[test_start:test_end]))
        probabilities_parts.append(calibrator.predict(test_logits))
        labels_parts.append(dataset.y[test_start:test_end])
        train_rate = float(np.mean(dataset.y[:train_end]))
        baseline_probabilities_parts.append(
            np.full(test_end - test_start, train_rate, dtype=np.float64)
        )
        fold_count += 1
        train_end += test_size
    if not probabilities_parts:
        raise ValueError("Walk-forward fold olusturulamadi")
    probabilities = np.concatenate(probabilities_parts)
    labels = np.concatenate(labels_parts)
    baseline_probabilities = np.concatenate(baseline_probabilities_parts)
    predictions = (probabilities >= 0.5).astype(np.float64)
    baseline_predictions = (baseline_probabilities >= 0.5).astype(np.float64)
    accuracy = float(np.mean(predictions == labels))
    baseline_accuracy = float(np.mean(baseline_predictions == labels))
    brier = float(np.mean((probabilities - labels) ** 2))
    baseline_brier = float(np.mean((baseline_probabilities - labels) ** 2))
    ece = expected_calibration_error(probabilities, labels)
    signal_mask = (probabilities >= signal_threshold) | (
        probabilities <= 1 - signal_threshold
    )
    signal_count = int(np.sum(signal_mask))
    if signal_count:
        successes = int(np.sum(predictions[signal_mask] == labels[signal_mask]))
        signal_accuracy = float(successes / signal_count)
        ci_low, ci_high = wilson_interval(successes, signal_count)
        # Six fixed symbol/horizon models are assessed together.  This z value is
        # the two-sided Bonferroni 95% family-wise level: Phi^-1(1 - .05/(2*6)).
        family_low, family_high = wilson_interval(
            successes, signal_count, z=2.638257273476751
        )
    else:
        signal_accuracy = 0.0
        ci_low, ci_high = 0.0, 1.0
        family_low, family_high = 0.0, 1.0
    reasons: list[str] = []
    if signal_count < minimum_signal_count:
        reasons.append(f"yuksek guven ornegi {minimum_signal_count} altinda")
    if signal_accuracy < minimum_signal_accuracy:
        reasons.append(f"sinyal dogrulugu %{minimum_signal_accuracy * 100:.1f} altinda")
    if family_low <= 0.50:
        reasons.append("alti model icin duzeltilmis %95 guven araligi yazı-turadan iyi degil")
    if brier >= baseline_brier:
        reasons.append("Brier skoru tarihsel taban olasiliktan iyi degil")
    if ece > maximum_ece:
        reasons.append(f"kalibrasyon hatasi %{maximum_ece * 100:.1f} ustunde")
    return BacktestMetrics(
        folds=fold_count,
        sample_count=int(labels.size),
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        brier_score=brier,
        baseline_brier_score=baseline_brier,
        expected_calibration_error=ece,
        signal_threshold=signal_threshold,
        signal_count=signal_count,
        signal_coverage=signal_count / labels.size,
        signal_accuracy=signal_accuracy,
        signal_ci95_low=ci_low,
        signal_ci95_high=ci_high,
        signal_familywise_ci95_low=family_low,
        signal_familywise_ci95_high=family_high,
        passed_research_gate=not reasons,
        gate_reasons=tuple(reasons),
    )


def fit_final_bundle(
    dataset: SupervisedDataset,
    *,
    symbol: str,
    interval: str,
    backtest: BacktestMetrics,
) -> ModelBundle:
    n = len(dataset)
    train_end = int(n * 0.80)
    calibration_start = train_end + 1
    if n - calibration_start < 100:
        raise ValueError("Nihai kalibrasyon icin yetersiz veri")
    scaler = Standardizer.fit(dataset.x[:train_end])
    model = fit_logistic(scaler.transform(dataset.x[:train_end]), dataset.y[:train_end])
    calibration_logits = model.logits(scaler.transform(dataset.x[calibration_start:]))
    calibrator = fit_platt(calibration_logits, dataset.y[calibration_start:])
    calibrated = calibrator.predict(calibration_logits)
    scenarios = build_scenarios(
        calibrated,
        dataset.atr_pct[calibration_start:],
        dataset.future_return_atr[calibration_start:],
        dataset.future_up_atr[calibration_start:],
        dataset.future_down_atr[calibration_start:],
    )
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return ModelBundle(
        symbol=symbol,
        interval=interval,
        trained_at_utc=trained_at,
        training_first_open_ms=int(dataset.open_time_ms[0]),
        training_last_close_ms=int(dataset.close_time_ms[-1]),
        feature_names=FEATURE_NAMES,
        standardizer=scaler,
        model=model,
        calibrator=calibrator,
        scenarios=scenarios,
        backtest=backtest,
    )


def build_scenarios(
    probabilities: np.ndarray,
    atr_pct: np.ndarray,
    future_return_atr: np.ndarray,
    future_up_atr: np.ndarray,
    future_down_atr: np.ndarray,
) -> dict[str, Any]:
    vol_edges = np.quantile(atr_pct, [1 / 3, 2 / 3]).astype(float)
    p_bins = np.digitize(probabilities, PROBABILITY_EDGES[1:-1], right=False)
    v_bins = np.digitize(atr_pct, vol_edges, right=False)
    cells: dict[str, Any] = {}
    for p_index in range(len(PROBABILITY_EDGES) - 1):
        p_mask = p_bins == p_index
        if np.any(p_mask):
            cells[f"p{p_index}"] = _scenario_summary(
                future_return_atr[p_mask], future_up_atr[p_mask], future_down_atr[p_mask]
            )
        for v_index in range(3):
            mask = p_mask & (v_bins == v_index)
            if np.any(mask):
                cells[f"p{p_index}v{v_index}"] = _scenario_summary(
                    future_return_atr[mask], future_up_atr[mask], future_down_atr[mask]
                )
    return {
        "probability_edges": list(PROBABILITY_EDGES),
        "volatility_edges": vol_edges.tolist(),
        "target_atr_multiple": 0.5,
        "global": _scenario_summary(future_return_atr, future_up_atr, future_down_atr),
        "cells": cells,
    }


def select_scenario(
    scenarios: dict[str, Any],
    probability: float,
    atr_pct: float,
    *,
    minimum_count: int,
) -> dict[str, float | int]:
    p_edges = np.asarray(scenarios["probability_edges"], dtype=np.float64)
    v_edges = np.asarray(scenarios["volatility_edges"], dtype=np.float64)
    p_index = int(np.clip(np.digitize([probability], p_edges[1:-1])[0], 0, len(p_edges) - 2))
    v_index = int(np.clip(np.digitize([atr_pct], v_edges)[0], 0, 2))
    cells = scenarios["cells"]
    detailed = cells.get(f"p{p_index}v{v_index}")
    if detailed and int(detailed["count"]) >= minimum_count:
        return dict(detailed)
    probability_only = cells.get(f"p{p_index}")
    if probability_only and int(probability_only["count"]) >= minimum_count:
        return dict(probability_only)
    return dict(scenarios["global"])


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        count = int(np.sum(mask))
        if count:
            error += (count / total) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return float(error)


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Gecersiz Wilson araligi girdisi")
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt((rate * (1 - rate) / trials) + (z * z / (4 * trials * trials)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    _validate_bundle(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        bundle.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def load_bundle(path: Path) -> ModelBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"Model dosyasi okunamadi: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError("Model dosyasi JSON nesnesi degil")
    return ModelBundle.from_dict(payload)


def _scenario_summary(
    returns: np.ndarray, up_excursions: np.ndarray, down_excursions: np.ndarray
) -> dict[str, float | int]:
    return {
        "count": int(returns.size),
        "close_return_atr_p10": float(np.quantile(returns, 0.10)),
        "close_return_atr_p50": float(np.quantile(returns, 0.50)),
        "close_return_atr_p90": float(np.quantile(returns, 0.90)),
        "touch_up_half_atr_probability": float(np.mean(up_excursions >= 0.5)),
        "touch_down_half_atr_probability": float(np.mean(down_excursions >= 0.5)),
    }


def _validate_bundle(bundle: ModelBundle) -> None:
    size = len(bundle.feature_names)
    arrays = (
        bundle.standardizer.mean,
        bundle.standardizer.scale,
        bundle.model.coefficients,
    )
    if any(array.shape != (size,) for array in arrays):
        raise ValueError("Model dizi boyutu gecersiz")
    numeric = np.concatenate(arrays + (np.array([
        bundle.model.intercept,
        bundle.calibrator.slope,
        bundle.calibrator.intercept,
    ]),))
    if not np.isfinite(numeric).all() or (bundle.standardizer.scale <= 0).any():
        raise ValueError("Model sayilari gecersiz")


__all__ = [
    "BacktestMetrics",
    "ModelBundle",
    "Standardizer",
    "fit_final_bundle",
    "fit_logistic",
    "fit_platt",
    "load_bundle",
    "save_bundle",
    "select_scenario",
    "walk_forward_backtest",
    "wilson_interval",
]
