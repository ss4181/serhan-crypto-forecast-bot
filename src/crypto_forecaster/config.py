from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVALS = ("5m", "15m", "1h")
INTERVAL_MILLISECONDS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}
INTERVAL_LABELS = {"5m": "5 dakika", "15m": "15 dakika", "1h": "1 saat"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts/models")
    report_dir: Path = Path("artifacts/reports")
    telegram_state_dir: Path = Path("state/telegram")
    signal_threshold: float = 0.60
    minimum_signal_count: int = 100
    minimum_signal_accuracy: float = 0.53
    maximum_ece: float = 0.10
    scenario_minimum_count: int = 40
    maximum_model_age_days: int = 8


def validate_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if normalized not in SYMBOLS:
        raise ValueError(f"Sembol yalnizca {', '.join(SYMBOLS)} olabilir")
    return normalized


def validate_interval(interval: str) -> str:
    if interval not in INTERVALS:
        raise ValueError(f"Zaman dilimi yalnizca {', '.join(INTERVALS)} olabilir")
    return interval


def cache_path(data_dir: Path, symbol: str, interval: str) -> Path:
    return data_dir / f"{validate_symbol(symbol)}_{validate_interval(interval)}.csv"


def model_path(model_dir: Path, symbol: str, interval: str) -> Path:
    return model_dir / f"{validate_symbol(symbol)}_{validate_interval(interval)}.json"
