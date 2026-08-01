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
    outcome_state_dir: Path = Path("state/outcomes")
    signal_threshold: float = 0.60
    minimum_signal_count: int = 100
    minimum_signal_accuracy: float = 0.53
    maximum_ece: float = 0.10
    scenario_minimum_count: int = 40
    maximum_model_age_days: int = 8
    # Binance Spot taker fee is 0.10% per side, so a round trip costs 20 bps.
    # Set to 10.0 to price USD-M futures taker instead (0.05% per side).
    round_trip_cost_bps: float = 20.0
    # A model may notify only when the lower bound of its net edge clears this.
    minimum_net_edge_bps: float = 0.0
    # A signal is actionable only while this much of the target candle is left.
    minimum_remaining_fraction: float = 0.60
    # Every model reports to Telegram this often even when it cannot trade.
    observation_digest_hours: int = 24
    # Triple barrier: how far the take-profit and stop sit from entry, and how
    # long the trade may stay open before it is closed at market.
    barrier_atr_multiple: float = 1.0
    barrier_horizon_candles: int = 12
    # The barrier never sits closer than this multiple of the round trip, so a
    # winning trade always clears its own costs by a margin.
    barrier_cost_multiple: float = 2.0


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
