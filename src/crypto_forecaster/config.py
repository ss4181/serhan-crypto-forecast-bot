from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SYMBOLS = ("BTCUSDT", "ETHUSDT")
# Messages are read by a person, so they carry that person's clock.  Reading
# "14:19" on a phone showing 17:19 makes a current report look three hours old.
DISPLAY_TIMEZONE_ENV = "CRYPTO_DISPLAY_TIMEZONE"
DEFAULT_DISPLAY_TIMEZONE = "Europe/Istanbul"
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
    # Binance USD-M futures taker is 0.05% per side, so a round trip costs
    # 10 bps.  Set to 20.0 to price Spot taker instead (0.10% per side).
    round_trip_cost_bps: float = 10.0
    # A model may notify only when the lower bound of its net edge clears this.
    minimum_net_edge_bps: float = 0.0
    # A signal is actionable only while this much of the target candle is left.
    minimum_remaining_fraction: float = 0.60
    # Every model reports to Telegram this often even when it cannot trade.
    observation_digest_hours: int = 24
    # Triple barrier.  The take-profit and stop sit this far from entry, in
    # basis points: the size of trade actually intended, not a number fitted to
    # the data.  A symmetric barrier needs a hit rate of
    #     50% + cost / (2 * barrier)
    # to break even, so barrier width relative to cost is what decides whether
    # the bar is reachable at all: 40 bps against 20 bps demands 75%, while
    # 100 bps against 10 bps demands 55%.
    barrier_target_bps: float = 100.0
    # Optional volatility floor; 0 keeps the barrier at exactly the target.
    barrier_atr_multiple: float = 0.0
    # Longest a trade may stay open before it is closed at market.  Expressed
    # in hours so every interval means the same thing in market time.
    barrier_horizon_hours: float = 24.0
    # Refuse to research a barrier too thin to survive its own costs.
    minimum_barrier_cost_multiple: float = 4.0


def validate_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if normalized not in SYMBOLS:
        raise ValueError(f"Sembol yalnizca {', '.join(SYMBOLS)} olabilir")
    return normalized


def validate_interval(interval: str) -> str:
    if interval not in INTERVALS:
        raise ValueError(f"Zaman dilimi yalnizca {', '.join(INTERVALS)} olabilir")
    return interval


def display_zone() -> tzinfo:
    """The clock messages are written in.  Falls back to UTC if tzdata is thin."""
    name = os.environ.get(DISPLAY_TIMEZONE_ENV, "").strip() or DEFAULT_DISPLAY_TIMEZONE
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return timezone.utc


def local_text(milliseconds: int, *, with_seconds: bool = True) -> str:
    """Format an instant for a reader, labelled so it cannot be misread."""
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=display_zone())
    pattern = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    offset = moment.strftime("%z")
    label = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    return f"{moment.strftime(pattern)} ({label})"


def barrier_horizon_candles(interval: str, hours: float) -> int:
    """Turn a holding-time limit into candles for this interval."""
    if hours <= 0:
        raise ValueError("Bariyer ufku pozitif olmali")
    step_ms = INTERVAL_MILLISECONDS[validate_interval(interval)]
    return max(1, round(hours * 60 * 60 * 1000 / step_ms))


def cache_path(data_dir: Path, symbol: str, interval: str) -> Path:
    return data_dir / f"{validate_symbol(symbol)}_{validate_interval(interval)}.csv"


def model_path(model_dir: Path, symbol: str, interval: str) -> Path:
    return model_dir / f"{validate_symbol(symbol)}_{validate_interval(interval)}.json"
