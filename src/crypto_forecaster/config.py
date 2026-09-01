from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SYMBOLS_ENV = "CRYPTO_SYMBOLS"
SCALP_OBSERVATION_ENV = "CRYPTO_SCALP_OBSERVATION"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9]{1,20}USDT\Z")


def _configured_symbols() -> tuple[str, ...]:
    """Which markets to research.  Fixed at import so every module agrees.

    Hardcoding BTC and ETH made the tool study markets its owner does not
    trade.  A comma-separated list is accepted, but the shape is still
    validated: an unchecked symbol becomes a path segment on disk.
    """
    raw = os.environ.get(SYMBOLS_ENV, "").strip()
    if not raw:
        return DEFAULT_SYMBOLS
    chosen: list[str] = []
    for item in raw.split(","):
        candidate = item.strip().upper()
        if not candidate:
            continue
        if not _SYMBOL_PATTERN.fullmatch(candidate):
            raise ValueError(f"{SYMBOLS_ENV} icinde gecersiz sembol: {item.strip()}")
        if candidate not in chosen:
            chosen.append(candidate)
    if not chosen:
        return DEFAULT_SYMBOLS
    return tuple(chosen)


SYMBOLS = _configured_symbols()
# The perpetual contract is what gets traded, so it is what gets modelled.
# Spot prices track it closely but are a different instrument, and a cache
# holding one must never be mistaken for the other -- hence the market in the
# filename rather than a flag somewhere.
MARKET_ENV = "CRYPTO_MARKET"
MARKETS = ("futures", "spot")


def market() -> str:
    choice = os.environ.get(MARKET_ENV, "").strip().lower() or MARKETS[0]
    if choice not in MARKETS:
        raise ValueError(f"{MARKET_ENV} yalnizca {', '.join(MARKETS)} olabilir")
    return choice


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
    scalp_data_dir: Path = Path("data/scalp")
    scalp_state_dir: Path = Path("state/scalp")
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
    # Full walk-forward research is deliberately slower than candle refresh;
    # model files carry the last successful run across service restarts.
    model_research_interval_hours: int = field(
        default_factory=lambda: _environment_int(
            "CRYPTO_MODEL_RESEARCH_INTERVAL_HOURS", 168, 24, 720
        )
    )
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
    # Broad-universe scalping stays separate from the verified-model path.  It
    # is deliberately opt-in and can only emit RESEARCH-ONLY observation
    # digests; no setting here can promote one to an actionable signal.
    scalp_observation_enabled: bool = field(
        default_factory=lambda: _environment_bool(SCALP_OBSERVATION_ENV, False)
    )
    scalp_top_k: int = field(
        default_factory=lambda: _environment_int("CRYPTO_SCALP_TOP_K", 5, 1, 10)
    )
    # Telegram only: keep every observation in the shadow ledger, but notify
    # only high-scoring, multi-family setups whose settled BT direction is exact.
    scalp_minimum_alert_score: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_MIN_ALERT_SCORE", 2.5, 0.0, 10.0
        )
    )
    scalp_cache_days: int = field(
        default_factory=lambda: _environment_int("CRYPTO_SCALP_CACHE_DAYS", 3, 2, 30)
    )
    scalp_maximum_bar_age_minutes: int = field(
        default_factory=lambda: _environment_int(
            "CRYPTO_SCALP_MAXIMUM_BAR_AGE_MINUTES", 15, 5, 60
        )
    )
    scalp_minimum_coverage: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_MINIMUM_COVERAGE", 0.90, 0.50, 1.0
        )
    )
    # Account-specific commission data needs a signed Binance request.  Keep
    # the public observer read-only and make the actual taker assumption
    # explicit instead of pretending the historical 12 bps is always current.
    scalp_taker_fee_bps: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_TAKER_FEE_BPS", 5.0, 0.0, 100.0
        )
    )
    scalp_slippage_bps_per_side: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_SLIPPAGE_BPS_PER_SIDE", 1.0, 0.0, 100.0
        )
    )
    scalp_maximum_spread_bps: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_MAXIMUM_SPREAD_BPS", 8.0, 0.1, 100.0
        )
    )
    scalp_bull_breadth_threshold: float = field(
        default_factory=lambda: _environment_float(
            "CRYPTO_SCALP_BULL_BREADTH", 0.60, 0.40, 0.90
        )
    )


def validate_symbol(symbol: str) -> str:
    normalized = validate_market_symbol(symbol)
    if normalized not in SYMBOLS:
        raise ValueError(
            f"Sembol yalnizca {', '.join(SYMBOLS)} olabilir; baskasi icin {SYMBOLS_ENV} degiskenini kullanin"
        )
    return normalized


def validate_market_symbol(symbol: str) -> str:
    """Validate a Binance-style symbol without adding it to the model universe."""
    if not isinstance(symbol, str):
        raise ValueError("Sembol metin olmali")
    normalized = symbol.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError(f"Gecersiz sembol: {symbol}")
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
    return (
        data_dir
        / f"{validate_symbol(symbol)}_{validate_interval(interval)}_{market()}.csv"
    )


def model_path(model_dir: Path, symbol: str, interval: str) -> Path:
    return model_dir / f"{validate_symbol(symbol)}_{validate_interval(interval)}.json"


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} true/false olmali")


def _environment_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None or not raw.strip() else int(raw)
    except ValueError:
        raise ValueError(f"{name} tam sayi olmali") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} {minimum} ile {maximum} arasinda olmali")
    return value


def _environment_float(
    name: str, default: float, minimum: float, maximum: float
) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None or not raw.strip() else float(raw)
    except ValueError:
        raise ValueError(f"{name} sayi olmali") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} {minimum} ile {maximum} arasinda olmali")
    return value
