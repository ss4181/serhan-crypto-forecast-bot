from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import (
    INTERVAL_MILLISECONDS,
    cache_path,
    market,
    validate_interval,
    validate_market_symbol,
    validate_symbol,
)


# Both are public, read-only kline endpoints; neither takes an API key.
MARKET_ENDPOINTS = {
    "futures": ("https://fapi.binance.com", "/fapi/v1/klines", 1500),
    "spot": ("https://data-api.binance.vision", "/api/v3/klines", 1000),
}
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
CSV_COLUMNS = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    # Binance sends these in every kline and they carry information OHLCV
    # cannot: which side was the aggressor, and whether the flow came from
    # many small trades or a few large ones.
    "quote_volume",
    "trade_count",
    "taker_buy_base",
)


class MarketDataError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BinanceMarketDataClient:
    """Read-only Binance kline client; no API key or trading surface.

    Reads the perpetual contract by default, because that is the instrument
    being traded; set CRYPTO_MARKET=spot for the spot series instead.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        request_pause_seconds: float = 0.06,
        opener: Callable[..., object] | None = None,
        market_name: str | None = None,
    ) -> None:
        if timeout_seconds <= 0 or request_pause_seconds < 0:
            raise ValueError("Gecersiz veri istemcisi zaman ayari")
        self.timeout_seconds = timeout_seconds
        self.request_pause_seconds = request_pause_seconds
        self._opener = opener or urlopen
        selected_market = market() if market_name is None else market_name.strip().lower()
        if selected_market not in MARKET_ENDPOINTS:
            raise ValueError("Piyasa yalnizca futures veya spot olabilir")
        self.market_name = selected_market
        self.origin, self.path, self.page_limit = MARKET_ENDPOINTS[selected_market]

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        return self._fetch_klines(
            validate_symbol(symbol),
            validate_interval(interval),
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def fetch_market_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch a validated public market outside the configured model set."""
        return self._fetch_klines(
            validate_market_symbol(symbol),
            validate_interval(interval),
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("Gecersiz kline tarih araligi")
        step_ms = INTERVAL_MILLISECONDS[interval]
        cursor = start_ms
        rows: list[list[object]] = []
        while cursor < end_ms:
            payload: list[list[object]] | None = None
            for attempt in range(3):
                try:
                    payload = self._request_page(
                        symbol=symbol,
                        interval=interval,
                        start_ms=cursor,
                        end_ms=end_ms,
                    )
                    break
                except MarketDataError:
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (2**attempt))
            if payload is None:
                raise MarketDataError("Binance piyasa verisi alinamadi")
            if not payload:
                break
            rows.extend(payload)
            last_open = _integer(payload[-1][0], "open time")
            next_cursor = last_open + step_ms
            if next_cursor <= cursor:
                raise MarketDataError("Binance sayfalama ilerlemedi")
            cursor = next_cursor
            if len(payload) < self.page_limit:
                break
            time.sleep(self.request_pause_seconds)
        return _rows_to_frame(rows, closed_before_ms=end_ms)

    def _request_page(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": self.page_limit,
            }
        )
        request = Request(
            f"{self.origin}{self.path}?{query}",
            headers={"Accept": "application/json", "User-Agent": "btc-eth-probability-bot/0.1"},
            method="GET",
        )
        try:
            response = self._opener(request, timeout=self.timeout_seconds)
            with response:  # type: ignore[attr-defined]
                status = response.getcode()  # type: ignore[attr-defined]
                raw = response.read(MAX_RESPONSE_BYTES + 1)  # type: ignore[attr-defined]
        except (HTTPError, URLError, TimeoutError, OSError):
            raise MarketDataError("Binance piyasa verisi alinamadi") from None
        except Exception:
            raise MarketDataError("Binance piyasa verisi alinamadi") from None
        if status != 200 or not raw or len(raw) > MAX_RESPONSE_BYTES:
            raise MarketDataError("Binance piyasa verisi yaniti gecersiz")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise MarketDataError("Binance piyasa verisi JSON degil") from None
        if not isinstance(decoded, list):
            raise MarketDataError("Binance kline yaniti liste degil")
        for row in decoded:
            if not isinstance(row, list) or len(row) < 11:
                raise MarketDataError("Binance kline satiri gecersiz")
        return decoded


def update_cache(
    data_dir: Path,
    symbol: str,
    interval: str,
    *,
    days: int,
    client: BinanceMarketDataClient | None = None,
    now: datetime | None = None,
    warn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)
    if days < 1:
        raise ValueError("Gun sayisi en az 1 olmali")
    now = (now or _utc_now()).astimezone(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    requested_start = int((now - timedelta(days=days)).timestamp() * 1000)
    path = cache_path(data_dir, symbol, interval)
    existing = _empty_frame()
    if path.exists():
        try:
            existing = load_cache(path)
        except MarketDataError:
            # An older cache predates the taker-flow columns, or is corrupt.
            # Re-downloading costs a few requests; guessing does not.
            if warn is not None:
                warn(f"{symbol} {interval}: onbellek okunamadi, bastan indiriliyor")
    if existing.empty:
        start_ms = requested_start
    else:
        # Re-fetch the last candle so a formerly open candle can never remain cached.
        start_ms = max(requested_start, int(existing["open_time_ms"].iloc[-1]))
    fetched = (client or BinanceMarketDataClient()).fetch_klines(
        symbol,
        interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        raise MarketDataError(f"{symbol} {interval} icin kapali mum bulunamadi")
    combined = _validate_frame(combined)
    combined = combined[combined["open_time_ms"] >= requested_start].reset_index(drop=True)
    # An exchange halt leaves a hole in the series.  Rejecting the whole cache
    # would wedge the bot until someone deletes the file by hand, so keep the
    # newest contiguous run instead and say how much was dropped.
    before = len(combined)
    combined = _trim_to_contiguous_tail(combined, INTERVAL_MILLISECONDS[interval])
    dropped = before - len(combined)
    if dropped and warn is not None:
        warn(
            f"{symbol} {interval}: mum araliginda kopukluk bulundu; en eski {dropped} mum "
            "atildi ve son kesintisiz dizi tutuldu"
        )
    if combined.empty:
        raise MarketDataError(f"{symbol} {interval} icin kesintisiz mum dizisi yok")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return combined


def update_market_cache(
    path: Path,
    symbol: str,
    interval: str,
    *,
    days: int,
    client: BinanceMarketDataClient,
    now: datetime | None = None,
    warn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Update a cache for a public symbol outside the configured model set.

    This is used by the isolated scalp observer.  The caller supplies an
    explicit, already-scoped path so broad-universe data can never overwrite a
    model cache.
    """
    symbol = validate_market_symbol(symbol)
    interval = validate_interval(interval)
    if client.market_name != "futures":
        raise ValueError("Scalp piyasa onbellegi futures istemcisi gerektirir")
    if days < 1:
        raise ValueError("Gun sayisi en az 1 olmali")
    now = (now or _utc_now()).astimezone(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    requested_start = int((now - timedelta(days=days)).timestamp() * 1000)
    existing = _empty_frame()
    if path.exists():
        try:
            existing = load_cache(path)
        except MarketDataError:
            if warn is not None:
                warn(f"{symbol} {interval}: scalp onbellegi okunamadi, bastan indiriliyor")
    start_ms = (
        requested_start
        if existing.empty
        else max(requested_start, int(existing["open_time_ms"].iloc[-1]))
    )
    fetched = client.fetch_market_klines(
        symbol,
        interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        raise MarketDataError(f"{symbol} {interval} icin kapali mum bulunamadi")
    combined = _validate_frame(combined)
    combined = combined[combined["open_time_ms"] >= requested_start].reset_index(drop=True)
    before = len(combined)
    combined = _trim_to_contiguous_tail(combined, INTERVAL_MILLISECONDS[interval])
    dropped = before - len(combined)
    if dropped and warn is not None:
        warn(
            f"{symbol} {interval}: scalp mum araliginda kopukluk bulundu; "
            f"en eski {dropped} mum atildi"
        )
    if combined.empty:
        raise MarketDataError(f"{symbol} {interval} icin kesintisiz mum dizisi yok")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return combined


def load_cache(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError):
        raise MarketDataError(f"Veri onbellegi okunamadi: {path}") from None
    return _validate_frame(frame)


def _rows_to_frame(rows: list[list[object]], *, closed_before_ms: int) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    normalized: list[dict[str, float | int]] = []
    for row in rows:
        close_time = _integer(row[6], "close time")
        if close_time >= closed_before_ms:
            continue
        normalized.append(
            {
                "open_time_ms": _integer(row[0], "open time"),
                "open": _positive_float(row[1], "open"),
                "high": _positive_float(row[2], "high"),
                "low": _positive_float(row[3], "low"),
                "close": _positive_float(row[4], "close"),
                "volume": _nonnegative_float(row[5], "volume"),
                "close_time_ms": close_time,
                "quote_volume": _nonnegative_float(row[7], "quote volume"),
                "trade_count": _integer(row[8], "trade count"),
                "taker_buy_base": _nonnegative_float(row[9], "taker buy volume"),
            }
        )
    return _validate_frame(pd.DataFrame(normalized, columns=CSV_COLUMNS))


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CSV_COLUMNS) - set(frame.columns)
    if missing:
        raise MarketDataError(f"Eksik veri sutunlari: {', '.join(sorted(missing))}")
    result = frame.loc[:, CSV_COLUMNS].copy()
    for column in ("open_time_ms", "close_time_ms", "trade_count"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    result = result.sort_values("open_time_ms").drop_duplicates("open_time_ms", keep="last")
    if result.empty:
        return _empty_frame()
    price_columns = ["open", "high", "low", "close"]
    if (result[price_columns] <= 0).any().any() or (result["volume"] < 0).any():
        raise MarketDataError("Negatif veya sifir piyasa verisi")
    if (result["trade_count"] < 0).any() or (result["quote_volume"] < 0).any():
        raise MarketDataError("Negatif islem sayisi veya hacmi")
    # A tolerance because Binance rounds the two volumes independently.
    if (result["taker_buy_base"] > result["volume"] * 1.0001 + 1e-8).any():
        raise MarketDataError("Taker alim hacmi toplam hacmi asiyor")
    if (result["high"] < result[price_columns].max(axis=1)).any():
        raise MarketDataError("Kline high alani tutarsiz")
    if (result["low"] > result[price_columns].min(axis=1)).any():
        raise MarketDataError("Kline low alani tutarsiz")
    if (result["close_time_ms"] <= result["open_time_ms"]).any():
        raise MarketDataError("Kline zaman sirasi tutarsiz")
    return result.reset_index(drop=True)


def _trim_to_contiguous_tail(frame: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if len(frame) < 2:
        return frame
    differences = frame["open_time_ms"].diff()
    broken = differences.ne(step_ms) & differences.notna()
    positions = broken.to_numpy().nonzero()[0]
    if positions.size == 0:
        return frame
    return frame.iloc[int(positions[-1]) :].reset_index(drop=True)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_COLUMNS)


def _integer(value: object, label: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise MarketDataError(f"Gecersiz {label}") from None
    if result < 0:
        raise MarketDataError(f"Gecersiz {label}")
    return result


def _positive_float(value: object, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result <= 0:
        raise MarketDataError(f"Gecersiz {label}")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise MarketDataError(f"Gecersiz {label}") from None
    if not pd.notna(result) or result < 0:
        raise MarketDataError(f"Gecersiz {label}")
    return result
