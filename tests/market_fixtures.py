"""Shared candle builders.

Binance sends more than OHLCV in every kline, so test frames have to carry the
same columns the cache does.  Keeping that in one place means the next schema
change touches one file instead of five.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_forecaster.data import CSV_COLUMNS


def with_flow(
    frame: pd.DataFrame,
    *,
    buy_ratio: float | np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Attach quote volume, trade count and taker buy volume to an OHLCV frame.

    ``buy_ratio`` is the share of volume that hit the ask; leave it None for a
    balanced, mildly noisy book, or pass a value to simulate one-sided flow.
    """
    rng = np.random.default_rng(seed)
    result = frame.copy()
    volume = result["volume"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    if buy_ratio is None:
        ratio = rng.uniform(0.35, 0.65, volume.size)
    else:
        ratio = np.broadcast_to(np.asarray(buy_ratio, dtype=float), volume.shape)
    # Trade count must not be a fixed multiple of volume, or average trade size
    # would be constant and its z-score undefined.
    result["quote_volume"] = volume * close
    result["trade_count"] = (
        np.maximum(1.0, volume * 3.0) + rng.integers(5, 60, volume.size)
    ).astype(np.int64)
    result["taker_buy_base"] = volume * ratio
    return result.loc[:, list(CSV_COLUMNS)]


def ohlcv(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_price: np.ndarray | None = None,
    start_ms: int = 1_700_000_000_000,
    step_ms: int = 300_000,
) -> pd.DataFrame:
    count = close.size
    open_time = start_ms + np.arange(count) * step_ms
    return pd.DataFrame(
        {
            "open_time_ms": open_time,
            "open": np.r_[close[0], close[:-1]] if open_price is None else open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time_ms": open_time + step_ms - 1,
        }
    )


__all__ = ["ohlcv", "with_flow"]
