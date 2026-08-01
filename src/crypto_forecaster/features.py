from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "momentum_1_atr",
    "momentum_3_atr",
    "momentum_12_atr",
    "ema_8_21_atr",
    "rsi_14_centered",
    "bollinger_z_20",
    "atr_pct_14_log",
    "volume_z_20",
    "candle_pressure",
)

FEATURE_LABELS_TR = {
    "momentum_1_atr": "1 mum momentumu",
    "momentum_3_atr": "3 mum momentumu",
    "momentum_12_atr": "12 mum momentumu",
    "ema_8_21_atr": "EMA(8/21) trend farki",
    "rsi_14_centered": "RSI(14)",
    "bollinger_z_20": "Bollinger(20) konumu",
    "atr_pct_14_log": "ATR(14) oynaklik rejimi",
    "volume_z_20": "20 mum hacim anomalisi",
    "candle_pressure": "Mum ici alici/satici baskisi",
}


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    x: np.ndarray
    y: np.ndarray
    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    atr_pct: np.ndarray
    future_return_atr: np.ndarray
    future_up_atr: np.ndarray
    future_down_atr: np.ndarray

    def __len__(self) -> int:
        return int(self.y.size)


def compute_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    if len(bars) < 40:
        raise ValueError("Belirtec hesabi icin en az 40 mum gerekli")
    frame = bars.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    log_close = np.log(close)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_pct = atr / close
    safe_atr_pct = atr_pct.clip(lower=1e-9)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(loss != 0, 100.0).where(gain != 0, 0.0)

    ema_8 = close.ewm(span=8, adjust=False, min_periods=8).mean()
    ema_21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    log_volume = np.log1p(volume)
    volume_mean = log_volume.rolling(20, min_periods=20).mean()
    volume_std = log_volume.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    candle_range = (high - low).replace(0, np.nan)

    frame["atr"] = atr
    frame["atr_pct"] = atr_pct
    frame["momentum_1_atr"] = log_close.diff(1) / safe_atr_pct
    frame["momentum_3_atr"] = log_close.diff(3) / safe_atr_pct
    frame["momentum_12_atr"] = log_close.diff(12) / safe_atr_pct
    frame["ema_8_21_atr"] = (ema_8 - ema_21) / atr.replace(0, np.nan)
    frame["rsi_14_centered"] = (rsi - 50.0) / 25.0
    frame["bollinger_z_20"] = (close - middle) / deviation
    frame["atr_pct_14_log"] = np.log(safe_atr_pct)
    frame["volume_z_20"] = (log_volume - volume_mean) / volume_std
    frame["candle_pressure"] = (2 * close - high - low) / candle_range
    return frame.replace([np.inf, -np.inf], np.nan)


def build_supervised_dataset(bars: pd.DataFrame) -> SupervisedDataset:
    featured = compute_feature_frame(bars)
    current_close = featured["close"].astype(float)
    atr = featured["atr"].astype(float)
    featured["target"] = (current_close.shift(-1) > current_close).astype(float)
    featured["future_return_atr"] = np.log(current_close.shift(-1) / current_close) / (
        atr / current_close
    ).clip(lower=1e-9)
    featured["future_up_atr"] = (featured["high"].shift(-1) - current_close) / atr
    featured["future_down_atr"] = (current_close - featured["low"].shift(-1)) / atr
    required = list(FEATURE_NAMES) + [
        "target",
        "future_return_atr",
        "future_up_atr",
        "future_down_atr",
        "atr",
        "atr_pct",
    ]
    valid = featured.iloc[:-1].dropna(subset=required).copy()
    if len(valid) < 200:
        raise ValueError("Model icin en az 200 kullanilabilir mum gerekli")
    return SupervisedDataset(
        x=valid.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64),
        y=valid["target"].to_numpy(dtype=np.float64),
        open_time_ms=valid["open_time_ms"].to_numpy(dtype=np.int64),
        close_time_ms=valid["close_time_ms"].to_numpy(dtype=np.int64),
        close=valid["close"].to_numpy(dtype=np.float64),
        atr=valid["atr"].to_numpy(dtype=np.float64),
        atr_pct=valid["atr_pct"].to_numpy(dtype=np.float64),
        future_return_atr=valid["future_return_atr"].to_numpy(dtype=np.float64),
        future_up_atr=valid["future_up_atr"].to_numpy(dtype=np.float64),
        future_down_atr=valid["future_down_atr"].to_numpy(dtype=np.float64),
    )


def latest_feature_vector(bars: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    featured = compute_feature_frame(bars)
    valid = featured.dropna(subset=list(FEATURE_NAMES) + ["atr", "atr_pct"])
    if valid.empty:
        raise ValueError("Son mum icin gecerli belirtec yok")
    row = valid.iloc[-1]
    vector = row.loc[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    return row, vector
