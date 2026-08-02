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
# Tried and rejected: taker buy/sell imbalance (raw level, 12-candle average,
# 20-candle z-score, and 3-candle change) plus average trade size.  On identical
# walk-forward folds every form made the net edge worse in 16 of 18 model
# comparisons -- the aggressor split is largely a restatement of
# candle_pressure, so it added variance without information.  The underlying
# columns are still cached, so a different model class can revisit them.

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


UPPER_FIRST = 1
LOWER_FIRST = -1
NO_TOUCH = 0
BOTH_IN_ONE_CANDLE = 2


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
    barrier_bps: np.ndarray
    first_touch: np.ndarray
    timeout_return_bps: np.ndarray
    exit_offset: np.ndarray

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def label_horizon(self) -> int:
        """How many candles ahead a label can look.

        Splits must be separated by at least this much: a barrier label reads
        the next `horizon` candles, so a one-candle embargo would let the
        training set see the same price action its neighbour is scored on.
        """
        return int(np.max(self.exit_offset)) if self.exit_offset.size else 1

    def outcome_bps(
        self, side: np.ndarray, cost_bps: float, *, indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Basis points a trade on `side` earns after fees.

        The ambiguous case — one candle that reaches both barriers — is charged
        as a loss to whichever side is being scored, because the candle does not
        say which level came first and a trader cannot assume the good one.
        """
        side = np.asarray(side, dtype=np.float64).reshape(-1)
        if indices is None:
            touch, barrier, timeout = self.first_touch, self.barrier_bps, self.timeout_return_bps
        else:
            selection = np.asarray(indices, dtype=np.int64).reshape(-1)
            touch = self.first_touch[selection]
            barrier = self.barrier_bps[selection]
            timeout = self.timeout_return_bps[selection]
        if side.size != touch.size:
            raise ValueError("Yon dizisi ile ornek sayisi uyusmuyor")
        favourable = np.where(side >= 0, UPPER_FIRST, LOWER_FIRST)
        gross = np.where(
            touch == favourable,
            barrier,
            np.where(
                touch == -favourable,
                -barrier,
                np.where(touch == BOTH_IN_ONE_CANDLE, -barrier, side * timeout),
            ),
        )
        return gross - cost_bps


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


def build_supervised_dataset(
    bars: pd.DataFrame,
    *,
    barrier_atr_multiple: float = 1.0,
    barrier_horizon_candles: int = 12,
    minimum_barrier_bps: float = 0.0,
) -> SupervisedDataset:
    """Label each candle by which barrier a trade opened there would hit first.

    The old label — "is the next close higher than this one" — treats a one
    basis point drift and a full swing as the same event, so a model could look
    accurate while losing money on fees.  Here the answer is worth a known
    amount before it is counted.
    """
    if barrier_horizon_candles < 1:
        raise ValueError("Bariyer ufku en az 1 mum olmali")
    if barrier_atr_multiple < 0 or minimum_barrier_bps < 0:
        raise ValueError("Gecersiz bariyer olcusu")
    if barrier_atr_multiple == 0 and minimum_barrier_bps <= 0:
        raise ValueError("Bariyer genisligi sifir olamaz")
    featured = compute_feature_frame(bars)
    current_close = featured["close"].astype(float)
    atr = featured["atr"].astype(float)
    atr_pct = featured["atr_pct"].astype(float)
    # Never place the target closer than the trade costs to reach and leave.
    barrier_bps = np.maximum(
        barrier_atr_multiple * atr_pct.to_numpy() * 10_000.0, minimum_barrier_bps
    )
    featured["barrier_bps"] = barrier_bps
    touch, timeout_bps, exit_offset = _first_barrier_touch(
        high=featured["high"].astype(float).to_numpy(),
        low=featured["low"].astype(float).to_numpy(),
        close=current_close.to_numpy(),
        barrier_bps=barrier_bps,
        horizon=barrier_horizon_candles,
    )
    featured["first_touch"] = touch
    featured["timeout_return_bps"] = timeout_bps
    featured["exit_offset"] = exit_offset
    resolved_up = touch == UPPER_FIRST
    resolved_down = touch == LOWER_FIRST
    featured["target"] = np.where(
        resolved_up, 1.0, np.where(resolved_down, 0.0, (timeout_bps > 0).astype(float))
    )
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
    # Drop the tail whose barriers cannot be resolved yet, not just one candle.
    valid = featured.iloc[:-barrier_horizon_candles].dropna(subset=required).copy()
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
        barrier_bps=valid["barrier_bps"].to_numpy(dtype=np.float64),
        first_touch=valid["first_touch"].to_numpy(dtype=np.int64),
        timeout_return_bps=valid["timeout_return_bps"].to_numpy(dtype=np.float64),
        exit_offset=valid["exit_offset"].to_numpy(dtype=np.int64),
    )


def _first_barrier_touch(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    barrier_bps: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = close.size
    upper = close * np.exp(barrier_bps / 10_000.0)
    lower = close * np.exp(-barrier_bps / 10_000.0)
    beyond = horizon + 1
    first_up = np.full(count, beyond, dtype=np.int64)
    first_down = np.full(count, beyond, dtype=np.int64)
    for offset in range(1, horizon + 1):
        future_high = _shift_back(high, offset)
        future_low = _shift_back(low, offset)
        reached_up = (first_up == beyond) & (future_high >= upper)
        reached_down = (first_down == beyond) & (future_low <= lower)
        first_up = np.where(reached_up, offset, first_up)
        first_down = np.where(reached_down, offset, first_down)
    touch = np.where(
        first_up < first_down,
        UPPER_FIRST,
        np.where(
            first_down < first_up,
            LOWER_FIRST,
            np.where(first_up <= horizon, BOTH_IN_ONE_CANDLE, NO_TOUCH),
        ),
    ).astype(np.int64)
    timeout_close = _shift_back(close, horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        timeout_bps = np.log(timeout_close / close) * 10_000.0
    exit_offset = np.minimum(np.minimum(first_up, first_down), horizon).astype(np.int64)
    return touch, timeout_bps, exit_offset


def _shift_back(values: np.ndarray, offset: int) -> np.ndarray:
    shifted = np.full(values.shape, np.nan, dtype=np.float64)
    if offset < values.size:
        shifted[: values.size - offset] = values[offset:]
    return shifted


def latest_feature_vector(bars: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    featured = compute_feature_frame(bars)
    valid = featured.dropna(subset=list(FEATURE_NAMES) + ["atr", "atr_pct"])
    if valid.empty:
        raise ValueError("Son mum icin gecerli belirtec yok")
    row = valid.iloc[-1]
    vector = row.loc[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    return row, vector
