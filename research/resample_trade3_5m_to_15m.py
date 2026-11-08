"""Build local 15m OHLCV bars from Trade3's continuous 5m CSV cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BAR_MS = 300_000
OUT_MS = 900_000


def resample(source: Path, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(source.glob("*_5m_futures.csv")):
        frame = pd.read_csv(path)
        required = {"open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
                    "quote_volume", "trade_count", "taker_buy_base"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if frame.open_time_ms.diff().dropna().ne(BAR_MS).any():
            raise ValueError(f"5m gaps in {path.name}")
        frame["bucket"] = frame.open_time_ms // OUT_MS * OUT_MS
        grouped = frame.groupby("bucket", sort=True).agg(
            bars=("open_time_ms", "size"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"), trade_count=("trade_count", "sum"),
            taker_buy_base=("taker_buy_base", "sum"),
        )
        grouped = grouped[grouped.bars == 3].drop(columns="bars")
        if grouped.empty:
            continue
        grouped.insert(0, "open_time_ms", grouped.index.astype("int64"))
        grouped["close_time_ms"] = grouped.open_time_ms + OUT_MS - 1
        output = grouped[["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
                          "quote_volume", "trade_count", "taker_buy_base"]].reset_index(drop=True)
        if output.open_time_ms.diff().dropna().ne(OUT_MS).any():
            raise ValueError(f"15m gaps after resampling {path.name}")
        symbol = path.name.removesuffix("_5m_futures.csv")
        target = destination / f"{symbol}_15m_futures.csv"
        output.to_csv(target, index=False)
        report[symbol] = {"bars": len(output), "first_open_ms": int(output.open_time_ms.iloc[0]),
                          "last_open_ms": int(output.open_time_ms.iloc[-1])}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = resample(args.source, args.destination)
    print({"symbols": len(result), "bars": result})
