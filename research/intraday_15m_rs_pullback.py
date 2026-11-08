"""Predeclared 15m long-only relative-strength pullback study.

Research-only. Uses local Trade3 futures OHLCV, a fixed chronological holdout,
next-bar entries, stop-first intrabar ambiguity, and conservative round-trip
cost haircuts. No live bot or Trade1 code is imported or modified.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MS = 900_000
HOLD_BARS = 48
TARGET_PCT = 2.0
STOP_PCT = 1.5


def load(directory: Path) -> dict[str, pd.DataFrame]:
    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close",
                "volume", "quote_volume", "taker_buy_base"}
    raw = {}
    for path in sorted(directory.glob("*_15m_futures.csv")):
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns):
            continue
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if len(frame) < 300 or frame.open_time_ms.diff().dropna().ne(BAR_MS).any():
            continue
        raw[path.name.removesuffix("_15m_futures.csv")] = frame
    if "BTCUSDT" not in raw or len(raw) < 3:
        raise ValueError("Need BTCUSDT and at least two continuous 15m symbol caches")
    # Keep each symbol's full cache. Benchmark features are aligned by the
    # actual bar timestamp below, so common-window clipping is unnecessary.
    return raw


def make_events(frames: dict[str, pd.DataFrame]) -> list[dict]:
    btc = frames["BTCUSDT"].copy()
    for span in (20, 50, 200):
        btc[f"ema{span}"] = btc.close.ewm(span=span, adjust=False, min_periods=span).mean()
    btc_bull = (btc.close > btc.ema50) & (btc.ema50 > btc.ema200) & (btc.ema50 > btc.ema50.shift(4))
    btc_by_time = btc.set_index("open_time_ms")
    events = []
    for symbol, source in frames.items():
        if symbol == "BTCUSDT":
            continue
        f = source.copy()
        for span in (20, 50, 200):
            f[f"ema{span}"] = f.close.ewm(span=span, adjust=False, min_periods=span).mean()
        benchmark = btc_by_time.reindex(f.open_time_ms)
        rs1h = f.close.pct_change(4).to_numpy() - benchmark.close.pct_change(4).to_numpy()
        rs4h = f.close.pct_change(16).to_numpy() - benchmark.close.pct_change(16).to_numpy()
        vol_med = f.quote_volume.shift(1).rolling(24, min_periods=18).median()
        vol_ratio = f.quote_volume / vol_med
        taker_ratio = f.taker_buy_base / f.volume.replace(0, np.nan)
        # Reindex by timestamps, not row numbers, to preserve benchmark alignment.
        bull = (benchmark.close.to_numpy() > benchmark.ema50.to_numpy()) & \
               (benchmark.ema50.to_numpy() > benchmark.ema200.to_numpy()) & \
               (benchmark.ema50.to_numpy() > benchmark.ema50.shift(4).to_numpy())
        own_trend = (f.close > f.ema200) & (f.ema20 > f.ema50) & (f.ema50 > f.ema50.shift(4))
        pullback = (f.low.shift(1).rolling(4, min_periods=1).min() <= f.ema20.shift(1)) & \
                   (f.close > f.ema20) & (f.close > f.open)
        mask = bull & own_trend.to_numpy() & pullback.to_numpy() & \
               (rs1h > 0) & (rs4h > 0) & \
               (vol_ratio.to_numpy() >= 1.2) & (taker_ratio.to_numpy() >= .50)
        for i in np.flatnonzero(mask & np.isfinite(vol_ratio.to_numpy()) & np.isfinite(rs4h)):
            events.append({"symbol": symbol, "i": int(i), "signal_ms": int(f.open_time_ms.iloc[i]), "bar": f})
    return sorted(events, key=lambda x: (x["signal_ms"], x["symbol"]))


def resolve(event: dict) -> dict | None:
    f, i = event["bar"], event["i"]
    entry_i, deadline = i + 1, i + HOLD_BARS
    if deadline >= len(f):
        return None
    entry = float(f.open.iloc[entry_i])
    target, stop = entry * (1 + TARGET_PCT / 100), entry * (1 - STOP_PCT / 100)
    exit_price, exit_i, reason = float(f.close.iloc[deadline]), deadline, "TIME_12H"
    for j in range(entry_i, deadline + 1):
        o, h, l = float(f.open.iloc[j]), float(f.high.iloc[j]), float(f.low.iloc[j])
        if o <= stop:
            exit_price, exit_i, reason = o, j, "STOP_GAP"
            break
        if o >= target:
            exit_price, exit_i, reason = o, j, "TARGET_GAP"
            break
        if l <= stop and h >= target:
            exit_price, exit_i, reason = stop, j, "AMBIGUOUS_STOP_FIRST"
            break
        if l <= stop:
            exit_price, exit_i, reason = stop, j, "STOP"
            break
        if h >= target:
            exit_price, exit_i, reason = target, j, "TARGET"
            break
    return {"symbol": event["symbol"], "signal_ms": event["signal_ms"],
            "entry_ms": int(f.open_time_ms.iloc[entry_i]), "exit_ms": int(f.close_time_ms.iloc[exit_i]),
            "gross_bps": (exit_price / entry - 1) * 10_000, "reason": reason}


def summarize(rows: list[dict], cost: float) -> dict:
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    daily = defaultdict(list)
    pnl = np.asarray([r["gross_bps"] - cost for r in rows])
    for row, value in zip(rows, pnl, strict=True):
        daily[row["signal_ms"] // 86_400_000].append(float(value))
    ci = None
    if len(daily) >= 12:
        rng = np.random.default_rng(20261006)
        days = np.array(sorted(daily))
        draws = [np.mean([v for d in rng.choice(days, len(days), replace=True) for v in daily[int(d)]])
                 for _ in range(3000)]
        ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    pos, neg = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len(daily), "symbols": len({r["symbol"] for r in rows}),
            "mean_net_bps": float(pnl.mean()), "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(pos.sum() / -neg.sum()) if len(neg) else None,
            "target_exits": sum(r["reason"].startswith("TARGET") for r in rows), "daily_ci95": ci}


def run(data: Path, output: Path, oos_start: str) -> dict:
    frames = load(data)
    events = make_events(frames)
    start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    cut = int(datetime.fromisoformat(oos_start.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1000)
    candidates = [resolve(e) for e in events if e["signal_ms"] >= cut]
    rows, active = [], []
    for trade in candidates:
        if trade is None:
            continue
        active = [t for t in active if t["exit_ms"] >= trade["signal_ms"]]
        if len(active) >= 2 or any(t["symbol"] == trade["symbol"] for t in active):
            continue
        rows.append(trade)
        active.append(trade)
    m30, m50 = summarize(rows, 30), summarize(rows, 50)
    ci = m30.get("daily_ci95")
    gates = {"positive_mean_30bps": bool(m30.get("mean_net_bps") is not None and m30["mean_net_bps"] > 0),
             "positive_mean_50bps": bool(m50.get("mean_net_bps") is not None and m50["mean_net_bps"] > 0),
             "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
             "at_least_100_trades": m30["n"] >= 100, "at_least_15_signal_days": m30["days"] >= 15}
    payload = {"symbols": list(frames), "window_utc": [datetime.fromtimestamp(start / 1000, UTC).isoformat(), datetime.fromtimestamp(end / 1000, UTC).isoformat()],
               "oos_start_utc": oos_start, "oos_event_count": len([e for e in events if e["signal_ms"] >= cut]),
               "rule": "BTC close>EMA50>EMA200 with rising EMA50; coin close>EMA200, EMA20>EMA50 and rising EMA50; coin 1h and 4h returns exceed BTC; pullback below prior EMA20 then bullish reclaim; quote-volume ratio>=1.2 and taker-buy ratio>=0.50.",
               "bracket": "+2% target / -1.5% stop; 12h max hold; max 2 concurrent; next 15m open; stop-first ambiguity",
               "metrics_30bps": m30, "metrics_50bps": m50, "gates": gates, "pass": all(gates.values()),
               "limitations": ["Exploratory only: the selected OOS interval has already been inspected by other studies.", "Coin-query cohort is user-selected, not representative of all futures.", "30/50 bps are round-trip stress assumptions; no historical spread, funding, or order-book replay.", "OHLC ambiguity resolves against the trade; independent forward shadow confirmation is mandatory before any live alerting."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "oos_trades.csv", index=False)
    (output / "report.md").write_text(
        f"# Trade3 15m relative-strength pullback (long-only)\n\n"
        f"Symbols: {', '.join(frames)}. Window {payload['window_utc'][0]}–{payload['window_utc'][1]}; OOS starts {oos_start}.\n\n"
        f"Rule: {payload['rule']}\n\nBracket: {payload['bracket']}.\n\n"
        f"| OOS trades/events | Days | Net mean 30/50 bps | PF @30 | Win @30 | Target exits | Daily 95% CI @30 | Gates pass? |\n|---:|---:|---:|---:|---:|---:|---|---|\n"
        f"| {m30['n']}/{payload['oos_event_count']} | {m30['days']} | {m30.get('mean_net_bps')}/{m50.get('mean_net_bps')} | {m30.get('profit_factor')} | {m30.get('win_pct')}% | {m30.get('target_exits')} | {m30.get('daily_ci95')} | {payload['pass']} |\n\n"
        f"Gates: `{json.dumps(gates)}`. No strategy was promoted and no live settings were changed.\n\n## Limitations\n\n"+
        "\n".join(f"- {item}" for item in payload["limitations"])+"\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oos-start", required=True)
    args = parser.parse_args()
    result = run(args.data, args.output, args.oos_start)
    print(json.dumps({"oos_events": result["oos_event_count"], "metrics_30bps": result["metrics_30bps"],
                      "metrics_50bps": result["metrics_50bps"], "gates": result["gates"], "pass": result["pass"]}, indent=2))
