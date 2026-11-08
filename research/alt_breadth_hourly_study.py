"""Fixed-rule, long-only hourly breakout gated by broad-market breadth.

Research-only Trade3 data. The single strategy is frozen before its OOS run:
BTC bull alignment, >=65% breadth among >=100 warmed-up symbols, a 12h high
breakout with trend/volume confirmation, next-hour entry, and +2/+3% target
with -1.5% stop. No live or Trade1 code is imported or changed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
OOS_START = "2026-09-01T00:00:00Z"
OOS_MID = "2026-09-17T00:00:00Z"
OOS_END = "2026-10-03T18:00:00Z"
TARGETS = (2.0, 3.0)
STOP = 1.5
HOLD_HOURS = 12
PORTFOLIO_CAP = 10
MIN_BREADTH_SYMBOLS = 100
BREADTH_THRESHOLD = 0.65
COSTS = (30.0, 50.0)


def load_frames(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close",
                "volume", "quote_volume", "taker_buy_base"}
    for path in sorted(directory.glob("*_1h.csv")):
        f = pd.read_csv(path)
        if not required.issubset(f.columns) or len(f) < 240:
            continue
        f = f.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        f["open_time_ms"] = f.open_time_ms.astype("int64")
        if f.open_time_ms.diff().dropna().ne(HOUR_MS).any():
            raise ValueError(f"Gapped hourly history: {path.name}")
        for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
            f[col] = pd.to_numeric(f[col], errors="coerce")
        f["ema20"] = f.close.ewm(span=20, adjust=False, min_periods=20).mean()
        f["ema50"] = f.close.ewm(span=50, adjust=False, min_periods=50).mean()
        f["ema200"] = f.close.ewm(span=200, adjust=False, min_periods=200).mean()
        f["ret24"] = f.close.pct_change(24)
        f["ema50_rising6"] = f.ema50 > f.ema50.shift(6)
        f["quote_vol_ratio"] = f.quote_volume / f.quote_volume.shift(1).rolling(24, min_periods=18).median()
        f["taker_ratio"] = f.taker_buy_base / f.volume.replace(0, np.nan)
        f["prior_high12"] = f.high.shift(1).rolling(12, min_periods=12).max()
        frames[path.name.removesuffix("_1h.csv")] = f
    if "BTCUSDT" not in frames or len(frames) < MIN_BREADTH_SYMBOLS:
        raise ValueError(f"Need BTCUSDT and >= {MIN_BREADTH_SYMBOLS} hourly markets")
    return frames


def build_breadth(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible_count: dict[int, int] = defaultdict(int)
    bull_count: dict[int, int] = defaultdict(int)
    for f in frames.values():
        valid = f.ema200.notna() & f.ret24.notna() & f.close.notna()
        bull = valid & (f.close > f.ema50) & (f.ema20 > f.ema50) & (f.ret24 > 0)
        for stamp in f.loc[valid, "open_time_ms"].to_numpy(dtype=np.int64):
            eligible_count[int(stamp)] += 1
        for stamp in f.loc[bull, "open_time_ms"].to_numpy(dtype=np.int64):
            bull_count[int(stamp)] += 1
    times = sorted(eligible_count)
    breadth = pd.DataFrame({
        "open_time_ms": times,
        "eligible_symbols": [eligible_count[t] for t in times],
        "bull_symbols": [bull_count.get(t, 0) for t in times],
    })
    breadth["breadth"] = breadth.bull_symbols / breadth.eligible_symbols
    return breadth.set_index("open_time_ms")


def make_events(frames: dict[str, pd.DataFrame], breadth: pd.DataFrame) -> list[dict]:
    btc = frames["BTCUSDT"]
    btc_bull = (btc.close > btc.ema50) & (btc.ema50 > btc.ema200) & btc.ema50_rising6
    btc_bull_by_time = pd.Series(btc_bull.to_numpy(dtype=bool), index=btc.open_time_ms)
    events = []
    for symbol, f in frames.items():
        if symbol == "BTCUSDT":
            continue
        times = f.open_time_ms
        broad = breadth.reindex(times)
        market_bull = btc_bull_by_time.reindex(times, fill_value=False).to_numpy(dtype=bool)
        own_trend = (f.close > f.ema50) & (f.ema20 > f.ema50) & f.ema50_rising6
        flow = (f.quote_vol_ratio >= 1.5) & (f.taker_ratio >= .55)
        breakout = f.close > f.prior_high12
        mask = (market_bull & (broad.eligible_symbols.to_numpy() >= MIN_BREADTH_SYMBOLS) &
                (broad.breadth.to_numpy() >= BREADTH_THRESHOLD) & own_trend.to_numpy() &
                flow.to_numpy() & breakout.to_numpy())
        finite = np.isfinite(f.quote_vol_ratio.to_numpy()) & np.isfinite(f.taker_ratio.to_numpy())
        for i in np.flatnonzero(mask & finite):
            events.append({"symbol": symbol, "i": int(i), "signal_ms": int(f.close_time_ms.iloc[i]),
                           "bar": f, "breadth": float(broad.breadth.iloc[i]),
                           "eligible_symbols": int(broad.eligible_symbols.iloc[i])})
    return sorted(events, key=lambda e: (e["signal_ms"], e["symbol"]))


def resolve(event: dict, target_pct: float) -> dict | None:
    f, i = event["bar"], event["i"]
    entry_i, last_i = i + 1, i + HOLD_HOURS
    if last_i >= len(f):
        return None
    entry = float(f.open.iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None
    target, stop = entry * (1 + target_pct / 100), entry * (1 - STOP / 100)
    exit_price, exit_i, reason = float(f.close.iloc[last_i]), last_i, "TIME_12H"
    for j in range(entry_i, last_i + 1):
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
            "gross_bps": (exit_price / entry - 1) * 10_000, "reason": reason,
            "breadth": float(event["breadth"]), "eligible_symbols": int(event["eligible_symbols"])}


def daily_ci(rows: list[dict], cost: float, reps: int = 5000):
    by_day: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_day[r["signal_ms"] // DAY_MS].append(r["gross_bps"] - cost)
    if len(by_day) < 12:
        return None
    days = np.asarray(sorted(by_day), dtype=np.int64)
    rng = np.random.default_rng(20261006)
    means = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(days, size=len(days), replace=True)
        means[i] = np.mean([v for d in sample for v in by_day[int(d)]])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def summarize(rows: list[dict], cost: float) -> dict:
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    pnl = np.asarray([r["gross_bps"] - cost for r in rows], dtype=float)
    pos, neg = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len({r["signal_ms"] // DAY_MS for r in rows}),
            "symbols": len({r["symbol"] for r in rows}), "mean_net_bps": float(pnl.mean()),
            "median_net_bps": float(np.median(pnl)), "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(pos.sum() / -neg.sum()) if len(neg) else None,
            "target_exits": sum(r["reason"].startswith("TARGET") for r in rows),
            "daily_ci95": daily_ci(rows, cost)}


def portfolio_for(events: list[dict], target_pct: float) -> list[dict]:
    rows, active = [], []
    for event in events:
        active = [r for r in active if r["exit_ms"] > event["signal_ms"]]
        if len(active) >= PORTFOLIO_CAP or any(r["symbol"] == event["symbol"] for r in active):
            continue
        trade = resolve(event, target_pct)
        if trade is not None:
            rows.append(trade)
            active.append(trade)
    return rows


def run(data: Path, out: Path) -> dict:
    frames = load_frames(data)
    breadth = build_breadth(frames)
    events = make_events(frames, breadth)
    start_ms = int(pd.Timestamp(OOS_START).timestamp() * 1000)
    mid_ms = int(pd.Timestamp(OOS_MID).timestamp() * 1000)
    end_ms = int(pd.Timestamp(OOS_END).timestamp() * 1000)
    oos_events = [e for e in events if start_ms <= e["signal_ms"] < end_ms]
    min_start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    max_end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    common_start = max(int(f.open_time_ms.iloc[0]) for f in frames.values())
    common_end = min(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    variants = []
    trade_records = []
    for target in TARGETS:
        rows = portfolio_for(oos_events, target)
        rows30, rows50 = summarize(rows, 30), summarize(rows, 50)
        ci = rows30.get("daily_ci95")
        gates = {"positive_mean_30bps": bool(rows30.get("mean_net_bps") is not None and rows30["mean_net_bps"] > 0),
                 "positive_mean_50bps": bool(rows50.get("mean_net_bps") is not None and rows50["mean_net_bps"] > 0),
                 "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
                 "at_least_100_trades": rows30["n"] >= 100,
                 "at_least_15_signal_days": rows30["days"] >= 15}
        fold1 = [r for r in rows if start_ms <= r["signal_ms"] < mid_ms]
        fold2 = [r for r in rows if mid_ms <= r["signal_ms"] < end_ms]
        variants.append({"target_pct": target, "stop_pct": STOP, "all_30bps": rows30,
                         "all_50bps": rows50, "fold1_30bps": summarize(fold1, 30),
                         "fold2_30bps": summarize(fold2, 30), "gates": gates,
                         "pass": all(gates.values())})
        trade_records.extend({**r, "target_pct": target,
                              "segment": "fold1" if r["signal_ms"] < mid_ms else "fold2"} for r in rows)
    payload = {"data_source": str(data), "symbols": len(frames), "window_union_utc": [
                   pd.to_datetime(min_start, unit="ms", utc=True).isoformat(),
                   pd.to_datetime(max_end + HOUR_MS - 1, unit="ms", utc=True).isoformat()],
               "window_common_utc": [pd.to_datetime(common_start, unit="ms", utc=True).isoformat(),
                                     pd.to_datetime(common_end + HOUR_MS - 1, unit="ms", utc=True).isoformat()],
               "oos_utc": [OOS_START, OOS_END], "fold_mid_utc": OOS_MID,
               "breadth_threshold": BREADTH_THRESHOLD, "minimum_breadth_symbols": MIN_BREADTH_SYMBOLS,
               "signal_events_oos": len(oos_events), "signal_symbols_oos": len({e["symbol"] for e in oos_events}),
               "max_oos_breadth_symbols": int(breadth.loc[(breadth.index >= start_ms) & (breadth.index < end_ms), "eligible_symbols"].max()),
               "rule": "BTC close>EMA50>EMA200 with EMA50 rising6; cross-sectional breadth >=65% of >=100 warmed-up symbols (coin close>EMA50, EMA20>EMA50, 24h return>0); own close>EMA50, EMA20>EMA50, EMA50 rising6; close breaks prior 12h high; quote-volume ratio>=1.5; taker-buy ratio>=0.55.",
               "execution": {"entry": "next hourly open", "max_hold_hours": HOLD_HOURS,
                             "same_bar_ambiguity": "stop first", "gap_fill": "at open",
                             "stop_pct": STOP, "targets_pct": list(TARGETS),
                             "max_concurrent_positions": PORTFOLIO_CAP,
                             "one_position_per_symbol": True, "round_trip_cost_bps": list(COSTS)},
               "variants": variants, "trades": trade_records,
               "limitations": ["The broad hourly universe has variable listing/coverage history and survivor bias.",
                               "The common all-symbol period is shorter than the union; breadth denominator is dynamic and requires 100 warmed-up symbols.",
                               "The OOS window and source data have been inspected by earlier research; this is exploratory, not independent proof.",
                               "30/50 bps are round-trip stress haircuts, not historical funding/spread/fill replay.",
                               "Hourly candles resolve ambiguous stop/target bars stop-first."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(trade_records).to_csv(out / "trades.csv", index=False)
    lines = ["# Trade3 hourly breadth-gated long breakout", "",
             f"Symbols: {len(frames)}. Union {payload['window_union_utc'][0]}–{payload['window_union_utc'][1]}; common overlap {payload['window_common_utc'][0]}–{payload['window_common_utc'][1]}.",
             f"OOS: {OOS_START} to {OOS_END}; fold split {OOS_MID}; OOS signals {len(oos_events)} across {payload['signal_symbols_oos']} symbols.",
             f"Fixed rule: {payload['rule']}",
             f"Execution: next-hour open, 12h max hold, stop-first ambiguity, 30/50 bps round-trip cost, max {PORTFOLIO_CAP} concurrent positions.", "",
             "| TP | OOS n/days | Net mean 30/50 bps | PF30 | Win30 | TP exits | CI95 daily30 | Fold1 / Fold2 mean30 | Pass? |",
             "|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for v in variants:
        a, b = v["all_30bps"], v["all_50bps"]
        lines.append(f"| +{v['target_pct']}% | {a['n']}/{a['days']} | {a.get('mean_net_bps')}/{b.get('mean_net_bps')} | {a.get('profit_factor')} | {a.get('win_pct')}% | {a.get('target_exits')} | {a.get('daily_ci95')} | {v['fold1_30bps'].get('mean_net_bps')}/{v['fold2_30bps'].get('mean_net_bps')} | {'YES' if v['pass'] else 'NO'} |")
    lines += ["", "Promotion gates: positive cost-adjusted mean at both 30 and 50 bps, positive lower daily-block 95% CI, >=100 OOS trades and >=15 signal days.", "", "## Limitations", ""] + [f"- {x}" for x in payload["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/research/server-long-scout-20261003/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/alt-breadth-hourly-20261006"))
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({"symbols": result["symbols"], "oos_events": result["signal_events_oos"],
                      "oos_symbols": result["signal_symbols_oos"], "variants": result["variants"],
                      "passes_any": any(v["pass"] for v in result["variants"]),
                      "output": str(args.output)}, indent=2))
