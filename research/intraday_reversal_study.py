"""Walk-forward test of fixed 5m mean-reversion rules on local Trade3 data.

No live settings or Trade1 data are used. Signals are confirmed at candle
close, entries use the following open, and OHLC ambiguities resolve against
the trade. The current historical window has already been inspected, so this
is a new exploratory family, not an untouched future holdout.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MS = 300_000
MAX_HOLD_BARS = 144  # 12h
BRACKETS = ((2.0, 1.5), (3.0, 1.5), (2.0, 2.5), (3.0, 2.5), (3.0, 3.0))
BOOTSTRAP_REPS = 3000


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw = {}
    for path in sorted(data_dir.glob("*_5m_futures.csv")):
        frame = pd.read_csv(path)
        if not {"open_time_ms", "close_time_ms", "open", "high", "low", "close"}.issubset(frame.columns):
            continue
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if (frame.open_time_ms.diff().dropna() != BAR_MS).any():
            raise ValueError(f"Gap in {path.name}")
        symbol = path.name.removesuffix("_5m_futures.csv")
        raw[symbol] = frame
    if len(raw) < 2:
        raise ValueError(f"Need two continuous 5m futures CSVs in {data_dir}")
    # Mean-reversion features are local to each asset; retain each continuous
    # cache and evaluate its full OOS tail rather than clipping to the shortest.
    frames = raw
    for symbol in frames:
        frame = frames[symbol]
        close, high, low = frame.close, frame.high, frame.low
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        frame["rsi14"] = 100 - 100 / (1 + rs)
        mid = close.rolling(20, min_periods=20).mean()
        sd = close.rolling(20, min_periods=20).std(ddof=0)
        frame["bb_low"] = mid - 2 * sd
        frame["bb_high"] = mid + 2 * sd
        frame["ema200"] = close.ewm(span=200, min_periods=200, adjust=False).mean()
    return frames


def make_events(frames: dict[str, pd.DataFrame]) -> dict[str, list[tuple[str, int, str]]]:
    events = {"RSI_EXHAUSTION": [], "BB_SNAPBACK": []}
    for symbol, f in frames.items():
        up = f.close > f.ema200
        down = f.close < f.ema200
        green = f.close > f.open
        red = f.close < f.open
        long_rsi = (f.rsi14.shift(1) <= 30) & (f.rsi14 > 30) & green & up
        short_rsi = (f.rsi14.shift(1) >= 70) & (f.rsi14 < 70) & red & down
        long_bb = (f.low < f.bb_low) & (f.close > f.bb_low) & green & up
        short_bb = (f.high > f.bb_high) & (f.close < f.bb_high) & red & down
        for name, long_mask, short_mask in (("RSI_EXHAUSTION", long_rsi, short_rsi),
                                           ("BB_SNAPBACK", long_bb, short_bb)):
            events[name].extend((symbol, int(i), "LONG") for i in np.flatnonzero(long_mask.fillna(False).to_numpy()))
            events[name].extend((symbol, int(i), "SHORT") for i in np.flatnonzero(short_mask.fillna(False).to_numpy()))
    return events


def resolve(frame: pd.DataFrame, signal_i: int, target_pct: float, stop_pct: float, side: str) -> dict | None:
    entry_i = signal_i + 1
    end_i = entry_i + MAX_HOLD_BARS - 1
    if end_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open)
    if entry <= 0:
        return None
    if side == "LONG":
        target, stop = entry * (1 + target_pct / 100), entry * (1 - stop_pct / 100)
    else:
        target, stop = entry * (1 - target_pct / 100), entry * (1 + stop_pct / 100)
    exit_price, exit_i, reason = float(frame.iloc[end_i].close), end_i, "TIME_12H"
    for i in range(entry_i, end_i + 1):
        row = frame.iloc[i]
        if side == "LONG":
            if row.open <= stop:
                exit_price, exit_i, reason = float(row.open), i, "STOP_GAP"
                break
            if row.open >= target:
                exit_price, exit_i, reason = float(row.open), i, "TARGET_GAP"
                break
            stop_hit, target_hit = row.low <= stop, row.high >= target
        else:
            if row.open >= stop:
                exit_price, exit_i, reason = float(row.open), i, "STOP_GAP"
                break
            if row.open <= target:
                exit_price, exit_i, reason = float(row.open), i, "TARGET_GAP"
                break
            stop_hit, target_hit = row.high >= stop, row.low <= target
        if stop_hit and target_hit:
            exit_price, exit_i, reason = stop, i, "AMBIGUOUS_STOP_FIRST"
            break
        if stop_hit:
            exit_price, exit_i, reason = stop, i, "STOP"
            break
        if target_hit:
            exit_price, exit_i, reason = target, i, "TARGET"
            break
    gross = ((exit_price - entry) if side == "LONG" else (entry - exit_price)) / entry * 10_000
    return {"symbol": "", "side": side, "signal_ms": int(frame.iloc[signal_i].close_time_ms),
            "entry_ms": int(frame.iloc[entry_i].open_time_ms), "exit_ms": int(frame.iloc[exit_i].close_time_ms),
            "gross_bps": gross, "reason": reason}


def nonoverlap(events, frames, target, stop):
    trades, next_free = [], {}
    ordered = sorted(events, key=lambda e: (int(frames[e[0]].iloc[e[1]].open_time_ms), e[0], e[2]))
    for symbol, i, side in ordered:
        stamp = int(frames[symbol].iloc[i].open_time_ms)
        if stamp <= next_free.get((symbol, side), -1):
            continue
        trade = resolve(frames[symbol], i, target, stop, side)
        if trade is None:
            continue
        trade["symbol"] = symbol
        trades.append(trade)
        next_free[(symbol, side)] = trade["exit_ms"]
    return trades


def summarize(trades, cost):
    if not trades:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    by_day = defaultdict(list)
    values = np.asarray([t["gross_bps"] - cost for t in trades], dtype=float)
    for trade, value in zip(trades, values, strict=True):
        by_day[trade["signal_ms"] // 86_400_000].append(float(value))
    ci = None
    if len(by_day) >= 5:
        rng = np.random.default_rng(20261003)
        days = np.asarray(sorted(by_day), dtype=np.int64)
        draws = [np.mean([v for day in rng.choice(days, len(days), replace=True) for v in by_day[int(day)]])
                 for _ in range(BOOTSTRAP_REPS)]
        ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    gains, losses = values[values > 0], values[values < 0]
    return {"n": len(trades), "symbols": sorted({t["symbol"] for t in trades}), "days": len(by_day),
            "win_pct": float(np.mean(values > 0) * 100), "mean_net_bps": float(values.mean()),
            "median_net_bps": float(np.median(values)),
            "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else None,
            "target_exit_pct": float(np.mean([t["reason"].startswith("TARGET") for t in trades]) * 100),
            "daily_ci95": ci}


def run(data_dir: Path, output: Path, holdout_start_ms: int | None = None):
    frames = load_data(data_dir)
    events = make_events(frames)
    start = max(int(f.open_time_ms.iloc[0]) for f in frames.values())
    end = min(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    cutoff = holdout_start_ms if holdout_start_ms is not None else start + int((end - start) * .70)
    results = []
    for name, candidates in events.items():
        oos_events = [(s, i, side) for s, i, side in candidates
                      if int(frames[s].iloc[i].close_time_ms) >= cutoff]
        for target, stop in BRACKETS:
            trades = nonoverlap(oos_events, frames, target, stop)
            m30, m50 = summarize(trades, 30), summarize(trades, 50)
            ci = m30["daily_ci95"]
            gates = {"mean_positive_30bps": bool(m30["mean_net_bps"] is not None and m30["mean_net_bps"] > 0),
                     "mean_positive_50bps": bool(m50["mean_net_bps"] is not None and m50["mean_net_bps"] > 0),
                     "ci_lower_above_zero": bool(ci and ci[0] > 0), "at_least_100_trades": m30["n"] >= 100,
                     "at_least_15_days": m30["days"] >= 15, "both_symbols": len(m30.get("symbols", [])) == 2}
            results.append({"rule": name, "target_pct": target, "stop_pct": stop,
                            "oos_30bps": m30, "oos_50bps": m50, "gates": gates,
                            "trades": [{**t, "rule": name, "target_pct": target, "stop_pct": stop}
                                       for t in trades]})
    union_start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    union_end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    payload = {"data_window_utc": [utc(union_start), utc(union_end + BAR_MS - 1)],
               "common_window_utc": [utc(start), utc(end + BAR_MS - 1)], "oos_start_utc": utc(cutoff),
               "symbols": list(frames), "split": "fixed-start chronological holdout" if holdout_start_ms is not None else "first 70% chronological, final 30% OOS",
               "entry": "next 5m open after signal close", "max_hold": "12 hours",
               "ambiguity": "same-bar target/stop stop-first; opening gaps filled at open",
               "costs": [30, 50], "results": results,
               "limitations": ["The ten coin-query alt symbols were user-selected, not a random futures-universe sample.",
                               "Several families have been screened on this same date block; any selected winner requires a future untouched confirmation period.",
                               "No historical spread/funding/orderbook data; 30/50 bps are stress haircuts."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for item in results:
        rows.extend({"rule": item["rule"], "target_pct": item["target_pct"], "stop_pct": item["stop_pct"],
                     "cost_bps": cost, **item[f"oos_{cost}bps"], "gates": item["gates"]}
                    for cost in (30, 50))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([trade for item in results for trade in item["trades"]]).to_csv(output / "trades.csv", index=False)
    lines = [f"# Trade3 5m mean-reversion walk-forward — {', '.join(frames)}", "",
             f"Per-symbol window union: {payload['data_window_utc'][0]}–{payload['data_window_utc'][1]}; common interval {payload['common_window_utc'][0]}–{payload['common_window_utc'][1]}; OOS starts {payload['oos_start_utc']}.",
             "Entry next 5m open; 12h max hold; gaps fill at open; same-bar TP/SL = stop first. Costs are round-trip haircuts.", "",
             "| Rule | TP/SL | OOS n/days | 30/50 bps net mean | PF 30 bps | TP exit % | Daily 95% CI | Pass? |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for item in results:
        a, b = item["oos_30bps"], item["oos_50bps"]
        passed = all(item["gates"].values())
        lines.append(f"| {item['rule']} | +{item['target_pct']}/−{item['stop_pct']}% | {a['n']}/{a['days']} | {a['mean_net_bps']}/{b['mean_net_bps']} | {a.get('profit_factor')} | {a.get('target_exit_pct')}% | {a.get('daily_ci95')} | {'YES' if passed else 'NO'} |")
    lines += ["", "## Fixed rules", "", "RSI_EXHAUSTION: RSI(14) crosses back above 30 with a green candle while close is above EMA200 (LONG); mirror: crosses below 70 with a red candle below EMA200 (SHORT). BB_SNAPBACK: candle pierces 2σ Bollinger band and closes back inside with reversal candle, gated by EMA200 regime.", "", "## Gates and limitations", "", "Every candidate must have positive net mean at 30 and 50 bps, positive lower 95% daily-block CI, at least 100 OOS trades over 15 days, and at least two symbols represented. Results are exploratory; the queried cohort is not representative of the full Trade3 universe.", ""] + [f"- {x}" for x in payload["limitations"]]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/intraday-reversal-20261003"))
    parser.add_argument("--oos-start", type=str, default=None, help="Fixed holdout start with timezone")
    args = parser.parse_args()
    oos_start_ms = None
    if args.oos_start:
        oos_start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
        if oos_start.tzinfo is None:
            raise SystemExit("--oos-start must include a timezone")
        oos_start_ms = int(oos_start.astimezone(UTC).timestamp() * 1000)
    result = run(args.data, args.output, holdout_start_ms=oos_start_ms)
    print(json.dumps({"oos_start": result["oos_start_utc"],
                      "passing_candidates": sum(all(x["gates"].values()) for x in result["results"]),
                      "output": str(args.output)}, indent=2))
