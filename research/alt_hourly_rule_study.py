"""Fixed-rule, offline 1h long scalp/short-swing study on Trade3's cache.

Uses only local server-long-scout CSVs. Signals are known after a closed 1h
bar; entry is the next bar open. Same-bar TP/SL ambiguity is resolved stop
first. This is exploratory: the current-symbol cache has survivor bias and
the period has been inspected in earlier studies. No live settings change.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
RULES = ("BULL_BREAKOUT_FLOW", "BULL_EMA_RECLAIM", "BULL_RELATIVE_STRENGTH",
         "BEAR_BREAKDOWN_FLOW", "BEAR_EMA_REJECT", "BEAR_RELATIVE_WEAKNESS")
BRACKETS = ((2.0, 1.5), (3.0, 1.5), (2.0, 2.5), (3.0, 2.5),
            (2.0, 3.0), (3.0, 3.0), (2.0, 4.0), (3.0, 4.0))
HOLDS = (6, 12, 24)


def load_frames(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("*_1h.csv")):
        try:
            frame = pd.read_csv(path)
            needed = {"open_time_ms", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"}
            if not needed.issubset(frame.columns) or len(frame) < 220:
                continue
            frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
            frame["open_time_ms"] = frame["open_time_ms"].astype("int64")
            for column in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["ema20"] = frame.close.ewm(span=20, adjust=False).mean()
            frame["ema50"] = frame.close.ewm(span=50, adjust=False).mean()
            frame["ema200"] = frame.close.ewm(span=200, adjust=False).mean()
            frame["ret24"] = frame.close.pct_change(24)
            frame["ret6"] = frame.close.pct_change(6)
            frame["quote_vol_ratio"] = frame.quote_volume / frame.quote_volume.shift(1).rolling(24, min_periods=18).median()
            frame["taker_ratio"] = frame.taker_buy_base / frame.volume.replace(0, np.nan)
            frame["prior_high24"] = frame.high.shift(1).rolling(24, min_periods=24).max()
            frame["ema50_rising6"] = frame.ema50 > frame.ema50.shift(6)
            frames[path.name.removesuffix("_1h.csv")] = frame
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    return frames


def build_signals(frames: dict[str, pd.DataFrame]) -> dict[str, list[tuple[str, int]]]:
    btc = frames.get("BTCUSDT")
    if btc is None:
        raise ValueError("BTCUSDT hourly file is required")
    btc_bull = (btc.close > btc.ema50) & (btc.ema50 > btc.ema200) & (btc.ema50 > btc.ema50.shift(6))
    btc_bull_by_time = pd.Series(btc_bull.to_numpy(), index=btc.open_time_ms)
    btc_bear = (btc.close < btc.ema50) & (btc.ema50 < btc.ema200) & (btc.ema50 < btc.ema50.shift(6))
    btc_bear_by_time = pd.Series(btc_bear.to_numpy(), index=btc.open_time_ms)
    btc_ret = btc.set_index("open_time_ms").ret24
    result = {rule: [] for rule in RULES}
    for symbol, f in frames.items():
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            continue
        stamps = f.open_time_ms
        market_bull = btc_bull_by_time.reindex(stamps).fillna(False).to_numpy(dtype=bool)
        market_bear = btc_bear_by_time.reindex(stamps).fillna(False).to_numpy(dtype=bool)
        own_uptrend = (f.close > f.ema50) & (f.ema20 > f.ema50) & f.ema50_rising6
        flow = (f.quote_vol_ratio >= 1.5) & (f.taker_ratio >= 0.55)
        breakout = own_uptrend & market_bull & (f.close > f.prior_high24) & flow
        reclaim = own_uptrend & market_bull & (f.close > f.ema20) & (f.close.shift(1) <= f.ema20.shift(1)) & (f.ret24 > 0)
        btcret = btc_ret.reindex(stamps).to_numpy(dtype=float)
        relative = own_uptrend & market_bull & (f.ret24 >= btcret + 0.03) & (f.quote_vol_ratio >= 1.0) & (f.taker_ratio >= 0.52)
        own_downtrend = (f.close < f.ema50) & (f.ema20 < f.ema50) & (f.ema50 < f.ema50.shift(6))
        sell_flow = (f.quote_vol_ratio >= 1.5) & (f.taker_ratio <= 0.45)
        breakdown = own_downtrend & market_bear & (f.close < f.low.shift(1).rolling(24, min_periods=24).min()) & sell_flow
        reject = own_downtrend & market_bear & (f.close < f.ema20) & (f.close.shift(1) >= f.ema20.shift(1)) & (f.ret24 < 0)
        weak = own_downtrend & market_bear & (f.ret24 <= btcret - 0.03) & (f.quote_vol_ratio >= 1.0) & (f.taker_ratio <= 0.48)
        for rule, mask in ((RULES[0], breakout), (RULES[1], reclaim), (RULES[2], relative),
                           (RULES[3], breakdown), (RULES[4], reject), (RULES[5], weak)):
            positions = np.flatnonzero(mask.fillna(False).to_numpy())
            result[rule].extend((symbol, int(i)) for i in positions)
    return result


def evaluate_one(frame: dict[str, np.ndarray], signal_i: int, target_pct: float, stop_pct: float, hold: int, cost_bps: float, side: str):
    entry_i = signal_i + 1
    end_i = entry_i + hold - 1
    stamps = frame["open_time_ms"]
    if end_i >= len(stamps):
        return None
    if np.any(np.diff(stamps[entry_i:end_i + 1]) != HOUR_MS):
        return None
    entry = float(frame["open"][entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None
    target = entry * (1 + target_pct / 100) if side == "LONG" else entry * (1 - target_pct / 100)
    stop = entry * (1 - stop_pct / 100) if side == "LONG" else entry * (1 + stop_pct / 100)
    exit_price, exit_ms, reason = float(frame["close"][end_i]), int(stamps[end_i] + HOUR_MS), "TIME"
    for j in range(entry_i, end_i + 1):
        touch_target = float(frame["high"][j]) >= target if side == "LONG" else float(frame["low"][j]) <= target
        touch_stop = float(frame["low"][j]) <= stop if side == "LONG" else float(frame["high"][j]) >= stop
        if touch_stop and touch_target:
            exit_price, exit_ms, reason = stop, int(stamps[j] + HOUR_MS), "AMBIGUOUS_STOP_FIRST"
            break
        if touch_stop:
            exit_price, exit_ms, reason = stop, int(stamps[j] + HOUR_MS), "STOP"
            break
        if touch_target:
            exit_price, exit_ms, reason = target, int(stamps[j] + HOUR_MS), "TARGET"
            break
    # Linear USD-M futures return: short P&L is (entry - exit) / entry.
    gross = ((exit_price / entry - 1) if side == "LONG" else (entry - exit_price) / entry) * 10_000
    return {"entry_ms": int(stamps[entry_i]), "exit_ms": exit_ms,
            "gross_bps": gross, "net_bps": gross - cost_bps, "reason": reason}


def simulate(events, bars, target, stop, hold, cost_bps, side):
    trades = []
    last_exit = {}
    for symbol, signal_i in sorted(events, key=lambda item: (int(bars[item[0]]["open_time_ms"][item[1]]), item[0])):
        f = bars[symbol]
        signal_ms = int(f["open_time_ms"][signal_i])
        if signal_ms < last_exit.get(symbol, 0):
            continue
        trade = evaluate_one(f, signal_i, target, stop, hold, cost_bps, side)
        if trade is None:
            continue
        trade.update(symbol=symbol, signal_ms=signal_ms)
        trades.append(trade)
        last_exit[symbol] = trade["exit_ms"]
    return trades


def daily_ci(trades, reps=2000):
    by_day = defaultdict(list)
    for trade in trades:
        by_day[trade["entry_ms"] // DAY_MS].append(trade["net_bps"])
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = np.random.default_rng(20261003)
    draws = np.empty(reps)
    for i in range(reps):
        sampled = rng.choice(days, len(days), replace=True)
        draws[i] = np.mean([v for day in sampled for v in by_day[int(day)]])
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def summarize(trades):
    if not trades:
        return {"n": 0, "days": 0, "symbols": 0, "mean_net_bps": None, "ci95_daily": None}
    returns = np.asarray([t["net_bps"] for t in trades])
    positive = returns[returns > 0].sum()
    negative = -returns[returns < 0].sum()
    return {"n": len(trades), "days": len({t["entry_ms"] // DAY_MS for t in trades}),
            "symbols": len({t["symbol"] for t in trades}), "mean_net_bps": float(returns.mean()),
            "median_net_bps": float(np.median(returns)), "win_pct": float((returns > 0).mean() * 100),
            "profit_factor": float(positive / negative) if negative else None,
            "target_exit_pct": float(np.mean([t["reason"] == "TARGET" for t in trades]) * 100),
            "stop_or_ambiguous_pct": float(np.mean([t["reason"] in {"STOP", "AMBIGUOUS_STOP_FIRST"} for t in trades]) * 100),
            "ci95_daily": daily_ci(trades)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prices", type=Path, nargs="?", default=Path("artifacts/research/server-long-scout-20261003/prices"))
    parser.add_argument("--oos-start", default="2026-08-25T00:00:00Z")
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/alt-hourly-rule-20261003"))
    args = parser.parse_args()
    frames = load_frames(args.prices)
    if not frames:
        raise SystemExit(f"No suitable hourly data found: {args.prices}")
    events_by_rule = build_signals(frames)
    bars = {symbol: {column: frame[column].to_numpy(dtype=np.int64 if column == "open_time_ms" else float)
                     for column in ("open_time_ms", "open", "high", "low", "close")}
            for symbol, frame in frames.items()}
    oos_ms = int(pd.Timestamp(args.oos_start).timestamp() * 1000)
    results = []
    trade_rows = []
    for rule in RULES:
        all_events = events_by_rule[rule]
        oos_events = [(s, i) for s, i in all_events if int(bars[s]["open_time_ms"][i]) >= oos_ms]
        for hold in HOLDS:
            for target, stop in BRACKETS:
                for cost in (30.0, 50.0):
                    side = "LONG" if rule.startswith("BULL_") else "SHORT"
                    trades = simulate(oos_events, bars, target, stop, hold, cost, side)
                    metrics = summarize(trades)
                    row = {"rule": rule, "side": side, "hold_h": hold, "target_pct": target, "stop_pct": stop,
                           "cost_bps": cost, **metrics}
                    results.append(row)
                    if cost == 30:
                        for t in trades:
                            trade_rows.append({"rule": rule, "side": side, "hold_h": hold, "target_pct": target,
                                               "stop_pct": stop, **t})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out / "results.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(args.out / "trades_30bps.csv", index=False)
    data_starts = [int(f.open_time_ms.iloc[0]) for f in frames.values()]
    data_ends = [int(f.open_time_ms.iloc[-1]) for f in frames.values()]
    meta = {"symbols": len(frames), "symbols_with_signals": {r: len({s for s, _ in events_by_rule[r]}) for r in RULES},
            "signals": {r: len(events_by_rule[r]) for r in RULES}, "oos_start": args.oos_start,
            "data_start": pd.to_datetime(min(data_starts), unit="ms", utc=True).isoformat(),
            "data_end": pd.to_datetime(max(data_ends), unit="ms", utc=True).isoformat(),
            "rules": {"BULL_BREAKOUT_FLOW": "BTC bull (EMA50>EMA200 and rising); coin EMA20>EMA50, above EMA50; close breaks prior 24h high; quote volume >=1.5x prior 24h median; taker-buy ratio >=0.55", "BULL_EMA_RECLAIM": "same bull/trend; close crosses back above EMA20; positive 24h return", "BULL_RELATIVE_STRENGTH": "same bull/trend; 24h coin return >= BTC+3%; quote volume >= prior 24h median; taker-buy ratio >=0.52", "BEAR_BREAKDOWN_FLOW": "BTC bear (EMA50<EMA200 and falling); coin EMA20<EMA50, below EMA50; close breaks prior 24h low; quote volume >=1.5x prior 24h median; taker-buy ratio <=0.45", "BEAR_EMA_REJECT": "same bear/trend; close crosses below EMA20; negative 24h return", "BEAR_RELATIVE_WEAKNESS": "same bear/trend; 24h coin return <= BTC-3%; quote volume >= prior 24h median; taker-buy ratio <=0.48"},
            "limitations": ["Exploratory fixed-rule screen; the available date range has been inspected in earlier research.", "Current cached universe creates survivor/listing bias.", "Hourly candles cannot resolve intrabar ordering; stop-first used.", "30/50 bps are round-trip cost stress assumptions, not historical spread/funding replay.", "Only the current local Trade3 data was used; no Trade1 data or live settings touched."]}
    (args.out / "metadata.json").write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")
    base = pd.DataFrame(results)
    condensed = base[base.cost_bps.eq(30)].copy()
    lines = ["# Trade3 fixed-rule hourly long/short strategy screen", "", f"Hourly symbols: {len(frames)}. Data: {meta['data_start']}–{meta['data_end']}. OOS begins {args.oos_start}.", "", "Entry is the next hourly open after a closed signal bar; same-bar target/stop ambiguity is stop-first. Each coin is non-overlapping; no cross-coin portfolio cap. Costs are round-trip stress haircuts.", "", "This period/universe has already been inspected in related studies, so these results are exploratory, not a fresh untouched holdout or profitability proof.", "", "| Fixed rule | Side | Hold | TP/SL | OOS n / days | Net mean 30/50 bps | PF (30bps) | Target / stop-or-ambiguous | Daily CI95 (30bps) | Pass gates? |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for _, row30 in condensed.iterrows():
        other = base[(base.rule == row30.rule) & (base.hold_h == row30.hold_h) & (base.target_pct == row30.target_pct) & (base.stop_pct == row30.stop_pct) & (base.cost_bps == 50)].iloc[0]
        ci = row30.ci95_daily
        mean30 = row30.mean_net_bps
        mean50 = other.mean_net_bps
        try:
            ci_arr = ast.literal_eval(ci) if isinstance(ci, str) and ci.startswith("[") else ci
        except Exception:
            ci_arr = None
        passed = bool(row30.n >= 100 and row30.days >= 15 and mean30 > 0 and mean50 > 0 and ci_arr and ci_arr[0] > 0)
        lines.append(f"| {row30.rule} | {row30.side} | {row30.hold_h}h | +{row30.target_pct:.0f}/−{row30.stop_pct:.1f}% | {row30.n} / {row30.days} | {mean30:.1f}/{mean50:.1f} | {row30.profit_factor:.2f} | {row30.target_exit_pct:.1f}% / {row30.stop_or_ambiguous_pct:.1f}% | {ci if ci is not None else '—'} | {'YES' if passed else 'NO'} |")
    lines += ["", "## Gates", "", "A candidate must have positive OOS mean after both 30 and 50 bps costs, daily-block 95% CI lower bound above zero, at least 100 closed trades over 15 signal days, and no unresolved data gaps. No candidate that misses any gate is eligible for live alerts.", "", "## Limitations", "", *[f"- {item}" for item in meta["limitations"]], ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out / 'report.md'} ({len(results)} metric rows)")


if __name__ == "__main__":
    main()
