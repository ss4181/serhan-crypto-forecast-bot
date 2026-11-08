"""Predeclared 5m long-rule walk-forward study on local Trade3 futures data.

Uses only OHLCV files in data/. No live bot or Trade1 files are imported or
changed. Rules, holding period, brackets, and promotion gates are fixed here.
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
HOLD_BARS = 144  # 12 hours on 5m candles.
BRACKETS = ((2.0, 1.0), (2.0, 1.5), (3.0, 1.5), (3.0, 2.0))
COSTS_BPS = (30.0, 50.0)
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20261003


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")


def load_data(directory: Path) -> dict[str, pd.DataFrame]:
    raw = {}
    for path in sorted(directory.glob("*_5m_futures.csv")):
        frame = pd.read_csv(path)
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume", "taker_buy_base"}
        if not required.issubset(frame.columns) or len(frame) < 1000:
            continue
        symbol = path.name.removesuffix("_5m_futures.csv")
        if (frame.open_time_ms.diff().dropna() != BAR_MS).any():
            raise ValueError(f"Gaps in {symbol} 5m data")
        raw[symbol] = frame
    if len(raw) < 2:
        raise ValueError(f"Need at least two continuous 5m futures CSVs in {directory}")
    # These rules calculate features per symbol, so retain each continuous
    # cache window instead of truncating all symbols to the shortest overlap.
    return raw


def make_events(frames: dict[str, pd.DataFrame]) -> dict[str, list[tuple[str, int]]]:
    events: dict[str, list[tuple[str, int]]] = {
        "TREND_BREAKOUT_12": [],
        "TREND_BREAKOUT_48": [],
        "BULL_VOLUME_BREAKOUT_12": [],
        "EMA20_RECLAIM": [],
        "BULL_SQUEEZE_RELEASE": [],
    }
    btc = frames.get("BTCUSDT")
    if btc is None:
        raise ValueError("BTCUSDT is required as the market-regime benchmark")
    btc_close = btc.close
    btc_ema50 = btc_close.ewm(span=50, adjust=False, min_periods=50).mean()
    btc_ema200 = btc_close.ewm(span=200, adjust=False, min_periods=200).mean()
    btc_bull = (btc_close > btc_ema50) & (btc_ema50 > btc_ema200) & (btc_ema50 > btc_ema50.shift(12))
    btc_bull_by_time = pd.Series(btc_bull.to_numpy(dtype=bool), index=btc.open_time_ms)
    for symbol, frame in frames.items():
        close = frame.close
        high = frame.high
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        trend = (close > ema200) & (ema50 > ema200) & (ema50 > ema50.shift(12))
        prior_volume_mean = frame.volume.shift(1).rolling(20, min_periods=20).mean()
        buyer_ratio = frame.taker_buy_base / frame.volume.replace(0, np.nan)
        for length, name in ((12, "TREND_BREAKOUT_12"), (48, "TREND_BREAKOUT_48")):
            prior_high = high.shift(1).rolling(length, min_periods=length).max()
            cross = (close > prior_high) & (close.shift(1) <= prior_high.shift(1))
            indices = np.flatnonzero((trend & cross).fillna(False).to_numpy())
            events[name].extend((symbol, int(i)) for i in indices)
        prior_high_12 = high.shift(1).rolling(12, min_periods=12).max()
        breakout_12 = (close > prior_high_12) & (close.shift(1) <= prior_high_12.shift(1))
        volume_confirmed = (frame.volume >= 1.5 * prior_volume_mean) & (buyer_ratio >= 0.55)
        indices = np.flatnonzero((trend & breakout_12 & volume_confirmed).fillna(False).to_numpy())
        events["BULL_VOLUME_BREAKOUT_12"].extend((symbol, int(i)) for i in indices)
        reclaim = (close > ema20) & (close.shift(1) <= ema20.shift(1)) & (close > ema50) & trend
        indices = np.flatnonzero(reclaim.fillna(False).to_numpy())
        events["EMA20_RECLAIM"].extend((symbol, int(i)) for i in indices)

        previous_close = frame.close.shift(1)
        true_range = pd.concat([(frame.high - frame.low), (frame.high - previous_close).abs(),
                                (frame.low - previous_close).abs()], axis=1).max(axis=1)
        atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_width = 2 * close.rolling(20, min_periods=20).std(ddof=0)
        squeeze_on = ((bb_mid + bb_width) < (ema20 + 1.5 * atr14)) & \
                     ((bb_mid - bb_width) > (ema20 - 1.5 * atr14))
        squeeze_release = squeeze_on.shift(1, fill_value=False) & ~squeeze_on
        market_bull = btc_bull_by_time.reindex(frame.open_time_ms, fill_value=False)
        squeeze_signal = (trend & market_bull.to_numpy(dtype=bool) & squeeze_release &
                          (close > frame.open) & (close > ema20) & volume_confirmed)
        indices = np.flatnonzero(squeeze_signal.fillna(False).to_numpy())
        events["BULL_SQUEEZE_RELEASE"].extend((symbol, int(i)) for i in indices)
    return events


def resolve(frame: pd.DataFrame, signal_i: int, target_pct: float, stop_pct: float) -> dict | None:
    entry_i = signal_i + 1
    deadline = entry_i + HOLD_BARS - 1
    if deadline >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open)
    target = entry * (1 + target_pct / 100)
    stop = entry * (1 - stop_pct / 100)
    exit_price = float(frame.iloc[deadline].close)
    exit_i = deadline
    reason = "TIME_12H"
    for i in range(entry_i, deadline + 1):
        row = frame.iloc[i]
        if row.open <= stop:
            exit_price, exit_i, reason = float(row.open), i, "STOP_GAP"
            break
        if row.open >= target:
            exit_price, exit_i, reason = float(row.open), i, "TARGET_GAP"
            break
        stop_hit = row.low <= stop
        target_hit = row.high >= target
        if stop_hit and target_hit:
            exit_price, exit_i, reason = stop, i, "AMBIGUOUS_STOP_FIRST"
            break
        if stop_hit:
            exit_price, exit_i, reason = stop, i, "STOP"
            break
        if target_hit:
            exit_price, exit_i, reason = target, i, "TARGET"
            break
    return {
        "symbol": "",
        "signal_ms": int(frame.iloc[signal_i].close_time_ms),
        "entry_ms": int(frame.iloc[entry_i].open_time_ms),
        "exit_ms": int(frame.iloc[exit_i].close_time_ms),
        "exit_i": exit_i,
        "entry": entry,
        "exit": exit_price,
        "gross_bps": (exit_price / entry - 1) * 10_000,
        "reason": reason,
    }


def nonoverlap(events: list[tuple[str, int]], frames: dict[str, pd.DataFrame],
               target: float, stop: float) -> list[dict]:
    trades = []
    next_free_ms: dict[str, int] = {}
    for symbol, signal_i in sorted(events, key=lambda item: int(frames[item[0]].iloc[item[1]].open_time_ms)):
        frame = frames[symbol]
        signal_open_ms = int(frame.iloc[signal_i].open_time_ms)
        if signal_open_ms <= next_free_ms.get(symbol, -1):
            continue
        trade = resolve(frame, signal_i, target, stop)
        if trade is None:
            continue
        trade["symbol"] = symbol
        trades.append(trade)
        # Permit the next signal only after the prior exit bar has closed.
        next_free_ms[symbol] = int(trade["exit_ms"])
    return trades


def summarize(trades: list[dict], cost: float) -> dict:
    if not trades:
        return {"n": 0, "signal_days": 0, "mean_net_bps": None, "daily_block_ci95": None}
    by_day: dict[int, list[float]] = defaultdict(list)
    values = np.asarray([float(t["gross_bps"]) - cost for t in trades], dtype=float)
    for trade, value in zip(trades, values, strict=True):
        by_day[trade["signal_ms"] // 86_400_000].append(float(value))
    pos, neg = values[values > 0], values[values < 0]
    ci = None
    if len(by_day) >= 5:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        keys = np.asarray(sorted(by_day), dtype=np.int64)
        means = []
        for _ in range(BOOTSTRAP_REPS):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            means.append(float(np.mean([v for day in chosen for v in by_day[int(day)]])))
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "n": len(trades),
        "coins": sorted({t["symbol"] for t in trades}),
        "signal_days": len(by_day),
        "wins": int((values > 0).sum()),
        "win_rate_pct": float(np.mean(values > 0) * 100),
        "mean_net_bps": float(values.mean()),
        "median_net_bps": float(np.median(values)),
        "profit_factor": float(pos.sum() / -neg.sum()) if len(neg) else None,
        "daily_block_ci95": ci,
        "exit_reasons": {str(k): int(v) for k, v in pd.Series([t["reason"] for t in trades]).value_counts().items()},
    }


def run(data_dir: Path, output: Path, holdout_start_ms: int | None = None,
        only_rule: str | None = None) -> dict:
    frames = load_data(data_dir)
    events = make_events(frames)
    if only_rule is not None:
        if only_rule not in events:
            raise ValueError(f"Unknown rule {only_rule!r}; choices are {', '.join(events)}")
        events = {only_rule: events[only_rule]}
    start = max(int(f.open_time_ms.iloc[0]) for f in frames.values())
    end = min(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    union_start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    union_end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    cut = holdout_start_ms if holdout_start_ms is not None else start + int((end - start) * 0.70)
    results = []
    trades_out = []
    gates = []
    for name, candidates in events.items():
        for target, stop in BRACKETS:
            trades = nonoverlap(candidates, frames, target, stop)
            train = [t for t in trades if t["signal_ms"] < cut and t["exit_ms"] < cut]
            purged = [t for t in trades if t["signal_ms"] < cut and t["exit_ms"] >= cut]
            oos = [t for t in trades if t["signal_ms"] >= cut]
            oos30 = summarize(oos, 30)
            oos50 = summarize(oos, 50)
            ci = oos30.get("daily_block_ci95")
            passed = {
                "positive_oos_mean_30bps": bool(oos30.get("mean_net_bps") is not None and oos30["mean_net_bps"] > 0),
                "positive_oos_mean_50bps": bool(oos50.get("mean_net_bps") is not None and oos50["mean_net_bps"] > 0),
                "daily_block_ci_lower_above_zero": bool(ci and ci[0] > 0),
                "at_least_100_oos_trades": bool(oos30.get("n", 0) >= 100),
                "at_least_15_oos_signal_days": bool(oos30.get("signal_days", 0) >= 15),
                "at_least_two_symbols_present_oos": bool(len(oos30.get("coins", [])) >= 2),
            }
            row = {
                "strategy": name,
                "target_pct": target,
                "stop_pct": stop,
                "training_30bps": summarize(train, 30),
                "purged_train_crossing_cut": len(purged),
                "oos_30bps": oos30,
                "oos_50bps": oos50,
                "gates": passed,
            }
            results.append(row)
            gates.append(all(passed.values()))
            trades_out.extend({**t, "strategy": name, "target_pct": target, "stop_pct": stop,
                               "segment": "oos" if t["signal_ms"] >= cut else
                               "train_purged" if t["exit_ms"] >= cut else "train"} for t in trades)
    selected_from_training = {}
    for name in sorted({row["strategy"] for row in results}):
        family = [row for row in results if row["strategy"] == name]
        selected = max(family, key=lambda row: (row["training_30bps"].get("mean_net_bps")
                                                if row["training_30bps"].get("mean_net_bps") is not None
                                                else float("-inf")))
        selected_from_training[name] = {
            "selection_rule": "select the fixed target/stop bracket with highest pre-cut training mean after 30bps; training outcomes crossing cutoff purged",
            "target_pct": selected["target_pct"],
            "stop_pct": selected["stop_pct"],
            "training_30bps": selected["training_30bps"],
            "oos_30bps": selected["oos_30bps"],
            "oos_50bps": selected["oos_50bps"],
            "gates": selected["gates"],
            "passes_all_gates": all(selected["gates"].values()),
        }
    payload = {
        "protocol": {
            "data_source": str(data_dir),
            "timeframe": "5m",
            "symbols": list(frames),
            "window_start_utc": utc(union_start),
            "window_end_utc": utc(union_end + BAR_MS - 1),
            "common_window_start_utc": utc(start),
            "common_window_end_utc": utc(end + BAR_MS - 1),
            "symbol_windows_utc": {symbol: [utc(int(frame.open_time_ms.iloc[0])),
                                            utc(int(frame.open_time_ms.iloc[-1]) + BAR_MS - 1)]
                                    for symbol, frame in frames.items()},
            "oos_start_utc": utc(cut),
            "split": "fixed-start chronological holdout; pre-holdout outcomes crossing cut purged" if holdout_start_ms is not None else "first 70% train, final 30% chronological OOS; training outcomes crossing cut purged",
            "entry": "next 5m open after confirmed signal close",
            "hold_max": "12 hours (144 x 5m bars)",
            "same_bar_stop_target": "stop first",
            "costs_round_trip_bps": list(COSTS_BPS),
            "rules": {
                "trend": "close > EMA200; EMA50 > EMA200; EMA50 > EMA50[12]",
                "TREND_BREAKOUT_12": "confirmed close crosses above prior 12-bar high while trend is true",
                "TREND_BREAKOUT_48": "confirmed close crosses above prior 48-bar high while trend is true",
                "BULL_VOLUME_BREAKOUT_12": "TREND_BREAKOUT_12 plus volume >= 1.5x previous 20-bar mean and taker-buy base volume >= 55%",
                "EMA20_RECLAIM": "confirmed close crosses above EMA20, close > EMA50, and trend is true",
                "BULL_SQUEEZE_RELEASE": "BTC close > EMA50 > EMA200 with rising EMA50; coin trend true; BB(20,2) exits a prior squeeze inside EMA20 +/- 1.5 ATR(14) Keltner; green close above EMA20; quote-volume >= 1.5x previous 20-bar mean and taker-buy base volume >= 55%",
            },
            "brackets_target_stop_pct": [list(x) for x in BRACKETS],
            "promotion_gate": "all six gates in each row must pass; no live promotion from these tests",
            "limitations": [
                "The queried-coin cohort is user-selected, not a random sample of the full futures universe.",
                "No historical spread/funding/orderbook data; 30/50 bps round-trip cost haircuts are stress assumptions.",
                "OHLC cannot resolve same-bar barrier order; those cases are conservatively stop-first.",
                "This fixed candidate set is exploratory; OOS has still been inspected by this study and is not a fresh future period.",
            ],
        },
        "results": results,
        "selected_by_training_only": selected_from_training,
        "any_candidate_passes_all_gates": any(gates),
        "trades": trades_out,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(trades_out).to_csv(output / "trades.csv", index=False)
    lines = [
        "# Trade3 5m long strategies — chronological walk-forward",
        "",
        f"Symbols: {', '.join(frames)}. Per-symbol window union {payload['protocol']['window_start_utc']}–{payload['protocol']['window_end_utc']}; common interval {payload['protocol']['common_window_start_utc']}–{payload['protocol']['common_window_end_utc']}; OOS starts {payload['protocol']['oos_start_utc']}. Entry next 5m open; max hold 12h; same-bar stop/target = stop first.",
        "Costs shown are round-trip haircuts, not historical fee/funding replay.",
        "",
        "| Rule | TP/SL | Train n / net mean 30bps | OOS n / days | OOS net mean 30/50bps | PF 30bps | Daily 95% CI | All gates? |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        tr, o30, o50 = row["training_30bps"], row["oos_30bps"], row["oos_50bps"]
        ci = o30.get("daily_block_ci95")
        lines.append(
            f"| {row['strategy']} | +{row['target_pct']}/−{row['stop_pct']}% | {tr.get('n', 0)} / {tr.get('mean_net_bps')} | {o30.get('n', 0)} / {o30.get('signal_days', 0)} | {o30.get('mean_net_bps')} / {o50.get('mean_net_bps')} | {o30.get('profit_factor')} | {ci} | {'YES' if all(row['gates'].values()) else 'NO'} |"
        )
    lines += ["", "## Gate definitions", "", "OOS mean must be positive at both 30 and 50 bps; daily-block 95% CI lower bound > 0; at least 100 closed OOS trades across at least 15 signal days; at least two symbols represented. No candidate that misses a gate is eligible for live activation.", "", "## Limitations", "", *[f"- {x}" for x in payload["protocol"]["limitations"]], ""]
    lines += ["", "## Bracket selected using training only", "", "Within each signal family, choose the bracket with the highest training mean after 30 bps; OOS outcomes are not used for bracket selection.", "", "| Family | Selected TP/SL | Train n / mean30 | OOS n/days | OOS mean 30/50 | Gates pass? |", "|---|---:|---:|---:|---:|---|"]
    for name, row in selected_from_training.items():
        train, o30, o50 = row["training_30bps"], row["oos_30bps"], row["oos_50bps"]
        lines.append(f"| {name} | +{row['target_pct']}/−{row['stop_pct']}% | {train.get('n')} / {train.get('mean_net_bps')} | {o30.get('n')}/{o30.get('signal_days')} | {o30.get('mean_net_bps')}/{o50.get('mean_net_bps')} | {'YES' if row['passes_all_gates'] else 'NO'} |")
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/intraday-long-20261003"))
    parser.add_argument("--oos-start", type=str, default=None, help="Fixed holdout start with timezone")
    parser.add_argument("--only-rule", type=str, default=None, help="Evaluate only one fixed signal family")
    args = parser.parse_args()
    oos_start_ms = None
    if args.oos_start:
        from datetime import UTC, datetime
        oos_start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
        if oos_start.tzinfo is None:
            raise SystemExit("--oos-start must include a timezone")
        oos_start_ms = int(oos_start.astimezone(UTC).timestamp() * 1000)
    result = run(args.data, args.output, holdout_start_ms=oos_start_ms, only_rule=args.only_rule)
    print(json.dumps({
        "window": result["protocol"]["window_start_utc"] + " to " + result["protocol"]["window_end_utc"],
        "oos_start": result["protocol"]["oos_start_utc"],
        "passes_any": result["any_candidate_passes_all_gates"],
        "candidates": [{"rule": r["strategy"], "bracket": [r["target_pct"], r["stop_pct"]],
                        "oos": r["oos_30bps"], "oos_50bps_mean": r["oos_50bps"]["mean_net_bps"],
                        "gates": r["gates"]} for r in result["results"]],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
