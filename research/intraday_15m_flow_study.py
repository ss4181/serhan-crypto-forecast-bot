"""Fixed-rule 15m flow-breakout screen using local Trade3 futures data only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MS = 900_000
DAY_MS = 86_400_000
TARGET_PCT = 2.0
STOP_PCT = 1.5
HOLD_BARS = 48  # 12 hours
COSTS = (30.0, 50.0)


def load(directory: Path) -> dict[str, pd.DataFrame]:
    raw = {}
    required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base"}
    for path in sorted(directory.glob("*_15m_futures.csv")):
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns) or len(frame) < 200:
            continue
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if frame.open_time_ms.diff().dropna().ne(BAR_MS).any():
            raise ValueError(f"{path.name}: gaps in local 15m cache")
        symbol = path.name.removesuffix("_15m_futures.csv")
        raw[symbol] = frame
    if "BTCUSDT" not in raw or len(raw) < 2:
        raise ValueError(f"Need BTCUSDT benchmark and another continuous 15m futures file in {directory}")
    # Preserve each symbol's complete cache. Indicators are local to each
    # asset, and the BTC regime is joined on timestamps in signals(); trimming
    # every asset to the shortest common window needlessly discards recent OOS.
    return raw


def signals(frames: dict[str, pd.DataFrame]) -> list[dict]:
    btc = frames["BTCUSDT"].copy()
    for span in (20, 50, 200):
        btc[f"ema{span}"] = btc.close.ewm(span=span, adjust=False, min_periods=span).mean()
    btc["ret12"] = btc.close.pct_change(12)
    btc_bull = (btc.close > btc.ema50) & (btc.ema50 > btc.ema200) & (btc.ema50 > btc.ema50.shift(4))
    btc_bear = (btc.close < btc.ema50) & (btc.ema50 < btc.ema200) & (btc.ema50 < btc.ema50.shift(4))
    btc_by_time = btc.set_index("open_time_ms")
    result = []
    for symbol, source in frames.items():
        if symbol == "BTCUSDT":
            f = btc.copy()
        else:
            f = source.copy()
            for span in (20, 50, 200):
                f[f"ema{span}"] = f.close.ewm(span=span, adjust=False, min_periods=span).mean()
            f["ret12"] = f.close.pct_change(12)
        f["vol_ratio"] = f.quote_volume / f.quote_volume.shift(1).rolling(24, min_periods=18).median()
        f["taker_ratio"] = f.taker_buy_base / f.volume.replace(0, np.nan)
        f["prior_high12"] = f.high.shift(1).rolling(12, min_periods=12).max()
        f["prior_low12"] = f.low.shift(1).rolling(12, min_periods=12).min()
        aligned = btc_by_time.reindex(f.open_time_ms)
        bull = (aligned.close.to_numpy() > aligned.ema50.to_numpy()) & (aligned.ema50.to_numpy() > aligned.ema200.to_numpy()) & (aligned.ema50.to_numpy() > aligned.ema50.shift(4).to_numpy())
        bear = (aligned.close.to_numpy() < aligned.ema50.to_numpy()) & (aligned.ema50.to_numpy() < aligned.ema200.to_numpy()) & (aligned.ema50.to_numpy() < aligned.ema50.shift(4).to_numpy())
        own_bull = (f.close > f.ema50) & (f.ema20 > f.ema50) & (f.ema50 > f.ema50.shift(4))
        own_bear = (f.close < f.ema50) & (f.ema20 < f.ema50) & (f.ema50 < f.ema50.shift(4))
        long = own_bull.to_numpy() & bull & (f.close.to_numpy() > f.prior_high12.to_numpy()) & (f.vol_ratio.to_numpy() >= 1.5) & (f.taker_ratio.to_numpy() >= .55)
        short = own_bear.to_numpy() & bear & (f.close.to_numpy() < f.prior_low12.to_numpy()) & (f.vol_ratio.to_numpy() >= 1.5) & (f.taker_ratio.to_numpy() <= .45)
        for side, mask in (("LONG", long), ("SHORT", short)):
            for i in np.flatnonzero(mask & np.isfinite(f.vol_ratio.to_numpy()) & np.isfinite(f.taker_ratio.to_numpy())):
                result.append({"symbol": symbol, "side": side, "i": int(i),
                               "signal_ms": int(f.open_time_ms.iloc[i]), "bar": f})
    return sorted(result, key=lambda row: (row["signal_ms"], row["symbol"]))


def replay(event: dict, target_pct: float = TARGET_PCT, stop_pct: float = STOP_PCT) -> dict | None:
    f, i, side = event["bar"], event["i"], event["side"]
    entry_i, deadline = i + 1, i + HOLD_BARS
    if deadline >= len(f):
        return None
    entry = float(f.open.iloc[entry_i])
    target = entry * (1 + target_pct / 100 if side == "LONG" else 1 - target_pct / 100)
    stop = entry * (1 - stop_pct / 100 if side == "LONG" else 1 + stop_pct / 100)
    exit_price, exit_i, reason = float(f.close.iloc[deadline]), deadline, "TIME_12H"
    for j in range(entry_i, deadline + 1):
        o, h, l = float(f.open.iloc[j]), float(f.high.iloc[j]), float(f.low.iloc[j])
        if (side == "LONG" and o <= stop) or (side == "SHORT" and o >= stop):
            exit_price, exit_i, reason = o, j, "STOP_GAP"
            break
        if (side == "LONG" and o >= target) or (side == "SHORT" and o <= target):
            exit_price, exit_i, reason = o, j, "TARGET_GAP"
            break
        stop_hit = l <= stop if side == "LONG" else h >= stop
        target_hit = h >= target if side == "LONG" else l <= target
        if stop_hit and target_hit:
            exit_price, exit_i, reason = stop, j, "AMBIGUOUS_STOP_FIRST"
            break
        if stop_hit:
            exit_price, exit_i, reason = stop, j, "STOP"
            break
        if target_hit:
            exit_price, exit_i, reason = target, j, "TARGET"
            break
    gross = (exit_price / entry - 1 if side == "LONG" else (entry - exit_price) / entry) * 10_000
    return {"symbol": event["symbol"], "side": side, "signal_ms": event["signal_ms"],
            "entry_ms": int(f.open_time_ms.iloc[entry_i]), "exit_ms": int(f.close_time_ms.iloc[exit_i]),
            "gross_bps": gross, "reason": reason}


def daily_ci(rows: list[dict], cost: float, reps: int = 3000):
    daily = defaultdict(list)
    for row in rows:
        daily[row["signal_ms"] // DAY_MS].append(row["gross_bps"] - cost)
    days = sorted(daily)
    if len(days) < 12:
        return None
    rng = np.random.default_rng(20261006)
    draws = np.empty(reps)
    for i in range(reps):
        chosen = rng.choice(days, len(days), replace=True)
        draws[i] = np.mean([value for day in chosen for value in daily[int(day)]])
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def summarize(rows: list[dict], cost: float) -> dict:
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    pnl = np.asarray([row["gross_bps"] - cost for row in rows])
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len({row["signal_ms"] // DAY_MS for row in rows}),
            "symbols": len({row["symbol"] for row in rows}), "mean_net_bps": float(pnl.mean()),
            "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else None,
            "target_exits": sum(row["reason"].startswith("TARGET") for row in rows),
            "daily_ci95": daily_ci(rows, cost)}


def run(data_dir: Path, output: Path, holdout_start_ms: int | None = None,
        target_pct: float = TARGET_PCT, stop_pct: float = STOP_PCT) -> dict:
    frames = load(data_dir)
    events = signals(frames)
    common_first = max(int(frame.open_time_ms.iloc[0]) for frame in frames.values())
    common_last = min(int(frame.open_time_ms.iloc[-1]) for frame in frames.values())
    data_first = min(int(frame.open_time_ms.iloc[0]) for frame in frames.values())
    data_last = max(int(frame.open_time_ms.iloc[-1]) for frame in frames.values())
    cut = holdout_start_ms if holdout_start_ms is not None else common_first + int((common_last - common_first) * .70)
    test_events = [event for event in events if event["signal_ms"] >= cut]
    portfolio, active = [], []
    for event in test_events:
        active = [trade for trade in active if trade["exit_ms"] >= event["signal_ms"]]
        if len(active) >= 2 or any(trade["symbol"] == event["symbol"] for trade in active):
            continue
        trade = replay(event, target_pct=target_pct, stop_pct=stop_pct)
        if trade:
            portfolio.append(trade)
            active.append(trade)
    long_rows = [row for row in portfolio if row["side"] == "LONG"]
    short_rows = [row for row in portfolio if row["side"] == "SHORT"]
    all30, all50 = summarize(portfolio, 30), summarize(portfolio, 50)
    ci = all30.get("daily_ci95")
    gates = {"positive_mean_30bps": bool(all30.get("mean_net_bps") is not None and all30["mean_net_bps"] > 0),
             "positive_mean_50bps": bool(all50.get("mean_net_bps") is not None and all50["mean_net_bps"] > 0),
             "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
             "at_least_100_trades": all30["n"] >= 100,
             "at_least_15_signal_days": all30["days"] >= 15}
    result = {"protocol": {"data_start_utc": datetime.fromtimestamp(data_first / 1000, UTC).isoformat(),
                            "data_end_utc": datetime.fromtimestamp(data_last / 1000, UTC).isoformat(),
                            "common_window_start_utc": datetime.fromtimestamp(common_first / 1000, UTC).isoformat(),
                            "common_window_end_utc": datetime.fromtimestamp(common_last / 1000, UTC).isoformat(),
                            "symbol_windows_utc": {symbol: [datetime.fromtimestamp(int(frame.open_time_ms.iloc[0]) / 1000, UTC).isoformat(),
                                                            datetime.fromtimestamp(int(frame.open_time_ms.iloc[-1]) / 1000, UTC).isoformat()]
                                                    for symbol, frame in frames.items()},
                            "oos_start_utc": datetime.fromtimestamp(cut / 1000, UTC).isoformat(),
                            "timeframe": "15m", "rules": "BTC and coin trend aligned; prior-12-bar breakout; quote-volume ratio >=1.5; buy/sell taker ratio >=0.55/<=0.45",
                            "entry": "next 15m open", "bracket": f"+{target_pct}% target / -{stop_pct}% stop (mirrored by side)", "max_hold": "12h",
                            "same_bar": "stop first", "costs_bps": list(COSTS), "portfolio_cap": 2,
                            "evaluation_mode": "fixed-start untouched chronological holdout" if holdout_start_ms is not None else "first 70% / final 30% chronological split",
                            "limitations": ["The coin-query cohort is user-selected, not a random sample of the full futures universe.",
                                            "30/50 bps are round-trip stress haircuts, not historical fees/funding/spread.",
                                            "OHLC cannot determine intrabar ordering; ambiguous bars are stop-first."]},
              "event_counts": {"all": len(events), "oos": len(test_events),
                               "oos_long": sum(event["side"] == "LONG" for event in test_events),
                               "oos_short": sum(event["side"] == "SHORT" for event in test_events)},
              "portfolio_metrics": {"all_30bps": all30, "all_50bps": all50,
                                    "long_30bps": summarize(long_rows, 30), "short_30bps": summarize(short_rows, 30)},
              "gates": gates, "passes_all_gates": all(gates.values()), "trades": portfolio}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(portfolio).to_csv(output / "oos_trades.csv", index=False)
    lines = ["# Trade3 15m aligned flow breakout", "",
             f"Symbols: {', '.join(frames)}. Per-symbol cache union {result['protocol']['data_start_utc']}–{result['protocol']['data_end_utc']}; common interval {result['protocol']['common_window_start_utc']}–{result['protocol']['common_window_end_utc']}; OOS starts {result['protocol']['oos_start_utc']}; {result['event_counts']}.",
             f"Fixed rule: next-bar entry, {result['protocol']['bracket']}, 12h max hold, max two concurrent positions. Stop-first for ambiguous bars.", "",
             "| Portfolio | n / days | Mean 30/50 bps | PF @30 | 30bps CI95 |", "|---|---:|---:|---:|---|"]
    for label, row30, row50 in (("All", all30, all50), ("Long", summarize(long_rows, 30), summarize(long_rows, 50)),
                                ("Short", summarize(short_rows, 30), summarize(short_rows, 50))):
        lines.append(f"| {label} | {row30['n']} / {row30['days']} | {row30.get('mean_net_bps')} / {row50.get('mean_net_bps')} | {row30.get('profit_factor')} | {row30.get('daily_ci95')} |")
    lines += ["", f"Gates: `{json.dumps(gates)}`. Pass: **{'YES' if all(gates.values()) else 'NO'}**.",
              "", "Exploratory OOS on a previously inspected historical interval; even a pass would require independent forward shadow testing. No live bot settings were changed.", "", "## Limitations", "", *[f"- {item}" for item in result["protocol"]["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/15m-flow-20261006"))
    parser.add_argument("--oos-start", type=str, default=None, help="Fixed holdout start with timezone, e.g. 2026-08-05T08:40:00+00:00")
    parser.add_argument("--target-pct", type=float, default=TARGET_PCT)
    parser.add_argument("--stop-pct", type=float, default=STOP_PCT)
    args = parser.parse_args()
    oos_start_ms = None
    if args.oos_start:
        oos_start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
        if oos_start.tzinfo is None:
            raise SystemExit("--oos-start must include a UTC offset")
        oos_start_ms = int(oos_start.astimezone(UTC).timestamp() * 1000)
    payload = run(args.data, args.output, holdout_start_ms=oos_start_ms,
                  target_pct=args.target_pct, stop_pct=args.stop_pct)
    print(json.dumps({"events": payload["event_counts"], "metrics": payload["portfolio_metrics"],
                      "gates": payload["gates"], "pass": payload["passes_all_gates"],
                      "output": str(args.output)}, indent=2))
