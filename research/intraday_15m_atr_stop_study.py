"""15m BTC/ETH flow-breakout with a causal ATR-based protective stop."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_15m_flow_study import COSTS, DAY_MS, HOLD_BARS, TARGET_PCT, load, signals

ATR_MULTIPLE = 1.5


def replay_atr(event: dict) -> dict | None:
    frame, i, side = event["bar"], event["i"], event["side"]
    entry_i, deadline = i + 1, i + HOLD_BARS
    if deadline >= len(frame):
        return None
    entry = float(frame.open.iloc[entry_i])
    stop_width = float(frame.atr14.iloc[i]) * ATR_MULTIPLE
    if not np.isfinite(stop_width) or entry <= 0 or stop_width <= 0:
        return None
    target = entry * (1 + TARGET_PCT / 100 if side == "LONG" else 1 - TARGET_PCT / 100)
    stop = entry - stop_width if side == "LONG" else entry + stop_width
    exit_price, exit_i, reason = float(frame.close.iloc[deadline]), deadline, "TIME_12H"
    for j in range(entry_i, deadline + 1):
        o, h, l = float(frame.open.iloc[j]), float(frame.high.iloc[j]), float(frame.low.iloc[j])
        stop_gap = o <= stop if side == "LONG" else o >= stop
        target_gap = o >= target if side == "LONG" else o <= target
        if stop_gap:
            exit_price, exit_i, reason = o, j, "STOP_GAP"
            break
        if target_gap:
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
            "entry_ms": int(frame.open_time_ms.iloc[entry_i]), "exit_ms": int(frame.close_time_ms.iloc[exit_i]),
            "gross_bps": gross, "stop_pct": stop_width / entry * 100, "reason": reason}


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
    pnl = np.asarray([r["gross_bps"] - cost for r in rows])
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len({r["signal_ms"] // DAY_MS for r in rows}),
            "symbols": len({r["symbol"] for r in rows}), "mean_net_bps": float(pnl.mean()),
            "median_net_bps": float(np.median(pnl)), "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else None,
            "target_exits": sum(r["reason"].startswith("TARGET") for r in rows),
            "mean_stop_pct": float(np.mean([r["stop_pct"] for r in rows])),
            "daily_ci95": daily_ci(rows, cost)}


def run(data_dir: Path, output: Path, holdout_start_ms: int | None = None) -> dict:
    frames = load(data_dir)
    for frame in frames.values():
        previous = frame.close.shift(1)
        true_range = pd.concat([(frame.high - frame.low), (frame.high - previous).abs(),
                                (frame.low - previous).abs()], axis=1).max(axis=1)
        frame["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    events = signals(frames)
    first = max(int(frame.open_time_ms.iloc[0]) for frame in frames.values())
    last = min(int(frame.open_time_ms.iloc[-1]) for frame in frames.values())
    cut = holdout_start_ms if holdout_start_ms is not None else first + int((last - first) * .70)
    oos_events = [event for event in events if event["signal_ms"] >= cut]
    portfolio, active = [], []
    for event in oos_events:
        active = [trade for trade in active if trade["exit_ms"] >= event["signal_ms"]]
        if len(active) >= 2 or any(trade["symbol"] == event["symbol"] for trade in active):
            continue
        trade = replay_atr(event)
        if trade:
            portfolio.append(trade)
            active.append(trade)
    metrics30, metrics50 = summarize(portfolio, 30), summarize(portfolio, 50)
    ci = metrics30.get("daily_ci95")
    gates = {"positive_mean_30bps": bool(metrics30.get("mean_net_bps") is not None and metrics30["mean_net_bps"] > 0),
             "positive_mean_50bps": bool(metrics50.get("mean_net_bps") is not None and metrics50["mean_net_bps"] > 0),
             "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
             "at_least_100_trades": metrics30["n"] >= 100,
             "at_least_15_days": metrics30["days"] >= 15}
    result = {"protocol": {"timeframe": "15m", "target": "+2%", "stop": "1.5 x Wilder ATR(14) measured at signal close",
                            "max_hold": "12h", "entry": "next bar open", "costs_bps": list(COSTS),
                            "oos_start_utc": datetime.fromtimestamp(cut / 1000, UTC).isoformat(),
                            "evaluation_mode": "fixed-start untouched chronological holdout" if holdout_start_ms is not None else "first 70% / final 30% chronological split",
                            "same_bar": "stop-first", "portfolio": "max 2 concurrent; one per symbol",
                            "data_source": "local Trade3 BTC/ETH 15m cache only",
                            "limitations": ["This inspected interval is exploratory, not a fresh independent holdout.",
                                            "Only BTC and ETH have year-long local 15m coverage.",
                                            "30/50 bps are stress haircuts; historical funding, spreads and fees are unavailable.",
                                            "Stop distance can vary considerably with volatility; no additional cap is tuned."]},
              "events": {"total": len(events), "oos": len(oos_events), "executed": len(portfolio)},
              "metrics_30bps": metrics30, "metrics_50bps": metrics50,
              "long_30bps": summarize([r for r in portfolio if r["side"] == "LONG"], 30),
              "short_30bps": summarize([r for r in portfolio if r["side"] == "SHORT"], 30),
              "gates": gates, "passes_all_gates": all(gates.values()), "trades": portfolio}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(portfolio).to_csv(output / "oos_trades.csv", index=False)
    lines = ["# Trade3 15m ATR-stop flow breakout", "",
             f"Fixed +2% target; stop = 1.5 × ATR(14) at signal close; 12h max hold; next-bar entry. OOS from {result['protocol']['oos_start_utc']}.",
             "ATR uses Wilder smoothing of true range and only data available by the signal close. OHLC ambiguous bars are stop-first.", "",
             "| Portfolio | n/days | Net mean 30/50 bps | Win% | PF @30 | Mean stop | Daily CI95 @30 |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, rows, c30, c50 in (("All", portfolio, metrics30, metrics50),
                                ("Long", [r for r in portfolio if r["side"] == "LONG"], summarize([r for r in portfolio if r["side"] == "LONG"], 30), summarize([r for r in portfolio if r["side"] == "LONG"], 50)),
                                ("Short", [r for r in portfolio if r["side"] == "SHORT"], summarize([r for r in portfolio if r["side"] == "SHORT"], 30), summarize([r for r in portfolio if r["side"] == "SHORT"], 50))):
        lines.append(f"| {name} | {c30['n']}/{c30['days']} | {c30.get('mean_net_bps')} / {c50.get('mean_net_bps')} | {c30.get('win_pct')} | {c30.get('profit_factor')} | {c30.get('mean_stop_pct')}% | {c30.get('daily_ci95')} |")
    lines += ["", f"Gates: `{json.dumps(gates)}`. Pass: **{'YES' if all(gates.values()) else 'NO'}**.",
              "", "No production bot setting was changed. A historical pass would still require new forward shadow data.", "", "## Limitations", "", *[f"- {x}" for x in result["protocol"]["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/15m-atr-stop-20261006"))
    parser.add_argument("--oos-start", type=str, default=None, help="Fixed holdout start with timezone, e.g. 2026-08-05T08:40:00+00:00")
    args = parser.parse_args()
    oos_start_ms = None
    if args.oos_start:
        oos_start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
        if oos_start.tzinfo is None:
            raise SystemExit("--oos-start must include a UTC offset")
        oos_start_ms = int(oos_start.astimezone(UTC).timestamp() * 1000)
    result = run(args.data, args.output, holdout_start_ms=oos_start_ms)
    print(json.dumps({"events": result["events"], "30bps": result["metrics_30bps"],
                      "50bps": result["metrics_50bps"], "gates": result["gates"],
                      "pass": result["passes_all_gates"], "output": str(args.output)}, indent=2))
