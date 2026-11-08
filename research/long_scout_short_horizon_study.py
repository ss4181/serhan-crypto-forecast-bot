"""Short-horizon bracket replay of saved Trade3 long-scout candidates.

Offline research only. Uses the current Trade3 candidate study and local 1h
OHLCV caches; it never modifies bot settings or sends notifications.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
HORIZONS_HOURS = (4, 12, 24)
BRACKETS = ((2.0, 1.5), (3.0, 1.5))
COSTS_BPS = (30.0, 50.0)
OOS_START_MS = int(datetime(2026, 9, 14, tzinfo=UTC).timestamp() * 1000)
MAX_ACTIVE = 50


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")


def load_prices(directory: Path, symbols: set[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in sorted(symbols):
        path = directory / f"{symbol}_1h.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        result[symbol] = frame
    return result


def replay(event: dict, frame: pd.DataFrame, target_pct: float, stop_pct: float,
           horizon_hours: int) -> dict | None:
    entry_ms = int(event["sourceCloseMs"]) + 1
    matches = np.flatnonzero(frame.open_time_ms.to_numpy(dtype=np.int64) == entry_ms)
    if len(matches) != 1:
        return None
    entry_i = int(matches[0])
    end_i = entry_i + horizon_hours - 1
    if end_i >= len(frame):
        return None
    future = frame.iloc[entry_i : end_i + 1]
    expected = entry_ms + np.arange(horizon_hours, dtype=np.int64) * HOUR_MS
    if len(future) != horizon_hours or not np.array_equal(future.open_time_ms.to_numpy(dtype=np.int64), expected):
        return None
    entry = float(future.iloc[0].open)
    target = entry * (1 + target_pct / 100)
    stop = entry * (1 - stop_pct / 100)
    exit_price = float(future.iloc[-1].close)
    exit_i_local = horizon_hours - 1
    reason = "TIME"
    for i, candle in enumerate(future.itertuples()):
        # OHLC cannot reveal intrabar path; when both levels hit, stop wins.
        if candle.open <= stop:
            exit_price, exit_i_local, reason = float(candle.open), i, "STOP_GAP"
            break
        if candle.open >= target:
            exit_price, exit_i_local, reason = float(candle.open), i, "TARGET_GAP"
            break
        stop_hit = candle.low <= stop
        target_hit = candle.high >= target
        if stop_hit and target_hit:
            exit_price, exit_i_local, reason = stop, i, "AMBIGUOUS_STOP_FIRST"
            break
        if stop_hit:
            exit_price, exit_i_local, reason = stop, i, "STOP"
            break
        if target_hit:
            exit_price, exit_i_local, reason = target, i, "TARGET"
            break
    return {
        "symbol": event["symbol"],
        "setup": event["strategy"],
        "signal_ms": int(event["sourceCloseMs"]),
        "entry_ms": entry_ms,
        "exit_ms": int(future.iloc[exit_i_local].close_time_ms),
        "entry": entry,
        "exit": exit_price,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "horizon_hours": horizon_hours,
        "gross_bps": (exit_price / entry - 1) * 10_000,
        "reason": reason,
        "source_stage": event.get("stage"),
        "source_score": event.get("score"),
    }


def portfolio(events: list[dict], frames: dict[str, pd.DataFrame], target: float,
              stop: float, horizon: int) -> tuple[list[dict], dict]:
    active: list[dict] = []
    accepted: list[dict] = []
    missing = overlap = cap_reject = 0
    for event in sorted(events, key=lambda row: int(row["sourceCloseMs"])):
        stamp = int(event["sourceCloseMs"])
        active = [trade for trade in active if trade["exit_ms"] >= stamp]
        if any(trade["symbol"] == event["symbol"] for trade in active):
            overlap += 1
            continue
        if len(active) >= MAX_ACTIVE:
            cap_reject += 1
            continue
        frame = frames.get(event["symbol"])
        if frame is None:
            missing += 1
            continue
        trade = replay(event, frame, target, stop, horizon)
        if trade is None:
            missing += 1
            continue
        accepted.append(trade)
        active.append(trade)
    return accepted, {"missing_or_incomplete": missing, "same_coin_overlap_rejected": overlap,
                      "max_active_rejected": cap_reject}


def summarize(trades: list[dict], cost_bps: float) -> dict:
    if not trades:
        return {"n": 0, "signal_days": 0, "mean_net_bps": None, "daily_block_ci95": None}
    values = np.asarray([trade["gross_bps"] - cost_bps for trade in trades], dtype=float)
    daily: dict[int, list[float]] = defaultdict(list)
    for trade, value in zip(trades, values, strict=True):
        daily[trade["signal_ms"] // DAY_MS].append(float(value))
    winners, losers = values[values > 0], values[values < 0]
    ci = None
    if len(daily) >= 5:
        rng = np.random.default_rng(20261003)
        days = np.asarray(sorted(daily), dtype=np.int64)
        means = []
        for _ in range(3000):
            picked = rng.choice(days, size=len(days), replace=True)
            means.append(float(np.mean([value for day in picked for value in daily[int(day)]])))
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "n": len(trades),
        "coins": len({trade["symbol"] for trade in trades}),
        "signal_days": len(daily),
        "wins": int((values > 0).sum()),
        "win_rate_pct": float(np.mean(values > 0) * 100),
        "mean_net_bps": float(values.mean()),
        "median_net_bps": float(np.median(values)),
        "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else None,
        "daily_block_ci95": ci,
        "reasons": {str(k): int(v) for k, v in pd.Series([trade["reason"] for trade in trades]).value_counts().items()},
    }


def run(study_path: Path, prices_dir: Path, output: Path) -> dict:
    study = json.loads(study_path.read_text(encoding="utf-8"))
    arms = {
        name: value["records"]
        for name, value in study["arms"].items()
    }
    all_symbols = {event["symbol"] for records in arms.values() for event in records}
    frames = load_prices(prices_dir, all_symbols)
    results = []
    all_trades = []
    for arm_name, events in arms.items():
        for horizon in HORIZONS_HOURS:
            for target, stop in BRACKETS:
                trades, selection = portfolio(events, frames, target, stop, horizon)
                train = [t for t in trades if t["signal_ms"] < OOS_START_MS and t["exit_ms"] < OOS_START_MS]
                purged = [t for t in trades if t["signal_ms"] < OOS_START_MS and t["exit_ms"] >= OOS_START_MS]
                test = [t for t in trades if t["signal_ms"] >= OOS_START_MS]
                test30 = summarize(test, 30)
                test50 = summarize(test, 50)
                ci = test30.get("daily_block_ci95")
                gates = {
                    "positive_oos_mean_30bps": bool(test30.get("mean_net_bps") is not None and test30["mean_net_bps"] > 0),
                    "positive_oos_mean_50bps": bool(test50.get("mean_net_bps") is not None and test50["mean_net_bps"] > 0),
                    "daily_block_ci_lower_above_zero": bool(ci and ci[0] > 0),
                    "at_least_100_oos_trades": bool(test30.get("n", 0) >= 100),
                    "at_least_15_oos_signal_days": bool(test30.get("signal_days", 0) >= 15),
                }
                results.append({
                    "arm": arm_name,
                    "horizon_hours": horizon,
                    "target_pct": target,
                    "stop_pct": stop,
                    "selection": selection,
                    "train_30bps": summarize(train, 30),
                    "purged_train_crossing_oos": len(purged),
                    "oos_30bps": test30,
                    "oos_50bps": test50,
                    "gates": gates,
                })
                all_trades.extend({**t, "arm": arm_name,
                                   "segment": "oos" if t["signal_ms"] >= OOS_START_MS else
                                   "train_purged" if t["exit_ms"] >= OOS_START_MS else "train"}
                                  for t in trades)
    payload = {
        "method": {
            "candidate_study": str(study_path),
            "price_cache": str(prices_dir),
            "entry": "next full 1h open following a confirmed candidate",
            "horizons_hours": list(HORIZONS_HOURS),
            "brackets_target_stop_pct": [list(pair) for pair in BRACKETS],
            "portfolio": f"one position per symbol; maximum {MAX_ACTIVE} concurrent; same-hour ties stop-first",
            "costs_round_trip_bps": [30, 50],
            "oos_start_utc": iso(OOS_START_MS),
            "universe_survivor_bias": True,
            "candidate_outcomes_from_saved_study_not_used_as_inputs": True,
            "limitations": [
                "Candidate universe and signal cohort were already seen in earlier 7d reports; this is another horizon analysis, not a fresh independent holdout.",
                "Current-universe survivor bias remains.",
                "No historical funding, spread, or orderbook costs; use 30/50 bps round-trip stress haircuts.",
                "Hourly OHLC same-bar barrier ambiguity is resolved against the strategy (stop first).",
                "A 1h entry and 24h maximum hold is intraday, not a 5m scalp.",
            ],
        },
        "data": {
            "candidate_count_by_arm": {name: len(records) for name, records in arms.items()},
            "candidate_symbols_with_prices": len(frames),
            "price_rows": sum(len(frame) for frame in frames.values()),
            "price_start_utc": iso(max(int(frame.open_time_ms.iloc[0]) for frame in frames.values())),
            "price_end_utc": iso(min(int(frame.open_time_ms.iloc[-1]) for frame in frames.values())),
        },
        "results": results,
        "any_candidate_passes_all_gates": any(all(row["gates"].values()) for row in results),
        "trades": all_trades,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_trades).to_csv(output / "trades.csv", index=False)
    lines = [
        "# Trade3 long-scout candidates — short-horizon bracket replay",
        "",
        f"Candidates: {payload['data']['candidate_count_by_arm']}; local hourly price symbols: {payload['data']['candidate_symbols_with_prices']}; price window {payload['data']['price_start_utc']} to {payload['data']['price_end_utc']}.",
        f"Entry next 1h open; OOS begins {iso(OOS_START_MS)}; max {MAX_ACTIVE} concurrent; overlap per symbol blocked; same-hour double-touch = stop first.",
        "30/50 bps are full round-trip stress costs. This previously reviewed candidate cohort is not independent validation.",
        "",
        "| Signal arm | Hold | TP/SL | Train n / mean net 30bps | OOS n / days | OOS net mean 30/50bps | PF (30bps) | Daily CI95 | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        tr, a, b = row["train_30bps"], row["oos_30bps"], row["oos_50bps"]
        lines.append(
            f"| {row['arm']} | {row['horizon_hours']}h | +{row['target_pct']}/−{row['stop_pct']}% | {tr.get('n',0)} / {tr.get('mean_net_bps')} | {a.get('n',0)} / {a.get('signal_days',0)} | {a.get('mean_net_bps')} / {b.get('mean_net_bps')} | {a.get('profit_factor')} | {a.get('daily_block_ci95')} | {'YES' if all(row['gates'].values()) else 'NO'} |"
        )
    lines += ["", "## Limits", "", *[f"- {item}" for item in payload["method"]["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/research/long-discovery-live-20261003/study.json"))
    parser.add_argument("--prices", type=Path, default=Path("artifacts/research/server-long-scout-20261003/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/long-scout-short-horizon-20261003"))
    args = parser.parse_args()
    result = run(args.study, args.prices, args.output)
    print(json.dumps({
        "data": result["data"],
        "passes_any": result["any_candidate_passes_all_gates"],
        "results": [{"arm": r["arm"], "horizon": r["horizon_hours"], "bracket": [r["target_pct"], r["stop_pct"]],
                     "oos30": r["oos_30bps"], "oos50_mean": r["oos_50bps"]["mean_net_bps"], "gates": r["gates"]}
                    for r in result["results"]],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
