"""Training-selected 5m long breakout holding-period test for Trade3.

This research-only test evaluates one fixed volume-breakout signal at fixed
2%/3% targets and predeclared holding windows. The holding window/bracket is
selected on training data only, then evaluated on the later holdout. It does
not import or change the live bot or Trade1.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_long_bracket_study import load_data, make_events, utc

BAR_MS = 5 * 60_000
HOLD_HOURS = (12, 24, 48)
BRACKETS = ((2.0, 1.5), (3.0, 1.5), (3.0, 2.0))
COSTS_BPS = (30.0, 50.0)
OOS_START = "2026-08-03T20:50:30Z"
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20261006


def stamp(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("OOS_START must include a timezone")
    return int(parsed.timestamp() * 1000)


def resolve(frame: pd.DataFrame, signal_i: int, target_pct: float,
            stop_pct: float, hold_hours: int) -> dict | None:
    entry_i = signal_i + 1
    deadline = entry_i + hold_hours * 12 - 1
    if deadline >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open)
    target = entry * (1 + target_pct / 100)
    stop = entry * (1 - stop_pct / 100)
    exit_price = float(frame.iloc[deadline].close)
    exit_i, reason = deadline, "TIME"
    for i in range(entry_i, deadline + 1):
        row = frame.iloc[i]
        if row.open <= stop:
            exit_price, exit_i, reason = float(row.open), i, "STOP_GAP"
            break
        if row.open >= target:
            exit_price, exit_i, reason = float(row.open), i, "TARGET_GAP"
            break
        stop_hit, target_hit = row.low <= stop, row.high >= target
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
        "entry": entry,
        "exit": exit_price,
        "gross_bps": (exit_price / entry - 1) * 10_000,
        "reason": reason,
    }


def replay(events: list[tuple[str, int]], frames: dict[str, pd.DataFrame],
           target_pct: float, stop_pct: float, hold_hours: int) -> list[dict]:
    trades: list[dict] = []
    next_free_ms: dict[str, int] = {}
    for symbol, signal_i in sorted(
        events, key=lambda event: int(frames[event[0]].iloc[event[1]].open_time_ms)
    ):
        frame = frames[symbol]
        signal_ms = int(frame.iloc[signal_i].open_time_ms)
        if signal_ms <= next_free_ms.get(symbol, -1):
            continue
        trade = resolve(frame, signal_i, target_pct, stop_pct, hold_hours)
        if trade is None:
            continue
        trade["symbol"] = symbol
        trades.append(trade)
        next_free_ms[symbol] = int(trade["exit_ms"])
    return trades


def summarize_fast(trades: list[dict], cost_bps: float, with_ci: bool = False) -> dict:
    if not trades:
        return {"n": 0, "coins": [], "signal_days": 0, "mean_net_bps": None,
                "median_net_bps": None, "profit_factor": None, "daily_block_ci95": None,
                "exit_reasons": {}}
    values = np.asarray([float(t["gross_bps"]) - cost_bps for t in trades], dtype=float)
    days = np.asarray([int(t["signal_ms"]) // 86_400_000 for t in trades], dtype=np.int64)
    unique_days, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    ci = None
    if with_ci and len(unique_days) >= 5:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        picks = rng.integers(0, len(unique_days), size=(BOOTSTRAP_REPS, len(unique_days)))
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    positive, negative = values[values > 0], values[values < 0]
    exit_reasons = pd.Series([t["reason"] for t in trades]).value_counts().to_dict()
    return {
        "n": len(trades),
        "coins": sorted({str(t["symbol"]) for t in trades}),
        "signal_days": len(unique_days),
        "wins": int((values > 0).sum()),
        "win_rate_pct": float(np.mean(values > 0) * 100),
        "mean_net_bps": float(values.mean()),
        "median_net_bps": float(np.median(values)),
        "profit_factor": float(positive.sum() / -negative.sum()) if len(negative) else None,
        "daily_block_ci95": ci,
        "exit_reasons": {str(k): int(v) for k, v in exit_reasons.items()},
    }


def run(data: Path, output: Path, oos_start: str) -> dict:
    frames = load_data(data)
    events = make_events(frames)["BULL_VOLUME_BREAKOUT_12"]
    cut = stamp(oos_start)
    rows: list[dict] = []
    for hours in HOLD_HOURS:
        for target, stop in BRACKETS:
            trades = replay(events, frames, target, stop, hours)
            train = [t for t in trades if t["signal_ms"] < cut and t["exit_ms"] < cut]
            purged = [t for t in trades if t["signal_ms"] < cut <= t["exit_ms"]]
            oos = [t for t in trades if t["signal_ms"] >= cut]
            train30 = summarize_fast(train, 30)
            oos30 = summarize_fast(oos, 30, with_ci=True)
            oos50 = summarize_fast(oos, 50)
            rows.append({
                "hold_hours": hours,
                "target_pct": target,
                "stop_pct": stop,
                "training_30bps": train30,
                "purged_train_crossing_cut": len(purged),
                "oos_30bps": oos30,
                "oos_50bps": oos50,
            })

    eligible = [
        row for row in rows
        if row["training_30bps"].get("n", 0) >= 100
        and row["training_30bps"].get("signal_days", 0) >= 20
        and row["training_30bps"].get("mean_net_bps") is not None
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["training_30bps"]["mean_net_bps"],
            row["training_30bps"]["n"],
            -row["hold_hours"],
        ),
        default=None,
    )
    if selected is not None:
        oos30, oos50 = selected["oos_30bps"], selected["oos_50bps"]
        ci = oos30.get("daily_block_ci95")
        selected["gates"] = {
            "positive_mean_30bps": bool(oos30.get("mean_net_bps") is not None and oos30["mean_net_bps"] > 0),
            "positive_mean_50bps": bool(oos50.get("mean_net_bps") is not None and oos50["mean_net_bps"] > 0),
            "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
            "at_least_100_trades": bool(oos30.get("n", 0) >= 100),
            "at_least_15_days": bool(oos30.get("signal_days", 0) >= 15),
            "at_least_two_symbols": bool(len(oos30.get("coins", [])) >= 2),
        }
        selected["passes_all_gates"] = all(selected["gates"].values())

    result = {
        "data_dir": str(data),
        "symbols": sorted(frames),
        "event_count": len(events),
        "holdout_start": oos_start,
        "training_selection": "max training mean net bps at 30 bps cost; eligible only at >=100 training trades and >=20 signal days",
        "selected_by_training_only": selected,
        "variants": rows,
        "warning": "The Aug-Sep 2026 holdout has been inspected in earlier Trade3 studies; this is exploratory, not a fresh confirmatory test.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        "# Trade3 5m volume-breakout holding-period study",
        "",
        f"Data symbols: {len(frames)}; candidate events: {len(events)}; holdout starts {utc(cut)}.",
        "Rule: fixed BULL_VOLUME_BREAKOUT_12 trigger; next-bar-open entry; targets +2%/+3%; stops -1.5%/-2%; fixed 12/24/48h horizons.",
        "Candidate chosen using training results only (30 bps cost), requiring >=100 training trades and >=20 training signal days.",
        "",
        "| Hold | TP / SL | Train n/days mean30 | OOS n/days | Mean net 30/50 bps | PF30 | daily CI95 at 30 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        tr, oos = row["training_30bps"], row["oos_30bps"]
        oos50 = row["oos_50bps"]
        report.append(
            f"| {row['hold_hours']}h | +{row['target_pct']}% / -{row['stop_pct']}% | "
            f"{tr.get('n', 0)}/{tr.get('signal_days', 0)} {tr.get('mean_net_bps')} | "
            f"{oos.get('n', 0)}/{oos.get('signal_days', 0)} | "
            f"{oos.get('mean_net_bps')} / {oos50.get('mean_net_bps')} | "
            f"{oos.get('profit_factor')} | {oos.get('daily_block_ci95')} |"
        )
    report.extend(["", "## Training-only selection", ""])
    if selected is None:
        report.append("No variant met the predeclared training eligibility floor; no candidate was selected.")
    else:
        report.append(
            f"Selected: {selected['hold_hours']}h, TP +{selected['target_pct']}%, SL -{selected['stop_pct']}%. "
            f"OOS gates: {selected['gates']}; pass all: {selected['passes_all_gates']}."
        )
    report.extend([
        "",
        "## Limitations",
        "",
        "- Holdout dates have been inspected by earlier studies; multiple research attempts mean this remains exploratory.",
        "- CSV universe is survivorship-biased and restricted to the available 12-symbol coin-query cohort.",
        "- Costs are fixed 30/50 bps round-trip stress haircuts; historical spread, funding and fill replay are unavailable.",
        "- On bars hitting both stop and target, the stop is assumed first.",
        "- Offline only; no live bot, Telegram, server configuration, or Trade1 changes.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/research/server-current-20261006/data/coin-query-cohort"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/long-horizon-20261006"))
    parser.add_argument("--oos-start", default=OOS_START)
    args = parser.parse_args()
    result = run(args.data, args.output, args.oos_start)
    selected = result["selected_by_training_only"]
    print(json.dumps({
        "events": result["event_count"],
        "selected_by_training_only": selected,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
