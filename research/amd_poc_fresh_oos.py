"""Untouched chronological holdout replay for Trade3's AMD + locked-POC baseline.

This fixes the evaluation start in advance, keeps indicator warm-up from the
full history, and resets all setup/position state at the holdout boundary.
It performs no parameter search and does not touch live Trade3 or Trade1.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from amd_poc_backtest import load_frames, replay_symbol, utc

REPS = 10_000
SEED = 20261006


def summarize_stress(trades: list[dict], haircut_bps: float) -> dict:
    values_by_day: dict[str, list[float]] = defaultdict(list)
    by_side: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        if trade["net_bps"] is None:
            continue
        # Remove the Pine baseline's modeled commission/slippage, then apply a
        # flat total-cost stress haircut to the pre-cost result.
        gross = float(trade["net_bps"]) + float(trade["commission_bps"] or 0) + float(trade["slippage_bps"] or 0)
        stressed = gross - haircut_bps
        day = datetime.fromtimestamp(trade["signal_ms"] / 1000, UTC).date().isoformat()
        values_by_day[day].append(stressed)
        by_side[trade["side"]].append(stressed)
    values = np.asarray([x for group in values_by_day.values() for x in group], dtype=float)
    ci = None
    if len(values_by_day) >= 5:
        rng = np.random.default_rng(SEED)
        days = np.asarray(sorted(values_by_day))
        draws = np.empty(REPS, dtype=float)
        for j in range(REPS):
            chosen = rng.choice(days, size=len(days), replace=True)
            sampled = [value for day in chosen for value in values_by_day[str(day)]]
            draws[j] = np.mean(sampled)
        ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    wins = values[values > 0]
    losses = values[values < 0]
    return {
        "cost_haircut_bps_round_trip": haircut_bps,
        "trades": int(len(values)),
        "signal_days": len(values_by_day),
        "win_rate_pct": float(np.mean(values > 0) * 100) if len(values) else None,
        "mean_net_bps": float(values.mean()) if len(values) else None,
        "median_net_bps": float(np.median(values)) if len(values) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and len(wins) else (None if not len(wins) else float("inf")),
        "daily_block_bootstrap_ci95_bps": ci,
        "mean_net_bps_by_side": {side: float(np.mean(vals)) for side, vals in by_side.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/amd-poc-fresh-oos-20261006"))
    parser.add_argument("--oos-start", default="2026-08-05T08:40:00+00:00")
    args = parser.parse_args()
    start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise SystemExit("--oos-start must include a UTC offset")
    start_ms = int(start.astimezone(UTC).timestamp() * 1000)

    frames, ticks, meta = load_frames(args.data)
    trades, event_totals, by_symbol = [], defaultdict(int), []
    for symbol, frame in frames.items():
        closed, events = replay_symbol(symbol, frame, ticks[symbol], start_ms=start_ms)
        trades.extend(closed)
        for key, value in events.items():
            if isinstance(value, int):
                event_totals[key] += value
        by_symbol.append({"symbol": symbol, **events})

    stress = {str(cost): summarize_stress([t.__dict__ for t in trades], cost) for cost in (30, 50)}
    main = stress["30"]
    ci = main["daily_block_bootstrap_ci95_bps"]
    gates = {
        "positive_oos_mean_after_30bps": bool(main["mean_net_bps"] is not None and main["mean_net_bps"] > 0),
        "positive_oos_mean_after_50bps": bool(stress["50"]["mean_net_bps"] is not None and stress["50"]["mean_net_bps"] > 0),
        "daily_block_ci_lower_above_zero_after_30bps": bool(ci and ci[0] > 0),
        "at_least_100_oos_trades": main["trades"] >= 100,
        "at_least_15_oos_signal_days": main["signal_days"] >= 15,
        "at_least_two_oos_symbols": len({t.symbol for t in trades}) >= 2,
    }
    payload = {
        "strategy": "AMD -> liquidity sweep -> range re-entry -> locked OHLCV POC cross -> 2R bracket",
        "evaluation": "predeclared untouched chronological holdout; no parameter search",
        "holdout_start_utc": start.astimezone(UTC).isoformat(timespec="seconds"),
        "data_end_utc": utc(meta["common_end_ms"] + 299_999),
        "symbols": list(frames),
        "bars_per_symbol": {s: len(f) for s, f in frames.items()},
        "indicator_warmup": "full available history for ATR; all setup and position state resets at holdout start",
        "defaults": {
            "accumulation_bars": "20 to 60, shortest qualifying window",
            "max_range_atr": 2.0, "atr_length": 14, "profile_rows": 40,
            "min_sweep_ticks": 2, "require_close_back_inside": True,
            "max_setup_validity_bars": 12, "stop_buffer_ticks": 2,
            "risk_reward": 2.0, "direction": "long and short",
            "entry": "confirmed-close signal; following 5m open with 2-tick slippage",
            "fill_ambiguity": "same-bar stop and target resolves stop-first; gaps fill at open",
            "default_commission": "0.05% per fill; 10 bps approximate round-trip",
        },
        "events": dict(event_totals),
        "closed_trades": len(trades),
        "traded_symbols": sorted({t.symbol for t in trades}),
        "open_trades_at_data_end": sum(row.get("open_at_end", 0) for row in by_symbol),
        "stress_results": stress,
        "gates": gates,
        "passed_all_gates": all(gates.values()),
        "limitations": [
            "Holdout contains only BTC and ETH, not the full altcoin universe.",
            "89 altcoin 5m files cover only about three days, not enough for independent validation.",
            "Volume profile distributes each candle's volume uniformly over its high-low range; this approximates, not reproduces, trade-level volume profile.",
            "Historical spread/funding are unavailable; 30/50 bps are fixed total-cost stress haircuts.",
            "Two months may be too short for 100 trades/15 days; a failed sample-size gate means insufficient evidence, not proof of no edge.",
            "Offline research only; no live bot, Telegram, or Trade1 changes.",
        ],
        "trades": [t.__dict__ for t in trades],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# AMD + locked POC — fresh Trade3 holdout", "",
        f"Holdout: {payload['holdout_start_utc']} to {payload['data_end_utc']}; no parameter search.",
        "", "| Cost haircut | Trades / signal days | Win rate | Mean net bps | PF | Daily-block 95% CI |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for cost, row in stress.items():
        report.append(f"| {cost} bps | {row['trades']} / {row['signal_days']} | {row['win_rate_pct']}% | {row['mean_net_bps']} | {row['profit_factor']} | {row['daily_block_bootstrap_ci95_bps']} |")
    report += ["", f"Event counts: `{json.dumps(dict(event_totals), ensure_ascii=False)}`", "",
               f"All predeclared gates passed: **{all(gates.values())}**", "",
               "## Gates", "", *[f"- {'PASS' if ok else 'FAIL'} — {name}" for name, ok in gates.items()],
               "", "## Limitations", "", *[f"- {item}" for item in payload["limitations"]], ""]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"holdout_start_utc": payload["holdout_start_utc"], "events": payload["events"],
                      "trades": payload["closed_trades"], "stress_results": stress,
                      "gates": gates, "passed_all_gates": payload["passed_all_gates"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
