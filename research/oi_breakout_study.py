"""Frozen OI-confirmed breakout ablation; see oi_breakout_protocol_20261022.md."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_long_bracket_study import make_events
from oi_coverage_audit import align_oi_features
from rolling_payoff_study import DAY, iso, load_snapshot, portfolio, resolve, timestamp
from squeeze_replication_20261022 import FRESH_START, measure

OOS = timestamp("2026-09-01T00:00:00Z")


def oi_confirms(btc_change: float, eth_change: float) -> bool:
    return bool(np.isfinite(btc_change) and np.isfinite(eth_change)
                and min(btc_change, eth_change) >= 0 and max(btc_change, eth_change) >= .005)


def benchmark_context(frames: dict, source: Path, lag_bars: int) -> pd.DataFrame:
    joined = []
    for symbol, prefix in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        f = frames[symbol]
        oi = pd.read_csv(source / f"{symbol}_open_interest.csv")
        aligned = align_oi_features(f.close_time_ms.to_numpy(), oi, lag_bars=lag_bars)
        aligned[f"{prefix}_return"] = f.close.pct_change(12, fill_method=None).to_numpy()
        aligned = aligned.set_index("signal_close_ms").rename(columns={
            "oi_1h_change": f"{prefix}_oi", "oi_window_complete": f"{prefix}_complete",
            "oi_source_ms": f"{prefix}_source_ms"})
        joined.append(aligned)
    # Inner exact-time join: no borrowed future benchmark values.
    return joined[0].join(joined[1], how="inner")


def run(data: Path, oi_source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    frames, manifest = load_snapshot(data)
    events = make_events(frames)["BULL_VOLUME_BREAKOUT_12"]
    liquidity = {s: f.quote_volume.rolling(288, min_periods=288).sum().to_numpy()
                 for s, f in frames.items() if "quote_volume" in f}
    arrays = {s: {c: f[c].to_numpy() for c in ("open_time_ms", "close_time_ms", "open", "high", "low", "close")}
              for s, f in frames.items()}
    arms, coverage = [], []
    for lag in (2, 1, 3):  # primary first, sensitivities not replacements
        context = benchmark_context(frames, oi_source, lag)
        baseline, confirmed = [], []
        missing, illiquid, not_rising = 0, 0, 0
        for symbol, i in events:
            close_ms = int(frames[symbol].close_time_ms.iloc[i])
            if close_ms not in context.index:
                missing += 1
                continue
            r = context.loc[close_ms]
            if not (r.btc_complete and r.eth_complete and np.isfinite(r.btc_return) and np.isfinite(r.eth_return)):
                missing += 1
                continue
            if symbol not in liquidity or not np.isfinite(liquidity[symbol][i]) or liquidity[symbol][i] < 5_000_000:
                illiquid += 1
                continue
            if r.btc_return <= 0 or r.eth_return <= 0:
                not_rising += 1
                continue
            baseline.append((symbol, i))
            if oi_confirms(float(r.btc_oi), float(r.eth_oi)):
                confirmed.append((symbol, i))
        coverage.append({"lag_minutes": lag * 5, "raw_events": len(events),
                         "missing_context": missing, "illiquid": illiquid,
                         "benchmark_price_not_rising": not_rising,
                         "baseline_events": len(baseline), "confirmed_events": len(confirmed)})
        for target in (2., 3.):
            for name, candidates in (("baseline", baseline), ("oi_confirmed", confirmed)):
                rows, missing_outcomes = [], 0
                for symbol, i in candidates:
                    row = resolve(frames[symbol], i, target, arrays=arrays[symbol])
                    if row is None:
                        missing_outcomes += 1
                    else:
                        rows.append({**row, "symbol": symbol})
                executed = portfolio(rows)
                test = [r for r in executed if r["signal_ms"] >= OOS]
                tail = [r for r in executed if r["signal_ms"] >= FRESH_START]
                weeks = []
                last = max(int(f.close_time_ms.iloc[-1]) for f in frames.values())
                for cut in range(OOS, last, 7 * DAY):
                    group = [r for r in test if cut <= r["signal_ms"] < cut + 7 * DAY]
                    weeks.append({"start": iso(cut), **measure(group)})
                arms.append({"arm": name, "lag_minutes": lag * 5, "primary_lag": lag == 2,
                             "target_pct": target, "missing_outcomes": missing_outcomes,
                             "oos": measure(test), "fresh_tail": measure(tail),
                             "weekly": weeks, "trades": executed})
                print(f"lag{lag*5} {name} TP{target:g}: n{len(test)} mean50={arms[-1]['oos']['at_50bps']['mean_net_bps']}", flush=True)
    files = [Path(__file__), Path(__file__).with_name("oi_coverage_audit.py"),
             Path(__file__).with_name("rolling_payoff_study.py"),
             Path(__file__).with_name("intraday_long_bracket_study.py"),
             Path(__file__).with_name("intraday_long_horizon_study.py"),
             Path(__file__).with_name("squeeze_replication_20261022.py")]
    result = {"protocol_sha256": hashlib.sha256(Path(__file__).with_name("oi_breakout_protocol_20261022.md").read_bytes()).hexdigest(),
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "oi_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(oi_source.glob("*_open_interest.csv"))},
              "source_manifest": manifest, "coverage": coverage, "arms": arms,
              "independent_validation": False, "live_activation_allowed": False}
    output.mkdir(parents=True)
    (output / "study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# OI-confirmed LONG breakout — fixed comparison", "",
             "Primary lag10min; lag5/15min are sensitivity checks, not selectable winners.",
             "OI is BTC/ETH market context, not each altcoin's own open interest.",
             "| Lag | Arm | TP | OOS n/days | Target first | Mean net30/50 bps | CI95 at30 | All gates |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for a in arms:
        r = a["oos"]; s = r["at_30bps"]
        lines.append(f"| {a['lag_minutes']}m | {a['arm']} | {a['target_pct']}% | {s['n']}/{s['signal_days']} | {r['target_before_stop_count']}/{s['n']} | {s['mean_net_bps']} / {r['at_50bps']['mean_net_bps']} | {s['daily_block_ci95']} | {all(r['gates'].values())} |")
    lines += ["", "Historical intervals were inspected previously: exploratory chronological replay, not independent confirmation.",
              "OI first-available/revision metadata missing; publication lag is an assumption. No live promotion.",
              "Costs are stress haircuts, not observed execution. Coin coverage is unequal; portfolio filtering changes capacity, so differences are not randomized causal estimates."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--oi", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.data, args.oi, args.output)
