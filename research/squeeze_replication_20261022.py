"""Unchanged squeeze rule replication, not a new optimized strategy.

Predeclared: existing BULL_SQUEEZE_RELEASE, TP 2% and 3% independently,
SL1.5%, hold12h, max3 positions. Both variants must be reported regardless
of results. No training/threshold search or production import/activation.
Existing historical OOS boundary retained; Oct11+ reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from intraday_long_bracket_study import make_events
from intraday_long_horizon_study import summarize_fast
from rolling_payoff_study import DAY, gates, iso, load_snapshot, portfolio, resolve, timestamp

HISTORICAL_START = timestamp("2026-08-03T20:50:30Z")
FRESH_START = timestamp("2026-10-11T00:00:00Z")


def measure(rows: list[dict]) -> dict:
    target_first = sum(r["reason"] in {"TARGET", "TARGET_GAP"} for r in rows)
    return {"at_30bps": summarize_fast(rows, 30, with_ci=True),
            "at_50bps": summarize_fast(rows, 50),
            "target_before_stop_count": target_first,
            "target_before_stop_rate": target_first / len(rows) if rows else None,
            "gates": gates(rows)}


def run(data: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve the prior result")
    frames, manifest = load_snapshot(data)
    events = make_events(frames)["BULL_SQUEEZE_RELEASE"]
    variants = []
    for target in (2., 3.):
        rows, unavailable = [], Counter()
        for symbol, i in events:
            row = resolve(frames[symbol], i, target)
            if row is None:
                unavailable[symbol] += 1
            else:
                rows.append({**row, "symbol": symbol})
        # Capacity is simulated through the whole history before splitting:
        # a trade opened before a reporting boundary can still occupy a slot.
        executed = portfolio(rows)
        oos = [r for r in executed if r["signal_ms"] >= HISTORICAL_START]
        tail = [r for r in executed if r["signal_ms"] >= FRESH_START]
        weekly = []
        anchor = timestamp("2026-08-03T00:00:00Z")
        last = max(int(f.close_time_ms.iloc[-1]) for f in frames.values())
        for cut in range(anchor, last, 7 * DAY):
            group = [r for r in oos if cut <= r["signal_ms"] < cut + 7 * DAY]
            weekly.append({"start": iso(cut), **measure(group)})
        variants.append({"target_pct": target, "stop_pct": 1.5,
                         "historical_oos": measure(oos), "fresh_tail": measure(tail),
                         "weekly": weekly, "unavailable_outcomes": dict(unavailable),
                         "trades": executed})
    sources = [Path(__file__), Path(__file__).with_name("rolling_payoff_study.py"),
               Path(__file__).with_name("intraday_long_bracket_study.py"),
               Path(__file__).with_name("intraday_long_horizon_study.py")]
    result = {"rule": "BULL_SQUEEZE_RELEASE", "source_manifest": manifest,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "historical_oos_start": iso(HISTORICAL_START), "fresh_tail_start": iso(FRESH_START),
              "variants": variants, "live_activation_allowed": False,
              "independent_validation": False,
              "limitations": ["Previously inspected historical data and multiple research trials.",
                              "Unequal coverage and query-selected coin universe, not all-market validation.",
                              "Fresh tail includes expanded data with only limited recent coin coverage.",
                              "Cost haircuts are not historical execution/funding replay.",
                              "TP-first differs from eventual touch after stop; only TP-first counted here."]}
    output.mkdir(parents=True)
    (output / "study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Unchanged squeeze replication", "", "Research only; no activation or parameter search.",
             "| Target | Segment | n/days | TP before stop | Mean net 30/50 bps | CI95 at30 | All gates |",
             "|---|---|---:|---:|---:|---|---|"]
    for variant in variants:
        for segment in ("historical_oos", "fresh_tail"):
            row = variant[segment]
            s = row["at_30bps"]
            lines.append(f"| {variant['target_pct']}% | {segment} | {s['n']}/{s['signal_days']} | {row['target_before_stop_count']}/{s['n']} | {s['mean_net_bps']} / {row['at_50bps']['mean_net_bps']} | {s['daily_block_ci95']} | {all(row['gates'].values())} |")
    lines += ["", *result["limitations"], "", "All individual weekly folds and executed trades are in study.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = run(args.data, args.output)
    print(json.dumps([{k: v[k] for k in ("target_pct", "historical_oos", "fresh_tail")} for v in p["variants"]], indent=2))
