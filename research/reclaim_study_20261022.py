"""Frozen sharp-fall/high-volume/reclaim LONG hypothesis. Research only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rolling_payoff_study import DAY, iso, load_snapshot, portfolio, resolve, timestamp
from squeeze_replication_20261022 import FRESH_START, HISTORICAL_START, measure


def signal_indices(frame: pd.DataFrame) -> np.ndarray:
    # The fall bar is t-1; every comparison level excludes that bar itself.
    prior_high = frame.high.shift(2).rolling(12, min_periods=12).max()
    prior_low = frame.low.shift(2).rolling(12, min_periods=12).min()
    prior_volume = frame.volume.shift(2).rolling(20, min_periods=20).median()
    liquid = frame.quote_volume.rolling(288, min_periods=288).sum() >= 5_000_000
    fall = ((frame.low.shift(1) <= .97 * prior_high)
            & (frame.low.shift(1) <= prior_low)
            & (frame.close.shift(1) < frame.open.shift(1))
            & (prior_volume > 0)
            & (frame.volume.shift(1) >= 2 * prior_volume))
    reclaim = (frame.close > frame.high.shift(1)) & (frame.close > frame.open)
    return np.flatnonzero((fall & reclaim & liquid).fillna(False).to_numpy())


def run(data: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    frames, manifest = load_snapshot(data)
    events, skipped_sources = [], []
    for symbol, frame in frames.items():
        if "quote_volume" not in frame or not np.isfinite(frame.quote_volume.to_numpy()).all():
            skipped_sources.append(symbol)
            continue
        events.extend((symbol, int(i)) for i in signal_indices(frame))
    variants = []
    last = max(int(f.close_time_ms.iloc[-1]) for f in frames.values())
    for target in (2., 3.):
        rows, missing = [], 0
        for symbol, i in events:
            row = resolve(frames[symbol], i, target)
            if row is None:
                missing += 1
            else:
                rows.append({**row, "symbol": symbol})
        executed = portfolio(rows)
        oos = [r for r in executed if r["signal_ms"] >= HISTORICAL_START]
        tail = [r for r in executed if r["signal_ms"] >= FRESH_START]
        weeks = []
        for cut in range(timestamp("2026-08-03T00:00:00Z"), last, 7 * DAY):
            group = [r for r in oos if cut <= r["signal_ms"] < cut + 7 * DAY]
            weeks.append({"start": iso(cut), **measure(group)})
        variants.append({"target_pct": target, "historical_oos": measure(oos),
                         "fresh_tail": measure(tail), "unavailable_outcomes": missing,
                         "weekly": weeks, "trades": executed})
    source = Path(__file__)
    code_files = [source, source.with_name("rolling_payoff_study.py"),
                  source.with_name("squeeze_replication_20261022.py"),
                  source.with_name("intraday_long_horizon_study.py")]
    result = {"strategy": "SHARP_FALL_VOLUME_RECLAIM", "protocol_sha256": hashlib.sha256(source.with_name("reclaim_protocol_20261022.md").read_bytes()).hexdigest(),
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
              "source_manifest": manifest, "signal_count": len(events),
              "missing_quote_volume_sources": skipped_sources, "variants": variants,
              "independent_validation": False, "live_activation_allowed": False}
    output.mkdir(parents=True)
    (output / "study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Sharp-fall reclaim — fixed exploratory experiment", "",
             f"Raw signals: {len(events)}. Missing quote-volume sources: {skipped_sources}.",
             "| Target | Segment | n/days | TP before stop | Mean net 30/50 bps | CI95 at30 | All gates |",
             "|---|---|---:|---:|---:|---|---|"]
    for variant in variants:
        for segment in ("historical_oos", "fresh_tail"):
            r = variant[segment]
            s = r["at_30bps"]
            lines.append(f"| {variant['target_pct']}% | {segment} | {s['n']}/{s['signal_days']} | {r['target_before_stop_count']}/{s['n']} | {s['mean_net_bps']} / {r['at_50bps']['mean_net_bps']} | {s['daily_block_ci95']} | {all(r['gates'].values())} |")
    lines += ["", "No live activation. Historical intervals informed hypothesis generation: not independent confirmation.",
              "Unequal coin histories, survivorship/query selection, fixed-cost assumptions and OHLC path ambiguity limit conclusions.",
              "Full per-week measurements, provenance and individual trades are in study.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({"signals": result["signal_count"], "variants": [
        {k: v[k] for k in ("target_pct", "historical_oos", "fresh_tail")} for v in result["variants"]]}, indent=2))
