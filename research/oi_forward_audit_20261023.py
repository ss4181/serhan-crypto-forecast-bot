"""Read-only-input forward audit of the frozen TP3/OI10m candidate.

No tuning, no source stitching, no production import or live activation.
Unresolved windows reserve capacity conservatively; they are never scored.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from intraday_long_bracket_study import make_events
from oi_breakout_study import benchmark_context, oi_confirms
from rolling_payoff_study import BAR, HOLD, iso, load_snapshot, portfolio, resolve, timestamp
from squeeze_replication_20261022 import measure

CUT = timestamp("2026-10-22T00:00:00Z")


def outcome(frame, i: int, snapshot_end: int) -> dict:
    """Require the frozen full outcome window, even for apparent early hits."""
    row = resolve(frame, i, 3.)
    if row is not None:
        return {**row, "outcome_status": "SETTLED"}
    signal_ms = int(frame.close_time_ms.iloc[i])
    deadline = signal_ms + HOLD * BAR
    return {"signal_ms": signal_ms, "entry_ms": signal_ms + 1,
            "exit_ms": deadline, "target_pct": 3.,
            "outcome_status": "PENDING" if deadline > snapshot_end else "UNKNOWN_DATA",
            "capacity_exit_is_upper_bound": True}


def evaluate_portfolio(rows: list[dict], cut: int = CUT) -> dict:
    # Unlike the historical runner's complete-case reference, do not erase
    # an unresolved position before allocating the three available slots.
    kept = portfolio(rows)
    forward = [r for r in kept if r["signal_ms"] >= cut]
    settled = [r for r in forward if r["outcome_status"] == "SETTLED"]
    reference = [r for r in portfolio([r for r in rows if r["outcome_status"] == "SETTLED"])
                 if r["signal_ms"] >= cut]
    return {"selected_forward": forward, "settled_measurement": measure(settled),
            "forward_status_counts": dict(Counter(r["outcome_status"] for r in forward)),
            "unresolved_pre_cut_carry": [r for r in kept if r["signal_ms"] < cut <= r["exit_ms"]
                                         and r["outcome_status"] != "SETTLED"],
            "complete_case_reference": measure(reference),
            "complete_case_reference_trades": reference}


def run(data: Path, oi: Path, frozen: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite an audit")
    previous = json.loads(frozen.read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in previous["code_sha256"]}
    if hashes != previous["code_sha256"]:
        raise ValueError("Frozen research code changed")
    protocol = Path(__file__).with_name("oi_breakout_protocol_20261022.md")
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != previous["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    frames, manifest = load_snapshot(data)
    end = max(int(f.close_time_ms.iloc[-1]) for f in frames.values())
    context = benchmark_context(frames, oi, 2)
    events = make_events(frames)["BULL_VOLUME_BREAKOUT_12"]
    liquidity = {s: f.quote_volume.rolling(288, min_periods=288).sum().to_numpy()
                 for s, f in frames.items() if "quote_volume" in f}
    counts, confirmed, raw_forward = Counter(), [], 0
    for symbol, i in events:
        signal_ms = int(frames[symbol].close_time_ms.iloc[i])
        # Replay all available history before splitting: prior occupancy can
        # affect which later position is carried into the holdout boundary.
        forward = signal_ms >= CUT
        raw_forward += int(forward)
        reason = None
        if signal_ms not in context.index:
            reason = "missing_context"
        else:
            r = context.loc[signal_ms]
            if not (r.btc_complete and r.eth_complete and np.isfinite(r.btc_return)
                    and np.isfinite(r.eth_return)):
                reason = "missing_context"
            elif symbol not in liquidity or not np.isfinite(liquidity[symbol][i]) or liquidity[symbol][i] < 5_000_000:
                reason = "liquidity_below_5m"
            elif r.btc_return <= 0 or r.eth_return <= 0:
                reason = "benchmark_prices_not_rising"
            elif not oi_confirms(float(r.btc_oi), float(r.eth_oi)):
                reason = "oi_not_confirmed"
        if reason:
            if forward:
                counts[reason] += 1
            continue
        confirmed.append({"symbol": symbol, **outcome(frames[symbol], i, end)})
        if forward:
            counts["confirmed_before_capacity"] += 1
    evaluated = evaluate_portfolio(confirmed)
    coverage = []
    for symbol, frame in frames.items():
        # Earliest possible full 24h liquidity feature (index 287).
        start = max(CUT, int(frame.close_time_ms.iloc[287]))
        last = int(frame.close_time_ms.iloc[-1])
        coverage.append({"symbol": symbol, "first_possible_signal": iso(start),
                         "last_observed_close": iso(last), "reaches_holdout": last >= start,
                         "last_full_window_signal": iso(last - HOLD * BAR) if last - HOLD * BAR >= start else None})
    limitations = [
        "Less than two days of new data; not enough for 100 trades / 15 signal days.",
        "Frozen longest-contiguous-source policy retained; stale longer query files can override fresh short caches.",
        "No gap stitching or parameter tuning; unequal coverage is not an all-market test.",
        "OI first-observed/revision metadata absent. Ten-minute availability is assumed, not verified.",
        "Pending/missing outcomes reserve capacity to their 12h deadline conservatively and are not scored.",
        "This reservation is an explicit audit correction to historical complete-case filtering; that reference is also reported.",
        "No exact portfolio claim when missing outcomes affect slot availability; no live execution or real fills.",
        "30/50 bps cost stress is not observed funding/spread/slippage.",
    ]
    result = {"cut": iso(CUT), "snapshot_end": iso(end), "code_sha256": hashes,
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_study_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
              "oi_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(oi.glob("*_open_interest.csv"))},
              "source_manifest": manifest, "symbol_coverage": coverage,
              "coins": len(frames), "coins_reaching_holdout": sum(r["reaches_holdout"] for r in coverage),
              "raw_forward_events": raw_forward, "filter_counts": dict(counts),
              "confirmed_events_including_carry": confirmed, **evaluated,
              "independent_validation": False, "live_activation_allowed": False,
              "limitations": limitations}
    output.mkdir(parents=True)
    (output / "study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    m = result["settled_measurement"]
    lines = ["# Trade3: sabit OI/%3 adayının yeni dönem sınaması", "",
             f"UTC dönem: {result['cut']} — {result['snapshot_end']}",
             "Yalnız yerel araştırma; sunucu, ayarlar ve bildirimler değiştirilmedi.", "",
             f"Kaynak: {len(manifest)} fiyat dosyası, {len(frames)} coin; yeni döneme ulaşan {result['coins_reaching_holdout']}.",
             f"Ham fiyat/hacim olayları: {raw_forward}. Filtreler: {dict(counts)}.",
             f"Kapasite sonrası durumlar: {result['forward_status_counts']}.",
             f"Sonuçlanan işlemler: {m['at_30bps']['n']}; stop öncesi %3 hedef: {m['target_before_stop_count']}.",
             f"Net ortalama 30/50 bps maliyet: {m['at_30bps']['mean_net_bps']} / {m['at_50bps']['mean_net_bps']} bps.",
             f"Günlük blok %95 aralık: {m['at_30bps']['daily_block_ci95']}.", "",
             "## Sınırlar", "", *[f"- {s}" for s in limitations]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "oi", "frozen", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    r = run(a.data, a.oi, a.frozen, a.output)
    print(json.dumps({k: r[k] for k in ("coins", "coins_reaching_holdout", "filter_counts",
                                      "forward_status_counts", "settled_measurement")}, indent=2))
