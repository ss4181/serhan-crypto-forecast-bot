"""Read-only, point-in-time replay of settled Trade3 multi-family setups.

Explains notification rejection; it does not promote strategies or send alerts.
Execution flags come from the original observation, not today's market data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypto_forecaster.scalping import (
    CONSENSUS_DIRECTION_POLICY, STRATEGY_HORIZON_DIRECTION_POLICY,
    BullRegime, ScalpObservation, ScalpScanReport,
    filter_scalp_notification_report, scalp_setup_assessment, scalp_setup_direction,
)

DAY = 86_400_000
HOUR = 3_600_000


def reconstruct(row: dict) -> ScalpObservation:
    names = {field.name for field in fields(ScalpObservation)}
    values = {key: value for key, value in row.items() if key in names}
    values.update(
        price=float(row.get("source_price", row["entry_price"])),
        bar_open_time_ms=int(row["bar_close_time_ms"]) - 299_999,
        estimated_cost_bps=float(row["round_trip_cost_bps"]), details=(),
    )
    return ScalpObservation(**values)


def run(root: Path, output: Path | None = None) -> dict:
    output = output or root
    summary = json.loads((root / "experiments/strategy-horizon-shadow-v1/summary.json").read_text())
    profile = root / "experiments/strategy-horizon-shadow-v1" / f"profile-{summary['profileId']}" / "config.json"
    options = json.loads(profile.read_text())["filters"]
    end = int(summary["evaluatedAtMs"])
    cutoff = end - 7 * DAY
    rows = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines() if line.strip()]
    grouped = defaultdict(dict)
    for row in rows:
        if (cutoff <= row["bar_close_time_ms"] <= end
                and row.get("horizon_minutes") == 60 and "entry_price" in row):
            grouped[(row["bar_close_time_ms"], row["perpetual_symbol"])][row["family"]] = row
    setups = [(key, list(families.values())) for key, families in sorted(grouped.items())
              if len(families) >= 2 and any(r.get("alert_tier") == "KURULUM" for r in families.values())]
    counts = defaultdict(Counter)
    details = []
    for number, ((close_ms, symbol), event_rows) in enumerate(setups, 1):
        # Every horizon becomes available together after the 60m outcome.
        # Preserve append order and the production reader's 20,000-row limit.
        known = [row for row in rows
                 if int(row.get("bar_close_time_ms", 0)) + HOUR <= close_ms
                 and int(row.get("exit_time_ms", 0)) <= close_ms][-20_000:]
        families = {row["family"] for row in event_rows}
        known = [row for row in known if row.get("family") in families]
        items = tuple(reconstruct(row) for row in event_rows)
        regime = items[0].regime_state
        report = ScalpScanReport(items[0].universe_version, 89, 89, 0, (), items,
                                close_ms + 1, BullRegime(regime, 0, 0, 0, False, 89),
                                newest_close_time_ms=close_ms)
        detail = {"symbol": symbol, "close_ms": close_ms, "regime": regime,
                  "families": sorted(families), "max_score": max(i.score for i in items),
                  "execution_eligible": any(i.execution_eligible for i in items)}
        for arm, policy in (("consensus", CONSENSUS_DIRECTION_POLICY),
                            ("strategy_horizon", STRATEGY_HORIZON_DIRECTION_POLICY)):
            assessment = scalp_setup_assessment(items, known, direction_policy=policy)
            direction, horizons = scalp_setup_direction(items, known, policy=policy)
            counts[f"{arm}_directions"][direction] += 1
            detail[arm] = {"direction": direction, "horizons": horizons,
                           "n": assessment.sample_count, "net_bps": assessment.expected_net_bps,
                           "probability": assessment.success_probability}
            # Cached assessment makes both score counterfactuals use identical evidence.
            with patch("crypto_forecaster.scalping.scalp_setup_assessment", return_value=assessment):
                for setting, score in (("current_score", options["minimum_score"]), ("score_1_5", 1.5)):
                    active = {**options, "minimum_score": score}
                    diagnostics = {}
                    result = filter_scalp_notification_report(
                        report, **active, ledger=known, direction_policy=policy,
                        diagnostics=diagnostics,
                    )
                    key = f"{arm}_{setting}"
                    counts[key].update(diagnostics)
                    detail[key] = diagnostics
                    if result.observations:
                        horizon = assessment.horizon_minutes
                        actual = [row for row in rows
                                  if row.get("bar_close_time_ms") == close_ms
                                  and row.get("perpetual_symbol") == symbol
                                  and row.get("horizon_minutes") == horizon
                                  and row.get("family") in families
                                  and isinstance(row.get("gross_bps"), (float, int))]
                        net = [(row["gross_bps"] if direction == "YUKARI" else -row["gross_bps"])
                               - row["round_trip_cost_bps"] for row in actual]
                        detail[key + "_realised_net_bps"] = min(net) if net else None
        details.append(detail)
        if number % 25 == 0:
            print(f"Replayed {number}/{len(setups)} setups", flush=True)
    result = {
        "cutoff_utc": datetime.fromtimestamp(cutoff / 1000, UTC).isoformat(),
        "snapshot_utc": datetime.fromtimestamp(end / 1000, UTC).isoformat(),
        "settled_multi_family_setups": len(setups),
        "scalping_code_sha256": sha256((Path(__file__).resolve().parents[1] / "src/crypto_forecaster/scalping.py").read_bytes()).hexdigest(),
        "counts": {key: dict(value) for key, value in counts.items()},
        "counterfactual_outcomes": {},
        "limitations": [
            "Only matured setups with readable 60m outcomes are replayed; the newest hour and missing outcomes are excluded.",
            "20,000 prior ledger rows are used; exact live settling delays were not persisted, so this is an approximate diagnostic replay.",
            "Existing execution_eligible flags are preserved; historical quote-volume numbers were not persisted.",
            "Changing base score to 1.5 leaves TRANSITION score at its existing floor (3) and all other gates unchanged.",
            "Post-hoc diagnostics, not an independent profitability test. No target-touch claim is made from terminal returns.",
        ],
    }
    for key in ("consensus_score_1_5", "strategy_horizon_current_score", "strategy_horizon_score_1_5"):
        accepted = [row for row in details if row[key].get("eligible")]
        net = [row[key + "_realised_net_bps"] for row in accepted if row.get(key + "_realised_net_bps") is not None]
        result["counterfactual_outcomes"][key] = {
            "n": len(net), "positive": sum(value > 0 for value in net),
            "mean_net_bps": sum(net) / len(net) if net else None,
            "examples": [{"symbol": row["symbol"], "close_ms": row["close_ms"],
                          "net_bps": row.get(key + "_realised_net_bps")} for row in accepted],
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "replay-details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
