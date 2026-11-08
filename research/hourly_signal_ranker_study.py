"""Walk-forward ranker for Trade3's local 1h long/short signal candidates.

This is offline research only. Features use decision-time candles; label is a
3% target / 3% stop / 24h max-hold realized linear-futures return after 30bps.
The current period was already inspected, so results are exploratory rather
than untouched external validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crypto_forecaster.model import Standardizer, fit_logistic, sigmoid
from alt_hourly_rule_study import RULES, build_signals, evaluate_one, load_frames

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
TARGET_PCT = 3.0
STOP_PCT = 3.0
HOLD_HOURS = 24
ROUND_TRIP_COST_BPS = 30.0
RULE_FEATURES = tuple(f"rule_{name}" for name in RULES)
FEATURES = (
    "direction", "directional_ret6", "directional_ret24", "directional_ret24_vs_btc",
    "directional_dist_ema20", "directional_dist_ema50", "directional_ema20_vs_ema50",
    "directional_ema50_vs_ema200", "directional_ema50_slope6", "directional_btc_ret24",
    "quote_volume_ratio", "directional_taker_imbalance", "realized_vol24", *RULE_FEATURES,
)


def prepare_candidates(frames):
    for frame in frames.values():
        frame["ret1"] = frame.close.pct_change()
    events_by_rule = build_signals(frames)
    grouped = defaultdict(set)
    for rule, events in events_by_rule.items():
        for symbol, index in events:
            grouped[(symbol, index)].add(rule)
    btc = frames["BTCUSDT"].set_index("open_time_ms")
    btc_ret24 = btc.ret24
    btc_dist_ema200 = btc.close / btc.ema200 - 1
    candidates = []
    bars = {symbol: {column: frame[column].to_numpy(dtype=np.int64 if column == "open_time_ms" else float)
                     for column in ("open_time_ms", "open", "high", "low", "close")}
            for symbol, frame in frames.items()}
    for (symbol, i), active_rules in sorted(
        grouped.items(), key=lambda item: (int(bars[item[0][0]]["open_time_ms"][item[0][1]]), item[0][0])
    ):
        f = frames[symbol]
        row = f.iloc[i]
        longs = any(rule.startswith("BULL_") for rule in active_rules)
        shorts = any(rule.startswith("BEAR_") for rule in active_rules)
        if longs == shorts:
            continue
        side = "LONG" if longs else "SHORT"
        sign = 1.0 if side == "LONG" else -1.0
        stamp = int(row.open_time_ms)
        btc_return = float(btc_ret24.get(stamp, np.nan))
        btc_dist = float(btc_dist_ema200.get(stamp, np.nan))
        ret6, ret24 = float(row.ret6), float(row.ret24)
        close = float(row.close)
        numeric = [
            sign,
            sign * ret6,
            sign * ret24,
            sign * (ret24 - btc_return),
            sign * (close / float(row.ema20) - 1),
            sign * (close / float(row.ema50) - 1),
            sign * (float(row.ema20) / float(row.ema50) - 1),
            sign * (float(row.ema50) / float(row.ema200) - 1),
            sign * (float(row.ema50) / float(f.ema50.iloc[max(0, i - 6)]) - 1),
            sign * btc_return,
            float(row.quote_vol_ratio),
            sign * (float(row.taker_ratio) - .5),
            float(f.ret1.iloc[max(0, i - 23):i + 1].std()),
            *[float(rule in active_rules) for rule in RULES],
        ]
        if not np.isfinite(numeric).all() or not np.isfinite(btc_dist):
            continue
        trade = evaluate_one(bars[symbol], i, TARGET_PCT, STOP_PCT, HOLD_HOURS, 0.0, side)
        if trade is None:
            continue
        candidates.append({"symbol": symbol, "side": side, "signal_ms": stamp,
                           "exit_ms": trade["exit_ms"], "gross_bps": trade["gross_bps"],
                           "features": numeric, "rules": sorted(active_rules),
                           "directional_btc_trend": sign * btc_dist})
    # Keep each unique coin/bar candidate. Position overlap is resolved only
    # after scoring, so an unselected signal cannot block a later selected one.
    candidates.sort(key=lambda row: (row["signal_ms"], row["symbol"]))
    return candidates, events_by_rule


def portfolio_select(scored_rows, threshold, max_active=50):
    active, selected = [], []
    candidates = sorted((row for row in scored_rows if row["score"] >= threshold),
                        key=lambda row: (row["signal_ms"], -row["score"], row["symbol"]))
    for row in candidates:
        active = [trade for trade in active if trade["exit_ms"] >= row["signal_ms"]]
        if len(active) >= max_active or any(trade["symbol"] == row["symbol"] for trade in active):
            continue
        selected.append(row)
        active.append(row)
    return selected


def fit_fold(train, calibration, test):
    x_train = np.asarray([row["features"] for row in train], dtype=float)
    x_cal = np.asarray([row["features"] for row in calibration], dtype=float)
    x_test = np.asarray([row["features"] for row in test], dtype=float)
    y_train = np.asarray([row["gross_bps"] > ROUND_TRIP_COST_BPS for row in train], dtype=float)
    median = np.median(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_cal = np.where(np.isfinite(x_cal), x_cal, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    scaler = Standardizer.fit(x_train)
    model = fit_logistic(scaler.transform(x_train), y_train, l2=10.0)
    calibration_scores = sigmoid(model.logits(scaler.transform(x_cal)))
    threshold = float(np.quantile(calibration_scores, .80))
    test_scores = sigmoid(model.logits(scaler.transform(x_test)))
    scored = [{**row, "score": float(score)} for row, score in zip(test, test_scores, strict=True)]
    return threshold, scored, portfolio_select(scored, threshold, max_active=50)


def daily_ci(rows, cost, reps=3000):
    daily = defaultdict(list)
    for row in rows:
        daily[row["signal_ms"] // DAY_MS].append(row["gross_bps"] - cost)
    days = sorted(daily)
    if len(days) < 5:
        return None
    rng = np.random.default_rng(20261003)
    draws = np.empty(reps)
    for i in range(reps):
        selected = rng.choice(days, len(days), replace=True)
        draws[i] = np.mean([value for day in selected for value in daily[int(day)]])
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def summarize(rows, cost):
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    values = np.asarray([row["gross_bps"] - cost for row in rows], dtype=float)
    gains, losses = values[values > 0], values[values < 0]
    return {"n": len(rows), "days": len({row["signal_ms"] // DAY_MS for row in rows}),
            "symbols": len({row["symbol"] for row in rows}),
            "mean_net_bps": float(values.mean()), "median_net_bps": float(np.median(values)),
            "win_pct": float(np.mean(values > 0) * 100),
            "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else None,
            "daily_ci95": daily_ci(rows, cost)}


def run(price_dir: Path, output: Path):
    frames = load_frames(price_dir)
    rows, events_by_rule = prepare_candidates(frames)
    if not rows:
        raise ValueError("No labeled candidate signals")
    first = min(row["signal_ms"] for row in rows)
    last = max(row["signal_ms"] for row in rows)
    span = last - first
    train_cut = first + int(span * .60)
    calibration_cut = first + int(span * .70)
    train = [row for row in rows if row["signal_ms"] < train_cut and row["exit_ms"] < train_cut]
    calibration = [row for row in rows if train_cut <= row["signal_ms"] < calibration_cut and row["exit_ms"] < calibration_cut]
    test = [row for row in rows if row["signal_ms"] >= calibration_cut]
    if min(len(train), len(calibration), len(test)) < 100:
        raise ValueError(f"Insufficient chronological samples: train={len(train)}, calibration={len(calibration)}, test={len(test)}")
    x_train = np.asarray([row["features"] for row in train], dtype=float)
    x_cal = np.asarray([row["features"] for row in calibration], dtype=float)
    x_test = np.asarray([row["features"] for row in test], dtype=float)
    y_train = np.asarray([row["gross_bps"] > ROUND_TRIP_COST_BPS for row in train], dtype=float)
    median = np.median(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_cal = np.where(np.isfinite(x_cal), x_cal, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    scaler = Standardizer.fit(x_train)
    model = fit_logistic(scaler.transform(x_train), y_train, l2=10.0)
    p_cal = sigmoid(model.logits(scaler.transform(x_cal)))
    threshold = float(np.quantile(p_cal, .80))
    p_test = sigmoid(model.logits(scaler.transform(x_test)))
    scored_test = [{**row, "score": float(probability)}
                   for row, probability in zip(test, p_test, strict=True)]
    raw_selected = [row for row in scored_test if row["score"] >= threshold]
    selected = portfolio_select(scored_test, threshold, max_active=50)
    rolling_test, rolling_selected, rolling_scored, fold_metrics = [], [], [], []
    for fold_number, (train_end_fraction, calibration_end_fraction, test_end_fraction) in enumerate(
        ((.40, .50, .60), (.60, .70, .80), (.80, .90, 1.00001)), start=1
    ):
        train_end = first + int(span * train_end_fraction)
        calibration_end = first + int(span * calibration_end_fraction)
        test_end = first + int(span * test_end_fraction)
        fold_train = [row for row in rows if row["signal_ms"] < train_end and row["exit_ms"] < train_end]
        fold_calibration = [row for row in rows if train_end <= row["signal_ms"] < calibration_end and row["exit_ms"] < calibration_end]
        fold_test = [row for row in rows if calibration_end <= row["signal_ms"] < test_end and row["exit_ms"] < test_end]
        if min(len(fold_train), len(fold_calibration), len(fold_test)) < 100:
            fold_metrics.append({"fold": fold_number, "status": "insufficient_sample",
                                 "train": len(fold_train), "calibration": len(fold_calibration),
                                 "test": len(fold_test)})
            continue
        fold_threshold, fold_scored, fold_selected = fit_fold(fold_train, fold_calibration, fold_test)
        rolling_test.extend(fold_test)
        rolling_selected.extend(fold_selected)
        selected_ids = {(row["symbol"], row["signal_ms"]) for row in fold_selected}
        rolling_scored.extend({**row, "fold": fold_number,
                               "portfolio_executed": (row["symbol"], row["signal_ms"]) in selected_ids}
                              for row in fold_scored)
        fold_metrics.append({"fold": fold_number,
                             "calibration_start_utc": pd.to_datetime(train_end, unit="ms", utc=True).isoformat(),
                             "test_start_utc": pd.to_datetime(calibration_end, unit="ms", utc=True).isoformat(),
                             "test_end_utc": pd.to_datetime(test_end, unit="ms", utc=True).isoformat(),
                             "train_n": len(fold_train), "calibration_n": len(fold_calibration),
                             "test_n": len(fold_test), "threshold": fold_threshold,
                             "selected_n": len(fold_selected),
                             "selected_30bps": summarize(fold_selected, 30),
                             "selected_50bps": summarize(fold_selected, 50)})
    rolling_metrics = {"test_all_30bps": summarize(rolling_test, 30),
                       "selected_30bps": summarize(rolling_selected, 30),
                       "selected_50bps": summarize(rolling_selected, 50)}
    roll30, roll50 = rolling_metrics["selected_30bps"], rolling_metrics["selected_50bps"]
    roll_ci = roll30["daily_ci95"]
    rolling_gates = {"positive_mean_30bps": bool(roll30["mean_net_bps"] is not None and roll30["mean_net_bps"] > 0),
                     "positive_mean_50bps": bool(roll50["mean_net_bps"] is not None and roll50["mean_net_bps"] > 0),
                     "ci_lower_above_zero": bool(roll_ci and roll_ci[0] > 0),
                     "at_least_100_selected": roll30["n"] >= 100,
                     "at_least_15_selected_days": roll30["days"] >= 15}
    result = {
        "protocol": {"window_start_utc": pd.to_datetime(first, unit="ms", utc=True).isoformat(),
                     "last_signal_utc": pd.to_datetime(last, unit="ms", utc=True).isoformat(),
                     "train_cut_utc": pd.to_datetime(train_cut, unit="ms", utc=True).isoformat(),
                     "calibration_cut_utc": pd.to_datetime(calibration_cut, unit="ms", utc=True).isoformat(),
                     "entry": "next 1h open after closed signal bar", "target_pct": TARGET_PCT,
                     "stop_pct": STOP_PCT, "max_hold_hours": HOLD_HOURS,
                     "label": "linear futures net return > 0 after 30bps round-trip cost",
                     "selection": "top 20% of calibration scores; cutoff is label-blind and frozen before OOS",
                     "purge": "training/calibration labels crossing each time boundary excluded",
                     "features": list(FEATURES), "rules": list(RULES)},
        "counts": {"total_candidates": len(rows), "train": len(train),
                   "calibration": len(calibration), "test": len(test), "raw_test_above_threshold": len(raw_selected),
                   "selected_test": len(selected), "selected_test_coverage": len(selected) / len(test),
                   "max_active_positions": 50,
                   "train_positive_rate": float(y_train.mean()), "calibration_score_threshold": threshold,
                   "test_days": len({row["signal_ms"] // DAY_MS for row in test}),
                   "selected_days": len({row["signal_ms"] // DAY_MS for row in selected})},
        "metrics": {"test_all_30bps": summarize(test, 30), "selected_30bps": summarize(selected, 30),
                    "selected_50bps": summarize(selected, 50)},
        "rolling_walk_forward": {"folds": fold_metrics, "metrics": rolling_metrics,
                                 "gates": rolling_gates, "any_gate_pass": all(rolling_gates.values()),
                                 "selected_days": roll30["days"]},
        "gates": {},
        "limitations": ["The full date range/universe has been inspected; this is not a fresh future holdout.",
                        "Current futures symbol list creates listing/survivor bias.",
                        "Historical funding, order-book spread and fill quality are unavailable; 30/50bps are stress assumptions.",
                        "The 1h model does not validate 5m broad-alt scalp performance.",
                        "Correlated markets and days remain; daily block intervals only partially address dependence."],
    }
    m30, m50 = result["metrics"]["selected_30bps"], result["metrics"]["selected_50bps"]
    ci = m30["daily_ci95"]
    result["gates"] = {"positive_mean_30bps": bool(m30["mean_net_bps"] is not None and m30["mean_net_bps"] > 0),
                       "positive_mean_50bps": bool(m50["mean_net_bps"] is not None and m50["mean_net_bps"] > 0),
                       "ci_lower_above_zero": bool(ci and ci[0] > 0),
                       "at_least_100_selected": m30["n"] >= 100,
                       "at_least_15_selected_days": m30["days"] >= 15}
    result["any_gate_pass"] = all(result["gates"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = []
    executed_ids = {(row["symbol"], row["signal_ms"]) for row in selected}
    for row, probability in zip(test, p_test, strict=True):
        predictions.append({"symbol": row["symbol"], "side": row["side"], "signal_ms": row["signal_ms"],
                            "rules": "+".join(row["rules"]), "score": float(probability),
                            "gross_bps": row["gross_bps"], "net_bps_30": row["gross_bps"] - 30,
                            "score_selected": probability >= threshold,
                            "portfolio_executed": (row["symbol"], row["signal_ms"]) in executed_ids,
                            "exit_ms": row["exit_ms"]})
    pd.DataFrame(predictions).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame([{"fold": row["fold"], "symbol": row["symbol"], "side": row["side"],
                   "signal_ms": row["signal_ms"], "score": row["score"], "rules": "+".join(row["rules"]),
                   "gross_bps": row["gross_bps"], "portfolio_executed": row["portfolio_executed"]}
                  for row in rolling_scored]).to_csv(output / "rolling_predictions.csv", index=False)
    lines = ["# Trade3 hourly signal ranker — walk-forward", "",
             f"Reproduce: `python research/hourly_signal_ranker_study.py --target-pct {TARGET_PCT:g} --stop-pct {STOP_PCT:g} --hold-hours {HOLD_HOURS} --output artifacts/research/hourly-ranker-{TARGET_PCT:g}pct-20261006`.",
             f"Time splits: train ends {result['protocol']['train_cut_utc']}; calibration ends {result['protocol']['calibration_cut_utc']}; OOS test ends {result['protocol']['last_signal_utc']}.",
             f"Trade label: +{TARGET_PCT:g}%/−{STOP_PCT:g}% bracket, next-hour-open entry, {HOLD_HOURS}h max hold; positive after 30bps round-trip cost. Same-bar ambiguity is stop-first.",
             f"Counts train/calibration/test/raw-selected/executed: {len(train)}/{len(calibration)}/{len(test)}/{len(raw_selected)}/{len(selected)}; threshold {threshold:.4f}; max 50 active, one position per coin.", "",
             "| Segment | n / days | Net mean bps | PF | Daily 95% CI |", "|---|---:|---:|---:|---|"]
    for name, metrics in (("All test candidates @30bps", result["metrics"]["test_all_30bps"]),
                          ("Selected @30bps", result["metrics"]["selected_30bps"]),
                          ("Selected @50bps", result["metrics"]["selected_50bps"])):
        lines.append(f"| {name} | {metrics['n']} / {metrics['days']} | {metrics.get('mean_net_bps')} | {metrics.get('profit_factor')} | {metrics.get('daily_ci95')} |")
    lines += ["", "## Three expanding chronological test blocks", "",
              "Each block refits on earlier closed outcomes, sets the top-20% score threshold on the immediately preceding calibration block without labels, and trades the next block with a 50-position cap and one position per coin.",
              "", "| Fold | Train / calibration / test n | Selected n / days | Net mean 30/50 bps | 30bps CI95 |", "|---:|---:|---:|---:|---|"]
    for fold in fold_metrics:
        if fold.get("status"):
            lines.append(f"| {fold['fold']} | {fold['train']} / {fold['calibration']} / {fold['test']} | insufficient | — | — |")
        else:
            a, b = fold["selected_30bps"], fold["selected_50bps"]
            lines.append(f"| {fold['fold']} | {fold['train_n']} / {fold['calibration_n']} / {fold['test_n']} | {a['n']} / {a['days']} | {a.get('mean_net_bps')} / {b.get('mean_net_bps')} | {a.get('daily_ci95')} |")
    lines += [f"", f"Combined rolling OOS metrics: 30bps {json.dumps(roll30)}; 50bps {json.dumps(roll50)}. Gates: `{json.dumps(rolling_gates)}`."]
    lines += ["", f"Single-split gate checks: `{json.dumps(result['gates'])}`. Overall pass: **{'YES' if result['any_gate_pass'] else 'NO'}**.",
              f"Rolling walk-forward gates: `{json.dumps(rolling_gates)}`. Overall pass: **{'YES' if all(rolling_gates.values()) else 'NO'}**.",
              "", "The single chronological split and the three expanding walk-forward blocks are separate exploratory evaluations on the same inspected historical interval; they are not independent fresh holdouts. No live alert or order was enabled.", "", "## Limitations", "", *[f"- {x}" for x in result["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("artifacts/research/server-long-scout-20261003/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/hourly-ranker-20261003"))
    parser.add_argument("--target-pct", type=float, default=TARGET_PCT)
    parser.add_argument("--stop-pct", type=float, default=STOP_PCT)
    parser.add_argument("--hold-hours", type=int, default=HOLD_HOURS)
    args = parser.parse_args()
    TARGET_PCT, STOP_PCT, HOLD_HOURS = args.target_pct, args.stop_pct, args.hold_hours
    result = run(args.prices, args.output)
    print(json.dumps({"counts": result["counts"], "metrics": result["metrics"],
                      "gates": result["gates"], "rolling_walk_forward": result["rolling_walk_forward"],
                      "output": str(args.output)}, indent=2))
