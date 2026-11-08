"""Trade3-only 5m BTC/ETH walk-forward ranker for a 2% target move.

Offline experiment. Fixed bracket: +2% target / -1.5% stop, 12h maximum
hold, next-bar-open entry, stop-first on ambiguous OHLC bars. Uses only local
Trade3 data; never changes the bot or sends alerts. The inspected history is
exploratory, not independent validation.
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
from intraday_long_bracket_study import BAR_MS, make_events, resolve

TARGET_PCT = 2.0
STOP_PCT = 1.5
HOLD_BARS = 144
COSTS = (30.0, 50.0)
RULES = ("TREND_BREAKOUT_12", "TREND_BREAKOUT_48", "BULL_VOLUME_BREAKOUT_12", "EMA20_RECLAIM")
RULE_FEATURES = tuple(f"rule_{rule}" for rule in RULES)
FEATURES = (
    "ret3", "ret12", "ret48", "close_ema20", "close_ema50", "ema20_50",
    "ema50_200", "ema50_slope12", "btc_ret48", "relative_ret48",
    "volume_ratio", "taker_imbalance", "realized_vol24", "realized_vol72",
    "rsi14", *RULE_FEATURES,
)
DAY_MS = 86_400_000


def load_data(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all aligned 5m futures CSVs; BTC is required as the market benchmark."""
    raw: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*_5m_futures.csv")):
        frame = pd.read_csv(path)
        required = {"open_time_ms", "close_time_ms", "open", "high", "low", "close",
                    "volume", "quote_volume", "taker_buy_base"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if len(frame) <= 200:
            continue
        symbol = path.name.removesuffix("_5m_futures.csv")
        gaps = frame.open_time_ms.diff().dropna().ne(BAR_MS)
        if gaps.any():
            raise ValueError(f"Gaps in {symbol} 5m data")
        raw[symbol] = frame
    if "BTCUSDT" not in raw or len(raw) < 2:
        raise ValueError(f"Need BTCUSDT benchmark and at least one other 5m market in {directory}")
    # Preserve per-symbol history. BTC returns are joined by timestamp in
    # prepare(), so the shortest asset must not truncate the recent OOS tail.
    return raw


def prepare(frames: dict[str, pd.DataFrame]) -> list[dict]:
    events = make_events(frames)
    by_bar: dict[tuple[str, int], set[str]] = defaultdict(set)
    for rule, items in events.items():
        for symbol, i in items:
            by_bar[(symbol, i)].add(rule)

    btc = frames["BTCUSDT"].copy()
    btc["ret48"] = btc.close.pct_change(48)
    btc_ret = pd.Series(btc.ret48.to_numpy(), index=btc.open_time_ms)
    ready: dict[str, pd.DataFrame] = {}
    for symbol, source in frames.items():
        f = source.copy()
        f["ema20"] = f.close.ewm(span=20, adjust=False, min_periods=20).mean()
        f["ema50"] = f.close.ewm(span=50, adjust=False, min_periods=50).mean()
        f["ema200"] = f.close.ewm(span=200, adjust=False, min_periods=200).mean()
        f["ret3"] = f.close.pct_change(3)
        f["ret12"] = f.close.pct_change(12)
        f["ret48"] = f.close.pct_change(48)
        f["volume_ratio"] = f.volume / f.volume.shift(1).rolling(20, min_periods=20).median()
        f["taker_ratio"] = f.taker_buy_base / f.volume.replace(0, np.nan)
        f["realized_vol24"] = f.close.pct_change().rolling(24, min_periods=24).std()
        f["realized_vol72"] = f.close.pct_change().rolling(72, min_periods=72).std()
        delta = f.close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        ready[symbol] = f

    bar_arrays = {
        symbol: {col: f[col].to_numpy(dtype=np.int64 if col in {"open_time_ms", "close_time_ms"} else float)
                 for col in ("open_time_ms", "close_time_ms", "open", "high", "low", "close")}
        for symbol, f in frames.items()
    }
    rows = []
    for (symbol, i), active in sorted(
        by_bar.items(), key=lambda item: (int(ready[item[0][0]].open_time_ms.iloc[item[0][1]]), item[0][0])
    ):
        f = ready[symbol]
        row = f.iloc[i]
        bret = float(btc_ret.get(int(row.open_time_ms), np.nan))
        ret48 = float(row.ret48)
        close = float(row.close)
        features = [
            float(row.ret3), float(row.ret12), ret48,
            close / float(row.ema20) - 1, close / float(row.ema50) - 1,
            float(row.ema20) / float(row.ema50) - 1,
            float(row.ema50) / float(row.ema200) - 1,
            float(row.ema50) / float(f.ema50.iloc[max(0, i - 12)]) - 1,
            bret, ret48 - bret, float(row.volume_ratio),
            float(row.taker_ratio) - 0.5, float(row.realized_vol24),
            float(row.realized_vol72), float(row.rsi14) / 100,
            *[float(rule in active) for rule in RULES],
        ]
        if not np.isfinite(features).all():
            continue
        trade = resolve(frames[symbol], i, TARGET_PCT, STOP_PCT)
        if trade is None:
            continue
        rows.append({"symbol": symbol, "signal_ms": int(row.close_time_ms),
                     "entry_ms": int(trade["entry_ms"]), "exit_ms": int(trade["exit_ms"]),
                     "gross_bps": float(trade["gross_bps"]), "reason": trade["reason"],
                     "features": features, "rules": sorted(active)})
    return sorted(rows, key=lambda row: (row["signal_ms"], row["symbol"]))


def score(train: list[dict], calibration: list[dict], test: list[dict]):
    x_train = np.asarray([r["features"] for r in train], dtype=float)
    x_cal = np.asarray([r["features"] for r in calibration], dtype=float)
    x_test = np.asarray([r["features"] for r in test], dtype=float)
    y = np.asarray([r["gross_bps"] > 30 for r in train], dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels contain only one outcome class")
    median = np.median(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_cal = np.where(np.isfinite(x_cal), x_cal, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    scaler = Standardizer.fit(x_train)
    model = fit_logistic(scaler.transform(x_train), y, l2=10.0)
    cal_scores = sigmoid(model.logits(scaler.transform(x_cal)))
    threshold = float(np.quantile(cal_scores, 0.80))
    test_scores = sigmoid(model.logits(scaler.transform(x_test)))
    scored = [{**row, "score": float(p)} for row, p in zip(test, test_scores, strict=True)]
    return threshold, scored


def select(scored: list[dict], threshold: float) -> list[dict]:
    active: list[dict] = []
    selected = []
    for row in sorted((r for r in scored if r["score"] >= threshold),
                      key=lambda r: (r["signal_ms"], -r["score"], r["symbol"])):
        active = [trade for trade in active if trade["exit_ms"] >= row["signal_ms"]]
        # Fixed portfolio risk cap: at most two simultaneous positions.
        if len(active) >= 2 or any(trade["symbol"] == row["symbol"] for trade in active):
            continue
        selected.append(row)
        active.append(row)
    return selected


def daily_ci(rows: list[dict], cost: float, reps: int = 3000):
    daily: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        daily[row["signal_ms"] // DAY_MS].append(row["gross_bps"] - cost)
    days = sorted(daily)
    if len(days) < 12:
        return None
    rng = np.random.default_rng(20261006)
    samples = np.empty(reps)
    for i in range(reps):
        picked = rng.choice(days, len(days), replace=True)
        samples[i] = np.mean([v for day in picked for v in daily[int(day)]])
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def summarize(rows: list[dict], cost: float) -> dict:
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    pnl = np.asarray([r["gross_bps"] - cost for r in rows], dtype=float)
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len({r["signal_ms"] // DAY_MS for r in rows}),
            "symbols": len({r["symbol"] for r in rows}), "mean_net_bps": float(pnl.mean()),
            "median_net_bps": float(np.median(pnl)), "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else None,
            "target_exits": sum(r["reason"].startswith("TARGET") for r in rows),
            "daily_ci95": daily_ci(rows, cost)}


def run(data_dir: Path, output: Path) -> dict:
    frames = load_data(data_dir)
    rows = prepare(frames)
    if not rows:
        raise ValueError("No eligible labelled candidate signals")
    first, last = min(r["signal_ms"] for r in rows), max(r["signal_ms"] for r in rows)
    span = last - first
    folds = []
    scored_all = []
    selected_all = []
    for number, (train_fraction, cal_fraction, end_fraction) in enumerate(
        ((.40, .50, .60), (.60, .70, .80), (.80, .90, 1.00001)), start=1
    ):
        train_end = first + int(span * train_fraction)
        cal_end = first + int(span * cal_fraction)
        test_end = first + int(span * end_fraction)
        train = [r for r in rows if r["signal_ms"] < train_end and r["exit_ms"] < train_end]
        calibration = [r for r in rows if train_end <= r["signal_ms"] < cal_end and r["exit_ms"] < cal_end]
        test = [r for r in rows if cal_end <= r["signal_ms"] < test_end and r["exit_ms"] < test_end]
        if min(len(train), len(calibration), len(test)) < 100:
            folds.append({"fold": number, "status": "insufficient", "train": len(train),
                          "calibration": len(calibration), "test": len(test)})
            continue
        threshold, scored = score(train, calibration, test)
        selected = select(scored, threshold)
        selected_keys = {(r["symbol"], r["signal_ms"]) for r in selected}
        scored_all.extend({**r, "fold": number,
                           "executed": (r["symbol"], r["signal_ms"]) in selected_keys} for r in scored)
        selected_all.extend({**r, "fold": number} for r in selected)
        folds.append({"fold": number,
                      "train_n": len(train), "calibration_n": len(calibration), "test_n": len(test),
                      "test_start_utc": pd.to_datetime(cal_end, unit="ms", utc=True).isoformat(),
                      "test_end_utc": pd.to_datetime(test_end, unit="ms", utc=True).isoformat(),
                      "threshold": threshold, "selected_n": len(selected),
                      "selected_30bps": summarize(selected, 30), "selected_50bps": summarize(selected, 50)})

    metrics = {"all_oos_candidates_30bps": summarize(scored_all, 30),
               "selected_30bps": summarize(selected_all, 30),
               "selected_50bps": summarize(selected_all, 50)}
    m30, m50 = metrics["selected_30bps"], metrics["selected_50bps"]
    ci = m30.get("daily_ci95")
    gates = {"positive_mean_30bps": bool(m30.get("mean_net_bps") is not None and m30["mean_net_bps"] > 0),
             "positive_mean_50bps": bool(m50.get("mean_net_bps") is not None and m50["mean_net_bps"] > 0),
             "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
             "at_least_100_selected": m30["n"] >= 100,
        "at_least_15_selected_days": m30["days"] >= 15,
        "at_least_two_selected_symbols": m30.get("symbols", 0) >= 2}
    common_start = max(int(f.open_time_ms.iloc[0]) for f in frames.values())
    common_end = min(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    data_start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    data_end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    result = {"protocol": {"data_start_utc": pd.to_datetime(data_start, unit="ms", utc=True).isoformat(),
                            "data_end_utc": pd.to_datetime(data_end, unit="ms", utc=True).isoformat(),
                            "common_window_start_utc": pd.to_datetime(common_start, unit="ms", utc=True).isoformat(),
                            "common_window_end_utc": pd.to_datetime(common_end, unit="ms", utc=True).isoformat(),
                            "symbols": list(frames), "candidate_rules": list(RULES),
                            "entry": "next 5m open after confirmed signal bar",
                            "target_stop": "+2% / -1.5%", "max_hold": "12h",
                            "costs_round_trip_bps": list(COSTS),
                            "threshold": "top 20% calibration scores, label blind",
                            "portfolio": "max 2 concurrent positions; one per symbol",
                            "purge": "exclude labels that cross train, calibration, and test boundaries",
                            "limitations": ["Same inspected Trade3-only local history; exploratory, not fresh external validation.",
                                            "Year-long 5m coverage locally contains only BTC and ETH.",
                                            "No historical spread/funding replay; 30/50 bps are cost stress assumptions.",
                                            "OHLC cannot resolve intrabar order; existing resolver assumes stop first.",
                                            "Universe/listing bias and correlation between BTC/ETH remain."]},
              "counts": {"candidate_signals": len(rows), "folds": len(folds),
                         "selected_oos": len(selected_all)},
              "folds": folds, "rolling_metrics": metrics, "gates": gates,
              "passes_all_gates": all(gates.values()),
              "selected_rows": selected_all}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in scored_all]).to_csv(output / "rolling_predictions.csv", index=False)
    lines = ["# Trade3 5m 2% target ranker — expanding walk-forward", "",
             f"Per-symbol window union {result['protocol']['data_start_utc']} through {result['protocol']['data_end_utc']}; common interval {result['protocol']['common_window_start_utc']} through {result['protocol']['common_window_end_utc']}; symbols {', '.join(frames)}.",
             "Fixed +2% target / -1.5% stop, next 5m open, 12h max hold; ambiguous bar uses stop-first. Scores use the top 20% of the immediately preceding calibration segment; no OOS labels select the threshold.",
             "", "| Fold | Train/cal/test | Selected n/days | Net mean 30/50 bps | 30bps daily CI95 |", "|---:|---:|---:|---:|---|"]
    for fold in folds:
        if fold.get("status"):
            lines.append(f"| {fold['fold']} | {fold['train']}/{fold['calibration']}/{fold['test']} | insufficient | — | — |")
        else:
            a, b = fold["selected_30bps"], fold["selected_50bps"]
            lines.append(f"| {fold['fold']} | {fold['train_n']}/{fold['calibration_n']}/{fold['test_n']} | {a['n']}/{a['days']} | {a['mean_net_bps']:.2f}/{b['mean_net_bps']:.2f} | {a['daily_ci95']} |")
    lines += ["", f"Combined rolling OOS: n={m30['n']}, days={m30['days']}, mean {m30['mean_net_bps']:.2f}/{m50['mean_net_bps']:.2f} bps at 30/50bps costs, CI95={ci}.",
              f"Gates: `{json.dumps(gates)}`. Pass: **{'YES' if all(gates.values()) else 'NO'}**.",
              "", "This is an exploratory screen on previously inspected history. Passing these gates would still require fresh forward shadow validation before alerts; no live bot settings were changed.", "", "## Limitations", "", *[f"- {item}" for item in result["protocol"]["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run_fresh_oos(data_dir: Path, output: Path, holdout_start_ms: int,
                  test_data_dir: Path | None = None) -> dict:
    """Fit/calibrate before the cutoff; optionally score a separate market cache."""
    frames = load_data(data_dir)
    rows = prepare(frames)
    pre = [r for r in rows if r["signal_ms"] < holdout_start_ms and r["exit_ms"] < holdout_start_ms]
    test_frames = load_data(test_data_dir) if test_data_dir is not None else frames
    test_source_rows = prepare(test_frames) if test_data_dir is not None else rows
    test = [r for r in test_source_rows if r["signal_ms"] >= holdout_start_ms]
    if not pre or not test:
        raise ValueError(f"Need labelled pre-holdout and holdout rows; got {len(pre)} and {len(test)}")
    first_pre = min(r["signal_ms"] for r in pre)
    end_pre = max(r["signal_ms"] for r in pre)
    train_end = first_pre + int((end_pre - first_pre) * .80)
    train = [r for r in pre if r["signal_ms"] < train_end and r["exit_ms"] < train_end]
    calibration = [r for r in pre if r["signal_ms"] >= train_end]
    if min(len(train), len(calibration), len(test)) < 20:
        raise ValueError(f"Insufficient fixed-holdout rows: train={len(train)}, calibration={len(calibration)}, test={len(test)}")
    threshold, scored = score(train, calibration, test)
    selected = select(scored, threshold)
    m30, m50 = summarize(selected, 30), summarize(selected, 50)
    ci = m30.get("daily_ci95")
    gates = {
        "positive_mean_30bps": bool(m30.get("mean_net_bps") is not None and m30["mean_net_bps"] > 0),
        "positive_mean_50bps": bool(m50.get("mean_net_bps") is not None and m50["mean_net_bps"] > 0),
        "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
        "at_least_100_selected": m30["n"] >= 100,
        "at_least_15_selected_days": m30["days"] >= 15,
    }
    test_start = min(r["signal_ms"] for r in test)
    test_end = max(r["signal_ms"] for r in test)
    result = {
        "protocol": {
            "mode": "fixed chronological outer holdout",
            "data_start_utc": pd.to_datetime(min(int(f.open_time_ms.iloc[0]) for f in frames.values()), unit="ms", utc=True).isoformat(),
            "data_end_utc": pd.to_datetime(max(int(f.open_time_ms.iloc[-1]) for f in frames.values()), unit="ms", utc=True).isoformat(),
            "holdout_start_utc": pd.to_datetime(holdout_start_ms, unit="ms", utc=True).isoformat(),
            "holdout_first_signal_utc": pd.to_datetime(test_start, unit="ms", utc=True).isoformat(),
            "holdout_last_signal_utc": pd.to_datetime(test_end, unit="ms", utc=True).isoformat(),
            "symbols": list(frames), "test_symbols": list(test_frames),
            "train_data_source": str(data_dir),
            "test_data_source": str(test_data_dir) if test_data_dir is not None else str(data_dir),
            "rules": list(RULES),
            "train_calibration_test_n": [len(train), len(calibration), len(test)],
            "candidate_threshold": threshold, "entry": "next 5m open after confirmed signal",
            "bracket": "+2% target / -1.5% stop", "max_hold": "12h",
            "calibration": "fixed 80th percentile of scores from pre-holdout calibration rows",
            "portfolio": "max 2 concurrent positions; at most one per symbol",
            "costs": list(COSTS), "selection": "fixed; OOS outcomes do not affect score threshold",
            "limitations": ["The train/calibration and test market cohorts may differ; this is deliberate transfer testing, not random sampling.",
                            "The 89-market test cache covers only a few days and cannot satisfy the minimum 15-day sample gate.",
                            "Historical spread/funding are unavailable; 30/50 bps are round-trip cost stress assumptions.",
                            "The train/calibration split is inside the pre-holdout period; it is not tuned on the holdout.",
                            "A positive result still requires more independent symbols/regimes and forward shadow validation."]
        },
        "oos_candidate_signals": len(scored), "selected_oos": len(selected),
        "all_candidates_30bps": summarize(scored, 30),
        "selected_30bps": m30, "selected_50bps": m50,
        "gates": gates, "passes_all_gates": all(gates.values()),
        "selected_rows": [{k: v for k, v in r.items() if k != "features"} for r in selected],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in scored]).to_csv(output / "oos_predictions.csv", index=False)
    pd.DataFrame(result["selected_rows"]).to_csv(output / "selected_oos.csv", index=False)
    lines = ["# Trade3 5m ranker — fixed fresh chronological holdout", "",
             f"Holdout starts {result['protocol']['holdout_start_utc']}; train/calibration/test rows = {len(train)}/{len(calibration)}/{len(test)}.",
             f"Train/calibration symbols: {', '.join(frames)}; OOS symbols: {', '.join(test_frames)}.",
             f"Train source: {data_dir}; OOS source: {test_data_dir if test_data_dir is not None else data_dir}.",
             f"Fixed target/stop +2%/−1.5%, 12h max hold. Training precedes calibration; calibration precedes OOS. No OOS outcomes set the threshold.", "",
             "| Selected | Trades / days | Mean net 30/50 bps | Win rate @30 | PF @30 | Daily CI @30 |",
             "|---|---:|---:|---:|---:|---|",
             f"| Ranker | {m30['n']} / {m30['days']} | {m30.get('mean_net_bps')} / {m50.get('mean_net_bps')} | {m30.get('win_pct')}% | {m30.get('profit_factor')} | {ci} |",
             "", f"Gates: `{json.dumps(gates)}`. Pass: **{'YES' if all(gates.values()) else 'NO'}**.", "",
             *[f"- {item}" for item in result["protocol"]["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/5m-2pct-ranker-20261006"))
    parser.add_argument("--oos-start", type=str, default=None, help="Fixed holdout start with timezone")
    parser.add_argument("--test-data", type=Path, default=None,
                        help="Optional separate Trade3 futures cache used only for the fixed OOS test")
    args = parser.parse_args()
    if args.oos_start:
        from datetime import UTC, datetime
        start = datetime.fromisoformat(args.oos_start.replace("Z", "+00:00"))
        if start.tzinfo is None:
            raise SystemExit("--oos-start must include a timezone")
        payload = run_fresh_oos(args.data, args.output, int(start.astimezone(UTC).timestamp() * 1000),
                                test_data_dir=args.test_data)
        print(json.dumps({"train_calibration_test_n": payload["protocol"]["train_calibration_test_n"],
                          "candidate_signals": payload["oos_candidate_signals"],
                          "selected_oos": payload["selected_oos"],
                          "selected_30bps": payload["selected_30bps"],
                          "selected_50bps": payload["selected_50bps"],
                          "gates": payload["gates"], "pass": payload["passes_all_gates"],
                          "output": str(args.output)}, indent=2))
        raise SystemExit(0)
    payload = run(args.data, args.output)
    print(json.dumps({"counts": payload["counts"], "metrics": payload["rolling_metrics"],
                      "gates": payload["gates"], "passes_all_gates": payload["passes_all_gates"],
                      "output": str(args.output)}, indent=2))
