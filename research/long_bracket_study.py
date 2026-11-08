"""Re-score the saved Trade3 long-discovery candidates with fixed 1h brackets.

Offline only. The setup list came from the existing discovery study; this is
not an independent universe or a live strategy activation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HOUR = 3_600_000
DAY = 24 * HOUR
HOLD_HOURS = 7 * 24
TARGETS = (2.0, 3.0, 5.0)
STOPS = (1.5, 2.0, 3.0, 4.0)
COSTS = (12, 30, 50, 70)
TEST_START = int(datetime(2026, 9, 14, tzinfo=UTC).timestamp() * 1000)


def load_frames(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in directory.glob("*_1h.csv"):
        frames[path.name.removesuffix("_1h.csv")] = pd.read_csv(path)
    return frames


def replay_trade(row: dict, frame: pd.DataFrame, target: float | None, stop: float) -> dict | None:
    start = int(row["sourceCloseMs"]) + 1
    expected = np.arange(start, start + HOLD_HOURS * HOUR, HOUR, dtype=np.int64)
    future = frame[frame.open_time_ms.isin(expected)].sort_values("open_time_ms")
    stamps = future.open_time_ms.to_numpy(dtype=np.int64)
    if len(stamps) != HOLD_HOURS or not np.array_equal(stamps, expected):
        return None
    entry = float(future.iloc[0].open)
    tp = entry * (1 + target / 100) if target is not None else None
    sl = entry * (1 - stop / 100)
    exit_price = float(future.iloc[-1].close)
    exit_i, reason = len(future) - 1, "TIME_7D"
    touch_before_stop = {str(level): False for level in (2, 3, 5)}
    for i, candle in enumerate(future.itertuples()):
        # Conservative hourly OHLC rule: if both barriers touch, stop wins.
        if candle.open <= sl:
            exit_price, exit_i, reason = float(candle.open), i, "STOP_GAP"
            break
        stop_hit = candle.low <= sl
        target_hit = tp is not None and candle.high >= tp
        if stop_hit and target_hit:
            exit_price, exit_i, reason = sl, i, "AMBIGUOUS_STOP_FIRST"
            break
        if stop_hit:
            exit_price, exit_i, reason = sl, i, "STOP"
            break
        for level in touch_before_stop:
            if candle.high >= entry * (1 + int(level) / 100):
                touch_before_stop[level] = True
        if target_hit:
            exit_price, exit_i, reason = tp, i, "TARGET"
            break
    gross_bps = (exit_price / entry - 1) * 10_000
    return {"symbol": row["symbol"], "setup": row["strategy"],
            "signal_ms": int(row["sourceCloseMs"]),
            "exit_ms": int(future.iloc[exit_i].close_time_ms),
            "entry": entry, "exit": exit_price, "gross_bps": gross_bps,
            "reason": reason, "touch_before_stop": touch_before_stop,
            "mae_pct": float(max(0, (1 - future.iloc[:exit_i + 1].low.min() / entry) * 100))}


def portfolio(records: list[dict], frames: dict[str, pd.DataFrame], target: float | None,
              stop: float, cap: int = 50) -> list[dict]:
    active: list[dict] = []
    accepted = []
    for row in sorted(records, key=lambda r: int(r["sourceCloseMs"])):
        signal_ms = int(row["sourceCloseMs"])
        active = [t for t in active if t["exit_ms"] >= signal_ms]
        if len(active) >= cap or any(t["symbol"] == row["symbol"] for t in active):
            continue
        frame = frames.get(row["symbol"])
        if frame is None:
            continue
        trade = replay_trade(row, frame, target, stop)
        if trade is None:
            continue
        accepted.append(trade)
        active.append(trade)
    return accepted


def day_bootstrap(trades: list[dict], cost: float, reps: int = 3000) -> list[float] | None:
    daily: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        daily[t["signal_ms"] // DAY].append(t["gross_bps"] - cost)
    days = sorted(daily)
    if len(days) < 5:
        return None
    rng = np.random.default_rng(20261003)
    means = []
    for _ in range(reps):
        sampled = rng.choice(days, size=len(days), replace=True)
        values = [v for day in sampled for v in daily[int(day)]]
        means.append(float(np.mean(values)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(trades: list[dict], cost: float) -> dict:
    if not trades:
        return {"n": 0}
    returns = np.asarray([t["gross_bps"] - cost for t in trades], dtype=float)
    pos, neg = returns[returns > 0], returns[returns < 0]
    return {"n": len(trades), "coins": len({t["symbol"] for t in trades}),
            "signal_days": len({t["signal_ms"] // DAY for t in trades}),
            "win_n": int((returns > 0).sum()), "win_rate": float(np.mean(returns > 0)),
            "mean_net_bps": float(returns.mean()), "median_net_bps": float(np.median(returns)),
            "profit_factor": float(pos.sum() / -neg.sum()) if len(neg) else None,
            "daily_bootstrap_ci95": day_bootstrap(trades, cost),
            "reasons": {str(k): int(v) for k, v in pd.Series([t["reason"] for t in trades]).value_counts().items()},
            "mean_mae_pct": float(np.mean([t["mae_pct"] for t in trades])),
            "targets_touched_before_stop": {level: {"hits": int(sum(t["touch_before_stop"][str(level)] for t in trades)),
                                                    "n": len(trades),
                                                    "rate": float(np.mean([t["touch_before_stop"][str(level)] for t in trades]))}
                                             for level in (2, 3, 5)}}


def run(source: Path, cache: Path, output: Path) -> dict:
    study = json.loads(source.read_text(encoding="utf-8"))
    rows = study["arms"]["without_bull_gate"]["records"]
    frames = load_frames(cache)
    valid = [r for r in rows if r["outcome"].get("complete") and r["outcome"].get("mature")]
    results = []
    for target in TARGETS:
        for stop in STOPS:
            trades = portfolio(valid, frames, target, stop)
            before = [t for t in trades if t["signal_ms"] < TEST_START]
            after = [t for t in trades if t["signal_ms"] >= TEST_START]
            results.append({"target_pct": target, "stop_pct": stop,
                            "train": {str(c): summarize(before, c) for c in COSTS},
                            "late_test": {str(c): summarize(after, c) for c in COSTS}})
    hold_results = []
    for stop in (6.0, 8.0, 10.0):
        trades = portfolio(valid, frames, None, stop)
        before = [t for t in trades if t["signal_ms"] < TEST_START]
        after = [t for t in trades if t["signal_ms"] >= TEST_START]
        hold_results.append({"target_pct": None, "stop_pct": stop,
                             "train": {str(c): summarize(before, c) for c in COSTS},
                             "late_test": {str(c): summarize(after, c) for c in COSTS}})
    payload = {"method": {"entry": "next 1h candle open", "horizon": "7d",
                          "same_hour_target_and_stop": "stop first",
                          "same_coin_overlap": "blocked until prior exit",
                          "maximum_concurrent_positions": 50,
                          "cost_scenarios_round_trip_bps": COSTS,
                          "late_test_start_utc": datetime.fromtimestamp(TEST_START / 1000, UTC).isoformat(),
                          "candidate_source": "saved Trade3 long-discovery without-bull-gate arm",
                          "limitations": ["survivor-biased current universe", "setup candidates and full-period summary were already inspected",
                                          "late test has fewer than 15 signal days and overlapping 7d outcomes",
                                          "funding and historical spread unavailable", "no live promotion"]},
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "candidate_count": len(rows), "mature_complete_candidates": len(valid),
               "results": results, "seven_day_hold_with_catastrophe_stop": hold_results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Trade3 LONG kurulumları — 7g hedef/stop yeniden oynatımı", "",
             "Giriş sonraki saat mumunun açılışı; hedef/stop ilk dokunuşla kapanır, aynı saat içinde ikisi dokunursa stop önce varsayılır. 7 gün içinde ikisi de olmazsa son kapanış. Pozisyon başına maliyet aşağıda düşülmüştür.",
             f"Aday kaynağı {len(rows)} kaydedilmiş no-BULL-gate LONG kurulumudur; olgun ve kesintisiz {len(valid)} aday. Geç test başlangıcı {payload['method']['late_test_start_utc']}. Bu kohort önceki raporda görüldüğü için bağımsız doğrulama değildir.", "",
             "| Hedef/stop | Erken dönem n / net ort 30bps | Geç test n / kârlı% | Net ort bps 12/30/50/70 | PF (30bps) | Günlük blok GA (30bps) |", "|---|---:|---:|---:|---:|---|"]
    for r in results:
        a, b = r["train"], r["late_test"]
        test30 = b["30"]
        cost_means = "/".join(f"{b[str(c)].get('mean_net_bps',float('nan')):.1f}" for c in COSTS)
        ci = test30.get("daily_bootstrap_ci95")
        lines.append(f"| +{r['target_pct']}/−{r['stop_pct']}% | {a['30'].get('n',0)} / {a['30'].get('mean_net_bps',float('nan')):.1f} | {test30.get('n',0)} / {100*test30.get('win_rate',0):.1f}% | {cost_means} | {test30.get('profit_factor',float('nan')):.2f} | {ci} |")
    lines += ["", "## 7 günlük elde tutma, yalnızca koruyucu stop", "",
              "Bu ayrı swing adayı kâr hedefi koymaz: hedefe değince çıkmak yerine 7 günlük kapanışa kadar tutar; stop ilk kez görülürse çıkar. Kısa vadeli scalp değildir ve %6–%10 stop yatırım başına büyük risk taşır.",
              "", "| Stop | Geç test n / gün | Net ort bps 12/30/50/70 | Kârlı% (30bps) | Medyan (30bps) | %2/%3/%5 önce stop | PF (30bps) | Günlük blok GA |",
              "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in hold_results:
        b = r["late_test"]
        q = b["30"]
        means = "/".join(f"{b[str(c)].get('mean_net_bps',float('nan')):.1f}" for c in COSTS)
        hit = q.get("targets_touched_before_stop", {})
        hit_text = "/".join(f"{100*hit[level]['rate']:.1f}%" for level in (2, 3, 5))
        lines.append(f"| −{r['stop_pct']}% | {q.get('n',0)} / {q.get('signal_days',0)} | {means} | {100*q.get('win_rate',0):.1f}% | {q.get('median_net_bps',float('nan')):.1f} | {hit_text} | {q.get('profit_factor',float('nan')):.2f} | {q.get('daily_bootstrap_ci95')} |")
    lines += ["", "## Ne sayılacak başarı?", "", "Geç testte net beklenen değer pozitif, 95% günlük blok güven aralığının altı sıfırın üstünde, en az 100 işlem ve 15 ayrı sinyal günü; maliyet 50 bps olduğunda da pozitiflik. Mevcut 7g testte en iyi görünümlü kural bu örnek/gün ve örtüşme kapılarını geçmediğinden doğrulanmış sayılmaz; yalnızca yeni, ileriye dönük sessiz test adayıdır.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/research/long-discovery-20260930/study.json"))
    parser.add_argument("--cache", type=Path, default=Path("state/scalp/experiments/long-discovery-v1/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/long-bracket-20261003"))
    args = parser.parse_args()
    result = run(args.source, args.cache, args.output)
    print(json.dumps({"candidate_count": result["candidate_count"], "results": result["results"]}, ensure_ascii=False))
