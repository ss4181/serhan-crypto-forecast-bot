"""Fixed-rule hourly squeeze-release long strategy on Trade3 broad history.

Signals use BTC bull alignment, each coin's uptrend, a volatility-squeeze
release, and confirmed green/volume/taker flow. This is offline research only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from alt_hourly_rule_study import HOUR_MS, load_frames

DAY_MS = 86_400_000
OOS_START = int(pd.Timestamp("2026-09-01T00:00:00Z").timestamp() * 1000)
OOS_MID = int(pd.Timestamp("2026-09-17T00:00:00Z").timestamp() * 1000)
OOS_END = int(pd.Timestamp("2026-10-03T18:00:00Z").timestamp() * 1000)
TARGETS = (2.0, 3.0)
STOP = 1.5
HOLD = 12
CAP = 10
COSTS = (30.0, 50.0)


def add_features(frames: dict[str, pd.DataFrame]) -> None:
    for f in frames.values():
        prior_close = f.close.shift(1)
        tr = pd.concat([(f.high - f.low), (f.high - prior_close).abs(),
                        (f.low - prior_close).abs()], axis=1).max(axis=1)
        f["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        middle = f.close.rolling(20, min_periods=20).mean()
        width = 2 * f.close.rolling(20, min_periods=20).std(ddof=0)
        upper_kc = f.ema20 + 1.5 * f.atr14
        lower_kc = f.ema20 - 1.5 * f.atr14
        f["squeeze_on"] = ((middle + width) < upper_kc) & ((middle - width) > lower_kc)
        f["squeeze_release"] = f.squeeze_on.shift(1, fill_value=False) & ~f.squeeze_on


def make_events(frames: dict[str, pd.DataFrame]) -> list[dict]:
    btc = frames.get("BTCUSDT")
    if btc is None:
        raise ValueError("BTCUSDT hourly benchmark required")
    btc_bull = (btc.close > btc.ema50) & (btc.ema50 > btc.ema200) & btc.ema50_rising6
    btc_by_time = pd.Series(btc_bull.to_numpy(dtype=bool), index=btc.open_time_ms)
    events = []
    for symbol, f in frames.items():
        if symbol == "BTCUSDT":
            continue
        market_bull = btc_by_time.reindex(f.open_time_ms, fill_value=False).to_numpy(dtype=bool)
        trend = (f.close > f.ema50) & (f.ema20 > f.ema50) & f.ema50_rising6
        green_release = f.squeeze_release & (f.close > f.open) & (f.close > f.ema20)
        flow = (f.quote_vol_ratio >= 1.5) & (f.taker_ratio >= .55)
        mask = market_bull & trend.to_numpy() & green_release.to_numpy() & flow.to_numpy()
        finite = np.isfinite(f.quote_vol_ratio.to_numpy()) & np.isfinite(f.taker_ratio.to_numpy())
        for i in np.flatnonzero(mask & finite):
            events.append({"symbol": symbol, "i": int(i), "signal_ms": int(f.open_time_ms.iloc[i]), "bar": f})
    return sorted(events, key=lambda e: (e["signal_ms"], e["symbol"]))


def resolve(event: dict, target_pct: float) -> dict | None:
    f, i = event["bar"], event["i"]
    entry_i, last_i = i + 1, i + HOLD
    if last_i >= len(f):
        return None
    # Reject a trade if its holding window crosses a data gap.
    if f.open_time_ms.iloc[entry_i:last_i + 1].diff().dropna().ne(HOUR_MS).any():
        return None
    entry = float(f.open.iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None
    target, stop = entry * (1 + target_pct / 100), entry * (1 - STOP / 100)
    exit_price, exit_i, reason = float(f.close.iloc[last_i]), last_i, "TIME_12H"
    for j in range(entry_i, last_i + 1):
        o, h, l = float(f.open.iloc[j]), float(f.high.iloc[j]), float(f.low.iloc[j])
        if o <= stop:
            exit_price, exit_i, reason = o, j, "STOP_GAP"
            break
        if o >= target:
            exit_price, exit_i, reason = o, j, "TARGET_GAP"
            break
        if l <= stop and h >= target:
            exit_price, exit_i, reason = stop, j, "AMBIGUOUS_STOP_FIRST"
            break
        if l <= stop:
            exit_price, exit_i, reason = stop, j, "STOP"
            break
        if h >= target:
            exit_price, exit_i, reason = target, j, "TARGET"
            break
    return {"symbol": event["symbol"], "signal_ms": int(f.close_time_ms.iloc[i]),
            "entry_ms": int(f.open_time_ms.iloc[entry_i]), "exit_ms": int(f.close_time_ms.iloc[exit_i]),
            "gross_bps": (exit_price / entry - 1) * 10_000, "reason": reason}


def summarize(rows: list[dict], cost: float) -> dict:
    if not rows:
        return {"n": 0, "days": 0, "mean_net_bps": None, "daily_ci95": None}
    pnl = np.asarray([r["gross_bps"] - cost for r in rows], dtype=float)
    daily: dict[int, list[float]] = defaultdict(list)
    for row, value in zip(rows, pnl, strict=True):
        daily[row["signal_ms"] // DAY_MS].append(float(value))
    ci = None
    if len(daily) >= 12:
        days = np.asarray(sorted(daily), dtype=np.int64)
        rng = np.random.default_rng(20261006)
        draws = np.empty(5000)
        for i in range(len(draws)):
            selected_days = rng.choice(days, len(days), replace=True)
            draws[i] = np.mean([v for d in selected_days for v in daily[int(d)]])
        ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {"n": len(rows), "days": len(daily), "symbols": len({r["symbol"] for r in rows}),
            "mean_net_bps": float(pnl.mean()), "median_net_bps": float(np.median(pnl)),
            "win_pct": float((pnl > 0).mean() * 100),
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else None,
            "target_exits": sum(r["reason"].startswith("TARGET") for r in rows), "daily_ci95": ci}


def run(data: Path, out: Path) -> dict:
    frames = load_frames(data)
    add_features(frames)
    events = [e for e in make_events(frames) if OOS_START <= e["signal_ms"] < OOS_END]
    variants, trades_out = [], []
    for target in TARGETS:
        portfolio, active = [], []
        for event in events:
            active = [t for t in active if t["exit_ms"] > event["signal_ms"]]
            if len(active) >= CAP or any(t["symbol"] == event["symbol"] for t in active):
                continue
            trade = resolve(event, target)
            if trade is not None:
                portfolio.append(trade)
                active.append(trade)
        m30, m50 = summarize(portfolio, 30), summarize(portfolio, 50)
        ci = m30.get("daily_ci95")
        gates = {"positive_mean_30bps": bool(m30.get("mean_net_bps") is not None and m30["mean_net_bps"] > 0),
                 "positive_mean_50bps": bool(m50.get("mean_net_bps") is not None and m50["mean_net_bps"] > 0),
                 "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
                 "at_least_100_trades": m30["n"] >= 100, "at_least_15_days": m30["days"] >= 15}
        fold1 = [t for t in portfolio if t["signal_ms"] < OOS_MID]
        fold2 = [t for t in portfolio if t["signal_ms"] >= OOS_MID]
        variants.append({"target_pct": target, "stop_pct": STOP, "oos_30bps": m30,
                         "oos_50bps": m50, "fold1_30bps": summarize(fold1, 30),
                         "fold2_30bps": summarize(fold2, 30), "gates": gates,
                         "passes_all_gates": all(gates.values())})
        trades_out.extend({**t, "target_pct": target,
                           "fold": "first_half" if t["signal_ms"] < OOS_MID else "second_half"}
                          for t in portfolio)
    union_start = min(int(f.open_time_ms.iloc[0]) for f in frames.values())
    union_end = max(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    common_start = max(int(f.open_time_ms.iloc[0]) for f in frames.values())
    common_end = min(int(f.open_time_ms.iloc[-1]) for f in frames.values())
    result = {"data_union_utc": [pd.to_datetime(union_start, unit="ms", utc=True).isoformat(),
                                  pd.to_datetime(union_end + HOUR_MS - 1, unit="ms", utc=True).isoformat()],
              "common_window_utc": [pd.to_datetime(common_start, unit="ms", utc=True).isoformat(),
                                    pd.to_datetime(common_end + HOUR_MS - 1, unit="ms", utc=True).isoformat()],
              "symbols": len(frames), "oos_utc": ["2026-09-01T00:00:00Z", "2026-10-03T18:00:00Z"],
              "oos_signal_events": len(events), "signal_symbols": len({e["symbol"] for e in events}),
              "rule": "BTC bull: close>EMA50>EMA200 and EMA50 rising6; coin trend: close>EMA50, EMA20>EMA50, EMA50 rising6; prior-bar BB(20,2) inside EMA20 +/- 1.5*ATR14 Keltner, current bar releases squeeze with green close above EMA20; quote-volume ratio>=1.5, taker buy>=0.55.",
              "execution": {"entry": "next hourly open", "target_pct": list(TARGETS),
                            "stop_pct": STOP, "max_hold_hours": HOLD, "gap_fill": "at open",
                            "same_bar": "stop-first", "max_concurrent": CAP,
                            "cost_bps": list(COSTS)},
              "variants": variants, "trades": trades_out,
              "limitations": ["OOS interval was inspected in prior Trade3 studies; exploratory only.",
                              "Only 266 currently cached symbols; listing/survivorship bias.",
                              "30/50bps are round-trip stress haircuts, not historical funding/spread/fill replay.",
                              "Target/stop ambiguity is resolved against the trade; hourly OHLC is coarse for scalping."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(trades_out).to_csv(out / "trades.csv", index=False)
    lines = ["# Trade3 1h squeeze-release long strategy", "",
             f"Symbols: {len(frames)}. Union {result['data_union_utc'][0]}–{result['data_union_utc'][1]}; common overlap {result['common_window_utc'][0]}–{result['common_window_utc'][1]}.",
             f"OOS {result['oos_utc'][0]}–{result['oos_utc'][1]}, {len(events)} signals across {result['signal_symbols']} symbols.",
             f"Rule: {result['rule']}", "", "| TP | n/days | Net mean 30/50 bps | PF30 | TP exits | CI95 daily | Fold means 30bps | Pass? |",
             "|---:|---:|---:|---:|---:|---|---:|---|"]
    for v in variants:
        a, b = v["oos_30bps"], v["oos_50bps"]
        lines.append(f"| +{v['target_pct']}% | {a['n']}/{a['days']} | {a.get('mean_net_bps')}/{b.get('mean_net_bps')} | {a.get('profit_factor')} | {a.get('target_exits')} | {a.get('daily_ci95')} | {v['fold1_30bps'].get('mean_net_bps')}/{v['fold2_30bps'].get('mean_net_bps')} | {'YES' if v['passes_all_gates'] else 'NO'} |")
    lines += ["", "Gates: positive cost-adjusted mean at both 30/50bps, positive lower 95% daily-block CI, >=100 trades, >=15 signal days.", "", "## Limitations", ""] + [f"- {x}" for x in result["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/research/server-long-scout-20261003/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/alt-squeeze-hourly-20261006"))
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({"oos_events": result["oos_signal_events"], "signal_symbols": result["signal_symbols"],
                      "variants": result["variants"], "passes_any": any(v["passes_all_gates"] for v in result["variants"]),
                      "output": str(args.output)}, indent=2))
