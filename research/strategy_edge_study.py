"""Exploratory, causal backtest of compact 5m LONG scalp rules.

This script reads only the local Trade3 research cache. It does not touch live
settings, send messages, or submit orders. The current cache lacks volume and
execution-book data, so results are hypotheses, not deployable edge claims.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypto_forecaster.model import Standardizer, fit_logistic, sigmoid

BAR_MS = 5 * 60_000
HOLD_BARS = 12 * 12
ROUND_TRIP_COST_BPS = 30.0
STRATEGIES = ("BULL_BREAKOUT", "BULL_PULLBACK", "RELATIVE_STRENGTH")
BRACKETS = ((2.0, 1.0), (2.0, 1.5), (3.0, 1.5))


def read_cache(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        a = np.asarray(payload["candles"], dtype=float)
        if a.ndim != 2 or a.shape[1] != 5 or len(a) < 300:
            continue
        frame = pd.DataFrame(
            {"open_ms": a[:, 0].astype(np.int64), "high": a[:, 1],
             "low": a[:, 2], "close": a[:, 3], "close_ms": a[:, 4].astype(np.int64)}
        )
        frame = frame.drop_duplicates("open_ms").sort_values("open_ms").reset_index(drop=True)
        frame["ret12"] = frame["close"].pct_change(12)
        frame["ret1"] = frame["close"].pct_change()
        frame["ret48"] = frame["close"].pct_change(48)
        frame["ret144"] = frame["close"].pct_change(144)
        frame["rv144"] = frame["ret1"].rolling(144, min_periods=48).std()
        frame["ema20"] = frame["close"].ewm(span=20, adjust=False).mean()
        frame["ema50"] = frame["close"].ewm(span=50, adjust=False).mean()
        frame["ema200"] = frame["close"].ewm(span=200, adjust=False).mean()
        frame["ema50_rise"] = frame["ema50"] > frame["ema50"].shift(12)
        frame["prior_high12"] = frame["high"].shift(1).rolling(12, min_periods=12).max()
        frame["prior_high48"] = frame["high"].shift(1).rolling(48, min_periods=48).max()
        frame["prior_low12"] = frame["low"].shift(1).rolling(12, min_periods=12).min()
        frame["reclaimed_ema20"] = (frame["close"] > frame["ema20"]) & (
            frame["close"].shift(1) <= frame["ema20"].shift(1)
        )
        frames[path.name.removesuffix(".json.gz")] = frame
    return frames


def add_market_regime(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    btc = frames.get("BTCUSDT")
    if btc is None:
        raise ValueError("BTCUSDT 5m candles are required for market regime")
    all_stamps = sorted(set().union(*(set(f.open_ms.tolist()) for f in frames.values())))
    breadth_rows = []
    aligned = []
    for symbol, f in frames.items():
        x = f[["open_ms", "close", "ema50", "ema200"]].copy()
        x["above_trend"] = x["ema50"] > x["ema200"]
        x["symbol"] = symbol
        aligned.append(x[["open_ms", "above_trend"]])
    breadth = pd.concat(aligned).groupby("open_ms")["above_trend"].mean()
    m = btc.set_index("open_ms")
    for stamp in all_stamps:
        if stamp not in m.index or stamp not in breadth.index:
            continue
        row = m.loc[stamp]
        breadth_rows.append((stamp, bool(row.close > row.ema200 and row.ema50 > row.ema200
                                         and row.ema50 > m.ema50.shift(12).get(stamp, np.nan)
                                         and breadth.loc[stamp] >= 0.55)))
    return pd.DataFrame(breadth_rows, columns=["open_ms", "bull"]).set_index("open_ms")


def signals(frames: dict[str, pd.DataFrame], regime: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    result = {name: [] for name in STRATEGIES}
    bull = regime["bull"]
    btc = frames["BTCUSDT"].set_index("open_ms")
    for symbol, f in frames.items():
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            continue
        for i, r in f.iterrows():
            stamp = int(r.open_ms)
            if stamp not in bull.index or not bull.loc[stamp] or not np.isfinite(r.ema200):
                continue
            uptrend = r.close > r.ema200 and r.ema50 > r.ema200 and bool(r.ema50_rise)
            if not uptrend:
                continue
            if r.close > r.prior_high12 and r.ret12 < 0.025:
                result["BULL_BREAKOUT"].append((symbol, i))
            if bool(r.reclaimed_ema20) and r.close > r.ema50:
                result["BULL_PULLBACK"].append((symbol, i))
            if stamp in btc.index:
                btc_ret = btc.loc[stamp, "close"] / btc.close.shift(12).get(stamp, np.nan) - 1
                if np.isfinite(btc_ret) and r.ret12 >= btc_ret + 0.01 and r.close > r.prior_high48:
                    result["RELATIVE_STRENGTH"].append((symbol, i))
    return result


def resolve_trade(frame: pd.DataFrame, signal_i: int, target_pct: float, stop_pct: float,
                  cost_bps: float = ROUND_TRIP_COST_BPS) -> dict | None:
    # At a signal close, wait for the following candle to close before entering.
    entry_i = signal_i + 1
    first_exit_i = entry_i + 1  # don't use the entry candle's unknown path
    if first_exit_i >= len(frame) or first_exit_i + HOLD_BARS >= len(frame):
        return None
    required = frame.iloc[signal_i + 1 : first_exit_i + HOLD_BARS + 1]["open_ms"].to_numpy()
    if len(required) != HOLD_BARS + 2 or np.any(np.diff(required) != BAR_MS):
        return None
    entry = float(frame.iloc[entry_i].close)
    deadline = min(first_exit_i + HOLD_BARS, len(frame) - 1)
    target = entry * (1 + target_pct / 100)
    stop = entry * (1 - stop_pct / 100)
    exit_price = float(frame.iloc[deadline].close)
    exit_i, reason = deadline, "TIME"
    for j in range(first_exit_i, deadline + 1):
        row = frame.iloc[j]
        hit_target, hit_stop = row.high >= target, row.low <= stop
        if hit_target and hit_stop:
            exit_price, exit_i, reason = stop, j, "AMBIGUOUS_STOP_FIRST"
            break
        if hit_stop:
            exit_price, exit_i, reason = stop, j, "STOP"
            break
        if hit_target:
            exit_price, exit_i, reason = target, j, "TARGET"
            break
    gross_bps = (exit_price / entry - 1) * 10_000
    return {"entry_ms": int(frame.iloc[entry_i].close_ms), "exit_ms": int(frame.iloc[exit_i].close_ms),
            "entry": entry, "exit": exit_price, "reason": reason,
            "gross_bps": gross_bps, "net_bps": gross_bps - cost_bps}


def select_trades(events: list[tuple[str, int]], frames: dict[str, pd.DataFrame],
                  target: float, stop: float) -> list[dict]:
    by_coin: dict[str, list[dict]] = defaultdict(list)
    last_exit: dict[str, int] = {}
    btc = frames["BTCUSDT"].set_index("open_ms")
    for symbol, i in sorted(events, key=lambda x: int(frames[x[0]].iloc[x[1]].open_ms)):
        f = frames[symbol]
        stamp = int(f.iloc[i].open_ms)
        if stamp < last_exit.get(symbol, 0):
            continue
        trade = resolve_trade(f, i, target, stop)
        if trade is None:
            continue
        r = f.iloc[i]
        ret1 = r.ret1
        ret48 = r.ret48
        ret144 = r.ret144
        btc_ret12 = btc.close.pct_change(12).get(stamp, np.nan)
        btc_ret48 = btc.close.pct_change(48).get(stamp, np.nan)
        trade.update(symbol=symbol, signal_ms=stamp, features=[
            float(r.ret12), float(ret1), float(ret48), float(ret144),
            float(r.close / r.ema20 - 1), float(r.close / r.ema50 - 1),
            float(r.close / r.ema200 - 1), float(r.ema50 / r.ema200 - 1),
            float(r.close / r.prior_high12 - 1), float(r.close / r.prior_high48 - 1),
            float(r.ret12 - btc_ret12) if np.isfinite(btc_ret12) else np.nan,
            float(ret48 - btc_ret48) if np.isfinite(btc_ret48) else np.nan,
            float(r.rv144),
        ])
        by_coin[symbol].append(trade)
        last_exit[symbol] = trade["exit_ms"]
    return sorted((x for values in by_coin.values() for x in values), key=lambda x: x["entry_ms"])


def rank_filter(trades: list[dict], first_ms: int, cutoff_ms: int) -> dict:
    """Train early, set a label-blind score cut, evaluate only the late block."""
    train_end = first_ms + int((cutoff_ms - first_ms) * 0.60)
    test_start = first_ms + int((cutoff_ms - first_ms) * 0.70)
    usable = [t for t in trades if np.isfinite(t["features"]).all()]
    train = [t for t in usable if t["exit_ms"] < train_end]
    calibration = [t for t in usable if train_end <= t["signal_ms"] < test_start and t["exit_ms"] < test_start]
    test = [t for t in usable if t["signal_ms"] >= test_start and t["exit_ms"] < cutoff_ms]
    if len(train) < 200 or len(calibration) < 50 or len(test) < 50:
        return {"status": "insufficient_sample", "train_n": len(train),
                "threshold_n": len(calibration), "test_n": len(test)}
    x_train = np.asarray([t["features"] for t in train], dtype=float)
    x_cal = np.asarray([t["features"] for t in calibration], dtype=float)
    x_test = np.asarray([t["features"] for t in test], dtype=float)
    median = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_cal = np.where(np.isfinite(x_cal), x_cal, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    scaler = Standardizer.fit(x_train)
    model = fit_logistic(
        scaler.transform(x_train),
        np.asarray([t["gross_bps"] > ROUND_TRIP_COST_BPS for t in train], dtype=float),
        l2=10,
    )
    p_cal = sigmoid(model.logits(scaler.transform(x_cal)))
    threshold = float(np.quantile(p_cal, 0.80))
    p_test = sigmoid(model.logits(scaler.transform(x_test)))
    selected = [t for t, p in zip(test, p_test, strict=True) if p >= threshold]
    return {"status": "exploratory_walk_forward", "train_n": len(train),
            "threshold_n": len(calibration), "test_n": len(test),
            "threshold_score_top20pct": threshold,
            "all_test_30bps": summarize(test, 30),
            "selected_test_30bps": summarize(selected, 30),
            "selected_test_50bps": summarize(selected, 50),
            "selected_test_70bps": summarize(selected, 70),
            "features": ["ret_5m", "ret_1h", "ret_4h", "ret_12h", "dist_ema20",
                         "dist_ema50", "dist_ema200", "ema50_vs_ema200", "dist_prior_high_1h",
                         "dist_prior_high_4h", "relative_strength_1h", "relative_strength_4h",
                         "realized_vol_12h"],
            "limitations": ["single chronological split", "score threshold from label-blind 10% block",
                            "same nonoverlap candidate simulation", "no volume/spread/funding features"]}


def bootstrap_ci(trades: list[dict], n: int = 2000) -> list[float] | None:
    if len(trades) < 10:
        return None
    # Resample UTC days so same-day altcoin moves do not masquerade as independent.
    days: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        days[t["entry_ms"] // 86_400_000].append(t["net_bps"])
    keys = sorted(days)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(20261002)
    means = []
    for _ in range(n):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = [v for k in sample_keys for v in days[int(k)]]
        means.append(float(np.mean(sample)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(trades: list[dict], cost_bps: float = ROUND_TRIP_COST_BPS) -> dict:
    if not trades:
        return {"trades": 0}
    adjusted = [{**t, "net_bps": t["gross_bps"] - cost_bps} for t in trades]
    returns = np.asarray([t["net_bps"] for t in adjusted], dtype=float)
    wins = int((returns > 0).sum())
    wins_sum = float(returns[returns > 0].sum())
    loss_sum = float(-returns[returns < 0].sum())
    return {"trades": len(trades), "coins": len({t["symbol"] for t in trades}),
            "days": len({t["entry_ms"] // 86_400_000 for t in trades}),
            "wins": wins, "win_rate": wins / len(trades),
            "mean_net_bps": float(returns.mean()), "median_net_bps": float(np.median(returns)),
            "profit_factor": wins_sum / loss_sum if loss_sum else None,
            "max_drawdown_bps": float(np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns)))),
            "mean_net_daily_bootstrap_95": bootstrap_ci(adjusted),
            "exits": {str(k): int(v) for k, v in pd.Series([t["reason"] for t in adjusted]).value_counts().items()},
            "from_utc": datetime.fromtimestamp(min(t["entry_ms"] for t in adjusted) / 1000, UTC).isoformat(),
            "to_utc": datetime.fromtimestamp(max(t["entry_ms"] for t in adjusted) / 1000, UTC).isoformat()}


def run(cache: Path, output: Path, cutoff: str | None) -> dict:
    frames = read_cache(cache)
    regime = add_market_regime(frames)
    events = signals(frames, regime)
    all_stamps = [int(f.open_ms.iloc[-1]) for f in frames.values()]
    latest = max(all_stamps)
    cutoff_ms = int(pd.Timestamp(cutoff).timestamp() * 1000) if cutoff else latest
    first = min(int(f.open_ms.iloc[0]) for f in frames.values())
    split = first + int((cutoff_ms - first) * 0.70)
    results = []
    model_filter = None
    for name, candidates in events.items():
        for target, stop in BRACKETS:
            trades = [t for t in select_trades(candidates, frames, target, stop) if t["exit_ms"] < cutoff_ms]
            oos = [t for t in trades if t["signal_ms"] >= split]
            results.append({"strategy": name, "target_pct": target, "stop_pct": stop,
                            "all": {str(cost): summarize(trades, cost) for cost in (30, 50, 70)},
                            "chronological_late_30pct": {str(cost): summarize(oos, cost) for cost in (30, 50, 70)}})
            if name == "BULL_PULLBACK" and target == 3.0 and stop == 1.5:
                model_filter = rank_filter(trades, first, cutoff_ms)
    payload = {"protocol": {"entry": "following 5m candle close; wait one additional closed candle before outcome scanning",
                             "hold_bars": HOLD_BARS, "cost_bps_round_trip": ROUND_TRIP_COST_BPS,
                             "slippage_stress_bps": [50, 70], "universe": "available 88 symbol cached survivors",
                             "data_limitations": ["OHLC only; no historical volume/spread/funding/orderbook",
                                                  "survivorship bias", "late-period results already partly inspected in earlier target-touch work",
                                                  "exploratory; no live promotion"]},
               "data": {"symbols": len(frames), "first_common_start_utc": datetime.fromtimestamp(first / 1000, UTC).isoformat(),
                        "cutoff_utc": datetime.fromtimestamp(cutoff_ms / 1000, UTC).isoformat(),
                        "late_split_utc": datetime.fromtimestamp(split / 1000, UTC).isoformat(),
                        "bull_market_bars": int(regime.bull.sum()),
                        "bars_per_symbol": {s: len(f) for s, f in frames.items()}}, "results": results,
               "walk_forward_rank_filter": model_filter}
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Trade3 5m strateji keşif testi", "", "Yalnızca OHLC araştırma önbelleği; canlı ayar ve bildirimlere dokunulmadı.",
             "Giriş, sinyalden sonraki mum kapanışında varsayıldı; pozisyon sonuçları bir sonraki kapanmış mumdan başlatıldı. Aynı mum hedef/stop çakışmasında stop varsayıldı.",
             "Maliyet: tur başına 30 bps; 50/70 bps stres kesitleri ayrıca JSON'da yeniden hesaplanmalıdır. Bu sonuçlar canlı başarı veya kâr kanıtı değildir.", "",
             f"Pencere: {payload['data']['first_common_start_utc']} – {payload['data']['cutoff_utc']}; geç dönem kesimi: {payload['data']['late_split_utc']}; {len(frames)} sembol.", "",
             "| Kural | Hedef/stop | Tüm n | Ort net bps (30/50/70 maliyet) | Geç %30 n | Ort net bps (30/50/70) | Geç dönem %95 günlük blok GA (30 bps) |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        a, b = r["all"], r["chronological_late_30pct"]
        all_costs = "/".join(f"{a[str(c)].get('mean_net_bps',float('nan')):.1f}" for c in (30, 50, 70))
        oos_costs = "/".join(f"{b[str(c)].get('mean_net_bps',float('nan')):.1f}" for c in (30, 50, 70))
        ci = b["30"].get("mean_net_daily_bootstrap_95")
        lines.append(f"| {r['strategy']} | +{r['target_pct']}/−{r['stop_pct']}% | {a['30'].get('trades',0)} | {all_costs} | {b['30'].get('trades',0)} | {oos_costs} | {ci} |")
    mf = model_filter or {"status": "not_run"}
    if mf.get("status") == "exploratory_walk_forward":
        selected, base = mf["selected_test_30bps"], mf["all_test_30bps"]
        lines += ["", "## Geçmişten öğrenen sıralama filtresi (keşif)", "",
                  f"LONG geri çekilme +3%/−1,5% kuralında {mf['train_n']} geçmiş işlemle lojistik sıralayıcı eğitildi; sonraki {mf['threshold_n']} işlem sadece skor kesimini (üst %20) belirledi; son dönem testinde filtresiz {base.get('trades',0)} işlemden {selected.get('trades',0)} seçildi.",
                  f"Test net ortalaması 30 bps maliyetle {base.get('mean_net_bps',float('nan')):.1f} bps → seçilenlerde {selected.get('mean_net_bps',float('nan')):.1f} bps; 50/70 bps maliyette seçilen ortalama {mf['selected_test_50bps'].get('mean_net_bps',float('nan')):.1f}/{mf['selected_test_70bps'].get('mean_net_bps',float('nan')):.1f} bps. Bu tek dönem, bağımsız ileri-test değildir."]
    lines += ["", "## Kabul kararı", "", "Bir aday başarı sayılmayacak; geç dönem sonuç net maliyet sonrası pozitif değilse, günlük blok GA alt sınırı sıfırı geçmiyorsa veya işlem sayısı 100'ün/bağımsız gün sayısı 15'in altındaysa terfi etmeyecek. Bu keşif penceresi daha önce kısmen incelendiğinden, hiçbir aday şu an canlı Telegram sinyaline çıkarılamaz; olumlu aday varsa ileriye dönük sessiz test gerekir.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, nargs="?", default=Path("artifacts/research/target-touch-20260928/prices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/strategy-edge-20261002"))
    parser.add_argument("--cutoff", help="UTC cutoff; labels must end before this time")
    args = parser.parse_args()
    result = run(args.cache, args.output, args.cutoff)
    print(json.dumps({"data": result["data"], "results": result["results"]}, ensure_ascii=False, indent=2))
