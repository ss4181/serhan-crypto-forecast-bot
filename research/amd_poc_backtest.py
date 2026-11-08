"""Causal Python replay of the AMD + locked accumulation POC Pine baseline.

Research only: reads Trade3's local 5m OHLCV cache, writes reports under
artifacts/research, and never imports or changes the live bot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MS = 5 * 60_000
FEE_RATE_PER_FILL = 0.0005  # Pine baseline commission_value = 0.05 percent.
PINE_SLIPPAGE_TICKS = 2
DEFAULTS = {
    "accumulation_min_bars": 20,
    "accumulation_max_bars": 60,
    "atr_length": 14,
    "max_range_atr": 2.0,
    "profile_rows": 40,
    "minimum_sweep_ticks": 2,
    "require_close_back_inside": True,
    "maximum_setup_validity_bars": 12,
    "stop_buffer_ticks": 2.0,
    "risk_reward": 2.0,
    "side": "both",
}


@dataclass
class Trade:
    symbol: str
    side: str
    signal_ms: int
    entry_ms: int
    exit_ms: int | None
    signal_close: float
    entry_fill: float
    exit_fill: float | None
    stop: float
    target: float
    exit_reason: str
    gross_bps: float | None
    commission_bps: float | None
    slippage_bps: float | None
    net_bps: float | None
    net_r: float | None


def utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")


def infer_tick_size(path: Path) -> float:
    """Infer only the price precision recorded in CSV (not exchange metadata)."""
    sample = pd.read_csv(path, usecols=["open", "high", "low", "close"], dtype=str)
    decimals = 0
    for column in sample.columns:
        for value in sample[column].dropna():
            decimals = max(decimals, len(value.partition(".")[2]))
    return 10.0 ** (-decimals) if decimals else 1.0


def load_frames(directory: Path) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict]:
    frames: dict[str, pd.DataFrame] = {}
    ticks: dict[str, float] = {}
    gaps: dict[str, int] = {}
    for path in sorted(directory.glob("*_5m_futures.csv")):
        frame = pd.read_csv(path)
        required = {"open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"]].copy()
        frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        if len(frame) < DEFAULTS["accumulation_max_bars"] + 10:
            continue
        symbol = path.name.removesuffix("_5m_futures.csv")
        delta = frame.open_time_ms.diff().dropna()
        gaps[symbol] = int((delta != BAR_MS).sum())
        frames[symbol] = frame
        ticks[symbol] = infer_tick_size(path)
    if not frames:
        raise ValueError(f"No qualifying 5m OHLCV CSV files found in {directory}")
    first = max(int(frame.open_time_ms.iloc[0]) for frame in frames.values())
    last = min(int(frame.open_time_ms.iloc[-1]) for frame in frames.values())
    common = {
        symbol: frame[(frame.open_time_ms >= first) & (frame.open_time_ms <= last)]
        .reset_index(drop=True)
        for symbol, frame in frames.items()
    }
    common = {s: f for s, f in common.items() if len(f) > DEFAULTS["accumulation_max_bars"] + 10}
    metadata = {
        "symbols": len(common),
        "bars_per_symbol_min": min(map(len, common.values())),
        "bars_per_symbol_median": int(np.median([len(f) for f in common.values()])),
        "bars_per_symbol_max": max(map(len, common.values())),
        "common_start_ms": first,
        "common_end_ms": last,
        "continuous_gap_symbols": [s for s, count in gaps.items() if count > 0],
        "gap_count_total": sum(gaps.values()),
    }
    return common, ticks, metadata


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    previous_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    tr[0] = high[0] - low[0]
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return out
    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def locked_poc(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
               end: int, length: int, range_low: float, range_high: float,
               rows: int) -> float | None:
    width = range_high - range_low
    bin_size = width / rows
    if not math.isfinite(bin_size) or bin_size <= 0:
        return None
    row_volumes = np.zeros(rows, dtype=float)
    start = end - length + 1
    for i in range(start, end + 1):
        candle_low, candle_high = low[i], high[i]
        candle_volume = max(0.0, float(volume[i]) if math.isfinite(volume[i]) else 0.0)
        if candle_high <= candle_low:
            row = min(rows - 1, max(0, int(math.floor((close[i] - range_low) / bin_size))))
            row_volumes[row] += candle_volume
            continue
        first = min(rows - 1, max(0, int(math.floor((candle_low - range_low) / bin_size))))
        last = min(rows - 1, max(0, int(math.floor((candle_high - range_low) / bin_size))))
        for row in range(first, last + 1):
            row_low = range_low + row * bin_size
            row_high = row_low + bin_size
            overlap = max(0.0, min(candle_high, row_high) - max(candle_low, row_low))
            if overlap > 0:
                row_volumes[row] += candle_volume * overlap / (candle_high - candle_low)
    best = float(row_volumes.max())
    if best <= 0:
        return None
    # numpy.argmax, like Pine's strict > update, selects the lowest row on ties.
    return range_low + (int(np.argmax(row_volumes)) + 0.5) * bin_size


def _resolve_fill(trade: dict, frame: pd.DataFrame, ticks: float, index: int) -> bool:
    """Fill pending market entry and bracket; return True once it is closed."""
    row = frame.iloc[index]
    side = trade["side"]
    slip = PINE_SLIPPAGE_TICKS * ticks
    raw_open = float(row.open)
    if trade["entry_fill"] is None:
        entry = raw_open + slip if side == "LONG" else raw_open - slip
        trade["entry_slippage"] = slip
        trade["entry_fill"] = entry
        trade["entry_ms"] = int(row.open_time_ms)
    else:
        entry = float(trade["entry_fill"])
    stop, target = trade["stop"], trade["target"]
    high, low = float(row.high), float(row.low)

    # Gap-through bracket levels fill at the open in TradingView's emulator.
    exit_slippage = 0.0
    if side == "LONG" and raw_open <= stop:
        exit_price, reason = raw_open - slip, "STOP_GAP"
        exit_slippage = slip
    elif side == "LONG" and raw_open >= target:
        exit_price, reason = raw_open, "TARGET_GAP"
    elif side == "SHORT" and raw_open >= stop:
        exit_price, reason = raw_open + slip, "STOP_GAP"
        exit_slippage = slip
    elif side == "SHORT" and raw_open <= target:
        exit_price, reason = raw_open, "TARGET_GAP"
    else:
        stop_hit = low <= stop if side == "LONG" else high >= stop
        target_hit = high >= target if side == "LONG" else low <= target
        if stop_hit and target_hit:
            exit_price = stop - slip if side == "LONG" else stop + slip
            reason = "AMBIGUOUS_STOP_FIRST"
            exit_slippage = slip
        elif stop_hit:
            exit_price = stop - slip if side == "LONG" else stop + slip
            reason = "STOP"
            exit_slippage = slip
        elif target_hit:
            exit_price, reason = target, "TARGET"
        else:
            return False

    direction = 1.0 if side == "LONG" else -1.0
    gross_cash = direction * (exit_price - entry)
    fee_cash = FEE_RATE_PER_FILL * (abs(entry) + abs(exit_price))
    net_cash = gross_cash - fee_cash
    planned_risk = abs(trade["signal_close"] - stop)
    total_slippage_cash = float(trade.get("entry_slippage", slip)) + exit_slippage
    trade.update(
        exit_fill=exit_price,
        exit_ms=int(row.close_time_ms),
        exit_reason=reason,
        gross_bps=gross_cash / abs(entry) * 10_000 if entry else None,
        commission_bps=fee_cash / abs(entry) * 10_000 if entry else None,
        slippage_bps=(total_slippage_cash / abs(entry) * 10_000) if entry else None,
        net_bps=net_cash / abs(entry) * 10_000 if entry else None,
        net_r=net_cash / planned_risk if planned_risk > 0 else None,
    )
    return True


def replay_symbol(symbol: str, frame: pd.DataFrame, tick: float,
                  start_ms: int | None = None) -> tuple[list[Trade], dict]:
    o = frame.open.to_numpy(float)
    h = frame.high.to_numpy(float)
    l = frame.low.to_numpy(float)
    c = frame.close.to_numpy(float)
    v = frame.volume.to_numpy(float)
    t = frame.open_time_ms.to_numpy(np.int64)
    close_t = frame.close_time_ms.to_numpy(np.int64)
    atr = atr_wilder(h, l, c, DEFAULTS["atr_length"])

    SEARCH, WAIT_SWEEP, WAIT_REENTRY, WAIT_POC, IN_POSITION = range(5)
    state, direction, sweep_i, reentry_i = SEARCH, 0, None, None
    range_hi = range_lo = poc = manip = None
    pending: dict | None = None
    closed: list[Trade] = []
    open_trades: list[Trade] = []
    unfilled_pending_at_end = 0
    counts = {"accumulations": 0, "down_sweeps": 0, "up_sweeps": 0,
              "reentries": 0, "poc_reclaims": 0, "poc_losses": 0,
              "expired": 0, "ambiguous_both_sweep_bars": 0}

    first_i = int(np.searchsorted(t, start_ms, side="left")) if start_ms is not None else 0
    for i in range(first_i, len(frame)):
        # Broker-emulator order: a close-generated market entry becomes active
        # at the next bar's open, and its bracket can exit on that same bar.
        if pending is not None and i > pending["signal_i"]:
            done = _resolve_fill(pending, frame, tick, i)
            if done:
                closed.append(Trade(**{k: pending.get(k) for k in Trade.__dataclass_fields__}))
                pending = None
                state, direction, sweep_i, reentry_i = SEARCH, 0, None, None
                range_hi = range_lo = poc = manip = None
            if pending is not None:
                # Keep position active; same exit resolver handles each next bar.
                pass
        if pending is not None:
            continue

        # No new setup is created until enough ATR/profile lookback exists.
        if state == SEARCH and math.isfinite(atr[i]):
            selected = None
            for length in range(DEFAULTS["accumulation_min_bars"], DEFAULTS["accumulation_max_bars"] + 1):
                if i < length - 1:
                    continue
                lo = float(np.min(l[i - length + 1 : i + 1]))
                hi = float(np.max(h[i - length + 1 : i + 1]))
                if hi > lo and hi - lo <= atr[i] * DEFAULTS["max_range_atr"]:
                    candidate_poc = locked_poc(h, l, c, v, i, length, lo, hi, DEFAULTS["profile_rows"])
                    if candidate_poc is not None:
                        selected = (hi, lo, candidate_poc)
                        break
            if selected is not None:
                range_hi, range_lo, poc = selected
                state = WAIT_SWEEP
                counts["accumulations"] += 1

        if state == WAIT_SWEEP:
            swept_down = l[i] < range_lo - DEFAULTS["minimum_sweep_ticks"] * tick
            swept_up = h[i] > range_hi + DEFAULTS["minimum_sweep_ticks"] * tick
            if swept_down and swept_up:
                counts["ambiguous_both_sweep_bars"] += 1
            elif swept_down:
                direction, manip, sweep_i = 1, float(l[i]), i
                counts["down_sweeps"] += 1
                state = WAIT_POC if DEFAULTS["require_close_back_inside"] and c[i] > range_lo else WAIT_REENTRY
                if state == WAIT_POC:
                    counts["reentries"] += 1
                    reentry_i = i
            elif swept_up:
                direction, manip, sweep_i = -1, float(h[i]), i
                counts["up_sweeps"] += 1
                state = WAIT_POC if DEFAULTS["require_close_back_inside"] and c[i] < range_hi else WAIT_REENTRY
                if state == WAIT_POC:
                    counts["reentries"] += 1
                    reentry_i = i

        if state == WAIT_REENTRY:
            if direction == 1:
                manip = min(manip, float(l[i]))
                if c[i] > range_lo:
                    state = WAIT_POC
                    counts["reentries"] += 1
                    reentry_i = i
            else:
                manip = max(manip, float(h[i]))
                if c[i] < range_hi:
                    state = WAIT_POC
                    counts["reentries"] += 1
                    reentry_i = i

        if state == WAIT_POC:
            if i - sweep_i > DEFAULTS["maximum_setup_validity_bars"]:
                counts["expired"] += 1
                state, direction, sweep_i, reentry_i = SEARCH, 0, None, None
                range_hi = range_lo = poc = manip = None
            else:
                after_reentry = reentry_i is not None and i > reentry_i
                long_cross = after_reentry and direction == 1 and c[i] > poc and c[i - 1] <= poc
                short_cross = after_reentry and direction == -1 and c[i] < poc and c[i - 1] >= poc
                side_allowed = (
                    DEFAULTS["side"] == "both"
                    or (DEFAULTS["side"] == "long" and direction == 1)
                    or (DEFAULTS["side"] == "short" and direction == -1)
                )
                if (long_cross or short_cross) and side_allowed:
                    side = "LONG" if direction == 1 else "SHORT"
                    stop = manip - DEFAULTS["stop_buffer_ticks"] * tick if direction == 1 else manip + DEFAULTS["stop_buffer_ticks"] * tick
                    signal_close = float(c[i])
                    risk = abs(signal_close - stop)
                    if risk > tick:
                        target = signal_close + risk * DEFAULTS["risk_reward"] if direction == 1 else signal_close - risk * DEFAULTS["risk_reward"]
                        pending = {
                            "symbol": symbol, "side": side, "signal_ms": int(close_t[i]),
                            "entry_ms": None, "exit_ms": None, "signal_close": signal_close,
                            "entry_fill": None, "exit_fill": None, "stop": float(stop),
                            "target": float(target), "exit_reason": "OPEN_AT_DATA_END",
                            "gross_bps": None, "commission_bps": None, "slippage_bps": None,
                            "net_bps": None, "net_r": None, "signal_i": i, "entry_i": i + 1,
                            "entry_slippage": None,
                        }
                        counts["poc_reclaims" if direction == 1 else "poc_losses"] += 1
                        state = IN_POSITION

    if pending is not None:
        # Mark any unclosed position at the final observed close, but keep it
        # censored and separate from closed-trade win/loss statistics.
        final_close = float(c[-1])
        side = pending["side"]
        if pending["entry_fill"] is None and len(frame) > pending["entry_i"]:
            pending["entry_fill"] = float(o[pending["entry_i"]] + (2 * tick if side == "LONG" else -2 * tick))
            pending["entry_ms"] = int(t[pending["entry_i"]])
        if pending["entry_fill"] is not None:
            direction_sign = 1 if side == "LONG" else -1
            gross_cash = direction_sign * (final_close - pending["entry_fill"])
            fee_cash = FEE_RATE_PER_FILL * (abs(pending["entry_fill"]) + abs(final_close))
            pending["exit_fill"] = final_close
            pending["exit_ms"] = int(close_t[-1])
            pending["gross_bps"] = gross_cash / abs(pending["entry_fill"]) * 10_000
            pending["commission_bps"] = fee_cash / abs(pending["entry_fill"]) * 10_000
            pending["slippage_bps"] = float(pending.get("entry_slippage") or 0.0) / abs(pending["entry_fill"]) * 10_000
            pending["net_bps"] = (gross_cash - fee_cash) / abs(pending["entry_fill"]) * 10_000
            pending["net_r"] = (gross_cash - fee_cash) / abs(pending["signal_close"] - pending["stop"])
            open_trades.append(Trade(**{k: pending.get(k) for k in Trade.__dataclass_fields__}))
        else:
            unfilled_pending_at_end = 1
    return closed, {**counts, "open_at_end": len(open_trades),
                    "unfilled_pending_at_end": unfilled_pending_at_end, "open_trades": open_trades}


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    net = np.asarray([float(t.net_bps) for t in trades if t.net_bps is not None])
    r = np.asarray([float(t.net_r) for t in trades if t.net_r is not None])
    wins = int(np.sum(net > 0))
    gross_win = float(net[net > 0].sum())
    gross_loss = float(-net[net < 0].sum())
    daily: dict[str, list[float]] = {}
    for trade in trades:
        day = utc(trade.signal_ms)[:10]
        daily.setdefault(day, []).append(float(trade.net_bps))
    daily_bootstrap_ci = None
    if len(daily) >= 5:
        rng = np.random.default_rng(20261003)
        keys = sorted(daily)
        means = []
        for _ in range(3000):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            means.append(float(np.mean([v for day in sampled for v in daily[str(day)]])))
        daily_bootstrap_ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "trades": len(trades),
        "coins": len({trade.symbol for trade in trades}),
        "days": len(daily),
        "daily_block_bootstrap_ci95_bps": daily_bootstrap_ci,
        "wins": wins,
        "win_rate_pct": wins / len(trades) * 100,
        "mean_net_bps": float(net.mean()),
        "median_net_bps": float(np.median(net)),
        "mean_net_r": float(r.mean()) if len(r) else None,
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "total_net_bps_additive": float(net.sum()),
        "exits": pd.Series([trade.exit_reason for trade in trades]).value_counts().to_dict(),
        "per_day_net_bps": {day: float(np.sum(values)) for day, values in sorted(daily.items())},
    }


def run(data_dir: Path, output_dir: Path, oos_start_ms: int | None = None) -> dict:
    frames, ticks, data_meta = load_frames(data_dir)
    all_closed: list[Trade] = []
    event_totals: dict[str, int] = {}
    open_marks: list[Trade] = []
    by_symbol = []
    for symbol, frame in frames.items():
        closed, events = replay_symbol(symbol, frame, ticks[symbol], start_ms=oos_start_ms)
        all_closed.extend(closed)
        open_marks.extend(events.pop("open_trades"))
        for name, value in events.items():
            if isinstance(value, int):
                event_totals[name] = event_totals.get(name, 0) + value
        by_symbol.append({"symbol": symbol, "tick_size_inferred": ticks[symbol], **events,
                          **summarize(closed)})

    oos_cut = oos_start_ms if oos_start_ms is not None else data_meta["common_start_ms"] + int(
        (data_meta["common_end_ms"] - data_meta["common_start_ms"]) * 0.70
    )
    training = [t for t in all_closed if t.signal_ms < oos_cut and t.exit_ms is not None and t.exit_ms < oos_cut]
    purged_train = [t for t in all_closed if t.signal_ms < oos_cut and (t.exit_ms is None or t.exit_ms >= oos_cut)]
    oos = [t for t in all_closed if t.signal_ms >= oos_cut]

    def cost_stress(trades: list[Trade]) -> dict:
        # Stress costs as a uniform round-trip haircut after reconstructing gross
        # return before commission and slippage.
        gross_pre_cost = np.asarray([
            float(t.net_bps) + float(t.commission_bps) + float(t.slippage_bps)
            for t in trades
        ])
        result = {}
        for cost_bps in (30.0, 50.0):
            stressed = gross_pre_cost - cost_bps
            result[str(int(cost_bps))] = {
                "mean_net_bps": float(stressed.mean()) if len(stressed) else None,
                "win_rate_pct": float(np.mean(stressed > 0) * 100) if len(stressed) else None,
                "total_net_bps_additive": float(stressed.sum()) if len(stressed) else 0.0,
            }
        return result

    # Stress costs as a uniform total round-trip haircut on the exact default
    # replay outcome. This is an extra robustness view, not another fill model.
    stress = cost_stress(all_closed)
    oos_stress = cost_stress(oos)

    result = {
        "strategy": "AMD -> liquidity sweep -> range re-entry -> locked OHLCV POC cross -> 2R bracket",
        "defaults": DEFAULTS,
        "execution_costs": {
            "commission_per_fill_pct": FEE_RATE_PER_FILL * 100,
            "round_trip_commission_bps_approx": FEE_RATE_PER_FILL * 2 * 10_000,
            "slippage_ticks": PINE_SLIPPAGE_TICKS,
            "slippage_tick_source": "inferred from maximum decimal precision in each local CSV; not historical exchangeInfo",
            "stress_round_trip_cost_bps": stress,
        },
        "data": {
            **data_meta,
            "from_utc": utc(data_meta["common_start_ms"]),
            "to_utc": utc(data_meta["common_end_ms"] + BAR_MS - 1),
            "chronological_oos_start_utc": utc(oos_cut),
            "timeframe": "5m",
            "source": str(data_dir),
            "volume_available": True,
        },
        "event_totals": event_totals,
        "closed_trade_summary_default_costs": summarize(all_closed),
        "train_closed_trade_summary": summarize(training),
        "oos_closed_trade_summary": summarize(oos),
        "purged_train_trade_count_crossing_oos_cut": len(purged_train),
        "open_at_data_end_count": len(open_marks),
        "unfilled_entry_orders_at_data_end": sum(row.get("unfilled_pending_at_end", 0) for row in by_symbol),
        "open_at_data_end_mark_to_market": summarize(open_marks),
        "oos_stress_costs": oos_stress,
        "by_symbol": by_symbol,
        "trades": [asdict(trade) for trade in all_closed],
        "open_trades": [asdict(trade) for trade in open_marks],
        "limitations": [
            "Full-year 5m OHLCV is present only for BTC and ETH; the 89-market altcoin OHLCV cache covers only three days.",
            "The 89-market cohort is the locally cached survivor universe and may have survivorship/universe-selection bias.",
            "Tick sizes are inferred from CSV decimal precision, not historical exchange metadata.",
            "5m OHLC cannot resolve the true order of intrabar stop/target touches; ambiguous bars conservatively count stop-first.",
            "Trade results are aggregated as independent per-symbol strategy replays; no shared portfolio capital or concurrent-risk cap is imposed.",
            "Entry and target/stop levels follow the Pine baseline: market entry next bar open, levels calculated from signal close; gaps can change realized R.",
            "Pine's Bar Magnifier may produce different intrabar fills; no synchronized 1m volume data is available in this local cache.",
            "Research output only; does not alter live Trade3 configuration or send Telegram alerts.",
        ],
    }
    if oos_start_ms is not None:
        result["evaluation_mode"] = "fixed untouched chronological holdout; all strategy states reset at OOS start; ATR retains historical warm-up only"
    summary = result["closed_trade_summary_default_costs"]
    oos_summary = result["oos_closed_trade_summary"]
    oos_ci = oos_summary.get("daily_block_bootstrap_ci95_bps")
    criteria = {
        "positive_oos_mean_net": bool(oos_summary.get("mean_net_bps", -math.inf) > 0),
        "oos_daily_ci_lower_above_zero": bool(oos_ci and oos_ci[0] > 0),
        "oos_minimum_100_trades": bool(oos_summary.get("trades", 0) >= 100),
        "oos_minimum_15_signal_days": bool(oos_summary.get("days", 0) >= 15),
        "positive_at_50bps_round_trip": bool(oos_stress["50"].get("mean_net_bps") is not None and oos_stress["50"]["mean_net_bps"] > 0),
    }
    result["predefined_promotion_gates"] = criteria
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result["trades"]).to_csv(output_dir / "trades.csv", index=False)
    pd.DataFrame(by_symbol).to_csv(output_dir / "by_symbol.csv", index=False)
    lines = [
        "# AMD + locked POC Trade3 local-data replay",
        "",
        "This is an offline, research-only replay of the Pine baseline. Live Trade3 and Trade1 were not modified.",
        "",
        f"- Data: {result['data']['symbols']} symbols, {result['data']['bars_per_symbol_min']}–{result['data']['bars_per_symbol_max']} 5m bars per symbol.",
        f"- Window: {result['data']['from_utc']} to {result['data']['to_utc']}.",
        f"- Chronological OOS starts: {result['data']['chronological_oos_start_utc']}; {result['purged_train_trade_count_crossing_oos_cut']} train signals whose outcome crossed the boundary were purged.",
        f"- Events: {result['event_totals']}.",
        f"- Closed trades: {summary.get('trades', 0)}; open at data end: {result['open_at_data_end_count']}.",
        f"- Closed-trade win rate after modeled costs: {summary.get('win_rate_pct', float('nan')):.1f}%.",
        f"- Mean/median net return: {summary.get('mean_net_bps', float('nan')):.1f}/{summary.get('median_net_bps', float('nan')):.1f} bps; mean net R: {summary.get('mean_net_r', float('nan')):.3f}.",
        f"- Profit factor: {summary.get('profit_factor')}.",
        f"- 30/50 bps total-cost stress mean: {stress['30']['mean_net_bps']} / {stress['50']['mean_net_bps']} bps.",
        f"- OOS: n={oos_summary.get('trades', 0)}, signal days={oos_summary.get('days', 0)}, net mean={oos_summary.get('mean_net_bps')}, 95% daily-block CI={oos_ci}.",
        f"- OOS 30/50 bps stress mean: {oos_stress['30']['mean_net_bps']} / {oos_stress['50']['mean_net_bps']} bps.",
        f"- Predefined promotion gates: `{json.dumps(criteria, ensure_ascii=False)}`.",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "## Default settings and event counts",
        "",
        f"- Defaults: `{json.dumps(DEFAULTS, ensure_ascii=False)}`",
        f"- Exit reasons: `{json.dumps(summary.get('exits', {}), ensure_ascii=False)}`",
        "",
        "These replays can check that the rule generates causal trades and expose cost sensitivity; the small trade/day counts do not establish profitability. A longer, multi-coin synchronized OHLCV history is needed before drawing a strategy-quality conclusion.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/scalp"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/amd-poc-20261003"))
    arguments = parser.parse_args()
    outcome = run(arguments.data, arguments.output)
    print(json.dumps({
        "data": outcome["data"],
        "events": outcome["event_totals"],
        "closed_trade_summary": outcome["closed_trade_summary_default_costs"],
        "open_at_data_end": outcome["open_at_data_end_count"],
        "stress_costs": outcome["execution_costs"]["stress_round_trip_cost_bps"],
        "oos": outcome["oos_closed_trade_summary"],
        "oos_stress_costs": outcome["oos_stress_costs"],
        "promotion_gates": outcome["predefined_promotion_gates"],
        "output": str(arguments.output),
    }, ensure_ascii=False, indent=2))
