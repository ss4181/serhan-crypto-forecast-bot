"""Frozen Trade3-only weekly payoff selector; see rolling_payoff_protocol_20261022.md."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_long_bracket_study import make_events
from intraday_long_horizon_study import summarize_fast

DAY = 86_400_000
BAR = 300_000
HOLD = 144
RULES = ("TREND_BREAKOUT_12", "TREND_BREAKOUT_48", "BULL_VOLUME_BREAKOUT_12",
         "EMA20_RECLAIM", "BULL_SQUEEZE_RELEASE")
TARGETS = (2.0, 3.0)


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, UTC).isoformat()


def load_snapshot(root: Path) -> tuple[dict, list]:
    """Longest valid contiguous source per coin, without joining across gaps."""
    frames, manifest = {}, []
    for path in sorted(root.rglob("*_5m_futures.csv")):
        row = {"path": str(path.relative_to(root)),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest.append(row)
        try:
            frame = pd.read_csv(path)
            required = ["open_time_ms", "close_time_ms", "open", "high", "low",
                        "close", "volume", "taker_buy_base"]
            if not set(required).issubset(frame):
                row["status"] = "SKIP_MISSING_COLUMNS"
                continue
            # Conflicting duplicate bars are not silently replaced.
            if frame.open_time_ms.duplicated().any():
                row["status"] = "SKIP_DUPLICATE_TIMESTAMP"
                continue
            frame = frame.sort_values("open_time_ms").reset_index(drop=True)
            valid = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
            valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
            valid &= (frame.high >= frame[["open", "close", "low"]].max(axis=1))
            valid &= (frame.low <= frame[["open", "close"]].min(axis=1))
            valid &= (frame.volume >= 0) & (frame.taker_buy_base >= 0)
            valid &= frame.taker_buy_base <= frame.volume
            valid &= frame.open_time_ms.mod(BAR).eq(0)
            valid &= frame.close_time_ms.eq(frame.open_time_ms + BAR - 1)
            row["invalid_bars"] = int((~valid).sum())
            frame = frame.loc[valid].copy()
            if frame.empty:
                row["status"] = "SKIP_NO_VALID_BARS"
                continue
            blocks = frame.open_time_ms.diff().ne(BAR).cumsum()
            segments = list(frame.groupby(blocks, sort=True))
            _, frame = max(segments, key=lambda item: len(item[1]))
            frame = frame.reset_index(drop=True)
            row.update({"segments": len(segments), "bars": len(frame),
                        "start": iso(int(frame.open_time_ms.iloc[0])),
                        "end": iso(int(frame.close_time_ms.iloc[-1]))})
            symbol = path.name.removesuffix("_5m_futures.csv")
            row["symbol"] = symbol
            if len(frame) < 200 + HOLD:
                row["status"] = "SKIP_SHORT"
                continue
            row["status"] = "AVAILABLE"
            if symbol not in frames or len(frame) > len(frames[symbol][0]):
                frames[symbol] = (frame, row)
        except (ValueError, TypeError, pd.errors.ParserError) as error:
            row["status"] = "SKIP_INVALID_FILE"
            row["error"] = type(error).__name__
    for _, row in frames.values():
        row["status"] = "SELECTED"
    if "BTCUSDT" not in frames:
        raise ValueError("BTC benchmark missing")
    return {symbol: pair[0] for symbol, pair in frames.items()}, manifest


def resolve(frame: pd.DataFrame, signal_i: int, target_pct: float, *, arrays: dict | None = None) -> dict | None:
    """Known closed bar -> following open, no incomplete windows or gap bridging."""
    start, end = signal_i + 1, signal_i + HOLD
    if end >= len(frame):
        return None
    if arrays is None:
        arrays = {column: frame[column].to_numpy() for column in
                  ("open_time_ms", "close_time_ms", "open", "high", "low", "close")}
    if np.any(np.diff(arrays["open_time_ms"][signal_i:end + 1]) != BAR):
        return None
    entry = float(arrays["open"][start])
    target, stop = entry * (1 + target_pct / 100), entry * .985
    o, h, l = (arrays[column][start:end + 1] for column in ("open", "high", "low"))
    hits = np.flatnonzero((l <= stop) | (h >= target))
    idx, price, reason = end, float(arrays["close"][end]), "TIME"
    if len(hits):
        k = int(hits[0])
        idx = start + k
        if o[k] <= stop:
            price, reason = float(o[k]), "STOP_GAP"
        elif o[k] >= target:
            price, reason = float(o[k]), "TARGET_GAP"
        elif l[k] <= stop:
            price, reason = stop, "AMBIGUOUS_STOP_FIRST" if h[k] >= target else "STOP"
        else:
            price, reason = target, "TARGET"
    return {"signal_ms": int(arrays["close_time_ms"][signal_i]),
            "entry_ms": int(arrays["open_time_ms"][start]),
            "exit_ms": int(arrays["close_time_ms"][idx]),
            "entry": entry, "exit": price, "gross_bps": (price / entry - 1) * 10_000,
            "reason": reason, "target_pct": target_pct}


def portfolio(rows: list[dict], busy: dict[str, int] | None = None) -> list[dict]:
    if busy is None:
        busy = {}
    kept = []
    for row in sorted(rows, key=lambda r: (r["entry_ms"], r["symbol"])):
        active = {symbol: exit_ms for symbol, exit_ms in busy.items() if exit_ms >= row["entry_ms"]}
        busy.clear()
        busy.update(active)
        if row["symbol"] in busy or len(busy) >= 3:
            continue
        kept.append(row)
        busy[row["symbol"]] = row["exit_ms"]
    return kept


def training_rows(rows: list[dict], cut: int) -> list[dict]:
    # Apply capacity BEFORE purging unsettled labels: an open position already
    # occupies its slot at the boundary, even though its outcome is not known.
    executed = portfolio([r for r in rows if cut - 28 * DAY <= r["signal_ms"] < cut])
    return [r for r in executed if r["exit_ms"] < cut]


def gates(rows: list[dict]) -> dict:
    stats = summarize_fast(rows, 30, with_ci=True)
    stressed = summarize_fast(rows, 50)
    ci = stats.get("daily_block_ci95")
    return {"positive_mean_30bps": bool(stats.get("mean_net_bps") is not None and stats["mean_net_bps"] > 0),
            "positive_mean_50bps": bool(stressed.get("mean_net_bps") is not None and stressed["mean_net_bps"] > 0),
            "daily_ci_lower_above_zero": bool(ci and ci[0] > 0),
            "at_least_100_trades": len(rows) >= 100,
            "at_least_15_signal_days": stats.get("signal_days", 0) >= 15,
            "at_least_two_symbols": len({r["symbol"] for r in rows}) >= 2}


def run(data: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output exists; refusing to overwrite experiment")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frames, manifest = load_snapshot(data)
    print(f"Loaded {len(frames)} coins from {len(manifest)} source files", flush=True)
    events = make_events(frames)
    arrays = {symbol: {column: frame[column].to_numpy() for column in
                      ("open_time_ms", "close_time_ms", "open", "high", "low", "close")}
              for symbol, frame in frames.items()}
    candidates, skipped = {}, {}
    for rule in RULES:
        for target in TARGETS:
            key = f"{rule}_TP{target:g}"
            rows, unavailable = [], 0
            for symbol, i in events[rule]:
                row = resolve(frames[symbol], i, target, arrays=arrays[symbol])
                if row is None:
                    unavailable += 1
                    continue
                rows.append({**row, "symbol": symbol, "rule": rule})
            candidates[key], skipped[key] = rows, unavailable
            print(f"Resolved {key}: {len(rows)}; unavailable {unavailable}", flush=True)
    end_ms = max(int(frame.close_time_ms.iloc[-1]) for frame in frames.values())
    folds, trades, busy = [], [], {}
    for cut in range(timestamp("2026-06-01T00:00:00Z"), end_ms, 7 * DAY):
        selection, eligible = [], []
        for key, rows in candidates.items():
            train = training_rows(rows, cut)
            stats = summarize_fast(train, 30, with_ci=True)
            ci = stats.get("daily_block_ci95")
            qualifies = (len(train) >= 50 and stats.get("signal_days", 0) >= 10
                         and stats.get("mean_net_bps") is not None and stats["mean_net_bps"] > 20
                         and ci is not None and ci[0] > 0)
            selection.append({"candidate": key, "eligible": qualifies, "training": stats})
            if qualifies:
                eligible.append((key, stats))
        eligible.sort(key=lambda pair: (-pair[1]["daily_block_ci95"][0], -pair[1]["mean_net_bps"], pair[0]))
        chosen = eligible[0][0] if eligible else None
        test = portfolio([r for r in candidates[chosen] if cut <= r["signal_ms"] < cut + 7 * DAY], busy) if chosen else []
        trades.extend(test)
        folds.append({"start": iso(cut), "end_exclusive": iso(cut + 7 * DAY),
                      "selected": chosen, "training_diagnostics": selection,
                      "test30": summarize_fast(test, 30), "test50": summarize_fast(test, 50)})
        print(f"Week {iso(cut)}: {chosen or 'ABSTAIN'}, {len(test)} trades", flush=True)
    fresh = [r for r in trades if r["signal_ms"] >= timestamp("2026-10-11T00:00:00Z")]
    checks = gates(trades)
    result = {"protocol_sha256": hashlib.sha256(Path(__file__).with_name("rolling_payoff_protocol_20261022.md").read_bytes()).hexdigest(),
              "script_sha256": script_hash,
              "manifest": manifest, "unavailable_outcomes": skipped,
              "folds": folds, "trades": trades,
              "oos30": summarize_fast(trades, 30, with_ci=True),
              "oos50": summarize_fast(trades, 50),
              "fresh_tail30": summarize_fast(fresh, 30, with_ci=True),
              "fresh_tail50": summarize_fast(fresh, 50),
              "gates": checks, "passes_exploratory_gates": all(checks.values()),
              "independent_validation": False, "live_activation_allowed": False}
    output.mkdir(parents=True)
    (output / "study.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    text = ["# Trade3 weekly payoff selector", "", "Exploratory only; no live changes.",
            f"Universe: {len(frames)} coins; {len(manifest)} source files; per-symbol coverage in study.json.",
            "Fixed TP2/TP3, SL1.5%, hold12h; three-position portfolio; training-only weekly choice.",
            "", "| Week | Choice | OOS n | Net mean 30/50 bps |", "|---|---|---:|---:|"]
    for f in folds:
        text.append(f"| {f['start'][:10]} | {f['selected'] or 'ABSTAIN'} | {f['test30']['n']} | {f['test30'].get('mean_net_bps')} / {f['test50'].get('mean_net_bps')} |")
    text += ["", f"Combined 30 bps: {result['oos30']}", f"Combined 50 bps: {result['oos50']}",
             f"October 11+ tail: {result['fresh_tail30']}", f"Gates: {checks}",
             "", "## Interpretation limits", "",
             "History has been reused by prior experiments. Neither the historical folds nor the short fresh tail are independent confirmation of this newly designed selector.",
             "Coins have unequal history and query-selection/survivorship bias. Short caches cannot support 28-day training; coverage is reported, not imputed.",
             "30/50 bps are cost stress, not observed funding/spread/fills. Multiple candidate selection and correlated returns require independent future validation even if gates pass."]
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({k: result[k] for k in ("oos30", "oos50", "fresh_tail30", "gates")}, indent=2))
