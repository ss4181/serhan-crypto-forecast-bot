"""OFF-regime SHORT bracket study. See off_short_protocol_20261109.md.

Usage: off_short_study_20261109.py fetch|run <output-dir>
Reads <output-dir>/observations.jsonl.gz, a server-ledger extract. Offline apart
from public Binance USD-M 5m candles. Never changes live settings.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

STEP = 300_000
DAY = 86_400_000
START = int(datetime(2026, 10, 17, tzinfo=UTC).timestamp() * 1000)
TARGET = 0.02
STOP = 0.02
BARS = 288
COST = 20.0
STRESS = 30.0
DAILY_CAP = 5
CI = 0.9875
KLINES = "https://fapi.binance.com/fapi/v1/klines"


def load_observations(root: Path) -> dict[tuple[int, str], list[dict]]:
    groups: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    with gzip.open(root / "observations.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["regime_state"] != "OFF" or row["bar_close_time_ms"] < START:
                continue
            groups[(row["bar_close_time_ms"], row["perpetual_symbol"])][row["family"]] = row
    return {
        key: list(families.values())
        for key, families in sorted(groups.items())
        if any(row.get("execution_eligible") for row in families.values())
    }


def fetch(root: Path) -> None:
    symbols = sorted({symbol for _, symbol in load_observations(root)})
    target = root / "prices"
    target.mkdir(parents=True, exist_ok=True)
    end = int(time.time() * 1000)

    def one(symbol: str) -> tuple[str, int]:
        path = target / f"{symbol}.json.gz"
        if path.exists():
            return symbol, -1
        rows, cursor = [], START - STEP
        while cursor < end:
            query = urlencode({"symbol": symbol, "interval": "5m", "startTime": cursor, "limit": 1500})
            request = Request(f"{KLINES}?{query}", headers={"User-Agent": "trade3-research/1.0"})
            with urlopen(request, timeout=30) as response:
                batch = json.load(response)
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][6]) + 1
            if len(batch) < 1500:
                break
            time.sleep(0.15)
        now = int(time.time() * 1000)
        keep = [
            {"open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "close_ms": int(r[6])}
            for r in rows
            if int(r[6]) < now
        ]
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(keep, handle)
        return symbol, len(keep)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, symbols))
    print("symbols", len(results), "bars", Counter(n for _, n in results).most_common(4))


class Prices:
    def __init__(self, root: Path) -> None:
        self.root = root / "prices"
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def get(self, symbol: str) -> dict[str, np.ndarray] | None:
        if symbol not in self.cache:
            path = self.root / f"{symbol}.json.gz"
            if not path.exists():
                return None
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.cache[symbol] = {
                key: np.array([row[key] for row in rows], dtype=np.float64)
                for key in ("open", "high", "low", "close")
            } | {"close_ms": np.array([row["close_ms"] for row in rows], dtype=np.int64)}
        return self.cache[symbol]


@dataclass(frozen=True)
class Trade:
    signal_ms: int
    exit_ms: int
    entry: float
    resolution: str
    net: float


def simulate_short(prices: dict[str, np.ndarray], signal_ms: int) -> Trade | str:
    index = int(np.searchsorted(prices["close_ms"], signal_ms))
    if index >= len(prices["close_ms"]) or prices["close_ms"][index] != signal_ms:
        return "no_signal_bar"
    first, last = index + 1, index + 1 + BARS
    if last > len(prices["close_ms"]):
        return "incomplete"
    times = prices["close_ms"][first:last]
    if not np.array_equal(times, signal_ms + STEP * np.arange(1, BARS + 1)):
        return "gap"
    entry = float(prices["open"][first])
    target, stop = entry * (1 - TARGET), entry * (1 + STOP)
    hit_target = prices["low"][first:last] <= target
    hit_stop = prices["high"][first:last] >= stop
    target_at = int(np.argmax(hit_target)) if hit_target.any() else BARS
    stop_at = int(np.argmax(hit_stop)) if hit_stop.any() else BARS
    if target_at == BARS and stop_at == BARS:
        close = float(prices["close"][last - 1])
        return Trade(signal_ms, int(times[-1]), entry, "TIME_EXIT", (entry - close) / entry * 1e4 - COST)
    if target_at < stop_at:
        return Trade(signal_ms, int(times[target_at]), entry, "TARGET", TARGET * 1e4 - COST)
    return Trade(signal_ms, int(times[stop_at]), entry, "STOP", -STOP * 1e4 - COST)


def bootstrap(values: np.ndarray, days: np.ndarray, level: float) -> tuple[float, float] | None:
    unique = np.unique(days)
    if len(unique) < 2:
        return None
    sums = np.array([values[days == day].sum() for day in unique])
    counts = np.array([(days == day).sum() for day in unique])
    draws = np.random.default_rng(7).integers(0, len(unique), (20_000, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    net = np.array([trade.net for trade in trades])
    days = np.array([trade.signal_ms // DAY for trade in trades])
    wins = sum(trade.resolution == "TARGET" for trade in trades)
    decided = sum(trade.resolution in {"TARGET", "STOP"} for trade in trades)
    win_flags = np.array([trade.resolution == "TARGET" for trade in trades if trade.resolution != "TIME_EXIT"], float)
    win_days = np.array([trade.signal_ms // DAY for trade in trades if trade.resolution != "TIME_EXIT"])
    low_ci = bootstrap(net, days, CI)
    result = {
        "n": len(trades),
        "days": len(np.unique(days)),
        "wins": wins,
        "decided": decided,
        "time_exit": len(trades) - decided,
        "win_rate": wins / decided if decided else None,
        "win_ci95": bootstrap(win_flags, win_days, 0.95),
        "mean_net_20": float(net.mean()),
        "mean_net_30": float(net.mean()) - (STRESS - COST),
        "net_ci95": bootstrap(net, days, 0.95),
        "net_ci9875": low_ci,
    }
    result["passed"] = bool(
        result["n"] >= 100
        and result["days"] >= 15
        and result["win_rate"] is not None
        and result["win_rate"] >= 0.55
        and result["mean_net_20"] > 0
        and result["mean_net_30"] > 0
        and low_ci is not None
        and low_ci[0] > 0
    )
    return result


ARMS = {
    "MULTI": lambda f: len(f) >= 2,
    "ALL": lambda f: True,
    "F2_CONTINUATION": lambda f: "F2" in f,
    "F1F3_FADE": lambda f: bool(f & {"F1", "F3"}) and "F2" not in f,
}


def portfolio(events: list[tuple], trades: dict[tuple[int, str], Trade], cap: int | None) -> list[Trade]:
    open_until: dict[str, int] = {}
    used: Counter = Counter()
    chosen = []
    for signal_ms, symbol, _score in sorted(events, key=lambda e: (e[0], -e[2], e[1])):
        trade = trades.get((signal_ms, symbol))
        if trade is None or open_until.get(symbol, 0) >= signal_ms:
            continue
        day = signal_ms // DAY
        if cap is not None and used[day] >= cap:
            continue
        used[day] += 1
        open_until[symbol] = trade.exit_ms
        chosen.append(trade)
    return chosen


def run(root: Path) -> None:
    groups = load_observations(root)
    prices = Prices(root)
    trades: dict[tuple[int, str], Trade] = {}
    skipped: Counter = Counter()
    mismatches = checked = 0
    for (signal_ms, symbol), rows in groups.items():
        series = prices.get(symbol)
        if series is None:
            skipped["no_prices"] += 1
            continue
        outcome = simulate_short(series, signal_ms)
        if isinstance(outcome, str):
            skipped[outcome] += 1
            continue
        trades[(signal_ms, symbol)] = outcome
        ledger_entry = rows[0].get("entry_price")
        if isinstance(ledger_entry, (int, float)):
            checked += 1
            mismatches += abs(ledger_entry / outcome.entry - 1) > 1e-6
    if not checked or mismatches / checked > 0.01:
        raise SystemExit(f"Entry cross-check failed: {mismatches}/{checked}")

    report: dict = {
        "setups": len(groups),
        "simulated": len(trades),
        "skipped": dict(skipped),
        "entry_check": {"checked": checked, "mismatches": mismatches},
        "families": Counter("+".join(sorted({r["family"] for r in rows})) for rows in groups.values()).most_common(),
    }
    for name, keep in ARMS.items():
        events = [
            (key[0], key[1], max(float(r["score"]) for r in rows))
            for key, rows in groups.items()
            if keep({r["family"] for r in rows})
        ]
        report[name] = {
            "capped": summarize(portfolio(events, trades, DAILY_CAP)),
            "uncapped": summarize(portfolio(events, trades, None)),
        }

    first_day = START // DAY
    last_signal = max(key[0] for key in groups)
    baseline: list[Trade] = []
    for symbol in sorted({key[1] for key in groups}):
        series = prices.get(symbol)
        if series is None:
            continue
        for day in range(first_day, last_signal // DAY + 1):
            for hour in (0, 6, 12, 18):
                outcome = simulate_short(series, day * DAY + hour * 3_600_000 - 1)
                if isinstance(outcome, Trade):
                    baseline.append(outcome)
    report["BASELINE"] = summarize(baseline)
    (root / "study.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]](Path(sys.argv[2]))
