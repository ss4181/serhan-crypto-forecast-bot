"""Offline, causal target-touch logistic study. Never changes live bot settings.

Protocol frozen before inspecting labels: 24h targets, L2=10, daily expanding
training with a 24h purge before a separate 5-day calibration window, then a
second 24h purge before testing. Select one side using P(2%), >=.70 and a .10
side-probability margin. 3/5% are evaluated on that same direction, never on
whichever side happened to win. No automatic promotion or Telegram delivery.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from crypto_forecaster.data import BinanceMarketDataClient
from crypto_forecaster.measurement import _wilson_interval
from crypto_forecaster.model import Standardizer, fit_logistic, fit_platt

STEP = 300_000
DAY = 86_400_000
LEVELS = (2, 3, 5)
SIDES = ("LONG", "SHORT")
FAMILIES = ("F1", "F2", "F3", "B1", "B2", "B3")
COHORT_START = int(datetime(2026, 9, 27, 17, 1, 36, tzinfo=UTC).timestamp() * 1000)
COHORT_END = COHORT_START + DAY
NUMERIC = (
    "score",
    "family_count",
    "regime_score",
    "breadth",
    "spread_bps",
    "funding_rate_bps",
    "return_24h_pct",
    "volume_1h_ratio",
    "volatility_bps",
    "taker_buy_ratio_1h",
)
FEATURES = (*NUMERIC, "BULL", "TRANSITION", *FAMILIES)
THRESHOLD = 0.70
MARGIN = 0.10


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")


def finite(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def load_setups(path):
    """Only original decision-time fields are allowed into predictors."""
    groups = defaultdict(dict)
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if row.get("alert_tier") != "KURULUM":
                continue
            key = (row["perpetual_symbol"], int(row["bar_close_time_ms"]))
            family = row["family"]
            if family not in groups[key] or row.get("horizon_minutes") == 15:
                groups[key][family] = row
    setups = []
    for (symbol, stamp), families in sorted(groups.items(), key=lambda x: x[0][1]):
        if len(families) < 2:
            continue
        records = list(families.values())
        source = records[0]
        prices = [finite(row.get("entry_price")) for row in records]
        if any(p is None or p <= 0 for p in prices):
            continue
        if max(prices) / min(prices) - 1 > 1e-8:
            raise ValueError(f"Inconsistent reference price {symbol} {stamp}")
        values = {key: finite(source.get(key)) for key in NUMERIC}
        values.update(
            score=max(float(r["score"]) for r in records), family_count=len(families)
        )
        values.update(
            {
                regime: float(source.get("regime_state") == regime)
                for regime in ("BULL", "TRANSITION")
            }
        )
        values.update({family: float(family in families) for family in FAMILIES})
        setups.append(
            {
                "symbol": symbol,
                "time_ms": stamp,
                "time_utc": utc(stamp),
                # The family outcome ledger stores NEXT candle open here,
                # not the original signal close. Never use it as a predictor
                # or quietly mislabel it as the Telegram reference price.
                "ledger_next_open_price": prices[0],
                "reference_price": None,
                "families": sorted(families),
                "regime": source.get("regime_state", "UNKNOWN"),
                "policy": source.get("policy_version", "legacy"),
                "cost_bps": max(
                    float(r.get("round_trip_cost_bps", 12)) for r in records
                ),
                "features": [values[name] for name in FEATURES],
            }
        )
    return setups, malformed


def fetch_prices(setups, directory, cutoff):
    """Public GETs only, local research files, two throttled workers."""
    directory.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in setups:
        grouped[row["symbol"]].append(row["time_ms"])

    def fetch(symbol, stamps):
        start = min(stamps) - STEP + 1
        end = min(max(stamps) + DAY + 1, cutoff)
        path = directory / f"{symbol}.json.gz"
        previous = []
        fetch_start = start
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved["requested_start"] <= start and saved["requested_end"] >= end:
                return symbol, len(saved["candles"]), "cached"
            if saved["requested_start"] <= start:
                previous = saved["candles"]
                fetch_start = max(start, saved["requested_end"] - STEP)
        client = BinanceMarketDataClient(
            market_name="futures", request_pause_seconds=0.8
        )
        frame = client.fetch_market_klines(
            symbol, "5m", start_ms=fetch_start, end_ms=end
        )
        rows = [
            [
                int(r.open_time_ms),
                float(r.high),
                float(r.low),
                float(r.close),
                int(r.close_time_ms),
            ]
            for r in frame.itertuples()
        ]
        by_time = {row[0]: row for row in previous + rows}
        rows = [by_time[stamp] for stamp in sorted(by_time)]
        payload = {
            "symbol": symbol,
            "requested_start": start,
            "requested_end": end,
            "candles": rows,
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False)
        return symbol, len(rows), "fetched"

    failures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(fetch, symbol, stamps): symbol
            for symbol, stamps in grouped.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                print(i, len(grouped), *future.result(), flush=True)
            except (OSError, RuntimeError, ValueError) as error:
                failures[symbol] = f"{type(error).__name__}: {error}"
                print(i, len(grouped), symbol, "FAILED", flush=True)
    return failures


def label_setup(setup, candles, cutoff):
    """Exclude source candle, never call immature/unobserved negatives misses."""
    stamp = setup["time_ms"]
    deadline = stamp + DAY
    end = min(deadline, cutoff - 1)
    source = candles[candles[:, 4] == stamp]
    future = candles[(candles[:, 0] > stamp) & (candles[:, 4] <= end)]
    expected = np.arange(stamp + 1, end - STEP + 2, STEP, dtype=np.int64)
    contiguous = len(future) == len(expected) and np.array_equal(
        future[:, 0].astype(np.int64), expected
    )
    price = finite(setup.get("reference_price"))
    reference_ok = (
        price is not None
        and price > 0
        and len(source) == 1
        and abs(source[0, 3] / price - 1) <= 1e-6
    )
    valid = bool(contiguous and reference_ok)
    price = price if price is not None and price > 0 else 1.0
    mature = deadline < cutoff
    result = {
        "mature": mature,
        "complete": valid,
        "bars": len(future),
        "deadline_ms": deadline,
        "sides": {},
    }
    for side in SIDES:
        favorable = (
            (future[:, 1] / price - 1) * 100
            if side == "LONG"
            else (1 - future[:, 2] / price) * 100
        )
        adverse = (
            (1 - future[:, 2] / price) * 100
            if side == "LONG"
            else (future[:, 1] / price - 1) * 100
        )
        touches = {}
        for level in LEVELS:
            indices = np.flatnonzero(favorable >= level - 1e-10)
            stops = np.flatnonzero(adverse >= 1 - 1e-10)
            first_hit = int(indices[0]) if len(indices) else None
            first_stop = int(stops[0]) if len(stops) else None
            ambiguous = first_hit is not None and first_hit == first_stop
            first_target = (
                None
                if not valid or ambiguous
                else bool(first_stop is None or first_hit < first_stop)
                if first_hit is not None
                else False
                if first_stop is not None or mature
                else None
            )
            touches[str(level)] = {
                "hit": (
                    bool(len(indices)) if mature else True if len(indices) else None
                )
                if valid
                else None,
                "touch_close_ms": int(future[indices[0], 4])
                if valid and len(indices)
                else None,
                "mae_through_touch_bar_pct": float(
                    max(0, adverse[: first_hit + 1].max())
                )
                if valid and first_hit is not None
                else None,
                "target_before_1pct_stop": first_target,
                "same_bar_target_stop_ambiguous": ambiguous if valid else None,
            }
        result["sides"][side] = {
            "touches": touches,
            "mfe_pct": float(max(0, favorable.max()))
            if valid and len(future)
            else None,
            "mae_pct": float(max(0, adverse.max())) if valid and len(future) else None,
            "close_net_bps": (
                (future[-1, 3] / price - 1) * (1 if side == "LONG" else -1) * 10_000
                - setup["cost_bps"]
            )
            if valid and mature and len(future)
            else None,
        }
    return result


def attach_labels(setups, directory, cutoff):
    cache = {}
    result = []
    for setup in setups:
        symbol = setup["symbol"]
        if symbol not in cache:
            path = directory / f"{symbol}.json.gz"
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    cache[symbol] = np.asarray(
                        json.load(handle)["candles"], dtype=float
                    ).reshape(-1, 5)
            else:
                cache[symbol] = np.empty((0, 5))
        source = cache[symbol][cache[symbol][:, 4] == setup["time_ms"]]
        reconstructed = {
            **setup,
            "reference_price": float(source[0, 3]) if len(source) == 1 else None,
            "reference_source": "historical_binance_signal_candle_close",
        }
        result.append(
            {
                **reconstructed,
                "outcome": label_setup(reconstructed, cache[symbol], cutoff),
            }
        )
    return result


def split_indices(rows, test_start):
    """24h outcome purges on both sides of the five-day calibration block."""
    cal_start = test_start - 6 * DAY
    train = [
        i
        for i, r in enumerate(rows)
        if r["time_ms"] + DAY < cal_start
        and r["outcome"]["mature"]
        and r["outcome"]["complete"]
    ]
    cal = [
        i
        for i, r in enumerate(rows)
        if cal_start <= r["time_ms"]
        and r["time_ms"] + DAY < test_start
        and r["outcome"]["mature"]
        and r["outcome"]["complete"]
    ]
    return train, cal


def preprocess_fit(x):
    median = np.array(
        [np.nanmedian(col) if np.any(np.isfinite(col)) else 0 for col in x.T]
    )
    filled = np.where(np.isfinite(x), x, median)
    # Train-only imputation/scaling; missingness is explicit, not hindsight-filled.
    design = np.column_stack([filled, ~np.isfinite(x)])
    return median, Standardizer.fit(design)


def transform(x, median, scaler):
    return scaler.transform(
        np.column_stack([np.where(np.isfinite(x), x, median), ~np.isfinite(x)])
    )


def predict_block(rows, train, cal, test):
    if len(train) < 150 or len(cal) < 60:
        return [], {
            "status": "insufficient_history",
            "train": len(train),
            "calibration": len(cal),
        }
    x = np.asarray([r["features"] for r in rows], dtype=float)
    median, scaler = preprocess_fit(x[train])
    xt, xc, xv = [transform(x[index], median, scaler) for index in (train, cal, test)]
    raw_probabilities = {side: {} for side in SIDES}
    baselines = {side: {} for side in SIDES}
    coefficients = {}
    statuses = {}
    for side in SIDES:
        for level in LEVELS:
            key = str(level)
            yt = np.array(
                [
                    rows[i]["outcome"]["sides"][side]["touches"][key]["hit"]
                    for i in train
                ],
                dtype=float,
            )
            yc = np.array(
                [rows[i]["outcome"]["sides"][side]["touches"][key]["hit"] for i in cal],
                dtype=float,
            )
            rate = (float(yc.sum()) + 1) / (len(yc) + 2)
            baselines[side][key] = rate
            if (
                min(yt.sum(), len(yt) - yt.sum()) < 10
                or min(yc.sum(), len(yc) - yc.sum()) < 5
            ):
                raw_probabilities[side][key] = np.full(len(test), rate)
                statuses[f"{side}_{level}"] = "base_rate_only_insufficient_classes"
                continue
            model = fit_logistic(xt, yt, l2=10)
            calibrator = fit_platt(model.logits(xc), yc)
            raw_probabilities[side][key] = calibrator.predict(model.logits(xv))
            statuses[f"{side}_{level}"] = (
                "calibrated"
                if calibrator.slope > 0
                else "base_rate_only_calibration_failed"
            )
            coefficients[f"{side}_{level}"] = {
                name: float(value)
                for name, value in zip(
                    (*FEATURES, *(f"{f}_missing" for f in FEATURES)),
                    model.coefficients,
                    strict=True,
                )
            }
    predictions = []
    for j, index in enumerate(test):
        probs = {side: {} for side in SIDES}
        for side in SIDES:
            ceiling = 1.0
            for level in LEVELS:
                ceiling = min(ceiling, float(raw_probabilities[side][str(level)][j]))
                probs[side][str(level)] = ceiling
        chosen = max(SIDES, key=lambda side: probs[side]["2"])
        other = "SHORT" if chosen == "LONG" else "LONG"
        margin = probs[chosen]["2"] - probs[other]["2"]
        qualified = statuses[f"{chosen}_2"] == "calibrated"
        predictions.append(
            {
                "index": index,
                "symbol": rows[index]["symbol"],
                "time_ms": rows[index]["time_ms"],
                "probabilities": probs,
                "side": chosen,
                "margin": margin,
                "selected": bool(
                    qualified and probs[chosen]["2"] >= THRESHOLD and margin >= MARGIN
                ),
                "base_probabilities": baselines,
                "calibrated_targets": [
                    str(k) for k in LEVELS if statuses[f"{chosen}_{k}"] == "calibrated"
                ],
            }
        )
    return predictions, {
        "status": "fit",
        "train": len(train),
        "calibration": len(cal),
        "models": statuses,
        "coefficients": coefficients,
    }


def rate_summary(values):
    values = [bool(v) for v in values if type(v) is bool]
    n, wins = len(values), sum(values)
    return {
        "n": n,
        "hits": wins,
        "rate": wins / n if n else None,
        "wilson95": list(_wilson_interval(wins, n)) if n else None,
    }


def summarize(predictions, rows):
    resolved = [
        p
        for p in predictions
        if rows[p["index"]]["outcome"]["mature"]
        and rows[p["index"]]["outcome"]["complete"]
    ]
    selected = [p for p in resolved if p["selected"]]
    result = {
        "predicted": len(predictions),
        "resolved": len(resolved),
        "selected": len(selected),
        "coverage": len(selected) / len(resolved) if resolved else None,
        "test_days": len({p["time_ms"] // DAY for p in resolved}),
        "selected_days": len({p["time_ms"] // DAY for p in selected}),
        "targets": {},
    }
    for level in LEVELS:
        key = str(level)

        def labels(group, target=key):
            return [
                rows[p["index"]]["outcome"]["sides"][p["side"]]["touches"][target][
                    "hit"
                ]
                for p in group
            ]

        all_labels, selected_labels = labels(resolved), labels(selected)
        target_selected = [
            p
            for p in selected
            if key in p["calibrated_targets"]
            and p["probabilities"][p["side"]][key] >= THRESHOLD
            and p["probabilities"][p["side"]][key]
            - p["probabilities"]["SHORT" if p["side"] == "LONG" else "LONG"][key]
            >= MARGIN
        ]
        prob = np.array([p["probabilities"][p["side"]][key] for p in resolved])
        base = np.array([p["base_probabilities"][p["side"]][key] for p in resolved])
        y = np.asarray(all_labels, dtype=float)
        result["targets"][key] = {
            "all_model_direction": rate_summary(all_labels),
            "selected": rate_summary(selected_labels),
            "target_specific_selected": rate_summary(labels(target_selected)),
            "false_alerts": len(selected_labels) - sum(selected_labels),
            "missed_hits": sum(all_labels) - sum(selected_labels),
            "recall": sum(selected_labels) / sum(all_labels)
            if sum(all_labels)
            else None,
            "brier": float(np.mean((prob - y) ** 2)) if len(y) else None,
            "baseline_brier": float(np.mean((base - y) ** 2)) if len(y) else None,
            "selected_target_before_1pct_stop": rate_summary(
                [
                    rows[p["index"]]["outcome"]["sides"][p["side"]]["touches"][key][
                        "target_before_1pct_stop"
                    ]
                    for p in selected
                ]
            ),
            "selected_same_bar_ambiguous": sum(
                rows[p["index"]]["outcome"]["sides"][p["side"]]["touches"][key][
                    "same_bar_target_stop_ambiguous"
                ]
                is True
                for p in selected
            ),
            "always_long": rate_summary(
                [
                    rows[p["index"]]["outcome"]["sides"]["LONG"]["touches"][key]["hit"]
                    for p in resolved
                ]
            ),
            "always_short": rate_summary(
                [
                    rows[p["index"]]["outcome"]["sides"]["SHORT"]["touches"][key]["hit"]
                    for p in resolved
                ]
            ),
            "calibration_bins": [
                {
                    "range": [lo, hi],
                    "n": int(((prob >= lo) & (prob < hi)).sum()),
                    "predicted": float(np.mean(prob[(prob >= lo) & (prob < hi)])),
                    "observed": float(np.mean(y[(prob >= lo) & (prob < hi)])),
                }
                for lo, hi in ((0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001))
                if ((prob >= lo) & (prob < hi)).any()
            ],
        }
    result["selected_mean_24h_close_net_bps"] = (
        float(
            np.mean(
                [
                    rows[p["index"]]["outcome"]["sides"][p["side"]]["close_net_bps"]
                    for p in selected
                ]
            )
        )
        if selected
        else None
    )
    result["selected_max_adverse_pct"] = max(
        (rows[p["index"]]["outcome"]["sides"][p["side"]]["mae_pct"] for p in selected),
        default=None,
    )
    # Calendar-day resampling keeps each day's correlated coins together.
    # Descriptive only: consecutive 24h outcomes still overlap between days.
    days = sorted({p["time_ms"] // DAY for p in resolved})
    if len(days) >= 5 and selected:
        rng = np.random.default_rng(20260928)
        by_day = [[p for p in resolved if p["time_ms"] // DAY == day] for day in days]
        lifts = []
        for _ in range(1000):
            sample = [
                p for k in rng.integers(0, len(days), size=len(days)) for p in by_day[k]
            ]
            chosen = [p for p in sample if p["selected"]]
            if not chosen:
                continue

            def hit(p):
                return rows[p["index"]]["outcome"]["sides"][p["side"]]["touches"]["2"][
                    "hit"
                ]

            lifts.append(
                float(
                    np.mean([hit(p) for p in chosen])
                    - np.mean([hit(p) for p in sample])
                )
            )
        result["descriptive_daily_bootstrap_2pct_lift_ci95"] = (
            np.quantile(lifts, [0.025, 0.975]).tolist() if lifts else None
        )
    return result


def run_analysis(rows, cutoff):
    predictions, folds = [], []
    # Fixed time blocks, not random rows. Hold the requested 17 out entirely.
    first_day = min(r["time_ms"] for r in rows) // DAY * DAY
    for day in range(first_day + 12 * DAY, COHORT_START, DAY):
        test = [
            i
            for i, r in enumerate(rows)
            if day <= r["time_ms"] < min(day + DAY, COHORT_START)
        ]
        if not test:
            continue
        train, cal = split_indices(rows, day)
        block, info = predict_block(rows, train, cal, test)
        predictions.extend(block)
        folds.append({"test_start": utc(day), **info})
    cohort = [
        i for i, r in enumerate(rows) if COHORT_START <= r["time_ms"] < COHORT_END
    ]
    train, cal = split_indices(rows, COHORT_START)
    cohort_predictions, cohort_model = predict_block(rows, train, cal, cohort)
    by_index = {p["index"]: p for p in cohort_predictions}
    return {
        "protocol": {
            "horizon_hours": 24,
            "threshold": THRESHOLD,
            "direction_margin": MARGIN,
            "l2": 10,
            "purge_hours": 24,
            "calibration_days": 5,
            "minimum_train": 150,
            "minimum_calibration": 60,
            "features": FEATURES,
            "no_live_changes": True,
            "label_cutoff_utc": utc(cutoff),
            "cohort_start": utc(COHORT_START),
            "cohort_end": utc(COHORT_END),
        },
        "data": {
            "setups": len(rows),
            "symbols": len({r["symbol"] for r in rows}),
            "mature_complete": sum(
                r["outcome"]["mature"] and r["outcome"]["complete"] for r in rows
            ),
            "incomplete": sum(not r["outcome"]["complete"] for r in rows),
            "pending": sum(not r["outcome"]["mature"] for r in rows),
            "policies": dict(Counter(r["policy"] for r in rows)),
        },
        "walk_forward": summarize(predictions, rows),
        "by_regime": {
            regime: summarize(
                [p for p in predictions if rows[p["index"]]["regime"] == regime], rows
            )
            for regime in sorted({r["regime"] for r in rows})
        },
        "by_policy": {
            policy: summarize(
                [p for p in predictions if rows[p["index"]]["policy"] == policy], rows
            )
            for policy in sorted({r["policy"] for r in rows})
        },
        "folds": folds,
        "out_of_sample_predictions": predictions,
        "cohort_model": cohort_model,
        "cohort_summary": summarize(cohort_predictions, rows),
        "cohort": [{**rows[i], "prediction": by_index.get(i)} for i in cohort],
    }


def markdown_report(result):
    def percent(value):
        return f"%{100 * value:.1f}" if value is not None else "—"

    def rate(value):
        interval = value["wilson95"]
        ci = f"; GA {percent(interval[0])}–{percent(interval[1])}" if interval else ""
        return f"{value['hits']}/{value['n']} · {percent(value['rate'])}{ci}"

    def number(value, digits=2):
        return f"{value:.{digits}f}" if value is not None else "—"

    wf = result["walk_forward"]
    lift_ci = wf.get("descriptive_daily_bootstrap_2pct_lift_ci95")
    lift_text = (
        f"{100 * lift_ci[0]:+.2f} ile {100 * lift_ci[1]:+.2f} yüzde puanı"
        if lift_ci is not None
        else "hesaplanamadı"
    )
    lines = [
        "# Trade3 — hedef dokunuşu lojistik regresyon araştırması",
        "",
        f"Veri kesimi: {result['protocol']['label_cutoff_utc']} (UTC). Canlı bot ve Telegram filtreleri değiştirilmedi.",
        "",
        "## Kısa sonuç",
        "",
        f"Birincil %2 filtresi {wf['targets']['2']['all_model_direction']['hits']}/{wf['resolved']} ({percent(wf['targets']['2']['all_model_direction']['rate'])}) isabeti {wf['targets']['2']['selected']['hits']}/{wf['selected']} ({percent(wf['targets']['2']['selected']['rate'])}) yaptı. Bu aday model canlıya hazır değildir; yalnız kazananları önceden ayırdığı gösterilemedi.",
        f"17 örneğin tamamlanmış ve eksiksiz sayısı: {result['cohort_summary']['resolved']}. Model {result['cohort_summary']['selected']} örneği seçti; %2 hedefi bunların {result['cohort_summary']['targets']['2']['selected']['hits']} tanesinde gerçekleşti. Bu seçim de başarılıların tamamını korumadı.",
        "",
        "## Yöntem",
        "",
        f"- {result['data']['setups']} ayrı coin/mum kurulumu; {result['data']['symbols']} coin. Aileler ve üç ileri ufuk ayrı örnek sayılmadı.",
        "- LONG ve SHORT için %2/%3/%5, sinyalin kapanış fiyatından sonraki 24 saatte kapalı 5m mum high/low verisiyle ölçüldü. Sinyal mumu hariçtir.",
        "- Eski aile sonuç defterindeki entry_price sonraki mumun açılışıdır. Bu çalışmada referans, sinyal anının Binance mum kapanışından yeniden oluşturuldu; entry_price girdi olarak kullanılmadı. Daha önce yapılmış yalnız fiyatı eşleşen alt-küme denemesi geçersizdir, burada raporlanmaz.",
        "- Sadece sinyalde mevcut skor, aile, rejim, genişlik, spread, funding, hacim, volatilite ve taker oranı kullanıldı. Gelecekteki getiri/çıkış fiyatı girdilere alınmadı.",
        "- L2=10 lojistik regresyon; eğitimde doldurma/ölçekleme; ayrı 5 günlük Platt kalibrasyonu. Eğitim→kalibrasyon ve kalibrasyon→test arasında 24 saatlik sonuç penceresi dışarıda bırakıldı. Testler günlük ve ileri zaman sıralıdır; rastgele bölme yapılmadı.",
        "- Denenmeden sabitlenen filtre: P(%2) ≥ %70, karşı yöne fark ≥ 10 yüzde puanı. Yön, %2 olasılığından seçilir; %3/%5 için sonradan yön değiştirilmez. Bu eşik test/17 örneğin başarılarına göre optimize edilmedi.",
        "- 17 örnek tüm model seçiminden ayrıldı; kohort modeli ilk örnekten önce sonuçlanmış geçmişle donduruldu. Ufku dolmamış negatifler başarısız sayılmadı, modelin eğitimine alınmadı.",
        "",
        "## Daha önce görülmemiş günlerde sonuç",
        "",
        f"OOS: {wf['resolved']} tamamlanmış kurulum / {wf['test_days']} takvim günü. Seçilen: {wf['selected']} / {wf['selected_days']} gün; kapsama {percent(wf['coverage'])}.",
        "",
        "| Hedef | Filtresiz, modelin seçtiği yön | Filtreli | Yanlış uyarı | Kaçırılan dokunuş | Yakalama |",
        "|---|---|---|---:|---:|---|",
    ]
    for key, metric in wf["targets"].items():
        lines.append(
            f"| %{key} | {rate(metric['all_model_direction'])} | {rate(metric['selected'])} | {metric['false_alerts']} | {metric['missed_hits']} | {percent(metric['recall'])} |"
        )
    lines.extend(
        [
            "",
            "GA: örnek düzeyinde %95 Wilson aralığı; coin/gün bağımlılığını düzeltmez.",
            "",
            "| Hedef | Brier (düşük iyi) | Sabit geçmiş olasılığı Brier | Hep LONG dokunuş | Hep SHORT dokunuş | Filtrelilerde hedeften önce %1 stopa değmeyen |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for key, metric in wf["targets"].items():
        lines.append(
            f"| %{key} | {number(metric['brier'], 4)} | {number(metric['baseline_brier'], 4)} | {rate(metric['always_long'])} | {rate(metric['always_short'])} | {rate(metric['selected_target_before_1pct_stop'])}; aynı mum belirsiz {metric['selected_same_bar_ambiguous']} |"
        )
    lines.extend(
        [
            "",
            "Her hedefe ayrıca aynı %70 olasılık ve 10 puan yön farkı şartı uygulanırsa (eşik araması yapılmadı):",
            "",
            "| Hedef | Geniş OOS test | Ayrı 17 örnek |",
            "|---|---|---|",
        ]
    )
    for key in ("2", "3", "5"):
        lines.append(
            f"| %{key} | {rate(wf['targets'][key]['target_specific_selected'])} | {rate(result['cohort_summary']['targets'][key]['target_specific_selected'])} |"
        )
    lines.extend(
        [
            "",
            "%1 stop burada yalnız bir duyarlılık kontrolüdür, canlı stop önerisi değildir. Aynı mumda hedef/stop varsa sıra bilinmez ve başarı sayılmaz. Dokunuşun kâr olmadığını göstermek için raporlanır.",
            f"Seçilenlerin 24 saat sonunda kapanıştan çıkış varsayımıyla ortalama maliyet sonrası hareketi: {number(wf['selected_mean_24h_close_net_bps'])} bps. Bu, hedefte satış simülasyonu değildir ve gerçek emir dolumu/funding ödemelerini içermez.",
            f"%2 isabet artışı için gün kümeli betimsel bootstrap %95 aralığı: {lift_text}. Günler arası 24h örtüşmesi nedeniyle kesin istatistiksel kanıt sayılmaz.",
            "",
            "## Ayrı tutulan 17 kurulum",
            "",
            "Aşağıdaki artı/eksi hareketler iki olası tarafın sonradan gözlenen uçlarıdır; iki yön birden önceden tahmin edilmiş gibi sayılmaz. Yalnız 24 saatini doldurmayan satırlar varsa bunların sonucu geçicidir.",
            "",
            "| UTC zaman | Coin | Rejim | Yukarı uç | Aşağı uç | LONG dokunan % | SHORT dokunan % | Model yönü / P2 / P3 / P5 | Uyarı seçimi | 24h doldu |",
            "|---|---|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in result["cohort"]:
        outcome, prediction = row["outcome"], row["prediction"]
        long, short = outcome["sides"]["LONG"], outcome["sides"]["SHORT"]
        touches = [
            ", ".join(key for key, v in side["touches"].items() if v["hit"] is True)
            or ("yok" if outcome["mature"] else "henüz yok")
            for side in (long, short)
        ]
        if prediction:
            chosen = prediction["side"]
            model = (
                chosen
                + " / "
                + " / ".join(
                    percent(prediction["probabilities"][chosen][str(k)]) for k in LEVELS
                )
            )
        else:
            model = "veri yetersiz"
        lines.append(
            f"| {row['time_utc']} | {row['symbol']} | {row['regime']} | {number(long['mfe_pct'])}% | {number(short['mfe_pct'])}% | {touches[0]} | {touches[1]} | {model} | {'EVET' if prediction and prediction['selected'] else 'HAYIR'} | {'Evet' if outcome['mature'] else 'Hayır'} |"
        )
    lines.extend(
        [
            "",
            "## Sınırlar ve karar",
            "",
            "- Bu geriye dönük araştırmadır; gerçek zamanlı ileri-test başarısı değildir. Geçmişteki dedektör/politika değişiklikleri, coinler arası korelasyon ve mevcut evren seçimi sonuçları etkiler. Rejim/politika alt grupları JSON raporunda ayrıdır.",
            "- %100 kazanan filtresi bulunmuş değildir. Dokunma olasılığı işlem kârı olasılığı değildir. Çok sıkı filtre başarısızların yanında başarılı adayları da susturur.",
            "- Tablodaki P2/P3/P5 model tahminleridir, güvenilirliği kanıtlanmış canlı olasılıklar değildir. Bu örnekleri artık gördüğümüz için sonraki modelin nihai doğrulaması yeni, bağımsız tarihlerde yapılmalıdır.",
            f"- Canlı aktivasyon yok. Önce farklı dönemlerde ve gerçek zamanlı sessiz testte doğrulama gerekir; 17 örnek tek başına yeterli değildir. Geniş testin TRANSITION örnek sayısı yalnız {result['by_regime'].get('TRANSITION', {}).get('resolved', 0)}; rejime özgü güvenilirlik iddiası yapılamaz.",
            "",
            "## Tekrar üretme",
            "",
            f"`python research/target_touch_study.py artifacts/research/target-touch-20260928 --cutoff {result['protocol']['label_cutoff_utc']}`",
            "",
            f"Gözlem kaydı SHA256: `{result['data']['ledger_sha256']}`. Fiyat verileri çalışma klasöründedir; `--fetch` yalnız yerel araştırma dosyalarını günceller, bot verisini değiştirmez.",
            "",
            "Yöntem kaynakları: [zaman sıralı ayrım](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [olasılık kalibrasyonu](https://scikit-learn.org/stable/modules/calibration.html).",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--cutoff", help="Freeze UTC cutoff, e.g. 2026-09-28T17:20:00+00:00"
    )
    args = parser.parse_args()
    root = args.directory
    setups, malformed = load_setups(root / "ledger.jsonl")
    cutoff = (
        int(
            (
                datetime.fromisoformat(args.cutoff)
                if args.cutoff
                else datetime.now(UTC)
            ).timestamp()
            * 1000
        )
        // STEP
        * STEP
    )
    print(
        "SETUPS",
        len(setups),
        "COHORT",
        sum(COHORT_START <= r["time_ms"] < COHORT_END for r in setups),
        "CUTOFF",
        utc(cutoff),
        flush=True,
    )
    failures = fetch_prices(setups, root / "prices", cutoff) if args.fetch else {}
    rows = attach_labels(setups, root / "prices", cutoff)
    result = run_analysis(rows, cutoff)
    result["data"]["malformed_ledger_lines"] = malformed
    result["data"]["ledger_sha256"] = hashlib.sha256(
        (root / "ledger.jsonl").read_bytes()
    ).hexdigest()
    result["fetch_failures"] = failures
    result["data"]["study_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    (root / "study.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (root / "labeled-setups.json").write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (root / "report.md").write_text(markdown_report(result), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("data", "walk_forward", "cohort_summary")},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
