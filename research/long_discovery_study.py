"""Exploratory LONG rule replay on hourly data; never deploys or sends alerts.

The present-day universe is selected before this replay. This is a survivor-
biased discovery study, NOT an independent test or a reconstructed old universe.
Live prospective tracking uses 5m bars; this preliminary replay uses 1h bars.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

from crypto_forecaster.config import Settings
from crypto_forecaster.data import load_cache
from crypto_forecaster.long_scout import (
    DAY,
    HOUR,
    LABELS,
    TARGETS,
    bull_history,
    candidate,
    features,
)
from crypto_forecaster.measurement import _wilson_interval


def label(row, frame, cutoff):
    start = row["sourceCloseMs"] + 1
    end = start + 7 * DAY
    future = frame[
        (frame["open_time_ms"] >= start) & (frame["close_time_ms"] < min(end, cutoff))
    ]
    mature = cutoff >= end
    expected = np.arange(start, min(end, cutoff), HOUR)
    complete = len(future) == len(expected) and np.array_equal(
        future["open_time_ms"].to_numpy(), expected
    )
    outcome = {"mature": mature, "complete": complete, "targets": {}}
    if future.empty or future.iloc[0]["open_time_ms"] != start:
        return outcome
    entry = float(future.iloc[0]["open"])
    highs = (future["high"].to_numpy() / entry - 1) * 100
    lows = (1 - future["low"].to_numpy() / entry) * 100
    stop_indices = np.flatnonzero(lows >= row["stopPct"] - 1e-9)
    stop = int(stop_indices[0]) if len(stop_indices) else None
    outcome.update(
        entryPrice=entry,
        mfePct=float(max(0, highs.max())),
        maePct=float(max(0, lows.max())),
        closeNetBps=float((future.iloc[-1]["close"] / entry - 1) * 10000 - 12),
    )
    for level in TARGETS:
        hits = np.flatnonzero(highs >= level - 1e-9)
        hit = int(hits[0]) if len(hits) else None
        outcome["targets"][str(level)] = {
            "hit": (hit is not None if mature else True if hit is not None else None)
            if complete
            else None,
            "first": None
            if not complete or (hit is not None and hit == stop)
            else (stop is None or hit < stop)
            if hit is not None
            else False
            if mature or stop is not None
            else None,
            "netBps": (
                level * 100 - 12
                if hit is not None and (stop is None or hit < stop)
                else -row["stopPct"] * 100 - 12
                if stop is not None
                else outcome["closeNetBps"]
                if mature
                else None
            )
            if complete and not (hit is not None and hit == stop)
            else None,
        }
    return outcome


def metrics(rows):
    resolved = [
        r
        for r in rows
        if r["outcome"]["complete"]
        and r["outcome"]["mature"]
        and r["outcome"].get("entryPrice")
    ]
    targets = {}
    for level in TARGETS:
        values = [r["outcome"]["targets"][str(level)] for r in resolved]
        hits = sum(v["hit"] is True for v in values)
        first = [v["first"] for v in values if type(v["first"]) is bool]
        net = [v["netBps"] for v in values if v.get("netBps") is not None]
        targets[str(level)] = {
            "hits": hits,
            "n": len(values),
            "rate": hits / len(values) if values else None,
            "wilson95": list(_wilson_interval(hits, len(values))) if values else None,
            "firstWins": sum(first),
            "firstN": len(first),
            "ambiguous": len(values) - len(first),
            "meanNetBps": float(np.mean(net)) if net else None,
            "netN": len(net),
        }
    return {
        "candidates": len(rows),
        "resolved": len(resolved),
        "pending": sum(not r["outcome"]["mature"] for r in rows),
        "symbols": len({r["symbol"] for r in resolved}),
        "days": len({r["sourceCloseMs"] // DAY for r in resolved}),
        "mean7dCloseNetBps": float(
            np.mean([r["outcome"]["closeNetBps"] for r in resolved])
        )
        if resolved
        else None,
        "targets": targets,
    }


def run(directory: Path, output: Path):
    universe = json.loads((directory / "universe.json").read_text(encoding="utf-8"))
    cutoff = universe["atMs"] // HOUR * HOUR
    frames = {
        s: load_cache(directory / "prices" / f"{s}_1h.csv")
        for s in universe["contracts"]
        if (directory / "prices" / f"{s}_1h.csv").exists()
    }
    tables = {s: features(f, frames["BTCUSDT"]) for s, f in frames.items()}
    macro = bull_history(tables)
    settings = Settings()
    events = []
    for symbol, table in tables.items():
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            continue
        for _, row in table[
            table["ready"] & table["trend"] & (table["rs7d"] >= 3)
        ].iterrows():
            item = candidate(row, symbol, universe["contracts"][symbol], settings)
            if item:
                item["establishedBull"] = bool(
                    item["sourceCloseMs"] in macro.index
                    and macro.loc[item["sourceCloseMs"], "establishedBull"]
                )
                events.append(item)
    grouped = defaultdict(list)
    for event in events:
        grouped[event["sourceCloseMs"]].append(event)
    arms = {}
    for bull_gate in (True, False):
        records, last_seen = [], {}
        for stamp in sorted(grouped):
            ranked = sorted(
                grouped[stamp],
                key=lambda r: (r["stage"] != "TEYİT", -r["score"], r["symbol"]),
            )
            ranked = [r for r in ranked if not bull_gate or r["establishedBull"]]
            for event in ranked[:5]:
                key = (event["symbol"], event["strategy"])
                if key in last_seen and stamp - last_seen[key] < DAY:
                    continue
                active = sum(r["sourceCloseMs"] + 1 + 7 * DAY > stamp for r in records)
                if active >= settings.long_scout_maximum_active:
                    continue
                record = {
                    **event,
                    "outcome": label(event, frames[event["symbol"]], cutoff),
                }
                records.append(record)
                last_seen[key] = stamp
        arms["bull_confirmed" if bull_gate else "without_bull_gate"] = {
            "overall": metrics(records),
            "byStrategy": {
                s: metrics([r for r in records if r["strategy"] == s]) for s in LABELS
            },
            "recentListing": metrics([r for r in records if r["newListing"]]),
            "records": records,
        }
    # A simple LONG benchmark: first established-BULL hour per coin/UTC day.
    controls, observed = [], set()
    for symbol, table in tables.items():
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            continue
        for stamp, row in table[table["ready"]].iterrows():
            if (
                stamp not in macro.index
                or not macro.loc[stamp, "establishedBull"]
                or row["quote24"] < settings.long_scout_minimum_volume_usdt
            ):
                continue
            key = (symbol, stamp // DAY)
            if key in observed:
                continue
            observed.add(key)
            item = {
                "symbol": symbol,
                "sourceCloseMs": int(stamp),
                "stopPct": min(6.0, max(2.0, 2 * row["atrPct"])),
            }
            controls.append({**item, "outcome": label(item, frames[symbol], cutoff)})
    result = {
        "version": "long-discovery-study-v1",
        "cutoffUtc": datetime.fromtimestamp(cutoff / 1000, UTC).isoformat(),
        "data": {
            "symbols": len(frames),
            "hourlyBars": sum(len(f) for f in frames.values()),
            "establishedBullHours": int(macro["establishedBull"].sum()),
            "scoreThreshold": settings.long_scout_minimum_score,
            "universeSha256": sha256(
                (directory / "universe.json").read_bytes()
            ).hexdigest(),
            "codeSha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "arms": arms,
        "dailyBullLongBenchmark": metrics(controls),
        "limitations": [
            "exploratory_not_independent_validation",
            "present_universe_survivorship_bias",
            "historical_spread_and_funding_unavailable",
            "hourly_entry_and_ambiguous_intrabar_order",
            "correlated_coins_and_days",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "study.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# Trade3 LONG keşif pilotu",
        "",
        f"Veri kesimi: {result['cutoffUtc']}. Bugünkü evrenden {len(frames)} piyasa / {result['data']['hourlyBars']} saatlik mum.",
        "",
        "Bu, sabit kuralların keşif amaçlı geçmiş taramasıdır. Önceki regresyon sonrasında geliştirildi; bağımsız doğrulama değildir. Gerçek zamanlı sessiz takip bundan sonra yeni kayıtlarla yapılır.",
        "",
        "Yerleşik LONG boğa: BTC/ETH EMA50 > EMA200, fiyat EMA50 üstünde, EMA50 eğimi ve 7 günlük getirileri pozitif; saatlik evrende 24 saatlik getiri genişliği ≥ %60. Bu koşullar altı ardışık saat sürer. Scalp 5 dakika rejiminden ayrı bir ufuktur.",
        "",
        "LA1: göreli güç + fiyat sıkışması + hacim; LB1: 7 günlük zirve üstü hacimli kapanış; LP1: yükselen trendde EMA20 geri alımı. Birikim skoru ≥50, teyitli kurulum ≥65/100. Skor olasılık değildir.",
        "",
        "Her saat teyitli adaylara öncelik verilir, en çok 5 aday alınır. Coin/strateji başına 24 saat tekrar aralığı ve 50 açık takip sınırı kullanılır. Giriş sonraki tam saat açılışı, takip 7 gün, stop ATR tabanlı %2–%6. Canlı sessiz takip daha hassas 5 dakika mumlarını kullanır.",
        "",
        f"Geçmiş veri içinde altı saatlik yerleşik boğa koşulunu sağlayan saat: {result['data']['establishedBullHours']}.",
        "",
        "| Kural | Olgun / aday | Coin / UTC gün | %2 dokunuş | %3 dokunuş | %5 dokunuş | %10 dokunuş | %20 dokunuş | %2 stop öncesi | %2 bracket ort. net |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def metric_text(m):
        return (
            f"{m['hits']}/{m['n']} (%{100 * m['rate']:.1f})" if m["n"] else "örnek yok"
        )

    sections = [("Boğa teyitli · tüm", arms["bull_confirmed"]["overall"])]
    sections += [
        (f"Boğa teyitli · {LABELS[s]}", m)
        for s, m in arms["bull_confirmed"]["byStrategy"].items()
    ]
    sections += [
        ("Boğa teyitli · yeni kontratlar", arms["bull_confirmed"]["recentListing"]),
        ("Boğa kapısı olmadan · tüm", arms["without_bull_gate"]["overall"]),
        ("Günlük boğa LONG karşılaştırması", result["dailyBullLongBenchmark"]),
    ]
    for name, m in sections:
        t = m["targets"]["2"]
        values = " | ".join(metric_text(m["targets"][str(k)]) for k in TARGETS)
        lines.append(
            f"| {name} | {m['resolved']}/{m['candidates']} | {m['symbols']} / {m['days']} | {values} | {t['firstWins']}/{t['firstN']} · belirsiz {t['ambiguous']} | {str(round(t['meanNetBps'], 1)) + ' bps' if t['meanNetBps'] is not None else 'örnek yok'} |"
        )
    lines += [
        "",
        "Tam 7 günlük pencere dolmadan oran hesaplanmaz; veri eksiği kayıp diye sayılmaz. Aynı saat mumunda stop/hedef sırası bilinmiyorsa stop öncesi başarı hesabına girmez. Ayrıntılı Wilson %95 aralıkları JSON raporundadır; coin/gün korelasyonunu düzeltmez.",
        "",
        "Günlük LONG karşılaştırması farklı giriş zamanları ve aday sayıları içerir; eşleştirilmiş bir nedensel üstünlük testi değildir. Geçmiş finansman ödemeleri, spread ve kayma verisi bulunmadığı için mevcut anlık veriler geçmişe yapıştırılmadı; kapanış net ölçüsü yalnız sabit 12 bps maliyet varsayımıdır.",
        "",
        "Bugünün hayatta kalan, likit evreni kullanıldı. Delist olmuş ve henüz listelenmemiş coinler eski tarihlerin evreninde yeniden oluşturulmadı. Bu sınır nedeniyle bu oranlar canlı başarı olasılığı diye yayımlanmaz veya otomatik Telegram işlem filtresine çevrilmez.",
        "",
        "## Karar",
        "",
        "Kurallar sürümü sabittir. Sonuçlara göre aynı geçmiş üzerinde eşik araması yapılmadı. LONG stratejisinin katkısı, sonraki yeni tarihlerdeki sessiz takipte hedef/stop ve maliyet verisiyle ayrıca doğrulanacaktır.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "data": result["data"],
                "arms": {k: v["overall"] for k, v in arms.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research/long-discovery-20260930"),
    )
    args = parser.parse_args()
    run(args.directory, args.output)
