"""Trade3 OI research coverage with explicit publication-lag assumption.

No exchange requests, model training, return-based selection or live writes.
Cached timestamp is not an observed first-available timestamp; lag is a
research assumption, never an assertion of verified historical availability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BAR = 300_000


def align_oi_features(candle_close_ms: np.ndarray, oi: pd.DataFrame, *, lag_bars: int = 1) -> pd.DataFrame:
    if lag_bars < 1:
        raise ValueError("A positive publication lag is required")
    fields = ["timestamp_ms", "open_interest", "open_interest_value"]
    if not set(fields).issubset(oi):
        raise ValueError("OI schema missing required fields")
    if oi.timestamp_ms.duplicated().any():
        raise ValueError("Conflicting/duplicate OI timestamps cannot be silently selected")
    source = oi.sort_values("timestamp_ms").set_index("timestamp_ms")
    source = source.where(np.isfinite(source) & (source > 0))
    # Never carry a stale observation forward or borrow the next one.
    cutoff = (np.asarray(candle_close_ms, dtype=np.int64) // BAR) * BAR - lag_bars * BAR
    windows = np.stack([source.open_interest.reindex(cutoff - j * BAR).to_numpy() for j in range(13)], axis=1)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    change = np.full(len(cutoff), np.nan)
    change[valid] = windows[valid, 0] / windows[valid, -1] - 1
    return pd.DataFrame({"signal_close_ms": candle_close_ms, "oi_source_ms": cutoff,
                         "oi_1h_change": change, "oi_window_complete": valid})


def audit(source: Path, candles: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output exists; refusing overwrite")
    results = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = source / f"{symbol}_open_interest.csv"
        pricepath = candles / f"{symbol}_5m_futures.csv"
        oi, prices = pd.read_csv(path), pd.read_csv(pricepath)
        diffs = oi.sort_values("timestamp_ms").timestamp_ms.diff().dropna()
        regular = oi.loc[oi.timestamp_ms.mod(BAR).eq(0)].sort_values("timestamp_ms")
        grid_diffs = regular.timestamp_ms.diff().dropna()
        coverage = {}
        for lag in (1, 2, 3):
            joined = align_oi_features(prices.close_time_ms.to_numpy(), oi, lag_bars=lag)
            start = int(oi.timestamp_ms.min()) + (12 + lag) * BAR
            in_scope = (prices.close_time_ms >= start) & (prices.close_time_ms <= int(oi.timestamp_ms.max()) + lag * BAR)
            valid = joined.oi_window_complete & in_scope
            coverage[str(lag * 5)] = {"lag_minutes_assumed": lag * 5,
                                     "eligible_price_bars": int(in_scope.sum()),
                                     "complete_oi_windows": int(valid.sum()),
                                     "missing_windows": int((in_scope & ~joined.oi_window_complete).sum()),
                                     "signal_days": len(set((joined.loc[valid, "signal_close_ms"] // 86_400_000).tolist()))}
        results.append({"symbol": symbol, "rows": len(oi), "start_ms": int(oi.timestamp_ms.min()),
                        "end_ms": int(oi.timestamp_ms.max()), "off_grid_timestamps": int(oi.timestamp_ms.mod(BAR).ne(0).sum()),
                        "irregular_step_count_in_mixed_source": int(diffs.ne(BAR).sum()),
                        "grid_rows": len(regular), "grid_gap_count": int(grid_diffs.ne(BAR).sum()),
                        "largest_grid_gap_ms": int(grid_diffs.max()),
                        "invalid_quantity_rows": int((~np.isfinite(oi.open_interest) | (oi.open_interest <= 0)).sum()),
                        "invalid_or_zero_usd_valuation_rows": int((~np.isfinite(oi.open_interest_value) | (oi.open_interest_value <= 0)).sum()),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "price_sha256": hashlib.sha256(pricepath.read_bytes()).hexdigest(),
                        "coverage_by_lag": coverage})
    payload = {"symbols": results, "first_observed_availability_recorded": False,
               "value_used": "contract/base-unit open_interest, not USD valuation (which also changes with price)",
               "forward_fill": False, "historical_revision_provenance_available": False,
               "claim": "Feature-coverage audit only. No edge, model validity or live readiness established."}
    output.mkdir(parents=True)
    (output / "coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Trade3 açık pozisyon verisi kapsamı", "",
             "Fiyat dışı yeni bir araştırma girdisi. Kârlılık/strateji testi değildir.", "",
             "| Coin | OI satırı | 5dk seride boşluk | Varsayılan yayın gecikmesi | Uygun mum | Eksiksiz 1s pencere | Gün |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        for c in r["coverage_by_lag"].values():
            lines.append(f"| {r['symbol']} | {r['rows']} | {r['grid_gap_count']} | {c['lag_minutes_assumed']} dk | {c['eligible_price_bars']} | {c['complete_oi_windows']} | {c['signal_days']} |")
    lines += ["", "## Sınırlar", "",
              "Kayıtlarda ilk görülme/yayın zamanı yok. 5/10/15 dakika gecikme varsayımdır; gerçek geçmiş erişilebilirlik kanıtlanmış değildir.",
              "Dosyalarda 5dk ızgarası dışındaki saniyelik kayıtlar da var. Bunlar düzenli serinin boşluğu olarak sayılmadı ve özellik hesabına alınmadı. Sıfır USD değerleri ayrıca sayıldı; pozitif OI miktarı ile karıştırılmadı.",
              "Eksik 5dk OI noktaları ileri doldurulmaz; bir saatlik pencere eksikse özellik UNKNOWN olur.",
              "USD değerindeki değişim yerine OI miktarı kullanılır; fiyat artışı OI artışı gibi sayılmaz.",
              "Bu iki coin kaydı altcoinlerin kendi OI verisi yerine kullanılamaz; ancak açıkça etiketli piyasa bağlamı olabilir."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--candles", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.source, args.candles, args.output), indent=2))
