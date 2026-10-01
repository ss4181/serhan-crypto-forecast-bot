"""Dated, source-backed fundamental watchlist, never an alert eligibility override."""

from __future__ import annotations

import json
from importlib.resources import files


def load_growth_watchlist() -> dict:
    return json.loads(
        files("crypto_forecaster.resources")
        .joinpath("growth_watchlist_v1.json")
        .read_text(encoding="utf-8")
    )


def format_growth_watchlist() -> str:
    data = load_growth_watchlist()
    lines = [
        f"🔭 TRADE3 • Büyüme izleme listesi\nİnceleme: {data['reviewedAt']} (sabit veri; canlı fiyat değil)",
        "100x tahmini veya alım sinyali DEĞİL. Sıra araştırma önceliğim; başarı olasılığı değildir.",
    ]
    for i, coin in enumerate(data["coins"], 1):
        cap = coin["marketCapM"]
        lines.append(
            f"\n{i}. {coin['symbol']} • {coin['theme']}\n"
            f"Değer ~${cap:.1f}m • sabit dolaşımda 100x ≈${cap / 10:.2f} milyar\n"
            f"Risk: {coin['risk']}"
        )
    lines.extend(
        [
            "\n100x hesabı yalnız fiyat × dolaşımdaki arz matematiğidir, fiyat hedefi DEĞİL. Yeni arz gerekli değeri artırır; FDV bazı kaynaklarda toplam arzı kullanır, maksimum arzı değil.",
            "Tam kayıp riski vardır. Liste likidite/funding/boğa filtrelerini aşmaz; bazı coinler bugün elenebilir. Sembolü göndererek teknik tahmin sorabilirsin.",
            "Kaynaklar ve arz bilgisi: https://github.com/ss4181/serhan-crypto-forecast-bot/blob/main/src/crypto_forecaster/resources/growth_watchlist_v1.json",
        ]
    )
    return "\n".join(lines)
