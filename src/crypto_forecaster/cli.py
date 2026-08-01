from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import INTERVALS, SYMBOLS, Settings, model_path
from .data import BinanceMarketDataClient, MarketDataError, update_cache
from .research import research_all
from .service import dashboard_snapshot, deliver_eligible, evaluate_all, format_prediction, serve_forever
from .telegram import CHAT_ID_ENV, TOKEN_ENV, TelegramError, TelegramNotifier


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-forecast",
        description="BTC/ETH 5m, 15m ve 1h kalibre olasilik arastirma botu",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Kapali Binance Spot mumlarini indir/guncelle")
    download.add_argument("--days", type=_positive_days, default=365)
    download.add_argument("--symbol", choices=SYMBOLS)
    download.add_argument("--interval", choices=INTERVALS)

    research = subparsers.add_parser(
        "research", help="Veriyi guncelle, walk-forward backtest yap ve modelleri yaz"
    )
    research.add_argument("--days", type=_positive_days, default=365)
    research.add_argument(
        "--offline", action="store_true", help="Ag cagirmadan mevcut CSV onbellegini kullan"
    )

    predict = subparsers.add_parser("predict", help="Son kapali mum icin tum tahminleri yaz")
    predict.add_argument("--refresh", action="store_true", help="Once veri onbellegini guncelle")
    predict.add_argument("--days", type=_positive_days, default=365)
    predict.add_argument("--send", action="store_true", help="Uygun sinyalleri Telegram'a gonder")

    serve = subparsers.add_parser("serve", help="Surekli veri yenile, gunluk arastir ve bildir")
    serve.add_argument("--days", type=_positive_days, default=365)
    serve.add_argument("--poll-seconds", type=int, default=60)

    cloud = subparsers.add_parser(
        "cloud-run", help="Bulutta veriyi yenile, gerekirse modeli arastir, bildir ve paneli guncelle"
    )
    cloud.add_argument("--days", type=_positive_days, default=365)
    cloud.add_argument("--force-research", action="store_true")

    subparsers.add_parser("telegram-test", help="Ucuncu Telegram kanalina sabit test mesaji gonder")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    settings = Settings()
    try:
        if args.command == "download":
            symbols = (args.symbol,) if args.symbol else SYMBOLS
            intervals = (args.interval,) if args.interval else INTERVALS
            _refresh(settings, days=args.days, symbols=symbols, intervals=intervals)
            return 0
        if args.command == "research":
            if not args.offline:
                _refresh(settings, days=args.days, symbols=SYMBOLS, intervals=INTERVALS)
            results = research_all(settings, progress=print)
            _print_metrics(results)
            return 0
        if args.command == "predict":
            if args.refresh:
                _refresh(settings, days=args.days, symbols=SYMBOLS, intervals=INTERVALS)
            predictions = evaluate_all(settings)
            for prediction in predictions:
                print(format_prediction(prediction))
                print("\n" + "=" * 72 + "\n")
            if args.send:
                deliveries = deliver_eligible(settings, predictions)
                if not deliveries:
                    print("Arastirma kapisini ve olasilik esigini gecen sinyal yok; mesaj gonderilmedi.")
                for prediction, delivery in deliveries:
                    print(f"{prediction.symbol} {prediction.interval}: {delivery.status}")
            return 0
        if args.command == "serve":
            serve_forever(
                settings,
                days=args.days,
                poll_seconds=args.poll_seconds,
                progress=print,
            )
            return 0
        if args.command == "cloud-run":
            _refresh(settings, days=args.days, symbols=SYMBOLS, intervals=INTERVALS)
            if args.force_research or _research_is_due(settings):
                print("Gunluk walk-forward arastirma yenileniyor...")
                research_all(settings, progress=print)
            predictions = evaluate_all(settings)
            deliveries = _deliver_cloud_eligible(settings, predictions)
            for prediction, delivery in deliveries:
                print(f"{prediction.symbol} {prediction.interval}: {delivery.status}")
            snapshot = dashboard_snapshot(predictions)
            _write_cloud_snapshot(settings, snapshot)
            posted = _post_cloud_snapshot(snapshot)
            print("Panel guncellendi." if posted else "Panel baglantisi tanimli degil; yerel ozet yazildi.")
            return 0
        if args.command == "telegram-test":
            message_id = TelegramNotifier().send_message(
                "BTC/ETH olasilik botu — ucuncu kanal baglanti testi.\n\n"
                "Yalnizca arastirma altyapisidir; yatirim tavsiyesi veya emir degildir."
            )
            print(f"Telegram test mesaji gonderildi (message_id={message_id}).")
            return 0
        raise RuntimeError("Bilinmeyen komut")
    except KeyboardInterrupt:
        print("Kullanici tarafindan durduruldu.", file=sys.stderr)
        return 130
    except (MarketDataError, TelegramError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Hata: {error}", file=sys.stderr)
        return 2


def _refresh(
    settings: Settings,
    *,
    days: int,
    symbols: Sequence[str],
    intervals: Sequence[str],
) -> None:
    client = BinanceMarketDataClient()
    for symbol in symbols:
        for interval in intervals:
            print(f"{symbol} {interval}: kapali mumlar guncelleniyor...")
            frame = update_cache(
                settings.data_dir,
                symbol,
                interval,
                days=days,
                client=client,
            )
            print(f"{symbol} {interval}: {len(frame)} mum hazir")


def _print_metrics(results):  # type: ignore[no-untyped-def]
    print("\nWalk-forward OOS ozet")
    for key, metric in results.items():
        gate = "GECTI" if metric.passed_research_gate else "KALDI"
        print(
            f"{key}: sinyal %{metric.signal_accuracy * 100:.2f} "
            f"(n={metric.signal_count}, %95 GA %{metric.signal_ci95_low * 100:.2f}–"
            f"%{metric.signal_ci95_high * 100:.2f}), tum yon %{metric.accuracy * 100:.2f}, "
            f"taban %{metric.baseline_accuracy * 100:.2f}, kapi={gate}"
        )


def _positive_days(value: str) -> int:
    parsed = int(value)
    if not 30 <= parsed <= 1500:
        raise argparse.ArgumentTypeError("Gun sayisi 30 ile 1500 arasinda olmali")
    return parsed


def _research_is_due(settings: Settings) -> bool:
    if any(
        not model_path(settings.model_dir, symbol, interval).exists()
        for symbol in SYMBOLS
        for interval in INTERVALS
    ):
        return True
    report = settings.report_dir / "latest_backtest.json"
    if not report.exists():
        return True
    age_seconds = datetime.now(timezone.utc).timestamp() - report.stat().st_mtime
    return age_seconds >= 20 * 60 * 60


def _write_cloud_snapshot(settings: Settings, snapshot: dict[str, object]) -> None:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    path = settings.report_dir / "cloud_snapshot.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _deliver_cloud_eligible(settings: Settings, predictions):  # type: ignore[no-untyped-def]
    token_configured = bool(os.environ.get(TOKEN_ENV, "").strip())
    chat_configured = bool(os.environ.get(CHAT_ID_ENV, "").strip())
    if token_configured != chat_configured:
        raise TelegramError("Telegram bulut ayarlari eksik; bot token ve kanal kimligi birlikte tanimlanmali")
    if not token_configured:
        if any(prediction.eligible for prediction in predictions):
            print("Telegram kanali henuz bagli degil; uygun sinyal panele yazildi ancak mesaj gonderilmedi.")
        return []
    return deliver_eligible(settings, predictions)


def _post_cloud_snapshot(snapshot: dict[str, object]) -> bool:
    endpoint = os.environ.get("PROJECT_HUB_INGEST_URL", "").strip()
    token = os.environ.get("PROJECT_HUB_INGEST_TOKEN", "").strip()
    if not endpoint and not token:
        return False
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("PROJECT_HUB_INGEST_URL gecerli bir HTTPS adresi degil")
    if len(token) < 32:
        raise ValueError("PROJECT_HUB_INGEST_TOKEN tanimli veya yeterince guclu degil")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "btc-eth-probability-bot/0.1",
    }
    bypass = os.environ.get("OAI_SITES_BYPASS_TOKEN", "").strip()
    if bypass:
        headers["OAI-Sites-Authorization"] = f"Bearer {bypass}"
    request = Request(
        endpoint,
        data=json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=20.0) as response:
            status = response.getcode()
            response.read(64 * 1024)
    except Exception:
        raise RuntimeError("Proje paneli guncellenemedi") from None
    if status != 202:
        raise RuntimeError("Proje paneli guncellemeyi kabul etmedi")
    return True


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


__all__ = ["build_parser", "main"]
