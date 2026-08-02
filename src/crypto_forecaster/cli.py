from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
from typing import Sequence
from urllib.request import HTTPRedirectHandler

from .commands import load_members, owner_id, safe_name, save_members
from .config import INTERVALS, SYMBOLS, Settings
from .data import BinanceMarketDataClient, MarketDataError, update_cache
from .hub import post_snapshot, write_snapshot
from .outcomes import format_scorecard, load_ledger, scorecard, settle_pending
from .research import research_all
from .service import (
    answer_commands,
    dashboard_snapshot,
    deliver_eligible,
    deliver_observation_digest,
    deliver_scorecard,
    evaluate_all,
    format_prediction,
    make_prediction,
    models_need_research,
    record_open_interest,
    serve_forever,
)
from .telegram import (
    CHAT_ID_ENV,
    OWNER_ID_ENV,
    TOKEN_ENV,
    TelegramError,
    TelegramNotifier,
    digest_signal_id,
    is_primary,
)


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

    verify = subparsers.add_parser(
        "verify-models",
        help="Alti modelin de Telegram mesaji uretebildigini dogrula",
    )
    verify.add_argument("--refresh", action="store_true", help="Once veri onbellegini guncelle")
    verify.add_argument("--days", type=_positive_days, default=365)
    verify.add_argument(
        "--send", action="store_true", help="Her model icin kanala bir dogrulama mesaji gonder"
    )

    card = subparsers.add_parser("scorecard", help="Gonderilen sinyallerin gercek sonucunu ozetle")
    card.add_argument("--days", type=int, default=30)
    card.add_argument("--send", action="store_true", help="Karneyi Telegram'a gonder")

    subparsers.add_parser(
        "commands", help="Telegram'dan gelen /durum, /performans gibi sorulari yanitla"
    )

    members = subparsers.add_parser("members", help="Sorgulama yetkisi olan kisileri listele")
    members.add_argument("--add", metavar="KIMLIK", type=int)
    members.add_argument("--name", default="")
    members.add_argument("--remove", metavar="KIMLIK", type=int)
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
            settled = settle_pending(
                settings.outcome_state_dir,
                settings.data_dir,
                round_trip_cost_bps=settings.round_trip_cost_bps,
            )
            if settled:
                hits = sum(1 for row in settled if row["correct"])
                print(f"{len(settled)} sinyal sonuclandi ({hits} yon dogru).")
            added = record_open_interest(settings)
            if added:
                print(f"Acik pozisyon kaydi: {added} yeni satir.")
            if args.force_research or _research_is_due(settings):
                print("Gunluk walk-forward arastirma yenileniyor...")
                research_all(settings, progress=print)
            predictions = evaluate_all(settings)
            deliveries = _deliver_cloud_eligible(settings, predictions)
            for prediction, delivery in deliveries:
                print(f"{prediction.symbol} {prediction.interval}: {delivery.status}")
            _deliver_cloud_digest(settings, predictions)
            _deliver_cloud_scorecard(settings)
            _answer_cloud_commands(settings, predictions)
            snapshot = dashboard_snapshot(predictions)
            write_snapshot(settings.report_dir, snapshot)
            posted = post_snapshot(snapshot)
            print("Panel guncellendi." if posted else "Panel baglantisi tanimli degil; yerel ozet yazildi.")
            return 0
        if args.command == "telegram-test":
            message_id = TelegramNotifier().send_message(
                "BTC/ETH olasilik botu — ucuncu kanal baglanti testi.\n\n"
                "Yalnizca arastirma altyapisidir; yatirim tavsiyesi veya emir degildir."
            )
            print(f"Telegram test mesaji gonderildi (message_id={message_id}).")
            return 0
        if args.command == "verify-models":
            if args.refresh:
                _refresh(settings, days=args.days, symbols=SYMBOLS, intervals=INTERVALS)
            return _verify_models(settings, send=args.send)
        if args.command == "scorecard":
            card = scorecard(load_ledger(settings.outcome_state_dir), days=args.days)
            print(format_scorecard(card))
            if args.send:
                if not _telegram_configured():
                    print("Telegram kanali tanimli degil; karne gonderilmedi.")
                    return 0
                delivery = deliver_scorecard(settings, days=args.days)
                print(f"Telegram karnesi: {delivery.status if delivery else 'GONDERILMEDI'}")
            return 0
        if args.command == "commands":
            if owner_id() is None:
                print(f"{OWNER_ID_ENV} tanimli degil; komut yanitlama kapali.", file=sys.stderr)
                return 2
            outcome = answer_commands(settings, evaluate_all(settings))
            print(
                f"{outcome.received} guncelleme okundu, {outcome.answered} yanit gonderildi, "
                f"{outcome.refused} yetkisiz istek yok sayildi."
            )
            return 0
        if args.command == "members":
            return _manage_members(settings, add=args.add, name=args.name, remove=args.remove)
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
                warn=print,
            )
            print(f"{symbol} {interval}: {len(frame)} mum hazir")


def _verify_models(settings: Settings, *, send: bool) -> int:
    """Prove every symbol/interval can build — and optionally deliver — a message.

    Delivery is gated on the research result, so a silent channel is ambiguous:
    it can mean "no model qualified" or "the pipeline is broken".  This checks
    the pipeline for all six models independently of the research gate.
    """
    notifier = TelegramNotifier() if send else None
    bucket = int(datetime.now(timezone.utc).timestamp() * 1000) // 60_000
    failures = 0
    print(f"{'Model':<16}{'Mesaj':>8}{'Tier':>9}{'Net bps':>10}  Durum")
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            key = f"{symbol}_{interval}"
            try:
                prediction = make_prediction(settings, symbol, interval)
                text = format_prediction(prediction)
            except (MarketDataError, OSError, RuntimeError, TypeError, ValueError) as error:
                failures += 1
                print(f"{key:<16}{'HATA':>8}{'-':>9}{'-':>10}  {error}")
                continue
            status = "hazir"
            if notifier is not None:
                delivery = notifier.deliver_once(
                    signal_id=digest_signal_id(f"verify|{key}", bucket),
                    text=text,
                    state_dir=settings.telegram_state_dir,
                )
                status = delivery.status + (f" ({delivery.detail})" if delivery.detail else "")
                if delivery.status == "UNCERTAIN":
                    failures += 1
            print(
                f"{key:<16}{len(text):>8}{prediction.tier:>9}"
                f"{prediction.backtest.net_edge_bps:>10.2f}  {status}"
            )
    if failures:
        print(f"\n{failures} model icin dogrulama basarisiz.", file=sys.stderr)
        return 2
    print("\nAlti modelin de Telegram mesaji uretildi.")
    if not send:
        print("Kanala gondermek icin: python run.py verify-models --refresh --send")
    return 0


def _deliver_cloud_digest(settings: Settings, predictions) -> None:  # type: ignore[no-untyped-def]
    if not _telegram_configured():
        return
    delivery = deliver_observation_digest(settings, predictions)
    if delivery is not None and delivery.status != "DEDUPLICATED":
        detail = f" ({delivery.detail})" if delivery.detail else ""
        print(f"Gozlem raporu: {delivery.status}{detail}")


def _manage_members(settings: Settings, *, add: int | None, name: str, remove: int | None) -> int:
    state_dir = settings.telegram_state_dir
    members = load_members(state_dir)
    if add is not None and remove is not None:
        print("Ayni anda hem ekleme hem silme yapilamaz.", file=sys.stderr)
        return 2
    if add is not None:
        if add <= 0:
            print("Gecersiz Telegram kimligi.", file=sys.stderr)
            return 2
        members[add] = safe_name(name)
        save_members(state_dir, members)
        print(f"{members[add]} ({add}) eklendi.")
    if remove is not None:
        if members.pop(remove, None) is None:
            print(f"{remove} listede yok.", file=sys.stderr)
            return 2
        save_members(state_dir, members)
        print(f"{remove} listeden cikarildi.")
    owner = owner_id()
    print(f"\nSahip: {owner if owner is not None else f'{OWNER_ID_ENV} tanimli degil'}")
    if not members:
        print("Baska yetkili kisi yok.")
    for identifier in sorted(members):
        print(f"• {members[identifier]} — {identifier}")
    return 0


def _answer_cloud_commands(settings: Settings, predictions) -> None:  # type: ignore[no-untyped-def]
    if not _telegram_configured() or owner_id() is None:
        return
    outcome = answer_commands(settings, predictions)
    if outcome.received:
        print(
            f"Komut: {outcome.received} guncelleme, {outcome.answered} yanit, "
            f"{outcome.refused} yetkisiz"
        )


def _deliver_cloud_scorecard(settings: Settings) -> None:
    """Once a UTC day, publish what actually happened to the sent signals."""
    if not _telegram_configured():
        return
    delivery = deliver_scorecard(settings)
    if delivery is not None and delivery.status != "DEDUPLICATED":
        print(f"Canli karne: {delivery.status}")


def _telegram_configured() -> bool:
    return (
        is_primary()
        and bool(os.environ.get(TOKEN_ENV, "").strip())
        and bool(os.environ.get(CHAT_ID_ENV, "").strip())
    )


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
    if models_need_research(settings):
        return True
    report = settings.report_dir / "latest_backtest.json"
    if not report.exists():
        return True
    age_seconds = datetime.now(timezone.utc).timestamp() - report.stat().st_mtime
    return age_seconds >= 20 * 60 * 60



def _deliver_cloud_eligible(settings: Settings, predictions):  # type: ignore[no-untyped-def]
    token_configured = bool(os.environ.get(TOKEN_ENV, "").strip())
    chat_configured = bool(os.environ.get(CHAT_ID_ENV, "").strip())
    if token_configured != chat_configured:
        raise TelegramError("Telegram bulut ayarlari eksik; bot token ve kanal kimligi birlikte tanimlanmali")
    if not token_configured:
        if any(prediction.eligible for prediction in predictions):
            print("Telegram kanali henuz bagli degil; uygun sinyal panele yazildi ancak mesaj gonderilmedi.")
        return []
    if not is_primary():
        print("Bu kosu yedek (standby) rolde; arastirma ve panel guncellendi, mesaj gonderilmedi.")
        return []
    return deliver_eligible(settings, predictions)



def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


__all__ = ["build_parser", "main"]
