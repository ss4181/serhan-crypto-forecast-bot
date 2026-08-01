from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from .config import INTERVALS, SYMBOLS, Settings, cache_path, model_path
from .data import load_cache
from .features import build_supervised_dataset
from .model import BacktestMetrics, fit_final_bundle, save_bundle, walk_forward_backtest


Progress = Callable[[str], None]


def research_one(
    settings: Settings,
    symbol: str,
    interval: str,
) -> BacktestMetrics:
    bars = load_cache(cache_path(settings.data_dir, symbol, interval))
    dataset = build_supervised_dataset(bars)
    metrics = walk_forward_backtest(
        dataset,
        signal_threshold=settings.signal_threshold,
        minimum_signal_count=settings.minimum_signal_count,
        minimum_signal_accuracy=settings.minimum_signal_accuracy,
        maximum_ece=settings.maximum_ece,
        round_trip_cost_bps=settings.round_trip_cost_bps,
        minimum_net_edge_bps=settings.minimum_net_edge_bps,
    )
    bundle = fit_final_bundle(
        dataset,
        symbol=symbol,
        interval=interval,
        backtest=metrics,
    )
    save_bundle(bundle, model_path(settings.model_dir, symbol, interval))
    return metrics


def research_all(
    settings: Settings,
    *,
    progress: Progress | None = None,
) -> dict[str, BacktestMetrics]:
    output = progress or (lambda _message: None)
    results: dict[str, BacktestMetrics] = {}
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            key = f"{symbol}_{interval}"
            output(f"{key}: kronolojik backtest basladi")
            results[key] = research_one(settings, symbol, interval)
            output(f"{key}: model ve metrikler kaydedildi")
    write_research_report(settings.report_dir, results)
    return results


def write_research_report(
    report_dir: Path, results: dict[str, BacktestMetrics]
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    json_path = report_dir / "latest_backtest.json"
    markdown_path = report_dir / "latest_backtest.md"
    payload = {
        "schema": "btc-eth-walk-forward-report-v1",
        "generated_at_utc": generated_at,
        "method": {
            "split": "expanding chronological train; later calibration; one-label embargo; later test",
            "target": "next closed candle close direction",
            "signal_threshold": next(iter(results.values())).signal_threshold if results else None,
            "gate": "net edge after round-trip cost must beat zero, day-block bootstrapped",
            "round_trip_cost_bps": (
                next(iter(results.values())).round_trip_cost_bps if results else None
            ),
            "warning": "Historical out-of-sample performance is not a guarantee of future performance.",
        },
        "results": {key: value.to_dict() for key, value in results.items()},
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# BTC/ETH olasilik botu — walk-forward backtest",
        "",
        f"Uretim zamani: `{generated_at}`",
        "",
        "Tum sonuclar kronolojik, modelden sonra gelen kalibrasyon ve test dilimlerinden uretilmistir. "
        "Train/kalibrasyon ve kalibrasyon/test arasinda bir hedef mumu embargo vardir.",
        "",
        "| Model | OOS mum | Tum yon | Taban | Yuksek guven | Aile-duz. %95 GA | Kapsama | "
        "Ort. kazanc/kayip | Brut bps | Net bps | Net %95 GA (gun blok) | ECE | Kapi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, metric in results.items():
        lines.append(
            f"| {key} | {metric.sample_count} | %{metric.accuracy * 100:.2f} | "
            f"%{metric.baseline_accuracy * 100:.2f} | %{metric.signal_accuracy * 100:.2f} "
            f"(n={metric.signal_count}, {metric.signal_days} gun) | "
            f"%{metric.signal_familywise_ci95_low * 100:.2f}–%{metric.signal_familywise_ci95_high * 100:.2f} | "
            f"%{metric.signal_coverage * 100:.2f} | "
            f"{metric.average_win_bps:+.1f} / {metric.average_loss_bps:+.1f} | "
            f"{metric.gross_edge_bps:+.2f} | {metric.net_edge_bps:+.2f} | "
            f"{metric.net_edge_ci95_low:+.2f} – {metric.net_edge_ci95_high:+.2f} | "
            f"%{metric.expected_calibration_error * 100:.2f} | "
            f"{'GECTI' if metric.passed_research_gate else 'KALDI'} |"
        )
    cost = next(iter(results.values())).round_trip_cost_bps if results else 0.0
    lines.extend(
        [
            "",
            "## Kapi",
            "",
            f"Bir modelin bildirim gonderebilmesi icin yon isabeti degil, {cost:.1f} baz puan "
            "gidis-donus maliyeti dusuldukten sonraki **beklentisi** pozitif olmalidir; ayrica bu "
            "beklentinin gun bloklu bootstrap ile hesaplanan %95 alt siniri sifirin ustunde olmalidir. "
            "Yon isabetinin %50 uzerinde olmasi tek basina bir sey ifade etmez: kazanan ve kaybeden "
            "islemlerin buyuklugu farklidir.",
            "",
            "## Uyari",
            "",
            "Bu oranlar yalniz incelenen Binance Spot verisi ve donemi icindir. Kayma ve emir icra "
            "sonucu olculmez; bot emir vermez. Gecmis basari gelecek basariyi garanti etmez.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
