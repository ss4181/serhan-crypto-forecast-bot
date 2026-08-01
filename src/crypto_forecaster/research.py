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
        "| Model | OOS mum | Tum yon | Taban | Yuksek guven | %95 GA | Aile-duz. %95 GA | Kapsama | Brier / taban | ECE | Kapi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, metric in results.items():
        lines.append(
            f"| {key} | {metric.sample_count} | %{metric.accuracy * 100:.2f} | "
            f"%{metric.baseline_accuracy * 100:.2f} | %{metric.signal_accuracy * 100:.2f} "
            f"(n={metric.signal_count}) | %{metric.signal_ci95_low * 100:.2f}–%{metric.signal_ci95_high * 100:.2f} | "
            f"%{metric.signal_familywise_ci95_low * 100:.2f}–%{metric.signal_familywise_ci95_high * 100:.2f} | "
            f"%{metric.signal_coverage * 100:.2f} | {metric.brier_score:.4f} / "
            f"{metric.baseline_brier_score:.4f} | %{metric.expected_calibration_error * 100:.2f} | "
            f"{'GECTI' if metric.passed_research_gate else 'KALDI'} |"
        )
    lines.extend(
        [
            "",
            "## Uyari",
            "",
            "Bu oranlar yalniz incelenen Binance Spot verisi ve donemi icindir. Komisyon, kayma veya "
            "emir icra sonucu olcmez; bot emir vermez. Gecmis basari gelecek basariyi garanti etmez.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
