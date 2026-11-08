"""Explain rejected frozen-rule events; diagnostic, not a new strategy search."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from intraday_long_bracket_study import make_events
from oi_breakout_study import benchmark_context, oi_confirms
from oi_forward_audit_20261023 import CUT, evaluate_portfolio, outcome
from rolling_payoff_study import HOLD, load_snapshot
from squeeze_replication_20261022 import measure


def classify(r, volume: float) -> str:
    if r is None or not (r.btc_complete and r.eth_complete
                         and np.isfinite(r.btc_return) and np.isfinite(r.eth_return)):
        return 'missing_context'
    if not np.isfinite(volume) or volume < 5_000_000:
        return 'liquidity_below_5m'
    if r.btc_return <= 0 or r.eth_return <= 0:
        return 'benchmark_prices_not_rising'
    if not oi_confirms(float(r.btc_oi), float(r.eth_oi)):
        return 'oi_not_confirmed'
    return 'confirmed_before_capacity'


def touch_diagnostic(frame, i: int, row: dict) -> dict:
    # Keep the same full-window denominator as the frozen resolver. Eventual
    # touches AFTER a stop are not winning trades or strategy returns.
    if row['outcome_status'] != 'SETTLED':
        return {'eventual_touch_pct': None}
    highs = frame.high.iloc[i+1:i+HOLD+1].to_numpy(dtype=float)
    return {'eventual_touch_pct': {str(p): bool(np.any(highs >= row['entry'] * (1+p/100)))
                                   for p in (2, 3, 5)}}


def summarize_events(rows: list[dict]) -> dict:
    settled = [r for r in rows if r['outcome_status'] == 'SETTLED']
    return {'total_events': len(rows), 'status_counts': dict(Counter(r['outcome_status'] for r in rows)),
            'overlapping_event_measurement_not_portfolio': measure(settled),
            'eventual_touch_counts_not_profit': {
                str(p): sum(r['eventual_touch_pct'][str(p)] for r in settled) for p in (2, 3, 5)}}


def run(data: Path, oi: Path, audit: Path, output: Path):
    if output.exists():
        raise ValueError('Output exists')
    old = json.loads(audit.read_text(encoding='utf-8'))
    frames, manifest = load_snapshot(data)
    if manifest != old['source_manifest']:
        raise ValueError('Price snapshot changed')
    for name, expected in old['code_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen code changed')
    for name, expected in old['oi_sha256'].items():
        if hashlib.sha256((oi/name).read_bytes()).hexdigest() != expected:
            raise ValueError('OI snapshot changed')
    context = benchmark_context(frames, oi, 2)
    volumes = {s: f.quote_volume.rolling(288, min_periods=288).sum().to_numpy()
               for s, f in frames.items() if 'quote_volume' in f}
    end = max(int(f.close_time_ms.iloc[-1]) for f in frames.values())
    forward, baseline, confirmed = [], [], []
    for symbol, i in make_events(frames)['BULL_VOLUME_BREAKOUT_12']:
        f = frames[symbol]
        close_ms = int(f.close_time_ms.iloc[i])
        r = context.loc[close_ms] if close_ms in context.index else None
        reason = classify(r, volumes[symbol][i] if symbol in volumes else float('nan'))
        # Preserve the complete history of both predeclared arms for capacity.
        if close_ms < CUT and reason not in ('oi_not_confirmed', 'confirmed_before_capacity'):
            continue
        row = {'symbol': symbol, 'first_filter_reason': reason, **outcome(f, i, end)}
        if close_ms >= CUT:
            row.update(touch_diagnostic(f, i, row))
            forward.append(row)
        if reason in ('oi_not_confirmed', 'confirmed_before_capacity'):
            baseline.append(row)
        if reason == 'confirmed_before_capacity':
            confirmed.append(row)
    counts = dict(Counter(r['first_filter_reason'] for r in forward))
    if counts != old['filter_counts'] or len(forward) != old['raw_forward_events']:
        raise ValueError('Filter reconstruction does not match original audit')
    groups = {reason: summarize_events([r for r in forward if r['first_filter_reason'] == reason])
              for reason in counts}
    result = {'source_audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'forward_audit_code_sha256': hashlib.sha256(Path(__file__).with_name('oi_forward_audit_20261023.py').read_bytes()).hexdigest(),
              'cut': old['cut'], 'snapshot_end': old['snapshot_end'], 'groups': groups,
              'events': forward, 'baseline_portfolio': evaluate_portfolio(baseline),
              'oi_portfolio': evaluate_portfolio(confirmed),
              'independent_validation': False, 'live_activation_allowed': False}
    output.mkdir(parents=True)
    (output/'study.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    labels = {'liquidity_below_5m': 'Hacim yetersiz', 'benchmark_prices_not_rising': 'BTC/ETH yönü uygun değil',
              'oi_not_confirmed': 'OI teyidi yok', 'missing_context': 'Veri eksik',
              'confirmed_before_capacity': 'Teyitli'}
    lines = ['# Elenen adaylar ne yaptı?', '', f"UTC: {old['cut']} — {old['snapshot_end']}", '',
             'Sabit %3 hedef, %1,5 stop, 12 saat. Yeni parametre seçimi yapılmadı.', '',
             '| İlk eleme nedeni | Toplam / sonuçlanan | Stop öncesi %3 | Sonradan %2 / %3 / %5 dokunuş | Net ort. 50 bps |',
             '|---|---:|---:|---:|---:|']
    for reason, g in groups.items():
        m = g['overlapping_event_measurement_not_portfolio']; s = m['at_50bps']
        touches = g['eventual_touch_counts_not_profit']
        lines.append(f"| {labels[reason]} | {g['total_events']} / {s['n']} | {m['target_before_stop_count']} | {touches['2']} / {touches['3']} / {touches['5']} | {s['mean_net_bps']} |")
    lines += ['', 'Olaylar aynı coin/zamanda örtüşebilir. Bu tablo portföy getirisi değildir; dokunuş stop sonrasında da olabilir.', '',
              '## Aynı üç pozisyon sınırıyla, önceden tanımlı iki kol', '',
              '| Kol | Sonuçlanan / gün | Stop öncesi %3 | Net ort. 30 / 50 bps |', '|---|---:|---:|---:|']
    for key, label in [('baseline_portfolio', 'Fiyat/hacim + BTC/ETH yönü, OI şartı yok'), ('oi_portfolio', 'Aynı taban + OI teyidi')]:
        m = result[key]['settled_measurement']; s = m['at_30bps']
        lines.append(f"| {label} | {s['n']} / {s['signal_days']} | {m['target_before_stop_count']} | {s['mean_net_bps']} / {m['at_50bps']['mean_net_bps']} |")
    lines += ['', 'İki günlük kesit başarı kanıtı değildir. En az 100 işlem/15 gün ve pozitif güven aralığı koşulları korunur.',
              'Elenme grupları sıra bağımlıdır; tabloda yalnız ilk elenme nedeni bulunur. Filtrenin nedensel etkisi ölçülmüş değildir.',
              'Eksik/bekleyen sonuçlar kayıp veya sıfır getiri sayılmadı. Veri kapsamı ve OI erişim zamanı sınırlamaları kaynak raporla aynıdır.',
              'Canlı ayarlar ve bildirimler değiştirilmedi.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'oi', 'audit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.oi, a.audit, a.output)
