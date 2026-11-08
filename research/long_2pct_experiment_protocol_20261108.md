# LONG +2% / -2% owner-only forward experiment (`long-2pct-v1`)

Frozen 2026-11-08, before the first live signal. Chosen by the owner after the
September check on 1,235 setups (LONG +/-2%, 24h: 55.4% win rate, daily-block
95% interval 46.7-62.9, about +1 bps at 20 bps cost). That cohort was almost
entirely BULL and is not evidence of an edge; this experiment exists to test it
on data nobody has seen.

## Signal

At each newly closed 5m bar, all of the following:

- broad-market regime is exactly `BULL`, evaluated by the existing bot rule;
- scan coverage meets `CRYPTO_SCALP_MINIMUM_COVERAGE`, manual kill switch off;
- at least two distinct families from F1, F2, F3, B1, B2, B3 on that coin/bar
  (the families present in the September cohort; B4 was not tested there);
- at least one observation is `KURULUM` and at least one `execution_eligible`.

No direction statistics, score floor or quality percentile are applied.

## Limits

At most 5 signals per UTC day, first come first served; within one bar the
highest detector score goes first, then symbol order. One open record per coin.

## Outcome

- Entry at the next 5m open. Target +2%, stop -2%, 24h limit.
- Target and stop inside the same bar count as STOP.
- A record still open at 24h exits at that bar's close (TIME_EXIT).
- Gaps in candles never produce an outcome; after two days they are
  DATA_MISSING and excluded.
- Fixed round-trip cost 20 bps; 30 bps is also reported.
- Every selected signal counts, whether or not Telegram delivered it.

Win rate is TARGET / (TARGET + STOP). Net return includes TIME_EXIT.
Break-even for this bracket at 20 bps is 55%.

## Decision

Reported by `run.py long-2pct` and the Telegram status, never applied
automatically. Clock starts at the experiment's first enabled scan.

| Verdict | Condition |
|---|---|
| SÜRÜYOR | Fewer than 28 days elapsed |
| VERİ YETERSİZ | 28+ days, but under 100 settled trades or under 15 signal days |
| KAPAT | 56+ days and still under those sample limits |
| BAŞARISIZ | Mean net at 20 bps <= 0, or win rate < 55% |
| GEÇTİ | Mean net > 0 at 30 bps and daily-block 95% lower bound at 20 bps > 0 |
| KARARSIZ | Anything else |

These thresholds are not changed after results are seen. GEÇTİ justifies
continued observation, not a claim of profitability.

## Scope

Owner-only private Telegram messages; opt-in through
`CRYPTO_LONG_2PCT_EXPERIMENT`. No orders. Existing scalp filters, ledgers and
subscriber alerts are unchanged.
