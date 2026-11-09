# OFF-regime SHORT bracket check

Frozen 2026-11-09, before any bracket outcome for this cohort was computed.
Research only: no live delivery, configuration or server change. Run once;
failed arms are rejected, not re-tuned.

## Cohort

Server scalp ledger, 60-minute rows only (one row per detector event), with
`regime_state == "OFF"` and signal bar close on or after 2026-10-17 00:00 UTC.
Rows are grouped into one setup per coin and bar. A setup needs at least one
`execution_eligible` row. Families in the cohort are whatever the live scanner
recorded in OFF (F1, F2, F3).

## Arms, all SHORT

| Arm | Setup filter |
|---|---|
| ALL | Every eligible setup |
| MULTI | At least two distinct families (mirrors the LONG experiment) |
| F2_CONTINUATION | F2 present |
| F1F3_FADE | F1 or F3 present, F2 absent |

MULTI is reported first because it mirrors `long-2pct-v1`, but all four count
toward the multiple-testing correction.

`BASELINE` shorts every cohort coin at 00:00, 06:00, 12:00 and 18:00 UTC for the
same dates, with no signal. It estimates what falling prices alone gave.

## Outcome

- Entry at the next 5m open, cross-checked against the ledger `entry_price`.
- Target -2%, stop +2%, 24-hour limit, same-bar touch counts as STOP.
- A record still open at 24h exits at that bar's close (TIME_EXIT).
- Missing candles or an incomplete 24h window: excluded and counted.
- Round-trip cost 20 bps, also reported at 30 bps. Funding is not modelled.
- Win rate is TARGET / (TARGET + STOP). Net return includes TIME_EXIT.

## Portfolio limits

Primary result applies live limits: one open position per coin, at most five
signals per UTC day, first come first served, highest detector score first
within a bar. The uncapped one-per-coin result is reported as a sensitivity.

## Gate (primary, per arm)

All required: at least 100 settled trades; at least 15 signal days; win rate at
least 55%; mean net above zero at 20 and 30 bps; and the lower bound of a
daily-block bootstrap interval for mean net at 20 bps above zero, at 98.75%
confidence (95% Bonferroni-corrected for four arms).

Passing justifies only a frozen forward experiment. Beating BASELINE is required
before any edge is claimed: otherwise the result is market direction.

## Limits

One 3.5-week falling market. Coins are the current 89-symbol universe. Ledger
rows were recorded live, but the arms were chosen after earlier studies had been
seen, and the regime label was known to be OFF when this was written.
