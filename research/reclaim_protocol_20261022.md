# Frozen sharp-fall reclaim experiment

Hypothesis: a high-volume downward excursion followed by recovery may have
different payoffs from the previously rejected trend-following families.
OHLCV only; not described as a measured liquidation event.

Freeze before computing signals or outcomes:

- All 99 available Trade3 coin sources via existing source-selection manifest.
- 5m closed bars, LONG only. At the signal close:
  - Previous bar's low is at least 3% below the highest high in the 12 bars
    preceding that previous bar.
  - Previous bar is red; its base volume >=2 times the median of its preceding
    20 bars; previous low <= lowest low of those preceding 12 bars.
  - Current close is above previous high and current open.
  - Last 288 closed bars have >=5 million USDT quote volume.
- No fitted parameter, score sweep, coin ranking, macro-regime override or
  hidden fallback. Signals occur after close; entry at next 5m open.
- Two fixed independent targets: +2%, +3%. Stop -1.5%; maxhold12h; stop-first
  ambiguous bars; next-open gap fills; one position per coin; max3 positions.
- Preserve reporting boundary August 3 2026 20:50:30 UTC, weekly breakdown,
  and October11+ sensitivity tail. These are exploratory, inspected intervals,
  not independent confirmation. No success claim based on choosing a period.
- Same final gates: positive mean at 30/50bps; daily-block 95% lower bound at
  30bps >0; >=100 settled OOS trades; >=15 signal days; >=2 coins.
- Incomplete windows and invalid/missing data are SKIP/UNKNOWN, not outcomes.
- Report both targets, both cost assumptions, coverage, every exit and failures.
  No live activation; a passing exploratory result requires future validation.

This is a separate frozen candidate, not a relaxation of the rolling selector.
Previous history has informed hypothesis generation. Query-selection bias,
unequal histories, no actual spread/funding and repeated research trials remain.
