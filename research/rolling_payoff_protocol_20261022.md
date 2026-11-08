# Trade3 rolling payoff gate — frozen exploratory protocol

Frozen before loading the October 22 price snapshot. Research only: no live
delivery, production configuration, server writes or Trade1 changes.

- Universe: all available Trade3 5m futures caches, including core scalp,
  coin-query and BTC/ETH main caches. Retain the longest contiguous source per
  symbol; report excluded/short/gapped coverage. Do not select coins by return.
- Candidate families: the five existing long rules in
  `intraday_long_bracket_study.make_events`, unchanged.
- Brackets: +2%/-1.5% and +3%/-1.5%; next-open entry, 12h holding limit,
  conservative stop-first when both barriers occur in one candle.
- At each seven-day boundary, use the previous 28 days. Only trades whose
  signal AND exit precede the boundary can train the selector. No test label
  changes the selected rule during its test week.
- Candidate training eligibility: >=50 closed trades and >=10 signal days;
  positive mean at both 30 and 50 bps round-trip cost; lower bound of daily
  block-bootstrap 95% confidence interval at 30 bps >0. Choose largest lower
  bound, then mean net return, then stable rule/bracket order. Otherwise abstain.
- Portfolio: one position per coin; at most three concurrent positions. Ties
  sorted by symbol, not future return. Carry open positions across week edges.
- OOS starts 2026-06-01 UTC; weekly boundaries anchored there. Historical
  intervals through October 10 have already informed research: exploratory.
  Report October 11 onward separately, but this is a fresh-tail sensitivity
  test, NOT independent validation of the newly designed selector.
- Final gates unchanged: >=100 settled OOS trades, >=15 signal days, >=2 coins;
  positive average net return at 30 and 50 bps; daily-block CI95 lower >0 at
  30 bps. Passing exploratory gates alone does not authorize activation.
- Missing bars, malformed prices, open/censored outcomes are UNKNOWN/SKIP;
  never turn them into zero returns or wins. Use complete 12h windows only.
- No parameter sweep after results. A failed result rejects this frozen
  candidate. A passing result requires an independent future holdout.

Limits: user-query selection/survivorship bias, uneven history, no historical
order books/funding/fills, multiple prior research trials and correlated
coins. Cost stress is not proof of actual execution cost.
