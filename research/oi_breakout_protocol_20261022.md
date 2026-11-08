# Frozen OI-confirmed LONG breakout comparison

Research only. Sources are the immutable Trade3 price snapshot and existing
Trade3 BTC/ETH OI files copied October22; no Trade1 or external price history.

Common baseline: existing BULL_VOLUME_BREAKOUT_12 rule unchanged, trailing
24h quote volume >=5M USDT, BTC and ETH last-1h price returns both >0,
and complete lagged OI windows for both benchmark coins. These BTC/ETH
measurements describe market context, not the altcoin's own OI.

OI candidate adds ONLY: BTC and ETH quantity OI changes over the previous hour
are both >=0, with at least one >=0.5%. No parameter fitting/search.

Primary assumed publication lag10min; sensitivity lags5/15min must both be
reported, never select the best one. Exact timestamp joins and all13 OI points
required; gaps are UNKNOWN. First-observed publication/revision metadata is
unavailable, so even positive results would remain provisional.

Fixed independent TP2% and TP3%, SL1.5%, hold12h, next5m-open entry,
ambiguous bar stop-first, max3 concurrent, one position/coin. Simulate full
history before reporting boundary so ongoing positions keep capacity.

Development August2026 (no tuning); test reporting September1 onward with
seven-day slices and separate October11+ tail. Previously inspected data:
exploratory chronological replay, NOT untouched confirmation.

Original gates remain: positive mean at30/50bps round-trip; daily-block95%
CI lower at30bps >0; >=100 settled OOS trades, >=15 signal days, >=2 coins.
Both targets reported; no live activation. Require new independent future
validation if exploratory gates ever pass. Portfolio comparisons are not
randomized causal estimates: filtering can change capacity/entry selection.
