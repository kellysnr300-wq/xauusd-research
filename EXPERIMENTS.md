# Experiments log

## 001 - MA crossover baseline (20/50, 5m)
- Data: 2016-2025 development set, 704,467 candles
- Costs: spread 0.30 + slippage 0.05 per side, 1 oz
- Result: 16,345 trades, win rate 30.5%, PnL -6,057, PF 0.84, max DD 7,057
- Losing years: 2016-2024; only 2025 profitable (+253)
- Conclusion: ~zero gross edge (+~480 before costs); costs dominate.
  This is the benchmark every later strategy must beat.

## 003 - Gold Trend (Donchian 55/20, 2 ATR), daily, 2008-2025
- 9 variants (entry 40/55/80, exit 15/20/30), split dev 2008-2015 / check 2016-2025.
- Base: 72 trades, +1151 $/oz total, long +1784, short -633. Buy&hold +3486.
- Pre-registered hypotheses: A shorts earn in 2011-15: FAILED (short -280 in dev).
  B longs earn before 2025: FAILED (long +16 in dev).
- Without the best year (2025) every variant loses (-10..-41 pts of %).
- 3 of 9 positive in both periods; 0 of 9 on both sides.
- Conclusion: no demonstrated edge; profit is long exposure + the 2025 rally.
  Single-market trend following cannot be validated in 17 years (needs ~44y for Sharpe 0.3).
