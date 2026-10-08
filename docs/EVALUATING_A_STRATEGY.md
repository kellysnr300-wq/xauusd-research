# Evaluating a strategy honestly

Decide the rules BEFORE looking at results, then follow them.

1. **Benchmark first.** `python -m src.benchmark 2016 2025` shows what simply
   holding 1 oz did. A long-biased strategy on a rising market looks good
   by default; judge it against this.
2. **Develop / check split.** Run the strategy once on the first period
   (e.g. 2016-2021) and once on the second (2022-2025) with the SAME
   parameters. Never adjust parameters between the runs. Keep 2026 untouched.
3. **Both sides.** Run with `{"allow_short": false}` and with
   `{"allow_long": false}`. If only the long side earns, the result is the
   gold bull market, not a trading edge.
4. **Neighbours.** Change each main parameter a little (for example
   entry/exit 40/15, 55/20, 80/30). A real edge works across neighbours; an
   edge that exists at a single setting is noise.
5. **Count the trades.** The Reliability panel shows the sample label. A
   daily system makes only a few trades a year, so a decade may be "small
   sample". Say so rather than trusting a percentage.
6. **Kill rule.** Treat it as NO EDGE if it loses in either period, or only
   one parameter set works, or it only wins on one side.

## Known limits of this engine for long-holding strategies

* No overnight financing / swap. Positions held for weeks or months pay
  (or earn) it in real life, so results are optimistic for longs.
* Fixed 1 oz size. Real trend systems size by volatility.
* Close-based exits are executed at the next open, so a sudden gap is
  taken in full.
* Costs are a fixed spread and slippage.
