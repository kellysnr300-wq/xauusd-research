# How far to trust a result

Every run shows a **Reliability** panel. It separates two different kinds
of uncertainty, because they have different cures.

## 1. Engine uncertainty (candle data hides detail)

A 5-minute candle only records open, high, low, close. If one candle
touches both a stop and a target, the candle cannot say which came first.

1. **Settle it with real data.** The 1-minute files in `data/raw` are read
   for just that candle. The first level touched wins. A candle that opens
   beyond a level fills there; there is nothing to guess.
2. **Bracket what is left.** If both levels fall inside the same minute (or
   the minute file is missing) the result is genuinely unknown. The same
   signals are re-run twice:
   * *pessimistic*: unknown exits go against you, and short stops/targets
     trigger on the ask price (bid + spread), since the data is bid-only;
   * *optimistic*: unknown exits go your way.

The distance between those runs is the **engine range**. Labels:
tight (<= 2 points of win rate), moderate (<= 5), wide (more).

## 2. Statistical uncertainty (few trades are noisy)

A win rate measured on N trades is only an estimate. The panel shows a
Wilson 95% interval. Rough guide: 30 trades give about +-15 points, 100
trades about +-9, 300 trades about +-5.

The headline "plausible" band combines both: the low end of the
pessimistic run's interval to the high end of the optimistic run's.

## What it does NOT cover

* A different broker's prices, spreads or execution.
* Costs that vary through the day (the spread here is a fixed input).
* Movement between 1-minute candles (a tick replay would be the next step).
* Overfitting: a strategy tuned on the same ten years looks better than it
  will. Keep a holdout period and judge the sample size shown here.
* The future.

## Checking the machinery

`python -m src.test_accuracy` includes a ground-truth test: a price path
is built minute by minute, and the engine (with 1-minute drill-down) must
agree with an independent minute-by-minute simulator on every trade.
