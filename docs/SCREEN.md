# Screening every strategy with the placebo yardstick

```
python -m src.screen --list                       # what is in strategies/
python -m src.screen 2016 2025 --runs 20          # all of them
python -m src.screen 2016 2025 --only smc_original,smc_basic --runs 50
python -m src.screen 2016 2025 --resume results/screen_<time>.json
```

For each strategy, in order: build it with its saved parameters, run the
look-ahead guard on real data, then the placebo test (docs/PLACEBO.md). One
bad strategy never stops the batch: crashes, invalid output, never trading,
and strategies that run longer than `--timeout-min` (default 30) are recorded
with the reason and skipped. The results file is rewritten after every
strategy, so an interrupted run loses nothing (`--resume`).

## The look-ahead guard (and what it rejects)

Besides comparing outputs on truncated data at a few cut points, the guard
now cuts the data RIGHT AFTER the strategy's own decisions (orders, or
changes of position). A decision that used later candles changes or vanishes
when they are removed. This was added after a test showed that a strategy
acting on only 0.35% of candles and peeking 3 candles ahead could slip past
the broad cuts alone.

The guard is deliberately strict. A strategy whose decisions depend on the
LENGTH of the data is rejected too, for example one that draws a random
sequence with `rng.choice(..., len(data))`: the same candle would get a
different decision depending on how much data exists after it. Use a
per-row hash or a fixed per-row rule instead.

## Reading the table

* `random` is what the same orders earn at random times; `edge` is real
  minus random, per trade. `p` is the chance random timing does as well.
* With 20 runs the smallest possible p is about 0.048. Anything with
  p <= 0.10 and a positive edge is only "worth a longer check": rerun it
  alone with `--runs 200`.
* Testing many strategies flags some by luck. Judge the whole table, not
  the best row.
* All strategies are run on the SAME timeframe (`--timeframe`). A daily
  strategy such as gold_trend should be screened with `--timeframe 1d`.
