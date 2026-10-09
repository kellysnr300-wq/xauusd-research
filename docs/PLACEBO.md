# The placebo test

Question: does this strategy's TIMING beat random timing?

```
python -m src.placebo smc_limit 2016 2025 --timeframe 5m --runs 50
```

## What it does

1. Runs the strategy normally and records its result with the engine
   settings used for the placebos (no 1-minute drill-down, so the
   comparison is like for like).
2. Builds `--runs` placebo versions of the same orders and simulates each
   with the same costs and engine:
   * **Event strategies** (entry / limit orders): each block of consecutive
     orders is moved to the SAME New York clock time on another day, up to
     `--max-shift-days` (default 90) away. Side, distance from the market,
     stop size, R target and lifetime are kept. So the killzone mix, the
     daylight-saving alignment and the rough volatility regime stay
     matched, while the structure that picked the moment (the zone, the
     trend, the sweep) is gone.
   * **State strategies** (a held position): the position series is
     rotated by a random amount, which keeps the time spent long, short
     and flat.
3. Compares the real average P&L per trade with the placebo distribution.

## Reading the output

* `placebo avg` is what random timing earns with this geometry and these
  costs. It is the baseline: usually slightly NEGATIVE, because of costs.
* `real minus placebo` is the edge the strategy's logic adds per trade.
* `p (random timing >= real)`: the share of placebos that did at least as
  well (with a +1 correction, so never exactly 0). With 50 runs the
  smallest possible value is about 0.02.
* **Verdict**
  * BETTER than random timing: p <= 0.05 and real above the placebo mean.
  * WORSE: p of "placebo <= real" <= 0.05.
  * INDISTINGUISHABLE: anything else. This is the normal outcome for a
    strategy with no edge.
  * too few trades to judge: under 30 real trades.

## How the test itself was checked (`python -m src.test_placebo`)

* Orders keep their exact geometry and NY clock time, move 1 to N days,
  cross daylight-saving changes correctly, and blocks stay consecutive.
* On 40 strategies with NO skill the average p-value is about 0.5 and
  almost none are flagged.
* A strategy that deliberately peeks 6 candles ahead is flagged at the
  maximum confidence the run count allows.

## Limits

* It tests timing, not whether the geometry is sensible. A strategy can
  still be unprofitable even if it beats random timing (costs).
* Random timing in a trending market still earns the drift on the side the
  strategy picked, so a long-only strategy in a bull market is compared
  with other long entries in the same bull market. That is intended.
* One test at p = 0.05 is weak evidence if you have tried many strategies.
  Ask for p <= 0.01 (needs 100+ runs) before believing a single success,
  and confirm on data the strategy has never seen.
* Block shifting preserves the order's distance from the market, not the
  exact price path, so very distant resting orders fill differently by
  design.
