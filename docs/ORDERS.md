# Resting orders (limit and stop entries)

Until now a strategy could only enter at the NEXT candle's open. A trader
who wants to buy a retest of a zone leaves a limit order AT the zone. This
engine now supports that.

## How a strategy posts an order

Event mode (an `entry` column) plus:

| Column | Meaning |
|---|---|
| `entry_price` (alias `limit_price`) | price of the resting order. Empty = market order at the next open |
| `valid_for` | candles the order stays live after the candle that posted it (default 1) |
| `cancel_beyond` | cancel if a candle CLOSES beyond this (long: below it, short: above it) |

Kind is worked out from the price: a buy below the close is a LIMIT, a buy
above it is a STOP entry (breakout); sells are the mirror. A stop and a
target are still required (`stop_price` or `stop_distance`, and `target_r`,
`target_distance` or `target_price`). With `target_r` the reward:risk is
exact, because the entry price is known in advance.

A common pattern is to re-post a one-candle order on every candle while a
zone is alive (see `strategies/smc_limit.py`).

## Fill rules

Chart prices are treated as mid; a buy pays the ask (half the spread above),
a sell receives the bid (half the spread below).

* **Buy limit at P** fills when the ask reaches P (mid <= P - half spread),
  at price P. **Sell limit** mirrored.
* **Buy stop at P** fills when the ask reaches P, at P plus slippage.
* A candle that OPENS beyond the trigger fills at the open (a better price
  for limits, a worse one for stops).
* `fill_margin` (price units) makes fills harder: price must trade that far
  through the level. Touching is not always filling.
* One position at a time. Orders posted while a trade is open wait until it
  closes, within their `valid_for`. When an order fills the others are
  cancelled (`cancel_on_fill`).
* Orders are skipped if the stop is on the wrong side of the fill.

## Entry and exit in the same candle

A candle only shows its extremes. After a fill, were the stop or target
reached BEFORE or AFTER it? The 1-minute data in `data/raw` decides when it
can. Where it cannot (the fill and an exit fall inside the same minute, or
the minute file is missing) a conservative rule is used and the trade is
flagged as ambiguous:

* a limit entry reaches its stop after the fill for certain, but its target
  maybe before it, so the target is not counted;
* a stop entry reaches its target after the fill for certain, but its stop
  maybe before it, so the stop is counted.

## What the Reliability panel does with it

It re-runs the same signals under a pessimistic case (resting orders must
trade 20 cents THROUGH their price, short stops trigger on the ask,
ambiguity goes against you) and an optimistic one (fills on a touch of the
chart price, ambiguity goes your way). A wide range means the result
depends on fill assumptions, which is common for tight stops.

## Limits

No partial fills or queue position, no price improvement beyond gaps, costs
fixed rather than real time-varying spreads, no stop slippage inside a
candle.

Tests: `python -m src.test_orders` includes a comparison against an
independent minute-by-minute simulator on 134 trades.
