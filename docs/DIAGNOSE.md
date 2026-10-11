# Why does a strategy trade rarely, never, or slowly?

```
python -m src.diagnose <strategy_id>                  # last two years, 5m
python -m src.diagnose <strategy_id> 2016 2025        # a longer window
python -m src.diagnose <strategy_id> --timeframe 1h
```

No need to read or paste the strategy's code. The tool measures three things
and turns them into plain-English likely reasons.

## 1. Speed

`generate_signals` is timed on growing slices (5k, 10k, 20k, 40k candles) and
a growth exponent is fitted. Time that doubles when the data doubles is
linear (exponent about 1); a strategy that re-scans history for every candle
shows an exponent near 2 and is projected to take hours on ten years. A
slice that does not finish within `--call-timeout` (default 120 s) is
reported as VERY SLOW without freezing the tool.

## 2. Output

How many order or signal rows, long vs short, resting vs market orders, how
far resting orders sit from the market, how big the stops are compared with
the typical candle range and with the round-trip cost. Orders re-posted
every candle (a zone kept alive) are counted as ONE setup, not as hundreds
of separate orders.

## 3. What the engine did with every order

`posted = filled + expired_unfilled + cancelled + dropped (trade open)
+ invalid + gap_blocked + overflow (+ a few still pending at the end)`. The
numbers always add up, which is tested.

| Reason | What it means |
|---|---|
| NEVER SIGNALS | the conditions are never true on this data |
| ORDERS REJECTED AS INVALID | stop/target on the wrong side of the entry (sign error) |
| RESTING ORDERS NEVER REACH THEIR PRICE | limit/stop prices too far from the market for how long they live |
| ORDERS DROPPED, A TRADE WAS ALREADY OPEN | signals constantly; the engine holds one position |
| ENTRIES BLOCKED BY MARKET GAPS | signals land right after weekends/holidays |
| STOPS ARE SMALL AGAINST COSTS | median stop under 5x the round-trip cost |
| TOO FEW TRADES TO JUDGE | under 100 trades: any win rate is mostly luck |
| SLOW / VERY SLOW | the run-time growth described above |

"None found" means it signals, fills and runs at a reasonable speed. It does
not mean it has an edge: that is what `src.placebo` and `src.screen` test.
