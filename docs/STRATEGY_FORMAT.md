# Strategy format

A strategy is a Python file with **one class** that has
`generate_signals(self, data)`, **or one function** named
`generate_signals`, `strategy` or `signals` (or the only public function in
the file). Extra settings go in the class constructor or as keyword
arguments of the function; the dashboard passes them from the JSON
parameters box.

`data` is a DataFrame of 5-minute candles with columns
`timestamp, open, high, low, close, volume`.

Return **one row per candle**. Whatever you return is normalized by
`src/normalize.py`, so several shapes work:

| You return | Treated as |
|---|---|
| a list, array or Series of 1 / -1 / 0 | `signal` |
| booleans (True = long) | `signal` 1 / 0 |
| words: `long`, `short`, `buy`, `sell`, `flat` | `signal` |
| other numbers (0.7, -3) | `sign()` of the number, with a note |
| a DataFrame with `signal`, `position`, `side` or `direction` | `signal` |
| a DataFrame with `entry` | event mode (see below) |

## Two modes

**State mode (`signal`)**: the position you want to hold *after* each
candle closes (1 long, -1 short, 0 flat). The engine trades it at the next
candle's open and reverses or exits when the signal changes.

**Event mode (`entry`)**: a one-off entry request (1 or -1). The engine
manages the trade until the stop or target is hit. One position at a time;
extra entries are ignored while a trade is open. Every entry row needs a
stop and a target.

## Exit levels (optional in state mode, required in event mode)

Add any of these columns. They are read on the candle that signals the
entry and resolved at the actual fill price (next open).

| Column | Meaning |
|---|---|
| `stop_price` / `target_price` | absolute price |
| `stop_distance` / `target_distance` | price units from the fill |
| `stop_pct` / `target_pct` | percent of the signal candle's close |
| `stop_atr` / `target_atr` | multiple of ATR(14) of the signal candle |
| `target_r` (aliases `rr`, `rr_ratio`, `risk_reward`) | reward:risk multiple of the real risk |

If several are given: price beats distance beats pct beats ATR. A stop or
target on the wrong side of the fill price skips that entry (counted in the
run notes). While ATR is warming up (first 14 candles) signals are ignored.

## Rules enforced automatically

* Use only current and past candles. Before every run the strategy is
  executed on truncated data; if earlier output changes, the run is
  rejected as look-ahead (`shift(-1)`, whole-sample statistics, ...).
* Exactly one output row per candle, no infinite values.
* Signals act on the next candle's open, never the candle that produced them.
* Unknown output columns are ignored and listed in the run notes.

## Examples

```python
# State mode, function style, 1% stop and 2R target
def generate_signals(data, length=20):
    average = data["close"].rolling(length).mean()
    out = (data["close"] > average).astype(int)
    return out.to_frame("signal").assign(stop_pct=1.0, target_r=2.0)
```

```python
# Event mode: enter long when close crosses above the average
def generate_signals(data, length=20):
    average = data["close"].rolling(length).mean()
    cross = (data["close"] > average) & (data["close"].shift(1) <= average.shift(1))
    out = cross.astype(int).to_frame("entry")
    return out.assign(stop_atr=1.5, target_r=3.0)
```

See `strategies/break_retest_ob.py` for a full class-based example.

## When a run produces no trades

The dashboard now says so ("No trades were generated") instead of showing
empty numbers. The usual causes, in order of likelihood:

1. **Filters too strict.** Every extra condition (killzone, higher-timeframe
   bias, liquidity sweep, CHoCH, order block, FVG) multiplies the odds down.
   A strategy can expose `self.diagnostics = {"label": count, ...}` after
   `generate_signals`; the dashboard shows it as a setup funnel so you can
   see which filter removed the setups. `strategies/smc_v1.py` is an example.
2. **A condition that can never be true** (wrong sign, a `shift` in the
   wrong direction, comparing against a column that is always NaN). The run
   notes then say "no entries at all on this data (every row is 0)".
3. **Stops on the wrong side of the entry**, so every entry is skipped. The
   notes show "entry attempts skipped".
4. **Timeframe / data mismatch**, e.g. a 1-hour setting on daily candles.
5. **Output format** problems are rejected with an explicit message before
   the run starts.

Entries can also exceed trades: the engine holds one position at a time and
ignores new entries while a trade is open. The funnel's last line shows how
many trades were actually taken.
