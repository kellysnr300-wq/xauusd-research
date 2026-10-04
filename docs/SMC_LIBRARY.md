# SMC building blocks (`src/smc.py`)

Import what you need inside a strategy:

```python
from src.smc import structure, fvg, order_blocks, in_sessions, sweeps
```

**Rule for every column:** the value on row *i* is known at the close of
candle *i*. So a strategy can read row *i* and enter at the next candle's
open without look-ahead. `python -m src.test_smc` proves this for every
block by truncating the data and checking that earlier rows never change.

## Blocks

| Function | Main columns |
|---|---|
| `swings(data, n=1)` | `swing_high`, `swing_low` (flagged `n` candles late), `last_swing_high/low`, `last_swing_high_idx` |
| `structure(data, n=1, break_on="close")` | `trend` (+1/-1/0), `bos`, `choch` (+1/-1), `break_level`, `break_swing_idx` |
| `fvg(data, min_gap=0, max_age=500)` | `fvg_bull_new`, `fvg_bull_top/bottom` (latest open gap), `fvg_bull_retest` + `fvg_bull_retest_top/bottom`, same for `bear` |
| `order_blocks(data, n=1, break_on="close", lookback=5, zone="range", kinds="both")` | same columns with prefix `ob_` |
| `in_sessions(data, windows=None)` | one boolean column per killzone: `asia`, `london`, `ny_am`, `ny_pm` |
| `session_levels(data, "asia")` | `asia_high/low` (running), `prev_asia_high/low` (last finished session) |
| `daily_levels(data)` | `pdh`, `pdl`, `pdc`, `pwh`, `pwl`, `day_high/low`, `week_high/low` |
| `sweeps(data, high_level, low_level)` | `sweep_high`, `sweep_low` (+ the swept level) |
| `align_htf(data, htf, "1h", columns)` | higher-timeframe columns, only from candles already finished |

## Definitions (original ICT / SMC teaching unless noted)

* **Swing**: higher (lower) than `n` candles on each side. `n=1` is the
  classic 3-candle short-term high/low. Flagged `n` candles after it forms.
* **BOS**: first close beyond the latest swing in the trend direction.
* **CHoCH**: first close beyond the latest swing against the trend; the
  trend flips. `break_on="wick"` uses wicks instead of closes.
* **FVG**: bullish when `low[i] > high[i-2]`, bearish when
  `high[i] < low[i-2]`, known on candle `i`.
* **Order block**: last opposite-coloured candle at the origin of the move
  that broke structure. `zone="range"` is the full candle, `"body"` is
  open-to-close. `kinds` picks BOS blocks, CHoCH blocks or both.
* **Zones** are retested when price trades back in, invalidated when a
  candle closes through the far side, and used once.
* **Killzones** (New York time, daylight saving handled, by candle open
  time): asia 20:00-00:00, london 02:00-05:00, ny_am 07:00-10:00,
  ny_pm 13:30-16:00. Pass your own `windows={"name": ("HH:MM", "HH:MM")}`.
* **Trading day** is 17:00-17:00 New York; weeks start Sunday 17:00.

## Multi-timeframe without peeking

```python
from src.timeframes import resample_ohlc
from src.smc import structure, align_htf

h1 = resample_ohlc(data, "1h")
h1["trend"] = structure(h1, n=1)["trend"]          # computed ON the 1h candles
ctx = align_htf(data, h1, "1h", ["trend"])         # -> column htf_trend
bullish_context = ctx["htf_trend"] == 1
```

`htf_trend` on a 5-minute candle is the trend as of the last 1-hour candle
that had already finished.

## Example recipe

See `strategies/smc_example.py` (trend + FVG retest inside a killzone). It
returns `entry`, `stop_price` and `target_r` columns (event mode), which
the engine fills at the next open with an exact reward:risk.

## Things to keep in mind

* Many concepts combined, tuned on one 10-year series, will find fake
  edge. Decide the rules first, keep 2026 untouched, check on a holdout.
* Zones are used once and each setup has a stop, but fixed 1 oz sizing
  means risk varies per trade.
