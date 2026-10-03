import numpy as np
import pandas as pd

from src.strategies import Strategy


class BreakRetestOB(Strategy):
    """Break of structure, then retest of the origin order block.

    Simple version:
    1. Swing highs/lows are fractals (`swing` candles each side), known
       only `swing` candles after they form.
    2. Bullish BOS: a candle CLOSES above the latest swing high.
       Origin = lowest low since that swing high; the order block (OB) is
       the last bearish candle at or before that low (bearish mirrored).
    3. Retest: after the break, a candle trades back into the OB zone
       (low <= OB high) without breaking the OB low minus `buffer`.
    4. Entry on the next open. Stop = OB far edge -/+ `buffer`.
       Target = `target_r` times the real risk (exact R, set by the engine).

    Setups expire after `max_wait` candles. Trades whose estimated risk is
    outside [min_risk, max_risk] (price units) are skipped.
    """

    def __init__(
        self,
        swing=3,
        ob_lookback=5,
        buffer=0.3,
        min_risk=1.0,
        max_risk=25.0,
        target_r=3.0,
        max_wait=100,
        allow_long=True,
        allow_short=True,
    ):
        self.swing = int(swing)
        self.ob_lookback = int(ob_lookback)
        self.buffer = float(buffer)
        self.min_risk = float(min_risk)
        self.max_risk = float(max_risk)
        self.target_r = float(target_r)
        self.max_wait = int(max_wait)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)

    def _confirmed_swings(self, high, low):
        """Boolean arrays: swing high / low confirmed at each index.

        A swing at candle i is flagged at index i + swing, when the last
        of its right-hand confirmation candles has closed.
        """
        n = self.swing
        h = pd.Series(high)
        l = pd.Series(low)

        left_h = h.shift(1).rolling(n).max()
        right_h = h[::-1].shift(1).rolling(n).max()[::-1]
        is_high = (h > left_h) & (h >= right_h)

        left_l = l.shift(1).rolling(n).min()
        right_l = l[::-1].shift(1).rolling(n).min()[::-1]
        is_low = (l < left_l) & (l <= right_l)

        return (
            is_high.shift(n, fill_value=False).to_numpy(dtype=bool),
            is_low.shift(n, fill_value=False).to_numpy(dtype=bool),
        )

    @staticmethod
    def _find_ob(open_, close, start, bearish, lookback):
        """Last bearish (or bullish) candle at or before `start`."""
        for k in range(start, max(start - lookback, -1), -1):
            if bearish and close[k] < open_[k]:
                return k
            if not bearish and close[k] > open_[k]:
                return k
        return None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        o = data["open"].to_numpy(dtype=float)
        h = data["high"].to_numpy(dtype=float)
        l = data["low"].to_numpy(dtype=float)
        c = data["close"].to_numpy(dtype=float)
        count = len(data)
        n = self.swing

        swing_high_at, swing_low_at = self._confirmed_swings(h, l)

        entry = np.zeros(count, dtype=int)
        stop = np.full(count, np.nan)
        multiple = np.full(count, np.nan)

        last_high = last_low = None   # (price, index)
        high_used = low_used = True
        long_setup = short_setup = None  # (zone_low, zone_high, expires)

        for j in range(count):
            if swing_high_at[j]:
                last_high = (h[j - n], j - n)
                high_used = False
            if swing_low_at[j]:
                last_low = (l[j - n], j - n)
                low_used = False

            long_trigger = short_trigger = None

            # --- retest of an existing setup (uses candle j) ---------
            if long_setup is not None:
                zone_low, zone_high, expires = long_setup
                stop_level = zone_low - self.buffer
                if j > expires or l[j] <= stop_level:
                    long_setup = None
                elif l[j] <= zone_high:
                    long_setup = None
                    risk = c[j] - stop_level
                    if self.min_risk <= risk <= self.max_risk:
                        long_trigger = stop_level

            if short_setup is not None:
                zone_low, zone_high, expires = short_setup
                stop_level = zone_high + self.buffer
                if j > expires or h[j] >= stop_level:
                    short_setup = None
                elif h[j] >= zone_low:
                    short_setup = None
                    risk = stop_level - c[j]
                    if self.min_risk <= risk <= self.max_risk:
                        short_trigger = stop_level

            if long_trigger is not None and short_trigger is None:
                if self.allow_long:
                    entry[j], stop[j], multiple[j] = 1, long_trigger, self.target_r
            elif short_trigger is not None and long_trigger is None:
                if self.allow_short:
                    entry[j], stop[j], multiple[j] = -1, short_trigger, self.target_r

            # --- new break of structure (setup active from j + 1) -----
            if (
                last_high is not None and not high_used
                and c[j] > last_high[0]
            ):
                high_used = True
                origin = last_high[1] + int(
                    np.argmin(l[last_high[1]: j + 1]))
                ob = self._find_ob(o, c, origin, True, self.ob_lookback)
                if ob is not None and self.allow_long:
                    long_setup = (l[ob], h[ob], j + self.max_wait)

            if (
                last_low is not None and not low_used
                and c[j] < last_low[0]
            ):
                low_used = True
                origin = last_low[1] + int(
                    np.argmax(h[last_low[1]: j + 1]))
                ob = self._find_ob(o, c, origin, False, self.ob_lookback)
                if ob is not None and self.allow_short:
                    short_setup = (l[ob], h[ob], j + self.max_wait)

        out = pd.DataFrame(index=data.index)
        out["entry"] = entry
        out["stop_price"] = stop
        out["target_r"] = multiple
        return out
