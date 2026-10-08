"""Look inside a candle using the 1-minute data.

When one candle touches both the stop and the target, the candle alone
cannot say which came first. The 1-minute candles inside it usually can.
Raw 1-minute files are loaded one day at a time, only when needed.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from src.paths import raw_path


class IntrabarResolver:
    def __init__(self, path_for_day=None, cache_days: int = 16):
        self.path_for_day = path_for_day or raw_path
        self.cache_days = cache_days
        self._cache: OrderedDict = OrderedDict()
        self.loaded_days: set = set()
        self.missing_days: set = set()
        self.queries = 0
        self.answered = 0

    # ------------------------------------------------------------------

    def _load_day(self, day):
        if day in self._cache:
            self._cache.move_to_end(day)
            return self._cache[day]

        arrays = None
        try:
            raw = pd.read_csv(self.path_for_day(day))
            stamps = pd.to_datetime(raw.iloc[:, 0], utc=True)
            arrays = (
                stamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                raw["Open"].to_numpy(dtype=float),
                raw["High"].to_numpy(dtype=float),
                raw["Low"].to_numpy(dtype=float),
            )
            self.loaded_days.add(day)
        except (FileNotFoundError, KeyError, ValueError):
            self.missing_days.add(day)

        self._cache[day] = arrays
        while len(self._cache) > self.cache_days:
            self._cache.popitem(last=False)
        return arrays

    def minutes(self, start, end):
        """1-minute (time, open, high, low) arrays with start <= t < end."""

        start_u = pd.Timestamp(start).tz_convert("UTC")
        end_u = pd.Timestamp(end).tz_convert("UTC")
        lo = start_u.tz_localize(None).to_datetime64()
        hi = end_u.tz_localize(None).to_datetime64()

        parts = []
        day = start_u.date()
        last_day = (end_u - pd.Timedelta(microseconds=1)).date()
        while day <= last_day:
            loaded = self._load_day(day)
            if loaded is not None:
                parts.append(loaded)
            day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()

        if not parts:
            return None

        t = np.concatenate([p[0] for p in parts])
        keep = (t >= lo) & (t < hi)
        return tuple(np.concatenate([p[i] for p in parts])[keep] for i in range(4))

    # ------------------------------------------------------------------

    def first_touch(self, side, start, end, stop, target, adj=0.0):
        """Which was touched first inside [start, end): "stop" / "target".

        Returns None when the minute data is missing, or when stop and
        target were both touched inside the same minute (still unknown).
        `adj` is the ask-side shift for short triggers (0 for bid-only).
        """

        self.queries += 1
        found = self.minutes(start, end)
        if found is None or len(found[0]) == 0:
            return None

        _, opens, highs, lows = found
        long = side == "long"

        for o, h, l in zip(opens, highs, lows):
            if long:
                stop_hit, take_hit = l <= stop, h >= target
                gap_stop, gap_take = o <= stop, o >= target
            else:
                stop_hit, take_hit = h + adj >= stop, l + adj <= target
                gap_stop, gap_take = o + adj >= stop, o + adj <= target

            if stop_hit and take_hit:
                if gap_stop:
                    self.answered += 1
                    return "stop"
                if gap_take:
                    self.answered += 1
                    return "target"
                return None
            if stop_hit:
                self.answered += 1
                return "stop"
            if take_hit:
                self.answered += 1
                return "target"

        return None

    def scan_order(self, side, kind, start, end, level, stop, target, adj=0.0):
        """Order of events for a resting order inside one candle.

        The order is filled the first minute price trades to `level`
        (downwards for a long limit / short stop, upwards otherwise). After
        the fill, which of stop / target comes first?

        Returns "stop", "target", "open" (filled, neither reached before the
        candle ends) or None (unknown: missing data, or the fill and an
        exit fall inside the same minute).
        """

        self.queries += 1
        found = self.minutes(start, end)
        if found is None or len(found[0]) == 0:
            return None

        _, opens, highs, lows = found
        long = side == "long"
        down = (long and kind == "limit") or (not long and kind == "stop")
        filled = False

        for o, h, l in zip(opens, highs, lows):
            if long:
                stop_hit, take_hit = l <= stop, h >= target
                gap_stop, gap_take = o <= stop, o >= target
            else:
                stop_hit, take_hit = h + adj >= stop, l + adj <= target
                gap_stop, gap_take = o + adj >= stop, o + adj <= target

            if not filled:
                touched = (l <= level) if down else (h >= level)
                if not touched:
                    continue
                filled = True
                if stop_hit or take_hit:
                    return None  # fill and exit in the same minute: unordered
                continue

            if stop_hit and take_hit:
                if gap_stop:
                    self.answered += 1
                    return "stop"
                if gap_take:
                    self.answered += 1
                    return "target"
                return None
            if stop_hit:
                self.answered += 1
                return "stop"
            if take_hit:
                self.answered += 1
                return "target"

        if not filled:
            return None
        self.answered += 1
        return "open"
