import pandas as pd

from src.execution import ExecutionModel
from src.normalize import normalize_output
from src.portfolio import Portfolio
from src.strategies import Strategy
from src.trades import Position, Trade


class Backtester:
    """Strategy-independent historical backtesting engine.

    Signal convention
    -----------------
    The strategy returns one row per candle with a "signal" column:
    the position it wants to hold AFTER that candle closes
    (1 = long, -1 = short, 0 = flat).

    Optional exit-level columns (see src/normalize.py) are read from the
    signal row and resolved at the fill price: stop_price, stop_distance,
    target_price, target_distance, target_r. Strategy output is passed
    through the normalizer first, so looser formats are accepted.

    Execution rules (no look-ahead)
    -------------------------------
    - A signal on candle i is executed at the OPEN of candle i+1.
    - The entry candle itself can already hit the stop or target.
    - If stop and target are both touched in one candle, the 1-minute
      data (if a resolver is given) decides which came first; otherwise
      the `ambiguity` policy does ("stop_first" by default). A candle
      that OPENS beyond the stop (or target) fills there, no ambiguity.
    - If a candle opens beyond a stop/target, the fill is at the open.
    - After a stop/target exit, the same signal value is ignored until
      the strategy signal changes (prevents instant re-entry).
    - Entries are skipped if the next candle is more than
      max_entry_gap after the signal candle (weekend/holiday gaps).
    - A position still open at the end is closed at the last close.

    Event mode (bracket-order strategies)
    -------------------------------------
    If the strategy output has an "entry" column (1 long, -1 short,
    0 none) the engine runs in event mode instead:
    - A stop (stop_price or stop_distance) and a target (target_r,
      target_distance or target_price) are required on every entry row.
    - The entry is filled at the next candle's open. With target_r the
      target is open +/- target_r * risk, so reward:risk is exact.
      Entries whose stop/target is on the wrong side of the open are
      skipped (counted in skipped_entries).
    - One position at a time: entry events are ignored while a trade is
      open. Trades end only at the stop, the target, or the end of data.
    - Stop/target fill rules are the same as in the default mode.
    - With an "entry_price" column the order RESTS: it is filled the moment
      price trades to it (see _try_fill) during the next `valid_for`
      candles, at that price, instead of at the next open. Entry and exit
      inside one candle are ordered with the 1-minute resolver when given.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 10_000.0,
        execution: ExecutionModel | None = None,
        quantity: float = 1.0,
        max_entry_gap: pd.Timedelta = pd.Timedelta(minutes=30),
        ambiguity: str = "stop_first",
        resolver=None,
        ask_triggers: bool = False,
        bar_delta: pd.Timedelta | None = None,
        fill_margin: float = 0.0,
        cancel_on_fill: bool = True,
        max_orders: int = 50,
    ):
        if ambiguity not in ("stop_first", "target_first"):
            raise ValueError("ambiguity must be 'stop_first' or 'target_first'")
        self.data = data.copy().reset_index(drop=True)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.execution = execution or ExecutionModel()
        self.quantity = quantity
        self.max_entry_gap = max_entry_gap
        self.ambiguity = ambiguity
        self.resolver = resolver
        self.ask_triggers = ask_triggers
        self.fill_margin = fill_margin
        self.cancel_on_fill = cancel_on_fill
        self.max_orders = max_orders
        self.bar_delta = bar_delta or self._infer_bar_delta()
        self.trades: list[Trade] = []
        self.trade_meta: list[dict] = []
        self.skipped_entries = 0
        self.warnings: list[str] = []
        self._signals = None

    def _infer_bar_delta(self) -> pd.Timedelta:
        stamps = pd.DatetimeIndex(self.data["timestamp"])[:200]
        steps = stamps[1:] - stamps[:-1]
        steps = steps[steps > pd.Timedelta(0)]
        return steps.min() if len(steps) else pd.Timedelta(minutes=5)

    def run(self) -> pd.DataFrame:
        """Run the strategy and return generated signals."""

        raw = self.strategy.generate_signals(self.data.copy())
        normalized = normalize_output(raw, self.data)
        self.warnings = normalized.warnings
        return normalized.frame

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade and update portfolio."""

        self.trades.append(trade)
        self.portfolio.record_pnl(trade.pnl)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, reuse_signals: bool = False) -> list[Trade]:
        """Simulate the strategy candle by candle and return trades.

        reuse_signals=True skips running the strategy again (used to rerun
        the same signals under different execution assumptions).
        """

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Data is missing columns: {missing}")

        if reuse_signals and self._signals is not None:
            signals = self._signals
        else:
            signals = self.run()
            self._signals = signals

        # Reset state so simulate() can be called more than once.
        self.trades = []
        self.trade_meta = []
        self.skipped_entries = 0
        self.portfolio = Portfolio(self.initial_capital)

        if "entry" in signals.columns:
            return self._simulate_events(signals)

        if "signal" not in signals.columns:
            raise ValueError(
                "Strategy output needs a 'signal' (or 'entry') column."
            )

        n = len(self.data)
        if n == 0:
            return self.trades

        ts = list(self.data["timestamp"])
        opens = self.data["open"].to_numpy(dtype=float)
        highs = self.data["high"].to_numpy(dtype=float)
        lows = self.data["low"].to_numpy(dtype=float)
        closes = self.data["close"].to_numpy(dtype=float)

        sig = (
            signals["signal"].fillna(0).to_numpy(dtype=float)
        ).clip(-1, 1).round().astype(int)

        specs = self._spec_arrays(signals)

        position: Position | None = None
        pending = None
        stopped_side = None  # side (1/-1) whose signal is ignored

        for j in range(n):
            # 1. Execute the order generated on the previous candle.
            if pending is not None:
                target, spec_row = pending
                pending = None

                current = self._side_value(position)

                if target != current:
                    if position is not None:
                        self._close(position, ts[j], opens[j], "signal")
                        position = None

                    if target != 0:
                        gap = ts[j] - ts[j - 1]
                        if gap <= self.max_entry_gap:
                            position = self._open_position(
                                target, ts[j], opens[j], spec_row, specs,
                                require_stop=False,
                            )
                            if position is None:
                                self.skipped_entries += 1

            # 2. Check stop / target inside this candle.
            if position is not None:
                exit_fill = self._check_exits(
                    position, opens[j], highs[j], lows[j], ts[j]
                )
                if exit_fill is not None:
                    fill, reason, meta = exit_fill
                    stopped_side = self._side_value(position)
                    self._close(position, ts[j], fill, reason, meta)
                    position = None

            # 3. Read this candle's close signal (acted on next candle).
            if j < n - 1:
                if stopped_side is not None:
                    if sig[j] == stopped_side:
                        continue
                    stopped_side = None

                pending = (sig[j], j)

        # Close anything still open at the final close.
        if position is not None:
            self._close(position, ts[-1], closes[-1], "end_of_data")

        return self.trades

    def _simulate_events(self, signals: pd.DataFrame) -> list[Trade]:
        n = len(self.data)
        if n == 0:
            return self.trades

        ts = list(self.data["timestamp"])
        opens = self.data["open"].to_numpy(dtype=float)
        highs = self.data["high"].to_numpy(dtype=float)
        lows = self.data["low"].to_numpy(dtype=float)
        closes = self.data["close"].to_numpy(dtype=float)

        entry = (
            signals["entry"].fillna(0).to_numpy(dtype=float)
        ).clip(-1, 1).round().astype(int)
        specs = self._spec_arrays(signals)

        def column(name):
            if name in signals.columns:
                return signals[name].to_numpy(dtype=float)
            return None

        prices = column("entry_price")
        lives = column("valid_for")
        cancels = column("cancel_beyond")

        position: Position | None = None
        entry_meta = {"ambiguous": False, "resolved": False}
        orders: list[dict] = []

        for j in range(n):
            filled = None  # None, "open" (market / gap fill) or "touch"
            fill_order = None
            gap_ok = j > 0 and (ts[j] - ts[j - 1]) <= self.max_entry_gap

            # 1. Orders from earlier candles: expire, then try to fill.
            orders = [o for o in orders if j <= o["last"]]

            if position is not None:
                # market requests are dropped while a trade is open
                orders = [o for o in orders if o["price"] is not None]
            elif gap_ok:
                for order in list(orders):
                    if order["price"] is None:
                        orders.remove(order)
                        position = self._open_position(
                            order["side"], ts[j], opens[j], order["row"],
                            specs, require_stop=True)
                        if position is None:
                            self.skipped_entries += 1
                            continue
                        filled, fill_order = "open", order
                    else:
                        hit = self._try_fill(order, opens[j], highs[j], lows[j])
                        if hit is None:
                            continue
                        orders.remove(order)
                        mid, entry_price, gap, level = hit
                        position = self._open_position_at(
                            order["side"], ts[j], mid, entry_price,
                            order["row"], specs)
                        if position is None:
                            self.skipped_entries += 1
                            continue
                        order["level"] = level
                        filled = "open" if gap else "touch"
                        fill_order = order
                    break

                if filled is not None:
                    entry_meta = {"ambiguous": False, "resolved": False}
                    if self.cancel_on_fill:
                        orders = []

            # 2. Stop / target inside this candle.
            if position is not None:
                if filled == "touch":
                    hit = self._exits_after_touch(
                        position, fill_order, opens[j], highs[j], lows[j],
                        ts[j], entry_meta)
                else:
                    hit = self._check_exits(
                        position, opens[j], highs[j], lows[j], ts[j])

                if hit is not None:
                    fill, reason, meta = hit
                    self._close(position, ts[j], fill, reason,
                                self._merge_meta(entry_meta, meta))
                    position = None

            # 3. Cancel resting orders whose zone was closed through.
            if orders:
                orders = [
                    o for o in orders
                    if o["cancel"] is None
                    or (o["side"] == 1 and closes[j] >= o["cancel"])
                    or (o["side"] == -1 and closes[j] <= o["cancel"])
                ]

            # 4. New order from this candle's close.
            if j < n - 1 and entry[j] != 0:
                side = int(entry[j])
                price = None
                if prices is not None and prices[j] == prices[j]:
                    price = float(prices[j])

                if price is None:
                    order = {"side": side, "price": None, "kind": "market",
                             "row": j, "last": j + 1, "cancel": None}
                else:
                    if side == 1:
                        kind = "limit" if price <= closes[j] else "stop"
                    else:
                        kind = "limit" if price >= closes[j] else "stop"
                    life = 1
                    if lives is not None and lives[j] == lives[j]:
                        life = max(int(lives[j]), 1)
                    cancel = None
                    if cancels is not None and cancels[j] == cancels[j]:
                        cancel = float(cancels[j])
                    order = {"side": side, "price": price, "kind": kind,
                             "row": j, "last": j + life, "cancel": cancel}

                orders = (orders + [order])[-self.max_orders:]

        if position is not None:
            self._close(position, ts[-1], closes[-1], "end_of_data",
                        self._merge_meta(entry_meta, None))

        return self.trades

    # ------------------------------------------------------------------
    # Resting orders
    # ------------------------------------------------------------------

    def _try_fill(self, order, o, h, l):
        """Does this candle fill the order? -> (mid, entry_price, gap, level).

        Chart prices are treated as mid; a buy pays the ask (mid + half the
        spread), a sell receives the bid. A buy limit at P fills when the ask
        reaches P, i.e. mid <= P - half. fill_margin makes fills harder
        (price must trade through by that much); a negative value makes them
        easier. A candle that opens beyond the trigger fills at the open.
        """

        half = self.execution.spread / 2.0
        slip = self.execution.slippage
        margin = self.fill_margin
        side, kind, price = order["side"], order["kind"], order["price"]

        if side == 1 and kind == "limit":
            level = price - half - margin
            triggered, gap = l <= level, o <= level
            mid = o if gap else price - half
            entry = mid + half + (slip if gap else 0.0)
        elif side == 1:
            level = price - half + margin
            triggered, gap = h >= level, o >= level
            mid = o if gap else price - half
            entry = mid + half + slip
        elif kind == "limit":
            level = price + half + margin
            triggered, gap = h >= level, o >= level
            mid = o if gap else price + half
            entry = mid - half - (slip if gap else 0.0)
        else:
            level = price + half - margin
            triggered, gap = l <= level, o <= level
            mid = o if gap else price + half
            entry = mid - half - slip

        if not triggered:
            return None
        return mid, entry, gap, level

    def _open_position_at(self, side, time, mid, entry_price, row, specs):
        levels = self._resolve_levels(side, mid, row, specs, True)
        if levels is None:
            return None
        stop, target = levels
        return Position(
            side="long" if side == 1 else "short",
            entry_time=time,
            entry_price=entry_price,
            quantity=self.quantity,
            stop_loss=stop,
            take_profit=target,
        )

    def _exits_after_touch(self, position, order, o, h, l, start, entry_meta):
        """Stop / target in the candle where a resting order was filled.

        The fill happened at an unknown moment inside the candle, so a level
        reached by the candle's extremes may have been reached BEFORE the
        fill. 1-minute data settles the order of events. Without it:
        * a limit entry (price came toward the position) reaches its stop
          only after the fill (certain) but its target maybe before it;
        * a stop entry reaches its target only after the fill (certain) but
          its stop maybe before it.
        Uncertain legs count against you under "stop_first" and for you
        under "target_first".
        """

        stop, take = position.stop_loss, position.take_profit
        long = position.side == "long"
        adj = self.execution.spread if (self.ask_triggers and not long) else 0.0

        if long:
            stop_hit, take_hit = l <= stop, h >= take
        else:
            stop_hit, take_hit = h + adj >= stop, l + adj <= take

        if not stop_hit and not take_hit:
            return None

        def result(choice, meta):
            if choice == "stop":
                return stop, "stop_loss", meta
            return take, "take_profit", meta

        if self.resolver is not None and start is not None:
            outcome = self.resolver.scan_order(
                position.side, order["kind"], start, start + self.bar_delta,
                order["level"], stop, take, adj)
            if outcome is not None:
                entry_meta["ambiguous"], entry_meta["resolved"] = True, True
                if outcome == "open":
                    return None
                return result(outcome, {"ambiguous": False, "resolved": False})

        limit_type = order["kind"] == "limit"
        certain_stop, certain_take = limit_type, not limit_type
        stop_first = self.ambiguity == "stop_first"

        counts_stop = stop_hit and (certain_stop or stop_first)
        counts_take = take_hit and (certain_take or not stop_first)

        uncertain = (stop_hit and not certain_stop) or (take_hit and not certain_take)
        if uncertain or (stop_hit and take_hit):
            entry_meta["ambiguous"], entry_meta["resolved"] = True, False

        if counts_stop and counts_take:
            return result("stop" if stop_first else "target",
                          {"ambiguous": False, "resolved": False})
        if counts_stop:
            return result("stop", {"ambiguous": False, "resolved": False})
        if counts_take:
            return result("target", {"ambiguous": False, "resolved": False})
        return None

    @staticmethod
    def _merge_meta(entry_meta, meta):
        meta = meta or {"ambiguous": False, "resolved": False}
        ambiguous = entry_meta["ambiguous"] or meta["ambiguous"]
        resolved = ambiguous and (
            (not entry_meta["ambiguous"] or entry_meta["resolved"])
            and (not meta["ambiguous"] or meta["resolved"]))
        return {"ambiguous": ambiguous, "resolved": resolved}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_arrays(signals: pd.DataFrame) -> dict:
        """Exit-level columns as numpy arrays (missing columns omitted)."""
        names = ("stop_price", "stop_distance", "target_price",
                 "target_distance", "target_r")
        return {
            name: signals[name].to_numpy(dtype=float)
            for name in names if name in signals.columns
        }

    @staticmethod
    def _side_value(position: Position | None) -> int:
        if position is None:
            return 0
        return 1 if position.side == "long" else -1

    def _resolve_levels(self, side, market_open, row, specs, require_stop):
        """Return (stop, target) prices for a fill, or None if invalid.

        Priority: stop_price > stop_distance; target_price >
        target_distance > target_r. Levels on the wrong side of the fill
        price make the entry invalid.
        """
        sign = 1 if side == 1 else -1

        def value(name):
            array = specs.get(name)
            if array is None:
                return None
            v = array[row]
            return None if v != v else float(v)

        stop = None
        stop_price = value("stop_price")
        stop_distance = value("stop_distance")
        if stop_price is not None:
            stop = stop_price
        elif stop_distance is not None:
            stop = market_open - sign * stop_distance

        if stop is None and require_stop:
            return None
        if stop is not None and sign * (market_open - stop) <= 0:
            return None

        target = None
        target_price = value("target_price")
        target_distance = value("target_distance")
        target_r = value("target_r")
        if target_price is not None:
            target = target_price
        elif target_distance is not None:
            target = market_open + sign * target_distance
        elif target_r is not None:
            if stop is None:
                return None
            target = market_open + sign * target_r * abs(market_open - stop)

        if target is not None and sign * (target - market_open) <= 0:
            return None
        if require_stop and target is None:
            return None

        return stop, target

    def _open_position(self, side, time, market_open, row, specs,
                       require_stop):
        levels = self._resolve_levels(
            side, market_open, row, specs, require_stop)
        if levels is None:
            return None

        stop, target = levels

        if side == 1:
            name, price = "long", self.execution.buy_price(market_open)
        else:
            name, price = "short", self.execution.sell_price(market_open)

        return Position(
            side=name,
            entry_time=time,
            entry_price=price,
            quantity=self.quantity,
            stop_loss=stop,
            take_profit=target,
        )

    def _check_exits(self, position: Position, o, h, l, start=None):
        """Return (market_fill_price, reason, meta) if stop/target is hit.

        meta = {"ambiguous": bool, "resolved": bool}: ambiguous means the
        candle touched both levels and its open did not decide the order.
        """

        stop = position.stop_loss
        take = position.take_profit

        if stop is None and take is None:
            return None

        long = position.side == "long"
        adj = self.execution.spread if (self.ask_triggers and not long) else 0.0

        if long:
            stop_hit = stop is not None and l <= stop
            take_hit = take is not None and h >= take
            gap_stop = stop is not None and o <= stop
            gap_take = take is not None and o >= take
        else:
            stop_hit = stop is not None and h + adj >= stop
            take_hit = take is not None and l + adj <= take
            gap_stop = stop is not None and o + adj >= stop
            gap_take = take is not None and o + adj <= take

        if not stop_hit and not take_hit:
            return None

        meta = {"ambiguous": False, "resolved": False}

        if stop_hit and take_hit:
            if gap_stop:
                choice = "stop"
            elif gap_take:
                choice = "target"
            else:
                meta["ambiguous"] = True
                choice = None
                if self.resolver is not None and start is not None:
                    choice = self.resolver.first_touch(
                        position.side, start, start + self.bar_delta,
                        stop, take, adj,
                    )
                    meta["resolved"] = choice is not None
                if choice is None:
                    choice = "target" if self.ambiguity == "target_first" else "stop"
        else:
            choice = "stop" if stop_hit else "target"

        if choice == "stop":
            fill = min(stop, o) if long else max(stop, o)
            return fill, "stop_loss", meta

        fill = max(take, o) if long else min(take, o)
        return fill, "take_profit", meta

    def _close(self, position: Position, time, market_price, reason,
               meta=None) -> None:
        if position.side == "long":
            exit_price = self.execution.exit_long_price(market_price)
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            exit_price = self.execution.exit_short_price(market_price)
            pnl = (position.entry_price - exit_price) * position.quantity

        self.trade_meta.append(meta or {"ambiguous": False, "resolved": False})
        self.record_trade(
            Trade(
                side=position.side,
                entry_time=position.entry_time,
                exit_time=time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=float(pnl),
                exit_reason=reason,
            )
        )
