import pandas as pd

from src.execution import ExecutionModel
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

    Optional columns "stop_distance" and "target_distance" (price units,
    measured from the entry price) set a stop-loss / take-profit when a
    position is opened from that signal row.

    Execution rules (no look-ahead)
    -------------------------------
    - A signal on candle i is executed at the OPEN of candle i+1.
    - The entry candle itself can already hit the stop or target.
    - If stop and target are both touched in one candle, the stop wins.
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
    - "stop_price" (absolute price) and "target_r" (reward:risk multiple)
      are required on every entry row.
    - The entry is filled at the next candle's open. Risk is the distance
      from that open to stop_price; the target is open +/- target_r * risk,
      so the reward:risk is exact. Entries whose stop is on the wrong side
      of the open are skipped (counted in skipped_entries).
    - One position at a time: entry events are ignored while a trade is
      open. Trades end only at the stop, the target, or the end of data.
    - Stop/target fill rules are the same as in the default mode.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 10_000.0,
        execution: ExecutionModel | None = None,
        quantity: float = 1.0,
        max_entry_gap: pd.Timedelta = pd.Timedelta(minutes=30),
    ):
        self.data = data.copy().reset_index(drop=True)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.execution = execution or ExecutionModel()
        self.quantity = quantity
        self.max_entry_gap = max_entry_gap
        self.trades: list[Trade] = []
        self.skipped_entries = 0

    def run(self) -> pd.DataFrame:
        """Run the strategy and return generated signals."""

        signals = self.strategy.generate_signals(self.data)

        if len(signals) != len(self.data):
            raise ValueError(
                "Strategy must return one signal row per candle."
            )

        return signals

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade and update portfolio."""

        self.trades.append(trade)
        self.portfolio.record_pnl(trade.pnl)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self) -> list[Trade]:
        """Simulate the strategy candle by candle and return trades."""

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Data is missing columns: {missing}")

        signals = self.run()

        # Reset state so simulate() can be called more than once.
        self.trades = []
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

        stop_dist = self._optional_column(signals, "stop_distance")
        target_dist = self._optional_column(signals, "target_distance")

        position: Position | None = None
        pending = None
        stopped_side = None  # side (1/-1) whose signal is ignored

        for j in range(n):
            # 1. Execute the order generated on the previous candle.
            if pending is not None:
                target, sd, td = pending
                pending = None

                current = self._side_value(position)

                if target != current:
                    if position is not None:
                        self._close(position, ts[j], opens[j], "signal")
                        position = None

                    if target != 0:
                        gap = ts[j] - ts[j - 1]
                        if gap <= self.max_entry_gap:
                            position = self._open(
                                target, ts[j], opens[j], sd, td
                            )

            # 2. Check stop / target inside this candle.
            if position is not None:
                exit_fill = self._check_exits(
                    position, opens[j], highs[j], lows[j]
                )
                if exit_fill is not None:
                    fill, reason = exit_fill
                    stopped_side = self._side_value(position)
                    self._close(position, ts[j], fill, reason)
                    position = None

            # 3. Read this candle's close signal (acted on next candle).
            if j < n - 1:
                if stopped_side is not None:
                    if sig[j] == stopped_side:
                        continue
                    stopped_side = None

                pending = (sig[j], stop_dist[j], target_dist[j])

        # Close anything still open at the final close.
        if position is not None:
            self._close(position, ts[-1], closes[-1], "end_of_data")

        return self.trades

    def _simulate_events(self, signals: pd.DataFrame) -> list[Trade]:
        for column in ("stop_price", "target_r"):
            if column not in signals.columns:
                raise ValueError(
                    f"Event strategies need a '{column}' column."
                )

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
        stops = signals["stop_price"].to_numpy(dtype=float)
        multiples = signals["target_r"].to_numpy(dtype=float)

        position: Position | None = None
        pending = None

        for j in range(n):
            # 1. Fill the entry requested on the previous candle.
            if pending is not None:
                side, stop_price, multiple = pending
                pending = None

                if position is None:
                    if ts[j] - ts[j - 1] <= self.max_entry_gap:
                        position = self._open_event(
                            side, ts[j], opens[j], stop_price, multiple
                        )
                        if position is None:
                            self.skipped_entries += 1

            # 2. Stop / target inside this candle (entry candle included).
            if position is not None:
                hit = self._check_exits(
                    position, opens[j], highs[j], lows[j]
                )
                if hit is not None:
                    fill, reason = hit
                    self._close(position, ts[j], fill, reason)
                    position = None

            # 3. New entry request, filled at the next open.
            if j < n - 1 and entry[j] != 0:
                pending = (entry[j], stops[j], multiples[j])

        if position is not None:
            self._close(position, ts[-1], closes[-1], "end_of_data")

        return self.trades

    def _open_event(self, side, time, market_open, stop_price, multiple):
        if pd.isna(stop_price) or pd.isna(multiple) or multiple <= 0:
            return None

        if side == 1:
            risk = market_open - stop_price
            if risk <= 0:
                return None
            name = "long"
            price = self.execution.buy_price(market_open)
            target = market_open + multiple * risk
        else:
            risk = stop_price - market_open
            if risk <= 0:
                return None
            name = "short"
            price = self.execution.sell_price(market_open)
            target = market_open - multiple * risk

        return Position(
            side=name,
            entry_time=time,
            entry_price=price,
            quantity=self.quantity,
            stop_loss=float(stop_price),
            take_profit=float(target),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_column(signals: pd.DataFrame, name: str):
        if name in signals.columns:
            return signals[name].to_numpy(dtype=float)
        return [float("nan")] * len(signals)

    @staticmethod
    def _side_value(position: Position | None) -> int:
        if position is None:
            return 0
        return 1 if position.side == "long" else -1

    def _open(self, target, time, market_open, stop_d, target_d) -> Position:
        if target == 1:
            side = "long"
            price = self.execution.buy_price(market_open)
            stop = None if pd.isna(stop_d) else market_open - stop_d
            take = None if pd.isna(target_d) else market_open + target_d
        else:
            side = "short"
            price = self.execution.sell_price(market_open)
            stop = None if pd.isna(stop_d) else market_open + stop_d
            take = None if pd.isna(target_d) else market_open - target_d

        return Position(
            side=side,
            entry_time=time,
            entry_price=price,
            quantity=self.quantity,
            stop_loss=stop,
            take_profit=take,
        )

    def _check_exits(self, position: Position, o, h, l):
        """Return (market_fill_price, reason) if stop/target is hit."""

        stop = position.stop_loss
        take = position.take_profit

        if position.side == "long":
            if stop is not None and l <= stop:
                return min(stop, o), "stop_loss"
            if take is not None and h >= take:
                return max(take, o), "take_profit"
        else:
            if stop is not None and h >= stop:
                return max(stop, o), "stop_loss"
            if take is not None and l <= take:
                return min(take, o), "take_profit"

        return None

    def _close(self, position: Position, time, market_price, reason) -> None:
        if position.side == "long":
            exit_price = self.execution.exit_long_price(market_price)
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            exit_price = self.execution.exit_short_price(market_price)
            pnl = (position.entry_price - exit_price) * position.quantity

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
