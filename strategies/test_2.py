import pandas as pd

from src.strategies import Strategy


class MyStrategy(Strategy):
    """Pure FVG retest strategy."""

    def __init__(
        self,
        length=20,
        atr_length=14,
        stop_atr=1.0,
        target_r=2.5,
    ):
        # length is accepted for engine compatibility.
        self.length = length
        self.atr_length = atr_length
        self.stop_atr = stop_atr
        self.target_r = target_r

    def generate_signals(self, data: pd.DataFrame):

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # -------------------------
        # ATR
        # -------------------------

        previous_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - previous_close).abs()
        tr3 = (low - previous_close).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        atr = true_range.rolling(self.atr_length).mean()

        # -------------------------
        # FVG STATE
        # -------------------------

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        bullish_low = None
        bullish_high = None

        bearish_low = None
        bearish_high = None

        # -------------------------
        # PROCESS CANDLES CAUSALLY
        # -------------------------

        for i in range(2, len(data)):

            # Candle indices:
            # i-2 = candle 1
            # i-1 = candle 2
            # i   = candle 3

            c1_high = high.iloc[i - 2]
            c1_low = low.iloc[i - 2]

            c3_high = high.iloc[i]
            c3_low = low.iloc[i]

            # --------------------------------
            # FIRST: TEST EXISTING FVG RETEST
            # --------------------------------

            current_close = close.iloc[i]

            # Bullish FVG:
            # price must retrace into its stored range.
            if (
                bullish_low is not None
                and bullish_high is not None
                and bullish_low <= current_close <= bullish_high
            ):
                signals.iloc[i, signals.columns.get_loc("signal")] = 1

                # FVG has now been used.
                bullish_low = None
                bullish_high = None

            # Bearish FVG:
            elif (
                bearish_low is not None
                and bearish_high is not None
                and bearish_low <= current_close <= bearish_high
            ):
                signals.iloc[i, signals.columns.get_loc("signal")] = -1

                # FVG has now been used.
                bearish_low = None
                bearish_high = None

            # --------------------------------
            # THEN: DETECT NEW FVG
            # --------------------------------

            # Bullish FVG:
            # candle 1 high < candle 3 low
            if c1_high < c3_low:

                bullish_low = c1_high
                bullish_high = c3_low

                # A newly created FVG cannot be traded
                # on its formation candle.

            # Bearish FVG:
            # candle 1 low > candle 3 high
            elif c1_low > c3_high:

                bearish_low = c3_high
                bearish_high = c1_low

                # A newly created FVG cannot be traded
                # on its formation candle.

        # -------------------------
        # RISK MANAGEMENT
        # -------------------------

        signals["stop_atr"] = self.stop_atr
        signals["target_r"] = self.target_r

        return signals