import pandas as pd

from src.strategies import Strategy


class MyStrategy(Strategy):
    """
    SMC structure detector — Step 2.

    Detects:
    - Internal swing highs/lows
    - External swing highs/lows
    - HH / HL / LH / LL
    - Bullish BOS
    - Bearish BOS
    - Bullish CHoCH
    - Bearish CHoCH

    Rules:
    - Swings are only usable after confirmation.
    - BOS/CHoCH requires a candle CLOSE through the level.
    - A wick through a level does not constitute a structural break.
    - No trading signals are generated yet.
    """

    def __init__(
        self,
        length=5,
        internal_swing_length=2,
        external_swing_length=None,
    ):
        self.length = length

        self.internal_swing_length = internal_swing_length

        self.external_swing_length = (
            external_swing_length
            if external_swing_length is not None
            else length
        )

    @staticmethod
    def _detect_swings(
        data: pd.DataFrame,
        length: int,
    ) -> pd.DataFrame:
        """
        Detect confirmed swing highs/lows.

        A candidate swing at candle i is confirmed at:

            i + length

        Therefore the swing is not exposed to the strategy
        before sufficient future candles have actually closed.
        """

        highs = data["high"]
        lows = data["low"]

        result = pd.DataFrame(index=data.index)

        result["swing_high"] = False
        result["swing_low"] = False

        result["swing_high_price"] = float("nan")
        result["swing_low_price"] = float("nan")

        for i in range(length, len(data) - length):

            current_high = highs.iloc[i]
            current_low = lows.iloc[i]

            left_highs = highs.iloc[i - length:i]
            right_highs = highs.iloc[i + 1:i + length + 1]

            left_lows = lows.iloc[i - length:i]
            right_lows = lows.iloc[i + 1:i + length + 1]

            # ----------------------------------------------
            # Swing high
            # ----------------------------------------------

            if (
                current_high > left_highs.max()
                and current_high > right_highs.max()
            ):
                confirmation_index = i + length

                result.loc[
                    result.index[confirmation_index],
                    "swing_high",
                ] = True

                result.loc[
                    result.index[confirmation_index],
                    "swing_high_price",
                ] = current_high

            # ----------------------------------------------
            # Swing low
            # ----------------------------------------------

            if (
                current_low < left_lows.min()
                and current_low < right_lows.min()
            ):
                confirmation_index = i + length

                result.loc[
                    result.index[confirmation_index],
                    "swing_low",
                ] = True

                result.loc[
                    result.index[confirmation_index],
                    "swing_low_price",
                ] = current_low

        return result

    @staticmethod
    def _build_structure(
        data: pd.DataFrame,
        swings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify confirmed swings as:

            HH = Higher High
            LH = Lower High
            HL = Higher Low
            LL = Lower Low

        Only previously confirmed swings are used.
        """

        result = pd.DataFrame(index=data.index)

        result["structure_high"] = ""
        result["structure_low"] = ""

        last_high = None
        last_low = None

        for i in range(len(data)):

            # ----------------------------------------------
            # Confirmed swing high
            # ----------------------------------------------

            if swings["swing_high"].iloc[i]:

                price = swings["swing_high_price"].iloc[i]

                if last_high is None:
                    structure = "HH"
                elif price > last_high:
                    structure = "HH"
                else:
                    structure = "LH"

                result.loc[
                    result.index[i],
                    "structure_high",
                ] = structure

                last_high = price

            # ----------------------------------------------
            # Confirmed swing low
            # ----------------------------------------------

            if swings["swing_low"].iloc[i]:

                price = swings["swing_low_price"].iloc[i]

                if last_low is None:
                    structure = "HL"
                elif price > last_low:
                    structure = "HL"
                else:
                    structure = "LL"

                result.loc[
                    result.index[i],
                    "structure_low",
                ] = structure

                last_low = price

        return result

    @staticmethod
    def _detect_breaks(
        data: pd.DataFrame,
        swings: pd.DataFrame,
        structure: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect BOS and CHoCH.

        Structural break requires CLOSE beyond the confirmed
        structural level.

        Wick-only penetration is ignored.

        CHoCH logic:

        Bullish structure:
            protected HL
            close below HL
            -> bearish CHoCH

        Bearish structure:
            protected LH
            close above LH
            -> bullish CHoCH

        BOS:

        Bullish:
            close above relevant confirmed swing high

        Bearish:
            close below relevant confirmed swing low
        """

        result = pd.DataFrame(index=data.index)

        result["bullish_bos"] = False
        result["bearish_bos"] = False

        result["bullish_choch"] = False
        result["bearish_choch"] = False

        result["bullish_bos_level"] = float("nan")
        result["bearish_bos_level"] = float("nan")

        result["bullish_choch_level"] = float("nan")
        result["bearish_choch_level"] = float("nan")

        # --------------------------------------------------
        # State
        # --------------------------------------------------

        market_structure = 0
        #  1 = bullish
        # -1 = bearish
        #  0 = not established

        last_swing_high = None
        last_swing_low = None

        protected_high = None
        protected_low = None

        broken_high = None
        broken_low = None

        for i in range(len(data)):

            close = data["close"].iloc[i]

            # ==================================================
            # First process newly CONFIRMED swings.
            # ==================================================

            if swings["swing_high"].iloc[i]:

                last_swing_high = swings[
                    "swing_high_price"
                ].iloc[i]

            if swings["swing_low"].iloc[i]:

                last_swing_low = swings[
                    "swing_low_price"
                ].iloc[i]

            # ==================================================
            # Determine structure from confirmed swing labels.
            # ==================================================

            high_label = structure["structure_high"].iloc[i]
            low_label = structure["structure_low"].iloc[i]

            # --------------------------------------------------
            # New HH
            # --------------------------------------------------

            if high_label == "HH":

                if market_structure == 1:
                    # In bullish structure, the most recently
                    # confirmed meaningful HL becomes protected.
                    if last_swing_low is not None:
                        protected_low = last_swing_low

                elif market_structure == 0:
                    market_structure = 1

            # --------------------------------------------------
            # New HL
            # --------------------------------------------------

            if low_label == "HL":

                if market_structure == 1:
                    protected_low = last_swing_low

                elif market_structure == 0:
                    market_structure = 1

            # --------------------------------------------------
            # New LL
            # --------------------------------------------------

            if low_label == "LL":

                if market_structure == -1:
                    if last_swing_high is not None:
                        protected_high = last_swing_high

                elif market_structure == 0:
                    market_structure = -1

            # --------------------------------------------------
            # New LH
            # --------------------------------------------------

            if high_label == "LH":

                if market_structure == -1:
                    protected_high = last_swing_high

                elif market_structure == 0:
                    market_structure = -1

            # ==================================================
            # Bullish BOS
            # ==================================================

            if (
                last_swing_high is not None
                and close > last_swing_high
                and broken_high != last_swing_high
            ):

                result.loc[
                    result.index[i],
                    "bullish_bos",
                ] = True

                result.loc[
                    result.index[i],
                    "bullish_bos_level",
                ] = last_swing_high

                broken_high = last_swing_high

                if market_structure <= 0:
                    market_structure = 1

            # ==================================================
            # Bearish BOS
            # ==================================================

            if (
                last_swing_low is not None
                and close < last_swing_low
                and broken_low != last_swing_low
            ):

                result.loc[
                    result.index[i],
                    "bearish_bos",
                ] = True

                result.loc[
                    result.index[i],
                    "bearish_bos_level",
                ] = last_swing_low

                broken_low = last_swing_low

                if market_structure >= 0:
                    market_structure = -1

            # ==================================================
            # Bearish CHoCH
            # ==================================================

            if (
                market_structure == 1
                and protected_low is not None
                and close < protected_low
            ):

                result.loc[
                    result.index[i],
                    "bearish_choch",
                ] = True

                result.loc[
                    result.index[i],
                    "bearish_choch_level",
                ] = protected_low

                market_structure = -1

                broken_low = protected_low

            # ==================================================
            # Bullish CHoCH
            # ==================================================

            if (
                market_structure == -1
                and protected_high is not None
                and close > protected_high
            ):

                result.loc[
                    result.index[i],
                    "bullish_choch",
                ] = True

                result.loc[
                    result.index[i],
                    "bullish_choch_level",
                ] = protected_high

                market_structure = 1

                broken_high = protected_high

        result["market_structure"] = 0

        # Reconstruct the structure state causally so it can be
        # inspected in the backtest output.
        state = 0

        for i in range(len(data)):

            if result["bullish_choch"].iloc[i]:
                state = 1

            elif result["bearish_choch"].iloc[i]:
                state = -1

            elif result["bullish_bos"].iloc[i]:
                state = 1

            elif result["bearish_bos"].iloc[i]:
                state = -1

            result.loc[
                result.index[i],
                "market_structure",
            ] = state

        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data = data.reset_index(drop=True).copy()

        # ======================================================
        # INTERNAL STRUCTURE
        # ======================================================

        internal = self._detect_swings(
            data,
            self.internal_swing_length,
        )

        internal_structure = self._build_structure(
            data,
            internal,
        )

        internal_breaks = self._detect_breaks(
            data,
            internal,
            internal_structure,
        )

        # ======================================================
        # EXTERNAL STRUCTURE
        # ======================================================

        external = self._detect_swings(
            data,
            self.external_swing_length,
        )

        external_structure = self._build_structure(
            data,
            external,
        )

        external_breaks = self._detect_breaks(
            data,
            external,
            external_structure,
        )

        # ======================================================
        # OUTPUT
        # ======================================================

        signals = pd.DataFrame(index=data.index)

        # Still no actual trades.
        signals["signal"] = 0

        # ======================================================
        # INTERNAL SWINGS
        # ======================================================

        signals["internal_swing_high"] = (
            internal["swing_high"]
        )

        signals["internal_swing_low"] = (
            internal["swing_low"]
        )

        signals["internal_swing_high_price"] = (
            internal["swing_high_price"]
        )

        signals["internal_swing_low_price"] = (
            internal["swing_low_price"]
        )

        # ======================================================
        # INTERNAL STRUCTURE
        # ======================================================

        signals["internal_structure_high"] = (
            internal_structure["structure_high"]
        )

        signals["internal_structure_low"] = (
            internal_structure["structure_low"]
        )

        signals["internal_bullish_bos"] = (
            internal_breaks["bullish_bos"]
        )

        signals["internal_bearish_bos"] = (
            internal_breaks["bearish_bos"]
        )

        signals["internal_bullish_choch"] = (
            internal_breaks["bullish_choch"]
        )

        signals["internal_bearish_choch"] = (
            internal_breaks["bearish_choch"]
        )

        signals["internal_bullish_bos_level"] = (
            internal_breaks["bullish_bos_level"]
        )

        signals["internal_bearish_bos_level"] = (
            internal_breaks["bearish_bos_level"]
        )

        signals["internal_bullish_choch_level"] = (
            internal_breaks["bullish_choch_level"]
        )

        signals["internal_bearish_choch_level"] = (
            internal_breaks["bearish_choch_level"]
        )

        signals["internal_market_structure"] = (
            internal_breaks["market_structure"]
        )

        # ======================================================
        # EXTERNAL SWINGS
        # ======================================================

        signals["external_swing_high"] = (
            external["swing_high"]
        )

        signals["external_swing_low"] = (
            external["swing_low"]
        )

        signals["external_swing_high_price"] = (
            external["swing_high_price"]
        )

        signals["external_swing_low_price"] = (
            external["swing_low_price"]
        )

        # ======================================================
        # EXTERNAL STRUCTURE
        # ======================================================

        signals["external_structure_high"] = (
            external_structure["structure_high"]
        )

        signals["external_structure_low"] = (
            external_structure["structure_low"]
        )

        signals["external_bullish_bos"] = (
            external_breaks["bullish_bos"]
        )

        signals["external_bearish_bos"] = (
            external_breaks["bearish_bos"]
        )

        signals["external_bullish_choch"] = (
            external_breaks["bullish_choch"]
        )

        signals["external_bearish_choch"] = (
            external_breaks["bearish_choch"]
        )

        signals["external_bullish_bos_level"] = (
            external_breaks["bullish_bos_level"]
        )

        signals["external_bearish_bos_level"] = (
            external_breaks["bearish_bos_level"]
        )

        signals["external_bullish_choch_level"] = (
            external_breaks["bullish_choch_level"]
        )

        signals["external_bearish_choch_level"] = (
            external_breaks["bearish_choch_level"]
        )

        signals["external_market_structure"] = (
            external_breaks["market_structure"]
        )

        return signals