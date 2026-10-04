import numpy as np
import pandas as pd

from src.smc import fvg, in_sessions, structure
from src.strategies import Strategy


class SmcLiquidityDisplacement(Strategy):
    """
    Selective SMC strategy.

    Setup:

        LONG
        1. Bullish market structure
        2. Sell-side liquidity sweep
        3. Bullish displacement
        4. Bullish FVG
        5. FVG retest
        6. Stop below swept liquidity
        7. Fixed asymmetric R target

        SHORT
        1. Bearish market structure
        2. Buy-side liquidity sweep
        3. Bearish displacement
        4. Bearish FVG
        5. FVG retest
        6. Stop above swept liquidity
        7. Fixed asymmetric R target

    The strategy is causal:
    all rolling liquidity and displacement measurements are shifted
    so the current candle is not used to calculate its own threshold.

    Engine compatibility:
    `length` is accepted because the strategy runner passes a
    `length` argument to strategy constructors.
    """

    def __init__(
        self,
        length=20,
        swing=3,
        liquidity_lookback=None,
        displacement_lookback=10,
        displacement_mult=1.5,
        min_gap=0.5,
        stop_buffer=0.3,
        target_r=3.0,
        min_risk=1.0,
        max_risk=25.0,
        killzones="london,ny_am",
    ):
        # ------------------------------------------------------------
        # Engine compatibility
        # ------------------------------------------------------------
        self.length = int(length)

        # If no separate liquidity lookback is supplied,
        # use the engine's standard `length`.
        self.liquidity_lookback = (
            self.length
            if liquidity_lookback is None
            else int(liquidity_lookback)
        )

        self.swing = int(swing)
        self.displacement_lookback = int(displacement_lookback)
        self.displacement_mult = float(displacement_mult)
        self.min_gap = float(min_gap)
        self.stop_buffer = float(stop_buffer)
        self.target_r = float(target_r)
        self.min_risk = float(min_risk)
        self.max_risk = float(max_risk)

        self.killzones = [
            z.strip()
            for z in str(killzones).split(",")
            if z.strip()
        ]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ============================================================
        # OHLC
        # ============================================================
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        open_ = data["open"].to_numpy(dtype=float)

        n = len(data)

        # ============================================================
        # SMC LIBRARY
        # ============================================================
        structure_data = structure(
            data,
            n=self.swing,
        )

        trend = structure_data["trend"].to_numpy()

        gaps = fvg(
            data,
            min_gap=self.min_gap,
        )

        zones = in_sessions(data)

        active_session = np.zeros(n, dtype=bool)

        for name in self.killzones:
            if name in zones.columns:
                active_session |= zones[name].to_numpy(dtype=bool)

        # ============================================================
        # CANDLE STATISTICS
        # ============================================================
        candle_range = high - low
        body = np.abs(close - open_)

        body_ratio = np.divide(
            body,
            candle_range,
            out=np.zeros(n, dtype=float),
            where=candle_range > 0,
        )

        # ============================================================
        # DISPLACEMENT
        #
        # The current candle is excluded from its own average.
        #
        # Example:
        #
        # average range at candle i =
        # candles i-lookback ... i-1
        #
        # NOT including candle i.
        # ============================================================
        range_series = pd.Series(
            candle_range,
            index=data.index,
        )

        avg_range = (
            range_series
            .rolling(self.displacement_lookback)
            .mean()
            .shift(1)
            .to_numpy()
        )

        displacement_threshold = np.where(
            np.isfinite(avg_range),
            avg_range * self.displacement_mult,
            np.inf,
        )

        bullish_displacement = (
            (close > open_)
            & (body >= displacement_threshold)
            & (body_ratio >= 0.60)
        )

        bearish_displacement = (
            (close < open_)
            & (body >= displacement_threshold)
            & (body_ratio >= 0.60)
        )

        # ============================================================
        # LIQUIDITY
        #
        # Previous liquidity only.
        #
        # We deliberately shift the rolling extrema by one candle.
        # Therefore candle i cannot define the liquidity level that
        # candle i itself is supposed to sweep.
        # ============================================================
        low_series = pd.Series(
            low,
            index=data.index,
        )

        high_series = pd.Series(
            high,
            index=data.index,
        )

        previous_low = (
            low_series
            .rolling(self.liquidity_lookback)
            .min()
            .shift(1)
            .to_numpy()
        )

        previous_high = (
            high_series
            .rolling(self.liquidity_lookback)
            .max()
            .shift(1)
            .to_numpy()
        )

        # ------------------------------------------------------------
        # Sell-side liquidity sweep
        #
        # Price trades below previous lows but closes back above them.
        # ------------------------------------------------------------
        sell_side_sweep = (
            np.isfinite(previous_low)
            & (low < previous_low)
            & (close > previous_low)
        )

        # ------------------------------------------------------------
        # Buy-side liquidity sweep
        #
        # Price trades above previous highs but closes back below them.
        # ------------------------------------------------------------
        buy_side_sweep = (
            np.isfinite(previous_high)
            & (high > previous_high)
            & (close < previous_high)
        )

        # ============================================================
        # FVG
        # ============================================================
        bullish_fvg = gaps[
            "fvg_bull_retest"
        ].to_numpy(dtype=bool)

        bearish_fvg = gaps[
            "fvg_bear_retest"
        ].to_numpy(dtype=bool)

        # ============================================================
        # SETUP
        #
        # We want the complete SMC sequence on the signal candle:
        #
        # LONG:
        # bullish structure
        # + sell-side sweep
        # + bullish displacement
        # + bullish FVG
        # + killzone
        #
        # SHORT:
        # bearish structure
        # + buy-side sweep
        # + bearish displacement
        # + bearish FVG
        # + killzone
        # ============================================================
        long_setup = (
            (trend == 1)
            & sell_side_sweep
            & bullish_displacement
            & bullish_fvg
            & active_session
        )

        short_setup = (
            (trend == -1)
            & buy_side_sweep
            & bearish_displacement
            & bearish_fvg
            & active_session
        )

        # ============================================================
        # STOP LOSS
        #
        # Long:
        # below the swept liquidity / current candle low.
        #
        # Short:
        # above the swept liquidity / current candle high.
        #
        # Because the sweep must occur on this candle, using the
        # extreme of this candle protects against placing the stop
        # inside the sweep.
        # ============================================================
        long_stop = (
            np.minimum(
                low,
                previous_low,
            )
            - self.stop_buffer
        )

        short_stop = (
            np.maximum(
                high,
                previous_high,
            )
            + self.stop_buffer
        )

        # ============================================================
        # RISK
        # ============================================================
        long_risk = close - long_stop
        short_risk = short_stop - close

        valid_long_risk = (
            np.isfinite(long_risk)
            & (long_risk >= self.min_risk)
            & (long_risk <= self.max_risk)
        )

        valid_short_risk = (
            np.isfinite(short_risk)
            & (short_risk >= self.min_risk)
            & (short_risk <= self.max_risk)
        )

        long_ok = (
            long_setup
            & valid_long_risk
        )

        short_ok = (
            short_setup
            & valid_short_risk
        )

        # ============================================================
        # ENTRY
        # ============================================================
        entry = np.where(
            long_ok,
            1,
            np.where(
                short_ok,
                -1,
                0,
            ),
        )

        # ============================================================
        # STOP
        # ============================================================
        stop = np.where(
            long_ok,
            long_stop,
            np.where(
                short_ok,
                short_stop,
                np.nan,
            ),
        )

        # ============================================================
        # TARGET
        #
        # The engine calculates the actual price target from:
        #
        #     entry +/- risk * target_r
        #
        # Default = 3R.
        # ============================================================
        target_r = np.where(
            entry != 0,
            self.target_r,
            np.nan,
        )

        # ============================================================
        # OUTPUT
        # ============================================================
        return pd.DataFrame(
            {
                "entry": entry,
                "stop_price": stop,
                "target_r": target_r,
            },
            index=data.index,
        )