"""Checks that a strategy's output is valid and uses no future data."""

import numpy as np
import pandas as pd

from src.normalize import (
    CANONICAL_COLUMNS,
    StrategyOutputError,
    normalize_output,
)


def _normalized(factory, data: pd.DataFrame):
    return normalize_output(factory().generate_signals(data.copy()), data)


def check_no_lookahead(factory, data: pd.DataFrame, cuts=(0.35, 0.7, 0.9)):
    """Return a list of problems (empty list = passed).

    Runs the strategy on all of `data`, then on truncated copies. If an
    earlier output changes when later candles are removed, the strategy is
    using future information. Outputs are compared after normalization.
    """

    data = data.reset_index(drop=True)

    try:
        full = _normalized(factory, data)
    except StrategyOutputError as error:
        return [str(error)]

    problems = []

    for fraction in cuts:
        k = max(int(len(data) * fraction), 1)
        short_data = data.iloc[:k].copy()

        try:
            part = _normalized(factory, short_data)
        except StrategyOutputError as error:
            return [f"Output is invalid on shorter data: {error}"]

        if part.mode != full.mode:
            return ["Output format changes when the data is shortened."]

        for column in CANONICAL_COLUMNS:
            in_full = column in full.frame.columns
            in_part = column in part.frame.columns
            if not in_full and not in_part:
                continue

            a = (full.frame[column].iloc[:k].to_numpy(dtype=float)
                 if in_full else np.full(k, np.nan))
            b = (part.frame[column].to_numpy(dtype=float)
                 if in_part else np.full(k, np.nan))

            same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
            if not same.all():
                first = int(np.flatnonzero(~same)[0])
                return [
                    f"LOOK-AHEAD: '{column}' at candle {first} changes when "
                    "later candles are removed. The strategy is using "
                    "future data (e.g. shift(-1), whole-sample statistics)."
                ]

    return problems
