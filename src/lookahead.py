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


def _same(a, b):
    return np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


def _decision_rows(frame, mode) -> np.ndarray:
    """Rows where the strategy acts: orders, or changes of position."""
    if mode == "event":
        return np.flatnonzero(frame["entry"].to_numpy() != 0)
    signal = frame["signal"].to_numpy()
    return np.flatnonzero(np.diff(signal, prepend=0) != 0)


def check_no_lookahead(factory, data: pd.DataFrame, cuts=(0.35, 0.7, 0.9),
                       targeted=12, min_rows=200, tail=3):
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

    # Targeted cuts: stop the data right after a decision row. If that
    # decision used later candles, it changes (or disappears) once they are
    # gone. Broad cuts alone can miss a sparse, short-range peek.
    candidates = _decision_rows(full.frame, full.mode)
    candidates = candidates[candidates >= min_rows]
    if len(candidates) > targeted:
        picks = np.random.default_rng(0).choice(candidates, targeted, replace=False)
    else:
        picks = candidates

    for row in sorted(int(r) for r in picks):
        k = row + 1
        try:
            part = _normalized(factory, data.iloc[:k].copy())
        except StrategyOutputError as error:
            return [f"Output is invalid on shorter data: {error}"]

        for column in CANONICAL_COLUMNS:
            in_full = column in full.frame.columns
            in_part = column in part.frame.columns
            if not in_full and not in_part:
                continue
            a = (full.frame[column].iloc[max(0, k - tail):k].to_numpy(dtype=float)
                 if in_full else np.full(min(tail, k), np.nan))
            b = (part.frame[column].iloc[max(0, k - tail):k].to_numpy(dtype=float)
                 if in_part else np.full(min(tail, k), np.nan))
            if not _same(a, b).all():
                return [
                    f"LOOK-AHEAD: the decision at candle {row} ('{column}') "
                    "changes when the candles after it are removed. The "
                    "strategy is using future data."
                ]

    return problems
