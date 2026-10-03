"""Checks that a strategy's output is valid and uses no future data."""

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = ("signal", "stop_distance", "target_distance")


def validate_signals(signals, n_rows: int) -> list[str]:
    if not isinstance(signals, pd.DataFrame):
        return ["generate_signals must return a pandas DataFrame."]

    problems = []

    if len(signals) != n_rows:
        problems.append(
            f"Returned {len(signals)} rows for {n_rows} candles "
            "(need exactly one row per candle)."
        )

    if "signal" not in signals.columns:
        problems.append("Output has no 'signal' column.")
    else:
        values = pd.to_numeric(signals["signal"], errors="coerce")
        if not values.fillna(0).isin([-1, 0, 1]).all():
            problems.append("'signal' must contain only -1, 0 or 1.")

    return problems


def check_no_lookahead(factory, data: pd.DataFrame, cuts=(0.35, 0.7, 0.9)):
    """Return a list of problems (empty list = passed).

    Runs the strategy on all of `data`, then on truncated copies. If an
    earlier signal changes when later candles are removed, the strategy
    is using future information.
    """

    data = data.reset_index(drop=True)

    full = factory().generate_signals(data.copy())
    problems = validate_signals(full, len(data))

    if problems:
        return problems

    columns = [c for c in SIGNAL_COLUMNS if c in full.columns]

    for fraction in cuts:
        k = max(int(len(data) * fraction), 1)

        part = factory().generate_signals(data.iloc[:k].copy())

        if not isinstance(part, pd.DataFrame) or len(part) != k:
            problems.append(
                "Output length changes when the data is shortened."
            )
            break

        for column in columns:
            if column not in part.columns:
                problems.append(
                    f"Column '{column}' disappears on shorter data."
                )
                continue

            a = pd.to_numeric(
                full[column].iloc[:k], errors="coerce"
            ).to_numpy(dtype=float)
            b = pd.to_numeric(part[column], errors="coerce").to_numpy(
                dtype=float
            )

            if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
                same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
                first = int(np.flatnonzero(~same)[0])
                problems.append(
                    f"LOOK-AHEAD: '{column}' at candle {first} changes when "
                    "later candles are removed. The strategy is using "
                    "future data (e.g. shift(-1), whole-sample statistics)."
                )
                return problems

    return problems
