"""Quick self-test of a saved strategy on synthetic data.

Usage:
    python -m src.check_strategy <strategy_id> [--params '{"x": 1}']

Prints one JSON line: {"ok": bool, "problems": [...], "info": {...}}
"""

import argparse
import json
import traceback

import numpy as np
import pandas as pd

from src import registry
from src.backtester import Backtester
from src.lookahead import check_no_lookahead
from src.normalize import normalize_output


def synthetic_data(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 2000 + rng.normal(0, 0.8, n).cumsum()
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.6, n)

    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2020-01-06", periods=n, freq="5min", tz="UTC"
        ),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })


def run_check(strategy_id: str, params: dict) -> dict:
    result = {"ok": False, "problems": [], "info": {}}

    try:
        def factory():
            return registry.build_strategy(strategy_id, params)

        factory()  # instantiate once to surface constructor errors
        data = synthetic_data()

        problems = check_no_lookahead(factory, data)
        result["problems"] = problems

        if not problems:
            strategy = factory()
            backtester = Backtester(data, strategy)
            trades = backtester.simulate()
            result["info"] = {
                "synthetic_trades": len(trades),
                "notes": backtester.warnings,
                "diagnostics": getattr(strategy, "diagnostics", None),
            }
            result["ok"] = True

    except Exception as error:
        result["problems"].append(f"{type(error).__name__}: {error}")
        result["info"]["traceback"] = traceback.format_exc()[-800:]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    print(json.dumps(run_check(args.strategy_id, json.loads(args.params))))


if __name__ == "__main__":
    main()
