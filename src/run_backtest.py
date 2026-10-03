"""Run any saved strategy on the development data and save the results.

Usage:
    python -m src.run_backtest ma_cross 2016 2025
    python -m src.run_backtest ma_cross 2016 2025 --params '{"fast": 10, "slow": 40}'
    python -m src.run_backtest ma_cross 2025 2025 --spread 0.5

Strategies are loaded from strategies/<id>.py (see src/registry.py).
Results are saved in results/<run_id>/ (git-ignored).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

from src import registry
from src.backtester import Backtester
from src.dataset import load_development
from src.equity import build_equity_curve
from src.execution import ExecutionModel
from src.lookahead import check_no_lookahead
from src.metrics import calculate_metrics

RESULTS_ROOT = Path("results")
LOOKAHEAD_SAMPLE = 20_000


def write_status(run_dir: Path, stage: str, **extra) -> None:
    """Atomically write results/<run_id>/status.json."""
    payload = {
        "stage": stage,
        "updated": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, run_dir / "status.json")


def full_metrics(trades, equity, initial_capital):
    summary = calculate_metrics(trades, initial_capital)

    if not trades:
        return summary

    pnls = pd.Series([float(t.pnl) for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    durations = pd.Series([t.exit_time - t.entry_time for t in trades])

    summary["wins"] = int(len(wins))
    summary["losses"] = int(len(losses))
    summary["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    summary["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    summary["payoff_ratio"] = (
        summary["avg_win"] / abs(summary["avg_loss"])
        if summary["avg_loss"] < 0 else None
    )
    summary["profit_factor"] = (
        gross_win / gross_loss if gross_loss > 0 else None
    )
    summary["max_drawdown"] = float(equity["drawdown"].max())
    summary["max_drawdown_pct"] = (
        summary["max_drawdown"] / initial_capital * 100.0
    )
    summary["avg_trade_duration"] = str(durations.mean())
    summary["long_trades"] = int(sum(t.side == "long" for t in trades))
    summary["short_trades"] = int(sum(t.side == "short" for t in trades))

    return summary


def yearly_pnl(trades) -> dict:
    rows = pd.DataFrame({
        "year": [t.exit_time.year for t in trades],
        "pnl": [float(t.pnl) for t in trades],
    })
    return {
        int(year): round(float(value), 2)
        for year, value in rows.groupby("year")["pnl"].sum().items()
    }


def run_experiment(
    strategy_id: str,
    params: dict,
    first_year: int,
    last_year: int,
    spread: float = 0.30,
    slippage: float = 0.05,
    quantity: float = 1.0,
    capital: float = 10_000.0,
    run_id: str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{stamp}_{strategy_id}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        write_status(run_dir, "loading", message="Loading data")
        data = load_development(first_year, last_year)

        def factory():
            return registry.build_strategy(strategy_id, params)

        factory()  # surfaces constructor / import errors early

        write_status(run_dir, "checking", message="Checking for look-ahead")
        problems = check_no_lookahead(factory, data.iloc[:LOOKAHEAD_SAMPLE])
        if problems:
            raise ValueError(" ".join(problems))

        write_status(run_dir, "simulating", message="Simulating trades")
        backtester = Backtester(
            data=data,
            strategy=factory(),
            initial_capital=capital,
            execution=ExecutionModel(spread=spread, slippage=slippage),
            quantity=quantity,
        )
        trades = backtester.simulate()

        write_status(run_dir, "saving", message="Saving results")
        equity = build_equity_curve(capital, trades)
        summary = full_metrics(trades, equity, capital)

        summary.update({
            "strategy_id": strategy_id,
            "params": params,
            "run_id": run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "first_year": first_year,
            "last_year": last_year,
            "candles": len(data),
            "lookahead_check": "passed",
            "normalizer_warnings": backtester.warnings,
            "skipped_entries": backtester.skipped_entries,
            "yearly_pnl": yearly_pnl(trades) if trades else {},
            "config": {
                "strategy": strategy_id,
                "spread": spread,
                "slippage": slippage,
                "quantity": quantity,
                "capital": capital,
            },
        })

        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        equity.to_csv(run_dir / "equity.csv", index=False)
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(
            run_dir / "trades.csv", index=False
        )

        write_status(run_dir, "done", message="Done")
        return run_dir

    except Exception as error:
        write_status(run_dir, "failed", message=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("--params", default=None,
                        help="JSON object; default = saved strategy params")
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.params is None:
        params = registry.get_strategy(args.strategy_id).get("params", {})
    else:
        params = json.loads(args.params)

    started = time.time()

    try:
        run_dir = run_experiment(
            args.strategy_id, params, args.first_year, args.last_year,
            args.spread, args.slippage, args.quantity, args.capital,
            args.run_id,
        )
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)

    summary = json.loads((run_dir / "summary.json").read_text())

    print(f"Finished in {time.time() - started:.1f}s")
    print()
    print("=== Summary ===")
    for key, value in summary.items():
        if key in ("config", "yearly_pnl", "params", "normalizer_warnings"):
            continue
        if isinstance(value, float):
            value = round(value, 4)
        print(f"{key:20s} {value}")

    if summary.get("yearly_pnl"):
        print()
        print("=== PnL by year (price units x quantity) ===")
        for year, value in summary["yearly_pnl"].items():
            print(f"{year}  {value:>10.2f}")

    print()
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
