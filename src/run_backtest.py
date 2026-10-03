"""Run a strategy on the development data and save the results.

Usage:
    python -m src.run_backtest ma_cross 2016 2025
    python -m src.run_backtest ma_cross 2016 2025 --fast 10 --slow 40
    python -m src.run_backtest ma_cross 2025 2025 --spread 0.5

Results are saved in results/<timestamp>_<strategy>/ (git-ignored).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd

from src.backtester import Backtester
from src.dataset import load_development
from src.equity import build_equity_curve
from src.execution import ExecutionModel
from src.metrics import calculate_metrics
from src.strategy_ma import MovingAverageCross

RESULTS_ROOT = Path("results")


def build_strategy(args):
    if args.strategy == "ma_cross":
        return MovingAverageCross(fast=args.fast, slow=args.slow)
    raise ValueError(f"Unknown strategy: {args.strategy}")


def full_metrics(trades, equity, initial_capital):
    summary = calculate_metrics(trades, initial_capital)

    if not trades:
        return summary

    pnls = pd.Series([float(t.pnl) for t in trades])
    gross_win = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())

    durations = pd.Series(
        [t.exit_time - t.entry_time for t in trades]
    )

    summary["profit_factor"] = (
        gross_win / gross_loss if gross_loss > 0 else float("inf")
    )
    summary["max_drawdown"] = float(equity["drawdown"].max())
    summary["max_drawdown_pct"] = (
        summary["max_drawdown"] / initial_capital * 100.0
    )
    summary["avg_trade_duration"] = str(durations.mean())
    summary["long_trades"] = int(sum(t.side == "long" for t in trades))
    summary["short_trades"] = int(sum(t.side == "short" for t in trades))

    return summary


def yearly_pnl(trades) -> pd.Series:
    rows = pd.DataFrame({
        "year": [t.exit_time.year for t in trades],
        "pnl": [float(t.pnl) for t in trades],
    })
    return rows.groupby("year")["pnl"].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["ma_cross"])
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args()

    print("Loading data...", flush=True)
    started = time.time()
    data = load_development(args.first_year, args.last_year)
    print(f"Loaded {len(data):,} candles in {time.time() - started:.1f}s")

    strategy = build_strategy(args)
    execution = ExecutionModel(spread=args.spread, slippage=args.slippage)

    backtester = Backtester(
        data=data,
        strategy=strategy,
        initial_capital=args.capital,
        execution=execution,
        quantity=args.quantity,
    )

    print("Simulating...", flush=True)
    started = time.time()
    trades = backtester.simulate()
    print(f"Simulated in {time.time() - started:.1f}s")

    equity = build_equity_curve(args.capital, trades)
    summary = full_metrics(trades, equity, args.capital)

    summary["config"] = vars(args)
    summary["candles"] = len(data)

    print()
    print("=== Summary ===")
    for key, value in summary.items():
        if key == "config":
            continue
        if isinstance(value, float):
            value = round(value, 4)
        print(f"{key:20s} {value}")

    if trades:
        print()
        print("=== PnL by year (price units x quantity) ===")
        print(yearly_pnl(trades).round(2).to_string())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / f"{stamp}_{args.strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    equity.to_csv(out_dir / "equity.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(
        out_dir / "trades.csv", index=False
    )

    print()
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
