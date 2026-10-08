"""Parameter sweep with a built-in develop / check split.

Usage:
    python -m src.sweep gold_trend 2016 2025 --split 2021
    python -m src.sweep gold_trend 2016 2025 --split 2021 \
        --grid entry_n=40,55,80 exit_n=15,20,30

Every variant is simulated ONCE over the whole period (so indicators keep
their history) and each trade is assigned to the period in which it
CLOSED: "dev" up to the split year, "check" after it. The point is to
judge a strategy by how stable it is across variants, periods and sides,
not by its single best number.
"""

import argparse
import itertools
import json
import time

import numpy as np

from src import registry
from src.backtester import Backtester
from src.dataset import load_development
from src.execution import ExecutionModel
from src.timeframes import (
    TIMEFRAMES,
    max_entry_gap_for,
    resample_ohlc,
)

MAX_VARIANTS = 60


def parse_grid(items) -> dict:
    grid = {}
    for item in items or []:
        name, _, values = item.partition("=")
        if not name or not values:
            raise ValueError(f"Bad grid item '{item}'. Use name=v1,v2,v3")
        parsed = []
        for text in values.split(","):
            try:
                parsed.append(json.loads(text.strip().lower()
                                         if text.strip().lower() in ("true", "false")
                                         else text.strip()))
            except json.JSONDecodeError:
                parsed.append(text.strip())
        grid[name.strip()] = parsed
    return grid


def variants(base: dict, grid: dict):
    """The saved parameters first, then every grid combination."""
    yield "base", dict(base)
    if not grid:
        return
    names = list(grid)
    seen = {json.dumps(base, sort_keys=True)}
    for combo in itertools.product(*(grid[n] for n in names)):
        params = {**base, **dict(zip(names, combo))}
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        yield ", ".join(f"{n}={v}" for n, v in zip(names, combo)), params


def evaluate(data, factory, split_year, execution, max_gap) -> dict:
    """Simulate once and summarise by period, side and year."""

    backtester = Backtester(
        data, factory(), execution=execution, max_entry_gap=max_gap)
    trades = backtester.simulate()

    pnl = np.array([float(t.pnl) for t in trades])
    entry = np.array([float(t.entry_price) for t in trades])
    years = np.array([t.exit_time.year for t in trades])
    sides = np.array([t.side for t in trades])
    n = len(trades)
    # Each trade's result as % of its entry price: comparable across eras
    # (a $30 move means far more at $1,000 than at $4,000).
    pct = (pnl / entry * 100.0) if n else np.array([])

    wins = pnl[pnl > 0].sum() if n else 0.0
    losses = -pnl[pnl < 0].sum() if n else 0.0
    equity = np.cumsum(pnl) if n else np.array([0.0])
    drawdown = float((np.maximum.accumulate(equity) - equity).max())

    in_dev = years <= split_year
    yearly = {int(y): float(pnl[years == y].sum()) for y in sorted(set(years))}
    yearly_pct = {int(y): float(pct[years == y].sum()) for y in sorted(set(years))}
    total_pct = float(pct.sum()) if n else 0.0
    best_pct = max(yearly_pct.values()) if yearly_pct else 0.0

    return {
        "dev_pct": float(pct[in_dev].sum()) if n else 0.0,
        "check_pct": float(pct[~in_dev].sum()) if n else 0.0,
        "total_pct": total_pct,
        "long_pct": float(pct[sides == "long"].sum()) if n else 0.0,
        "short_pct": float(pct[sides == "short"].sum()) if n else 0.0,
        "ex_best_year_pct": total_pct - best_pct,
        "best_year": max(yearly_pct, key=yearly_pct.get) if yearly_pct else None,
        "yearly_pct": yearly_pct,
        "long_dev": float(pnl[in_dev & (sides == "long")].sum()) if n else 0.0,
        "short_dev": float(pnl[in_dev & (sides == "short")].sum()) if n else 0.0,
        "long_check": float(pnl[~in_dev & (sides == "long")].sum()) if n else 0.0,
        "short_check": float(pnl[~in_dev & (sides == "short")].sum()) if n else 0.0,
        "trades": n,
        "total": float(pnl.sum()) if n else 0.0,
        "dev": float(pnl[years <= split_year].sum()) if n else 0.0,
        "check": float(pnl[years > split_year].sum()) if n else 0.0,
        "long": float(pnl[sides == "long"].sum()) if n else 0.0,
        "short": float(pnl[sides == "short"].sum()) if n else 0.0,
        "profit_factor": (wins / losses) if losses > 0 else None,
        "win_rate": float((pnl > 0).mean() * 100) if n else 0.0,
        "max_drawdown": drawdown,
        "yearly": yearly,
        "warnings": backtester.warnings,
    }


def buy_and_hold(data, split_year) -> dict:
    year = data["timestamp"].dt.year
    def change(mask, percent=False):
        part = data[mask]
        if not len(part):
            return 0.0
        start, end = float(part["open"].iloc[0]), float(part["close"].iloc[-1])
        return (end / start - 1) * 100.0 if percent else end - start
    dev, check, whole = year <= split_year, year > split_year, year > 0
    return {
        "dev": change(dev), "check": change(check), "total": change(whole),
        "dev_pct": change(dev, True), "check_pct": change(check, True),
        "total_pct": change(whole, True),
    }


def fmt(value, width=9):
    return f"{value:>+{width}.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("--split", type=int, required=True,
                        help="last year of the development period")
    parser.add_argument("--timeframe", default="1d", choices=list(TIMEFRAMES))
    parser.add_argument("--grid", nargs="*", default=[])
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    args = parser.parse_args()

    if not args.first_year <= args.split < args.last_year:
        raise SystemExit("--split must be at least first_year and before last_year")

    base = registry.get_strategy(args.strategy_id).get("params", {})
    grid = parse_grid(args.grid)
    combos = list(variants(base, grid))
    if len(combos) > MAX_VARIANTS:
        raise SystemExit(f"{len(combos)} variants is too many (max {MAX_VARIANTS}). "
                         "Fewer variants also means less chance of fooling yourself.")

    print("Loading data...", flush=True)
    started = time.time()
    data = resample_ohlc(load_development(args.first_year, args.last_year), args.timeframe)
    print(f"{len(data):,} {args.timeframe} candles in {time.time() - started:.1f}s")

    execution = ExecutionModel(args.spread, args.slippage)
    gap = max_entry_gap_for(args.timeframe)

    rows, skipped = [], []
    for label, params in combos:
        try:
            rows.append((label, params, evaluate(
                data, lambda p=params: registry.build_strategy(args.strategy_id, p),
                args.split, execution, gap)))
        except ValueError as error:
            skipped.append((label, str(error)))

    hold = buy_and_hold(data, args.split)
    dev_label = f"dev<={args.split}"
    check_label = f"check>{args.split}"

    print()
    print(f"{'variant':<28}{'trades':>7}{dev_label:>13}{check_label:>13}"
          f"{'total':>9}{'long':>9}{'short':>9}{'PF':>6}{'maxDD':>8}")
    for label, _, r in rows:
        pf = "-" if r["profit_factor"] is None else f"{r['profit_factor']:.2f}"
        print(f"{label[:27]:<28}{r['trades']:>7}{fmt(r['dev'], 13)}{fmt(r['check'], 13)}"
              f"{fmt(r['total'])}{fmt(r['long'])}{fmt(r['short'])}{pf:>6}{r['max_drawdown']:>8.0f}")
    print(f"{'buy and hold (1 oz)':<28}{'':>7}{fmt(hold['dev'], 13)}{fmt(hold['check'], 13)}{fmt(hold['total'])}")

    print()
    print("Same results as % of price (sum of each trade's % return; fair across eras)")
    print(f"{'variant':<28}{dev_label:>13}{check_label:>13}{'total':>9}"
          f"{'long':>9}{'short':>9}{'no best yr':>12}")
    for label, _, r in rows:
        print(f"{label[:27]:<28}{fmt(r['dev_pct'], 13)}{fmt(r['check_pct'], 13)}"
              f"{fmt(r['total_pct'])}{fmt(r['long_pct'])}{fmt(r['short_pct'])}"
              f"{fmt(r['ex_best_year_pct'], 12)}")
    print(f"{'buy and hold':<28}{fmt(hold['dev_pct'], 13)}{fmt(hold['check_pct'], 13)}"
          f"{fmt(hold['total_pct'])}")
    print("('no best yr' = total % without the single best year)")

    for label, reason in skipped:
        print(f"skipped {label}: {reason}")

    results = [r for _, _, r in rows]
    both = sum(1 for r in results if r["dev"] > 0 and r["check"] > 0)
    print()
    print(f"variants tried:                    {len(results)}")
    print(f"positive in BOTH periods:          {both} of {len(results)}")
    print(f"positive on BOTH sides (long+short): "
          f"{sum(1 for r in results if r['long'] > 0 and r['short'] > 0)} of {len(results)}")
    print(f"median total P&L across variants:  {np.median([r['total'] for r in results]):+.0f} $/oz")
    best = max(results, key=lambda r: r["total"])
    print(f"best single variant total:         {best['total']:+.0f} $/oz "
          "(the best of many is flattered by luck: trust the median)")

    base_result = results[0]
    print()
    print("Base variant, side by period ($/oz):")
    print(f"  {dev_label:<14} long {base_result['long_dev']:>+7.0f}   short {base_result['short_dev']:>+7.0f}")
    print(f"  {check_label:<14} long {base_result['long_check']:>+7.0f}   short {base_result['short_check']:>+7.0f}")
    print()
    print("Base variant by year (by exit year):      $/oz    % of price")
    for year, value in base_result["yearly"].items():
        print(f"  {year}  {value:>+8.0f}   {base_result['yearly_pct'][year]:>+8.1f}%")
    for note in base_result["warnings"]:
        print("note:", note)


if __name__ == "__main__":
    main()
