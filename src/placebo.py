"""Placebo test: does a strategy's TIMING beat random timing?

The strategy's own orders are replayed at random times with the same
geometry (side, distance from the market, stop size, R target, lifetime),
the same costs and the same engine. If the real result is not clearly
better than what random timing earns, the strategy's logic adds nothing.

Event strategies (an `entry` column): each block of consecutive orders is
moved to the SAME New York clock time on another day (up to
`max_shift_days` away), so the killzone mix and daylight saving stay
matched. State strategies (a `signal` column): the position series is
rotated by a random amount, which keeps long / short / flat durations.

Usage:
    python -m src.placebo smc_limit 2016 2025 --timeframe 5m --runs 50
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import registry
from src.accuracy import summarize
from src.backtester import Backtester
from src.dataset import load_development
from src.execution import ExecutionModel
from src.normalize import normalize_output
from src.timeframes import TIMEFRAMES, max_entry_gap_for, resample_ohlc

NEW_YORK = "America/New_York"
PRICE_COLUMNS = ("entry_price", "stop_price", "target_price", "cancel_beyond")
RESULTS_ROOT = Path("results")


class Replay:
    """A strategy that just returns precomputed signals."""

    def __init__(self, frame):
        self.frame = frame

    def generate_signals(self, data):
        return self.frame.copy()


def _stamps(data):
    return pd.DatetimeIndex(data["timestamp"]).astype("datetime64[ns, UTC]")


def transplant_events(signals, data, rng, max_shift_days=90, tries=8):
    """Move blocks of orders to the same New York clock time on other days.

    Returns (new_signals, moved, total_rows).
    """

    n = len(data)
    entry = signals["entry"].to_numpy()
    rows = np.flatnonzero(entry != 0)
    out = pd.DataFrame(index=signals.index)
    out["entry"] = np.zeros(n, dtype=int)
    for column in signals.columns:
        if column != "entry":
            out[column] = np.nan
    if len(rows) == 0:
        return out, 0, 0

    stamps = _stamps(data)
    wall = stamps.tz_convert(NEW_YORK).tz_localize(None)
    close = data["close"].to_numpy(dtype=float)

    block = np.concatenate([[0], np.cumsum(np.diff(rows) > 1)])
    blocks = int(block[-1]) + 1

    def target_index(offset_days):
        shifted = wall[rows] + pd.to_timedelta(offset_days, unit="D")
        shifted = shifted.tz_localize(
            NEW_YORK, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        shifted = shifted.astype("datetime64[ns, UTC]")
        return stamps.get_indexer(shifted)

    # Several candidate offsets per block; keep the one that lands on the
    # most existing candles (weekends and holidays have none).
    candidates = (rng.integers(1, max_shift_days + 1, (blocks, tries))
                  * rng.choice([-1, 1], (blocks, tries)))
    best = np.zeros(blocks, dtype=int)
    best_count = np.full(blocks, -1.0)
    for k in range(tries):
        landed = target_index(candidates[block, k]) >= 0
        counts = np.bincount(block, weights=landed, minlength=blocks)
        better = counts > best_count
        best[better], best_count[better] = candidates[better, k], counts[better]

    destination = target_index(best[block])
    keep = (destination >= 0) & (destination < n)
    src, dst = rows[keep], destination[keep]

    out.loc[out.index[dst], "entry"] = entry[src]
    for column in signals.columns:
        if column == "entry":
            continue
        values = signals[column].to_numpy(dtype=float)
        if column in PRICE_COLUMNS:
            moved = close[dst] + (values[src] - close[src])
        else:
            moved = values[src]
        array = out[column].to_numpy(dtype=float).copy()
        array[dst] = moved
        out[column] = array
    return out, int(keep.sum()), len(rows)


def rotate_state(signals, data, rng):
    """Rotate a position series by a random amount (at least 10% of it)."""

    n = len(data)
    if n < 20:
        return signals.copy(), n, n
    close = data["close"].to_numpy(dtype=float)
    k = int(rng.integers(n // 10, n - n // 10))
    source = (np.arange(n) - k) % n
    out = pd.DataFrame(index=signals.index)
    for column in signals.columns:
        values = signals[column].to_numpy(dtype=float)
        if column in PRICE_COLUMNS:
            out[column] = close + (values[source] - close[source])
        else:
            out[column] = values[source]
    if "signal" in out.columns:
        out["signal"] = out["signal"].fillna(0).astype(int)
    return out, n, n


def run_placebo(data, factory, runs=50, seed=1, execution=None, max_gap=None,
                max_shift_days=90, progress=None):
    execution = execution or ExecutionModel()

    def simulate(frame):
        backtester = Backtester(
            data, Replay(frame), execution=execution,
            max_entry_gap=max_gap or pd.Timedelta(minutes=30))
        return summarize(backtester.simulate())

    normalized = normalize_output(factory().generate_signals(data.copy()), data)
    signals, mode = normalized.frame, normalized.mode
    real = simulate(signals)

    placebo, moved_fraction = [], []
    started = time.time()
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(runs)):
        rng = np.random.default_rng(child)
        if mode == "event":
            frame, moved, total = transplant_events(signals, data, rng, max_shift_days)
        else:
            frame, moved, total = rotate_state(signals, data, rng)
        placebo.append(simulate(frame))
        moved_fraction.append(moved / total if total else 0.0)
        if progress:
            done = i + 1
            progress(done, runs, (time.time() - started) / done * (runs - done))

    return build_report(real, placebo, mode, moved_fraction, max_shift_days)


def _avg(s):
    return s["total_pnl"] / s["trade_count"] if s["trade_count"] else float("nan")


def build_report(real, placebo, mode, moved_fraction=None, max_shift_days=90):
    valid = [p for p in placebo if p["trade_count"] > 0]
    avgs = np.array([_avg(p) for p in valid])
    totals = np.array([p["total_pnl"] for p in valid])
    real_avg = _avg(real)
    n = len(valid)

    report = {
        "mode": mode,
        "runs": len(placebo),
        "valid_runs": n,
        "max_shift_days": max_shift_days,
        "real": {**real, "avg_pnl": real_avg},
        "placebo_trades_mean": float(np.mean([p["trade_count"] for p in valid])) if n else 0.0,
        "placebo_win_rate_mean": float(np.mean([p["win_rate"] for p in valid])) if n else 0.0,
        "orders_moved_fraction": float(np.mean(moved_fraction)) if moved_fraction else None,
    }

    if n == 0 or real["trade_count"] == 0:
        report["verdict"] = "no trades to compare"
        return report

    report["placebo_avg"] = {
        "mean": float(avgs.mean()), "std": float(avgs.std(ddof=1)) if n > 1 else 0.0,
        "p05": float(np.percentile(avgs, 5)), "p50": float(np.percentile(avgs, 50)),
        "p95": float(np.percentile(avgs, 95)),
    }
    report["placebo_total"] = {
        "mean": float(totals.mean()), "p05": float(np.percentile(totals, 5)),
        "p95": float(np.percentile(totals, 95)),
    }
    std = report["placebo_avg"]["std"]
    report["z_score"] = float((real_avg - avgs.mean()) / std) if std > 0 else None
    # one-sided, with the +1 correction (never exactly zero)
    report["p_better"] = float((1 + np.sum(avgs >= real_avg)) / (n + 1))
    report["p_worse"] = float((1 + np.sum(avgs <= real_avg)) / (n + 1))
    report["edge_over_random_per_trade"] = float(real_avg - avgs.mean())

    if real["trade_count"] < 30:
        report["verdict"] = "too few trades to judge"
    elif report["p_better"] <= 0.05:
        report["verdict"] = "BETTER than random timing"
    elif report["p_worse"] <= 0.05:
        report["verdict"] = "WORSE than random timing"
    else:
        report["verdict"] = "INDISTINGUISHABLE from random timing"
    return report


def print_report(report):
    real = report["real"]
    print()
    print(f"mode: {report['mode']} | runs: {report['valid_runs']}/{report['runs']} | "
          f"orders moved: {report['orders_moved_fraction'] or 0:.0%} "
          f"(up to +-{report['max_shift_days']} days, same NY clock time)")
    print(f"real       {real['trade_count']:>6} trades  total {real['total_pnl']:>+9.1f}  "
          f"avg {real['avg_pnl']:>+7.3f}/trade  win {real['win_rate']:.1f}%")
    if "placebo_avg" not in report:
        print("verdict:", report["verdict"])
        return
    pa, pt = report["placebo_avg"], report["placebo_total"]
    print(f"placebo    {report['placebo_trades_mean']:>6.0f} trades  total {pt['mean']:>+9.1f}  "
          f"avg {pa['mean']:>+7.3f}/trade  win {report['placebo_win_rate_mean']:.1f}%")
    print(f"placebo avg/trade 5%-95% range: {pa['p05']:+.3f} to {pa['p95']:+.3f}")
    z = report["z_score"]
    print(f"real minus placebo: {report['edge_over_random_per_trade']:+.3f} per trade"
          + (f"  (z = {z:+.2f})" if z is not None else ""))
    print(f"p (random timing >= real): {report['p_better']:.3f}   "
          f"p (random timing <= real): {report['p_worse']:.3f}")
    print(f"verdict: {report['verdict']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("--timeframe", default="5m", choices=list(TIMEFRAMES))
    parser.add_argument("--params", default=None)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-shift-days", type=int, default=90)
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    args = parser.parse_args()

    if not 5 <= args.runs <= 1000:
        raise SystemExit("--runs must be between 5 and 1000")

    params = (registry.get_strategy(args.strategy_id).get("params", {})
              if args.params is None else json.loads(args.params))

    print("Loading data...", flush=True)
    data = resample_ohlc(load_development(args.first_year, args.last_year), args.timeframe)
    print(f"{len(data):,} {args.timeframe} candles")

    def progress(done, total, eta):
        print(f"  placebo {done}/{total}  (about {eta / 60:.1f} min left)", flush=True)

    print("Running the real strategy and the placebos...", flush=True)
    report = run_placebo(
        data, lambda: registry.build_strategy(args.strategy_id, params),
        runs=args.runs, seed=args.seed,
        execution=ExecutionModel(args.spread, args.slippage),
        max_gap=max_entry_gap_for(args.timeframe),
        max_shift_days=args.max_shift_days, progress=progress)
    print_report(report)

    RESULTS_ROOT.mkdir(exist_ok=True)
    path = RESULTS_ROOT / f"placebo_{datetime.now():%Y%m%d_%H%M%S}_{args.strategy_id}.json"
    report.update({"strategy_id": args.strategy_id, "params": params,
                   "first_year": args.first_year, "last_year": args.last_year,
                   "timeframe": args.timeframe, "seed": args.seed})
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
