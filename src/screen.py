"""Run the placebo yardstick over every saved strategy and rank them.

Usage:
    python -m src.screen --list
    python -m src.screen 2016 2025 --timeframe 5m --runs 20
    python -m src.screen 2016 2025 --only smc_original,smc_basic
    python -m src.screen 2016 2025 --resume results/screen_2026....json

For each strategy, in order:
  1. build it with its saved parameters (errors are recorded, not fatal);
  2. look-ahead guard on real data: a cheating strategy is excluded;
  3. the placebo test (see docs/PLACEBO.md);
and one results file is rewritten after every strategy.
"""

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path

from src import registry
from src.dataset import load_development
from src.execution import ExecutionModel
from src.lookahead import check_no_lookahead
from src.placebo import run_placebo
from src.timeframes import TIMEFRAMES, max_entry_gap_for, resample_ohlc

GUARD_SAMPLE = 20_000
RESULTS_ROOT = Path("results")


class StrategyTimeout(Exception):
    pass


def _alarm(seconds):
    if not hasattr(signal, "SIGALRM"):
        return
    if seconds:
        def handler(signum, frame):
            raise StrategyTimeout()
        signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds or 0))


def evaluate_strategy(data, sid, runs, seed, execution, max_gap,
                      max_shift_days, timeout_s, min_trades):
    """One leaderboard row. Never raises."""

    row = {"id": sid, "status": "?", "seconds": None}
    started = time.time()

    try:
        _alarm(timeout_s)
        try:
            meta = registry.get_strategy(sid)
            params = meta.get("params", {})
            row["name"] = meta.get("name", sid)

            def factory():
                return registry.build_strategy(sid, params)

            factory()
            problems = check_no_lookahead(factory, data.iloc[:GUARD_SAMPLE])
            if problems:
                text = problems[0]
                row["status"] = ("LOOK-AHEAD (excluded)" if "LOOK-AHEAD" in text
                                 else f"invalid output: {text[:90]}")
                return row

            report = run_placebo(data, factory, runs=runs, seed=seed,
                                 execution=execution, max_gap=max_gap,
                                 max_shift_days=max_shift_days)
        finally:
            _alarm(0)

        real = report["real"]
        row.update({
            "trades": real["trade_count"], "win_rate": real["win_rate"],
            "total_pnl": real["total_pnl"], "avg_pnl": real["avg_pnl"],
            "mode": report["mode"],
        })
        if real["trade_count"] == 0:
            row["status"] = "no trades"
        elif "placebo_avg" not in report:
            row["status"] = report["verdict"]
        else:
            row.update({
                "placebo_avg": report["placebo_avg"]["mean"],
                "edge": report["edge_over_random_per_trade"],
                "z": report["z_score"], "p": report["p_better"],
                "p_worse": report["p_worse"], "verdict": report["verdict"],
            })
            row["status"] = ("too few trades" if real["trade_count"] < min_trades
                             else "evaluated")
    except StrategyTimeout:
        row["status"] = f"timeout (> {timeout_s // 60} min)"
    except KeyboardInterrupt:
        raise
    except Exception as error:
        row["status"] = f"error: {type(error).__name__}: {str(error)[:100]}"

    row["seconds"] = round(time.time() - started, 1)
    return row


def screen(data, ids, runs=20, seed=1, execution=None, max_gap=None,
           max_shift_days=90, timeout_s=1800, min_trades=100, done=None,
           on_row=None, log=print):
    execution = execution or ExecutionModel()
    rows = list(done or [])
    finished = {r["id"] for r in rows}
    todo = [i for i in ids if i not in finished]

    for number, sid in enumerate(todo, start=1):
        log(f"[{number}/{len(todo)}] {sid} ...")
        row = evaluate_strategy(data, sid, runs, seed, execution, max_gap,
                                max_shift_days, timeout_s, min_trades)
        rows.append(row)
        detail = f" p={row['p']:.3f}" if row.get("p") is not None else ""
        log(f"      {row['status']}{detail}  ({row['seconds']}s)")
        if on_row:
            on_row(rows)
    return rows


def leaderboard(rows) -> str:
    evaluated = sorted((r for r in rows if r["status"] == "evaluated"),
                       key=lambda r: r["p"])
    others = [r for r in rows if r["status"] != "evaluated"]
    lines = []

    if evaluated:
        lines.append(f"{'strategy':<34}{'trades':>7}{'win%':>6}{'avg/trade':>10}"
                     f"{'random':>8}{'edge':>8}{'p':>7}  verdict")
        for r in evaluated:
            lines.append(
                f"{r['id'][:33]:<34}{r['trades']:>7}{r['win_rate']:>6.1f}"
                f"{r['avg_pnl']:>+10.3f}{r['placebo_avg']:>+8.3f}{r['edge']:>+8.3f}"
                f"{r['p']:>7.3f}  {r['verdict']}")
    if others:
        lines.append("")
        lines.append("Not evaluated:")
        for r in others:
            extra = f" ({r['trades']} trades)" if r.get("trades") else ""
            lines.append(f"  {r['id'][:40]:<41}{r['status']}{extra}")

    flagged = [r for r in evaluated if r["p"] <= 0.10 and r["edge"] > 0]
    lines.append("")
    if flagged:
        lines.append("Worth a longer check (p <= 0.10, edge > 0): "
                     + ", ".join(r["id"] for r in flagged))
        lines.append("Rerun those alone with --runs 200 before believing them: "
                     "testing many strategies flags some by luck.")
    else:
        lines.append("No strategy clearly beats random timing.")
    lines.append("random = what the same orders earn at random times; "
                 "edge = real minus random, per trade.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first_year", type=int, nargs="?")
    parser.add_argument("last_year", type=int, nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--timeframe", default="5m", choices=list(TIMEFRAMES))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip", default="")
    parser.add_argument("--timeout-min", type=int, default=30)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--max-shift-days", type=int, default=90)
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    available = registry.list_strategies()

    if args.list:
        for s in available:
            print(f"{s['id']:<50} {s.get('type', ''):<12} {json.dumps(s.get('params', {}))[:60]}")
        return

    if args.first_year is None or args.last_year is None:
        raise SystemExit("Usage: python -m src.screen <first_year> <last_year> [options]")
    if not 5 <= args.runs <= 1000:
        raise SystemExit("--runs must be between 5 and 1000")

    ids = [s["id"] for s in available]
    if args.only:
        wanted = [i.strip() for i in args.only.split(",") if i.strip()]
        unknown = [i for i in wanted if i not in ids]
        if unknown:
            raise SystemExit(f"Unknown strategy ids: {unknown}. Try --list.")
        ids = wanted
    skip = {i.strip() for i in args.skip.split(",") if i.strip()}
    ids = [i for i in ids if i not in skip]

    done, out = [], None
    if args.resume:
        out = Path(args.resume)
        done = json.loads(out.read_text())["rows"]
    else:
        RESULTS_ROOT.mkdir(exist_ok=True)
        out = RESULTS_ROOT / f"screen_{datetime.now():%Y%m%d_%H%M%S}.json"

    config = {"first_year": args.first_year, "last_year": args.last_year,
              "timeframe": args.timeframe, "runs": args.runs, "seed": args.seed}

    def save(rows):
        out.write_text(json.dumps({"config": config, "rows": rows}, indent=2, default=str))

    print("Loading data...", flush=True)
    data = resample_ohlc(load_development(args.first_year, args.last_year), args.timeframe)
    print(f"{len(data):,} {args.timeframe} candles; {len(ids)} strategies; "
          f"{args.runs} placebo runs each; results file: {out}\n", flush=True)

    rows = screen(
        data, ids, runs=args.runs, seed=args.seed,
        execution=ExecutionModel(args.spread, args.slippage),
        max_gap=max_entry_gap_for(args.timeframe),
        max_shift_days=args.max_shift_days, timeout_s=args.timeout_min * 60,
        min_trades=args.min_trades, done=done, on_row=save,
        log=lambda text: print(text, flush=True))
    save(rows)
    print("\n" + leaderboard(rows))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
