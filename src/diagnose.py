"""Explain why a strategy trades rarely, never, or slowly.

Usage:
    python -m src.diagnose smc_original                 # last two years
    python -m src.diagnose new 2023 2025 --timeframe 5m

It never needs the strategy's code. It measures:
  * speed: how run time grows with the amount of data;
  * output: how many orders, which kind, how big the stops are, how far
    resting orders sit from the market;
  * fate: what the engine did with every order (filled, expired unfilled,
    dropped because a trade was open, rejected as invalid, ...);
and turns the evidence into plain-English likely reasons.
"""

import argparse
import signal
import time

import numpy as np
import pandas as pd

from src import registry
from src.backtester import Backtester
from src.dataset import load_development
from src.execution import ExecutionModel
from src.normalize import StrategyOutputError, normalize_output
from src.placebo import Replay
from src.timeframes import TIMEFRAMES, max_entry_gap_for, resample_ohlc


class Timeout(Exception):
    pass


def _alarm(seconds):
    if not hasattr(signal, "SIGALRM"):
        return
    if seconds:
        def handler(signum, frame):
            raise Timeout()
        signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds or 0))


# ----------------------------------------------------------------------
# Speed
# ----------------------------------------------------------------------

def scaling_profile(factory, data, sizes=(5_000, 10_000, 20_000, 40_000),
                    call_timeout=120, escalate_below=20.0):
    """Time generate_signals on growing slices and fit time ~ size ** e."""

    points = []
    for size in sizes:
        if size > len(data):
            break
        sample = data.iloc[:size].copy()
        started = time.time()
        try:
            _alarm(call_timeout)
            try:
                factory().generate_signals(sample)
            finally:
                _alarm(0)
        except Timeout:
            points.append((size, float(call_timeout)))
            return _fit(points, len(data), timed_out=True)
        elapsed = time.time() - started
        points.append((size, elapsed))
        if elapsed > escalate_below:
            break
    return _fit(points, len(data))


def _fit(points, full_size, timed_out=False):
    result = {"points": points, "exponent": None, "projected_seconds": None,
              "timed_out": timed_out, "full_size": full_size}
    usable = [(s, t) for s, t in points if t > 0.05]
    if len(usable) >= 2 and usable[-1][1] > usable[0][1]:
        sizes = np.log([s for s, _ in usable])
        times = np.log([t for _, t in usable])
        exponent = float(np.polyfit(sizes, times, 1)[0])
        result["exponent"] = exponent
        size, seconds = points[-1]
        result["projected_seconds"] = float(seconds * (full_size / size) ** max(exponent, 1.0))
    elif points:
        size, seconds = points[-1]
        result["projected_seconds"] = float(seconds * full_size / size)
    return result


# ----------------------------------------------------------------------
# Output and fate
# ----------------------------------------------------------------------

def describe_output(frame, mode, data, execution):
    close = data["close"].to_numpy(dtype=float)
    candle_range = float(np.median((data["high"] - data["low"]).to_numpy(dtype=float)))
    cost = execution.spread + 2 * execution.slippage
    info = {"mode": mode, "candles": len(data), "median_candle_range": candle_range,
            "round_trip_cost": cost}

    if mode == "event":
        entry = frame["entry"].to_numpy()
        rows = np.flatnonzero(entry != 0)
        info.update({"order_rows": int(len(rows)),
                     "long": int((entry == 1).sum()), "short": int((entry == -1).sum())})
        # Runs of consecutive order rows are ONE setup re-posted every candle.
        if len(rows):
            info["episodes"] = int(1 + np.sum(np.diff(rows) > 1))
            info["repost_share"] = float(np.mean(np.diff(rows) == 1)) if len(rows) > 1 else 0.0
        else:
            info["episodes"], info["repost_share"] = 0, 0.0
        price = frame["entry_price"].to_numpy(dtype=float) if "entry_price" in frame else np.full(len(frame), np.nan)
        priced = rows[~np.isnan(price[rows])]
        info["priced_orders"] = int(len(priced))
        info["market_orders"] = int(len(rows) - len(priced))
        if len(priced):
            info["median_distance_from_market"] = float(np.median(np.abs(price[priced] - close[priced])))
            if "valid_for" in frame:
                life = frame["valid_for"].to_numpy(dtype=float)[priced]
                life = life[~np.isnan(life)]
                info["median_valid_for"] = float(np.median(life)) if len(life) else 1.0
            else:
                info["median_valid_for"] = 1.0

        ref = np.where(np.isnan(price), close, price)
        stop = np.full(len(frame), np.nan)
        if "stop_price" in frame:
            stop = np.abs(ref - frame["stop_price"].to_numpy(dtype=float))
        if "stop_distance" in frame:
            stop = np.where(np.isnan(stop), frame["stop_distance"].to_numpy(dtype=float), stop)
        stop = stop[rows]
        stop = stop[~np.isnan(stop)]
        if len(stop):
            info["median_stop"] = float(np.median(stop))
    else:
        signal_values = frame["signal"].to_numpy()
        info["position_changes"] = int((np.diff(signal_values, prepend=0) != 0).sum())
        info["time_in_market"] = float((signal_values != 0).mean())
        if "stop_distance" in frame:
            stops = frame["stop_distance"].dropna()
            if len(stops):
                info["median_stop"] = float(stops.median())
    return info


def engine_fate(data, frame, execution, max_gap):
    backtester = Backtester(data, Replay(frame), execution=execution, max_entry_gap=max_gap)
    trades = backtester.simulate()
    return {"trades": len(trades), "stats": dict(backtester.order_stats),
            "skipped_entries": backtester.skipped_entries,
            "warnings": list(backtester.warnings)}


# ----------------------------------------------------------------------
# Reasons
# ----------------------------------------------------------------------

def explain(speed, output, fate, years):
    reasons = []

    if speed and speed["timed_out"]:
        reasons.append(
            "VERY SLOW: generate_signals did not finish a "
            f"{speed['points'][-1][0]:,}-candle slice in {speed['points'][-1][1]:.0f} s. "
            "It cannot process ten years. Look for work that is redone for every "
            "candle (loops that re-scan history, repeated .iloc[:i] slices).")
    elif speed and speed["exponent"] is not None and speed["exponent"] > 1.5:
        minutes = (speed["projected_seconds"] or 0) / 60
        reasons.append(
            f"SLOW: run time grows like n^{speed['exponent']:.1f} (double the data = "
            f"{2 ** speed['exponent']:.1f}x the time). Projected for {speed['full_size']:,} "
            f"candles: about {minutes:,.0f} minutes. Replace per-candle re-scans with rolling/"
            "vectorised operations.")
    elif speed and (speed["projected_seconds"] or 0) > 600:
        reasons.append(
            f"SLOW: about {speed['projected_seconds'] / 60:.0f} minutes for "
            f"{speed['full_size']:,} candles, though it scales roughly linearly.")

    if output is None:
        return reasons

    stats, trades = fate["stats"], fate["trades"]
    posted = stats["posted"]

    if output["mode"] == "event" and output["order_rows"] == 0 or \
            output["mode"] == "state" and output["position_changes"] == 0:
        reasons.append("NEVER SIGNALS: the strategy produced no signals at all on this data. "
                       "Its conditions may never be true, or its filters stack up too far.")
        return reasons

    if output["mode"] == "event" and posted:
        share = lambda key: stats[key] / posted
        # Only RESTING orders can be one setup re-posted every candle; a
        # market entry on every candle is a stream of separate attempts.
        reposted = (output["repost_share"] > 0.5
                    and output["priced_orders"] > 0.5 * output["order_rows"])
        denominator = max(output["episodes"], 1) if reposted else posted
        unit = "setups" if reposted else "orders"

        if stats["filled"] / posted < 0.10:
            if share("invalid") >= 0.3:
                reasons.append(
                    f"ORDERS REJECTED AS INVALID: {stats['invalid']:,} of {posted:,} orders "
                    "have a stop or target on the wrong side of the entry price. Check the "
                    "sign of the stop for longs vs shorts.")
            if share("dropped_busy") >= 0.3:
                reasons.append(
                    f"ORDERS DROPPED, A TRADE WAS ALREADY OPEN: {stats['dropped_busy']:,} of "
                    f"{posted:,} market entries. The engine holds one position at a time, so a "
                    "strategy that signals constantly takes only a fraction of its signals.")
            if share("gap_blocked") >= 0.1:
                reasons.append(
                    f"ENTRIES BLOCKED BY MARKET GAPS: {stats['gap_blocked']:,} market orders "
                    "arrived just after a weekend, holiday or data gap and were skipped.")

        if share("expired_unfilled") >= 0.3 and stats["filled"] / denominator < 0.10:
            distance = output.get("median_distance_from_market")
            reasons.append(
                f"RESTING ORDERS NEVER REACH THEIR PRICE: only {stats['filled']:,} of "
                f"{denominator:,} {unit} were ever filled"
                + (f" (orders are re-posted every candle, {output['episodes']:,} separate setups)"
                   if reposted else "")
                + (f". They sit a median ${distance:.2f} from the market, against a typical "
                   f"{output['median_candle_range']:.2f} candle range, for "
                   f"{output.get('median_valid_for', 1):.0f} candle(s)." if distance is not None else ".")
                + " Move the price closer or let orders live longer (valid_for).")

    if trades and output.get("median_stop") is not None:
        ratio = output["median_stop"] / output["round_trip_cost"] if output["round_trip_cost"] else None
        if ratio is not None and ratio < 5:
            reasons.append(
                f"STOPS ARE SMALL AGAINST COSTS: median stop ${output['median_stop']:.2f} is only "
                f"{ratio:.1f}x the round-trip cost (${output['round_trip_cost']:.2f}). "
                "Costs alone will eat most of any edge.")

    if trades < 100 and years:
        reasons.append(
            f"TOO FEW TRADES TO JUDGE: {trades} trades in {years:.1f} years. "
            "A win rate or P&L from this few trades is mostly luck.")

    for warning in fate["warnings"]:
        reasons.append(f"NOTE: {warning}")
    return reasons


def diagnose(data, factory, execution=None, max_gap=None, sizes=(5_000, 10_000, 20_000, 40_000),
             call_timeout=120):
    execution = execution or ExecutionModel()
    max_gap = max_gap or pd.Timedelta(minutes=30)
    years = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).days / 365.25

    speed = scaling_profile(factory, data, sizes, call_timeout)
    report = {"speed": speed, "years": years, "output": None, "fate": None,
              "reasons": [], "error": None}

    if speed["timed_out"]:
        report["reasons"] = explain(speed, None, None, years)
        return report

    try:
        normalized = normalize_output(factory().generate_signals(data.copy()), data)
    except StrategyOutputError as error:
        report["error"] = f"The output cannot be used: {error}"
        report["reasons"] = [report["error"]]
        return report

    report["output"] = describe_output(normalized.frame, normalized.mode, data, execution)
    report["fate"] = engine_fate(data, normalized.frame, execution, max_gap)
    report["reasons"] = explain(speed, report["output"], report["fate"], years)
    return report


def print_report(report, name):
    speed, output, fate = report["speed"], report["output"], report["fate"]
    print(f"\nDiagnosis: {name}")
    print("\nSpeed (generate_signals):")
    for size, seconds in speed["points"]:
        print(f"  {size:>8,} candles  {seconds:>7.2f} s")
    if speed["exponent"] is not None:
        print(f"  grows like n^{speed['exponent']:.2f}; about "
              f"{(speed['projected_seconds'] or 0) / 60:.1f} min for {speed['full_size']:,} candles")

    if output:
        print(f"\nOutput ({output['mode']} mode, {output['candles']:,} candles, {report['years']:.1f} years):")
        if output["mode"] == "event":
            print(f"  order rows {output['order_rows']:,} (long {output['long']:,}, short {output['short']:,}); "
                  f"resting {output['priced_orders']:,}, market {output['market_orders']:,}")
            if output["repost_share"] > 0.5:
                print(f"  re-posted every candle: {output['episodes']:,} separate setups")
            if "median_distance_from_market" in output:
                print(f"  resting orders sit a median ${output['median_distance_from_market']:.2f} "
                      f"from the market, valid {output.get('median_valid_for', 1):.0f} candle(s)")
        else:
            print(f"  position changes {output['position_changes']:,}; "
                  f"in the market {output['time_in_market']:.0%} of the time")
        if "median_stop" in output:
            print(f"  median stop ${output['median_stop']:.2f} (candle range {output['median_candle_range']:.2f}, "
                  f"round-trip cost ${output['round_trip_cost']:.2f})")
    if fate:
        s = fate["stats"]
        print(f"\nWhat the engine did: {fate['trades']:,} trades")
        print(f"  posted {s['posted']:,} | filled {s['filled']:,} | expired unfilled {s['expired_unfilled']:,} | "
              f"cancelled {s['cancelled']:,} | dropped (trade open) {s['dropped_busy']:,} | "
              f"invalid {s['invalid']:,} | gap-blocked {s['gap_blocked']:,}")

    print("\nLikely reasons:")
    if report["reasons"]:
        for reason in report["reasons"]:
            print(f"  - {reason}")
    else:
        print("  none found: it signals, fills and runs at a reasonable speed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_id")
    parser.add_argument("first_year", type=int, nargs="?", default=2024)
    parser.add_argument("last_year", type=int, nargs="?", default=2025)
    parser.add_argument("--timeframe", default="5m", choices=list(TIMEFRAMES))
    parser.add_argument("--params", default=None)
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--call-timeout", type=int, default=120)
    args = parser.parse_args()

    import json
    params = (registry.get_strategy(args.strategy_id).get("params", {})
              if args.params is None else json.loads(args.params))

    print("Loading data...", flush=True)
    data = resample_ohlc(load_development(args.first_year, args.last_year), args.timeframe)
    print(f"{len(data):,} {args.timeframe} candles", flush=True)

    report = diagnose(
        data, lambda: registry.build_strategy(args.strategy_id, params),
        execution=ExecutionModel(args.spread, args.slippage),
        max_gap=max_entry_gap_for(args.timeframe), call_timeout=args.call_timeout)
    print_report(report, args.strategy_id)


if __name__ == "__main__":
    main()
