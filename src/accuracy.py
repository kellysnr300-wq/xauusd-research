"""How much should a backtest result be trusted?

Two kinds of uncertainty are measured separately:

* ENGINE uncertainty - candles hide what happened inside them. Exits that
  touched both stop and target are settled with 1-minute data where
  possible. Whatever stays unknown is bracketed by re-running the same
  signals under a pessimistic and an optimistic assumption.
* STATISTICAL uncertainty - a win rate measured on few trades is noisy.
  A Wilson 95% interval quantifies that.
"""

import copy
import math


def wilson_interval(wins: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a win rate (fractions, or None)."""
    if n <= 0:
        return None, None
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(trades) -> dict:
    n = len(trades)
    pnls = [float(t.pnl) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    return {
        "trade_count": n,
        "wins": wins,
        "win_rate": (wins / n * 100.0) if n else 0.0,
        "total_pnl": sum(pnls),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
    }


def verdict(spread_points: float) -> str:
    """Engine uncertainty: how far apart the scenarios are (win-rate points)."""
    if spread_points <= 2.0:
        return "tight"
    if spread_points <= 5.0:
        return "moderate"
    return "wide"


def sample_label(n: int) -> str:
    """Statistical uncertainty: is the number of trades enough?"""
    if n < 30:
        return "too few trades"
    if n < 100:
        return "small sample"
    return "adequate sample"


def accuracy_report(backtester, resolver=None) -> dict:
    """Bracket a finished run. `backtester` must have run simulate()."""

    central = summarize(backtester.trades)
    meta = list(backtester.trade_meta)
    ambiguous = sum(1 for m in meta if m["ambiguous"])
    resolved = sum(1 for m in meta if m["ambiguous"] and m["resolved"])

    scenarios = {}
    for name, ambiguity, ask in (
        ("pessimistic", "stop_first", True),
        ("optimistic", "target_first", False),
    ):
        clone = copy.copy(backtester)
        clone.ambiguity = ambiguity
        clone.ask_triggers = ask
        scenarios[name] = summarize(clone.simulate(reuse_signals=True))
    scenarios["central"] = central

    rates = [s["win_rate"] for s in scenarios.values()]
    pnls = [s["total_pnl"] for s in scenarios.values()]
    spread = max(rates) - min(rates)

    ci_lo, ci_hi = wilson_interval(central["wins"], central["trade_count"])
    pess, opt = scenarios["pessimistic"], scenarios["optimistic"]
    band_lo = wilson_interval(pess["wins"], pess["trade_count"])[0]
    band_hi = wilson_interval(opt["wins"], opt["trade_count"])[1]

    report = {
        "verdict": verdict(spread),
        "sample": sample_label(central["trade_count"]),
        "scenarios": scenarios,
        "win_rate_engine_range": [min(rates), max(rates)],
        "win_rate_ci95": [None if ci_lo is None else ci_lo * 100,
                          None if ci_hi is None else ci_hi * 100],
        "win_rate_band": [None if band_lo is None else band_lo * 100,
                          None if band_hi is None else band_hi * 100],
        "pnl_range": [min(pnls), max(pnls)],
        "ambiguous_exits": ambiguous,
        "resolved_with_1m": resolved,
        "unresolved": ambiguous - resolved,
        "scenario_notes": {
            "pessimistic": "unresolved exits go against you; short stops and "
                           "targets trigger on the ask price (bid + spread)",
            "optimistic": "unresolved exits go in your favour",
        },
    }

    if resolver is not None:
        report["one_minute_data"] = {
            "days_loaded": len(resolver.loaded_days),
            "days_missing": len(resolver.missing_days),
            "queries": resolver.queries,
            "answered": resolver.answered,
        }

    return report
