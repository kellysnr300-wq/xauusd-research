"""Buy-and-hold benchmark: what holding 1 oz would have done.

Usage:
    python -m src.benchmark 2016 2025

A strategy that earns less than this, with more risk, added nothing.
Figures are in dollars per ounce, like the backtest P&L (quantity 1).
"""

import sys

from src.dataset import load_development
from src.timeframes import resample_ohlc


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.benchmark <first_year> <last_year>")
        raise SystemExit(1)

    first, last = int(sys.argv[1]), int(sys.argv[2])
    daily = resample_ohlc(load_development(first, last), "1d")
    daily["year"] = daily["timestamp"].dt.year

    print(f"{'year':>6} {'start':>9} {'end':>9} {'change':>9}")
    for year, chunk in daily.groupby("year"):
        start, end = chunk["open"].iloc[0], chunk["close"].iloc[-1]
        print(f"{year:>6} {start:>9.2f} {end:>9.2f} {end - start:>+9.2f}")

    start, end = daily["open"].iloc[0], daily["close"].iloc[-1]
    peak = daily["close"].cummax()
    drawdown = (peak - daily["close"]).max()

    print()
    print(f"buy and hold {first}-{last}: {end - start:+.2f} $/oz "
          f"(from {start:.2f} to {end:.2f})")
    print(f"worst peak-to-trough drop on daily closes: {drawdown:.2f} $/oz")


if __name__ == "__main__":
    main()
