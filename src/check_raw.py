"""Quick integrity check for a single raw Dukascopy CSV.

Usage:
    python -m src.check_raw data/raw/2025/some_file.csv
"""

from pathlib import Path
import sys

import pandas as pd

from src.data import load_csv
from src.validation import validate_ohlc


def check_raw(path: str | Path) -> None:
    path = Path(path)

    print(f"Checking: {path}")
    print("-" * 50)

    if not path.exists():
        print("ERROR: File does not exist.")
        return

    if path.stat().st_size == 0:
        print("ERROR: File is empty.")
        return

    try:
        data = load_csv(path)
    except Exception as e:
        print(f"ERROR while loading: {e}")
        return

    print(f"Rows          : {len(data)}")
    print(f"Columns       : {list(data.columns)}")
    print(f"First timestamp: {data['timestamp'].iloc[0]}")
    print(f"Last timestamp : {data['timestamp'].iloc[-1]}")

    # Time span
    span = data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]
    print(f"Time span     : {span}")

    # Basic OHLC validation
    errors = validate_ohlc(data)

    if errors:
        print("\nOHLC problems found:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nOHLC validation: PASS")

    # Extra quick stats
    print("\nPrice summary (close):")
    print(f"  min  : {data['close'].min():.2f}")
    print(f"  max  : {data['close'].max():.2f}")
    print(f"  mean : {data['close'].mean():.2f}")

    print("-" * 50)
    print("Check finished.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.check_raw <path-to-csv>")
        raise SystemExit(1)

    check_raw(sys.argv[1])
