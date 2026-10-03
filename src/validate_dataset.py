from pathlib import Path

import pandas as pd

from src.validation import validate_ohlc


def load_processed_files(year: str) -> pd.DataFrame:
    """Load and combine all processed 5-minute files for a year."""

    directory = Path("data/processed") / year
    files = sorted(directory.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No processed CSV files found in {directory}"
        )

    frames = []

    for file in files:
        frame = pd.read_csv(file)
        frame["source_file"] = file.name
        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    return data


def validate_dataset(year: str):
    """Validate the combined processed dataset."""

    data = load_processed_files(year)

    errors = validate_ohlc(data)

    if errors:
        print("OHLC validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("OHLC validation: PASS")

    duplicate_timestamps = data["timestamp"].duplicated().sum()

    print(f"Rows: {len(data)}")
    print(f"Unique timestamps: {data['timestamp'].nunique()}")
    print(f"Duplicate timestamps: {duplicate_timestamps}")

    if data.empty:
        return

    data = data.sort_values("timestamp").reset_index(drop=True)

    first_timestamp = data["timestamp"].iloc[0]
    last_timestamp = data["timestamp"].iloc[-1]

    print(f"First timestamp: {first_timestamp}")
    print(f"Last timestamp: {last_timestamp}")

    gaps = data["timestamp"].diff()
    unexpected_gaps = gaps[
        gaps.notna()
        & (gaps != pd.Timedelta(minutes=5))
    ]

    print(
        f"Non-5-minute gaps: "
        f"{len(unexpected_gaps)}"
    )

    if len(unexpected_gaps) > 0:
        print("Gap examples:")

        for index in unexpected_gaps.head(10).index:
            previous_timestamp = data.loc[
                index - 1,
                "timestamp",
            ]

            current_timestamp = data.loc[
                index,
                "timestamp",
            ]

            gap = current_timestamp - previous_timestamp

            print(
                f"- {previous_timestamp} -> "
                f"{current_timestamp} "
                f"gap={gap}"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.validate_dataset <year>")
        raise SystemExit(1)

    validate_dataset(sys.argv[1])
