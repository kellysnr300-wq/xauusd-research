from pathlib import Path

import pandas as pd

from src.resample import resample_to_5m
from src.validation import validate_ohlc


def process_file(input_path: str | Path, output_path: str | Path):
    """Convert a Dukascopy 1-minute CSV into validated 5-minute data."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    data = pd.read_csv(input_path)

    processed = resample_to_5m(data)

    errors = validate_ohlc(processed)

    if errors:
        raise ValueError(
            f"Processed data failed validation: {errors}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        output_path,
        index=False,
    )

    return processed


if __name__ == "__main__":
    input_file = (
        "data/raw/"
        "XAU-USD_1Minute_BID_2026-10-02_00_00-23_59_Etc_UTC.csv"
    )

    output_file = (
        "data/processed/"
        "XAUUSD_5m_BID_2026-10-02.csv"
    )

    result = process_file(
        input_file,
        output_file,
    )

    print("Processed:", input_file)
    print("Saved:", output_file)
    print("Rows:", len(result))
    print("Columns:", list(result.columns))
