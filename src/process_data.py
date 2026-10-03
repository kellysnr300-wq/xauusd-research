from pathlib import Path
import sys

from src.data import load_csv
from src.resample import resample_to_5m
from src.validation import validate_ohlc


def process_file(
    input_path: str | Path,
    output_path: str | Path,
):
    """Convert a Dukascopy 1-minute CSV into validated 5-minute data."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    data = load_csv(input_path)

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
    if len(sys.argv) != 3:
        print(
            "Usage: python -m src.process_data "
            "<input.csv> <output.csv>"
        )
        raise SystemExit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    result = process_file(
        input_file,
        output_file,
    )

    print("Processed:", input_file)
    print("Saved:", output_file)
    print("Rows:", len(result))
    print("Columns:", list(result.columns))
