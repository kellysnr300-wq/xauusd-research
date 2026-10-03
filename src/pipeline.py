from pathlib import Path

from src.data import load_csv
from src.validation import validate_ohlc


def load_and_validate(path: str | Path):
    """Load OHLC data and return validation errors."""

    data = load_csv(path)
    errors = validate_ohlc(data)

    return data, errors
