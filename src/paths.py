"""Single source of truth for data file locations and names."""

from datetime import date
from pathlib import Path
import re

RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
QUARANTINE_ROOT = Path("data/quarantine_oos")

# Final untouched out-of-sample period starts here.
OOS_START = date(2026, 1, 1)

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_date(name: str) -> date | None:
    """Extract the YYYY-MM-DD date from a file name."""
    match = _DATE_PATTERN.search(name)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def is_oos(day: date) -> bool:
    return day >= OOS_START


def raw_name(day: date) -> str:
    return f"XAU-USD_1Minute_BID_{day.isoformat()}_00_00-23_59_Etc_UTC.csv"


def processed_name(day: date) -> str:
    return f"XAUUSD_5m_BID_{day.isoformat()}.csv"


def raw_path(day: date) -> Path:
    return RAW_ROOT / str(day.year) / raw_name(day)


def processed_path(day: date) -> Path:
    return PROCESSED_ROOT / str(day.year) / processed_name(day)


def assert_development(day: date) -> None:
    """Refuse to touch out-of-sample dates during development."""
    if is_oos(day):
        raise ValueError(
            f"{day} is in the out-of-sample period "
            f"(>= {OOS_START}). Refusing."
        )
