"""Download Dukascopy 1-minute XAU/USD BID data via dukascopy-node and
store it as one raw file per UTC day, in the website-export format.

Usage:
    python -m src.download 2025-01 2025-01     # one month (test first)
    python -m src.download 2016-01 2025-12     # full development range

Rules:
  - Never overwrites existing raw files (mismatches are reported).
  - Refuses out-of-sample dates (>= 2026-01-01).
  - Finished months are logged in config/download_done.txt (resumable).
"""

from datetime import date
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from src.paths import OOS_START, assert_development, raw_path

DONE_FILE = Path("config/download_done.txt")
VOLUME_SCALE = 1_000_000  # dukascopy-node volume is in millions of units
MAX_ATTEMPTS = 3


def parse_month(text: str) -> date:
    year, month = text.split("-")
    return date(int(year), int(month), 1)


def next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def months_between(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current = next_month(current)


def load_done() -> set[str]:
    if not DONE_FILE.exists():
        return set()
    return {
        line.strip()
        for line in DONE_FILE.read_text().splitlines()
        if line.strip()
    }


def mark_done(key: str) -> None:
    DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DONE_FILE, "a") as handle:
        handle.write(key + "\n")


def fetch_month(month_start: date) -> pd.DataFrame:
    """Run dukascopy-node for one month and return its raw rows."""
    month_end = next_month(month_start)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        tmp = Path(tempfile.mkdtemp(dir=Path.home(), prefix=".dl_"))
        try:
            result = subprocess.run(
                [
                    "npx", "dukascopy-node",
                    "-i", "xauusd",
                    "-from", month_start.isoformat(),
                    "-to", month_end.isoformat(),
                    "-t", "m1",
                    "-f", "csv",
                    "-p", "bid",
                    "-v",
                    "-dir", str(tmp),
                ],
                capture_output=True,
                text=True,
            )
            files = list(tmp.glob("*.csv"))

            if result.returncode == 0 and files:
                return pd.read_csv(files[0], dtype=str)

            print(
                f"  attempt {attempt} failed: "
                f"{(result.stderr or result.stdout).strip()[-200:]}"
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        time.sleep(5)

    raise RuntimeError(f"Download failed for {month_start:%Y-%m}")


def to_raw_text(day_rows: pd.DataFrame) -> str:
    """Format one day's rows exactly like the website export."""
    lines = ["Etc/UTC,Open,High,Low,Close,Volume"]

    for row in day_rows.itertuples(index=False):
        stamp = row.ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        volume = int(round(float(row.volume) * VOLUME_SCALE))
        lines.append(
            f"{stamp},{row.open},{row.high},{row.low},{row.close},{volume}"
        )

    return "\n".join(lines) + "\n"


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def save_month(month_start: date, frame: pd.DataFrame) -> dict:
    stats = {"written": 0, "same": 0, "mismatch": 0}

    if frame.empty:
        print("  WARNING: no rows returned for this month")
        return stats

    frame = frame.copy()
    frame["ts"] = pd.to_datetime(
        frame["timestamp"].astype("int64"), unit="ms", utc=True
    )
    frame = frame.sort_values("ts").drop_duplicates("ts")
    frame["day"] = frame["ts"].dt.date

    month_end = next_month(month_start)
    frame = frame[
        (frame["day"] >= month_start)
        & (frame["day"] < month_end)
        & (frame["day"] < OOS_START)
    ]

    for day, rows in frame.groupby("day"):
        assert_development(day)
        path = raw_path(day)
        text = to_raw_text(rows)

        if path.exists():
            if normalize(path.read_text()) == normalize(text):
                stats["same"] += 1
            else:
                stats["mismatch"] += 1
                print(f"  MISMATCH with existing file (kept): {path}")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        stats["written"] += 1

    return stats


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.download <YYYY-MM> <YYYY-MM>")
        raise SystemExit(1)

    start = parse_month(sys.argv[1])
    end = parse_month(sys.argv[2])

    if end >= OOS_START:
        print(f"Refusing: range reaches out-of-sample period ({OOS_START}).")
        raise SystemExit(1)

    done = load_done()

    for month in months_between(start, end):
        key = f"{month:%Y-%m}"

        if key in done:
            print(f"{key}: already done, skipping")
            continue

        print(f"{key}: downloading...", flush=True)
        frame = fetch_month(month)
        stats = save_month(month, frame)
        mark_done(key)

        print(
            f"{key}: written={stats['written']} "
            f"identical={stats['same']} mismatch={stats['mismatch']}"
        )

    print("Finished. Now run: python -m src.check_layout")


if __name__ == "__main__":
    main()
