from pathlib import Path

from src.process_data import process_file


def process_year(year: str):
    """Process all raw CSV files for a given year."""

    input_dir = Path("data/raw") / year
    output_dir = Path("data/processed") / year

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))

    if not files:
        print(f"No CSV files found in {input_dir}")
        return

    success = 0
    failed = 0

    for input_file in files:
        output_file = (
            output_dir
            / f"XAUUSD_5m_BID_{input_file.stem.split('_')[3]}.csv"
        )

        try:
            result = process_file(
                input_file,
                output_file,
            )

            print(
                f"OK: {input_file.name} -> "
                f"{len(result)} candles"
            )

            success += 1

        except Exception as error:
            print(
                f"FAILED: {input_file.name} -> {error}"
            )

            failed += 1

    print()
    print(f"Completed: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.process_year <year>")
        raise SystemExit(1)

    process_year(sys.argv[1])
