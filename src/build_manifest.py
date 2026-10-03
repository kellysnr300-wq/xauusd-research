from pathlib import Path

import pandas as pd


def build_manifest():
    rows = []

    raw_root = Path("data/raw")

    for year_dir in sorted(raw_root.iterdir()):
        if not year_dir.is_dir():
            continue

        for raw_file in sorted(year_dir.glob("*.csv")):
            try:
                data = pd.read_csv(raw_file)

                row_count = len(data)

                timestamp_column = "Etc/UTC"

                if timestamp_column in data.columns and row_count:
                    timestamps = pd.to_datetime(
                        data[timestamp_column],
                        utc=True,
                        errors="coerce",
                    )

                    first_timestamp = timestamps.min()
                    last_timestamp = timestamps.max()

                    date_text = raw_file.name.split("_")[3]

                    processed_file = (
                        Path("data/processed")
                        / year_dir.name
                        / f"XAUUSD_5m_BID_{date_text}.csv"
                    )

                    if processed_file.exists():
                        status = "processed"
                    else:
                        status = "downloaded"

                    notes = (
                        f"{first_timestamp} to "
                        f"{last_timestamp}"
                    )

                else:
                    date_text = raw_file.name.split("_")[3]
                    processed_file = ""
                    status = "empty"
                    notes = "Empty or invalid file"

                rows.append({
                    "date": date_text,
                    "year": year_dir.name,
                    "raw_file": raw_file.name,
                    "status": status,
                    "rows": row_count,
                    "processed_file": (
                        processed_file.name
                        if processed_file
                        else ""
                    ),
                    "notes": notes,
                })

            except Exception as error:
                rows.append({
                    "date": "",
                    "year": year_dir.name,
                    "raw_file": raw_file.name,
                    "status": "error",
                    "rows": 0,
                    "processed_file": "",
                    "notes": str(error),
                })

    manifest = pd.DataFrame(
        rows,
        columns=[
            "date",
            "year",
            "raw_file",
            "status",
            "rows",
            "processed_file",
            "notes",
        ],
    )

    output = Path("config/data_manifest.csv")

    manifest.to_csv(
        output,
        index=False,
    )

    print(f"Manifest saved: {output}")
    print(f"Files tracked: {len(manifest)}")


if __name__ == "__main__":
    build_manifest()
