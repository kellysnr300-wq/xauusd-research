"""Audit data folders for misplaced, duplicate, or OOS files.

Usage:
    python -m src.check_layout          # report only
    python -m src.check_layout --fix    # apply safe fixes

Safe fixes:
  - misplaced file, no canonical copy  -> move to canonical path
  - misplaced file, identical copy     -> delete the extra copy
  - out-of-sample file                 -> move to data/quarantine_oos/
Conflicts (same date, different content) are never touched.
"""

from collections import defaultdict
import hashlib
from pathlib import Path
import shutil
import sys

from src.paths import (
    PROCESSED_ROOT,
    QUARANTINE_ROOT,
    RAW_ROOT,
    is_oos,
    parse_date,
    processed_path,
    raw_path,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path, kind: str, canonical, fix: bool):
    issues = 0
    good_files = []

    for path in sorted(root.rglob("*.csv")):
        day = parse_date(path.name)

        if day is None:
            print(f"UNKNOWN NAME : {path}")
            issues += 1
            continue

        if is_oos(day):
            print(f"OOS FILE     : {path}")
            issues += 1
            if fix:
                dest = QUARANTINE_ROOT / kind / path.name
                if dest.exists():
                    print(f"  CONFLICT, not moved: {dest} exists")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest))
                    print(f"  moved -> {dest}")
            continue

        dest = canonical(day)

        if path == dest:
            good_files.append(path)
            continue

        issues += 1

        if not dest.exists():
            print(f"MISPLACED    : {path} -> {dest}")
            if fix:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
                print("  moved")
                good_files.append(dest)
        elif file_hash(path) == file_hash(dest):
            print(f"DUPLICATE    : {path} (identical to {dest})")
            if fix:
                path.unlink()
                print("  deleted extra copy")
        else:
            print(f"CONFLICT     : {path} differs from {dest}")

    # Same content under different dates (e.g. downloaded twice).
    by_hash = defaultdict(list)
    for path in good_files:
        if path.exists() and path.stat().st_size > 200:
            by_hash[file_hash(path)].append(path)

    for paths in by_hash.values():
        if len(paths) > 1:
            issues += 1
            print("SAME CONTENT : " + ", ".join(str(p) for p in paths))

    return issues


def main():
    fix = "--fix" in sys.argv

    total = 0
    total += audit(RAW_ROOT, "raw", raw_path, fix)
    total += audit(PROCESSED_ROOT, "processed", processed_path, fix)

    print()
    if total == 0:
        print("Layout OK: no issues.")
    elif fix:
        print(f"{total} issue(s) handled. Re-run without --fix to confirm.")
    else:
        print(f"{total} issue(s) found. Re-run with --fix to apply safe fixes.")

    raise SystemExit(1 if total and not fix else 0)


if __name__ == "__main__":
    main()
