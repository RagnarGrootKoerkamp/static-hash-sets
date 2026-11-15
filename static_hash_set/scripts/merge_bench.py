#!/usr/bin/env python3
"""Replace one method's rows in a combined benchmark CSV with a fresh run."""

import csv
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: merge_bench.py COMBINED.csv NEW.csv LABEL_SUFFIX")

    combined_path, new_path = map(Path, sys.argv[1:3])
    label_suffix = sys.argv[3]
    with combined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)
    with new_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fieldnames:
            raise SystemExit("benchmark CSV columns do not match")
        incoming = list(reader)

    for row in incoming:
        row["h"] += label_suffix
    replaced = {row["h"] for row in incoming}
    rows = [row for row in existing if row["h"] not in replaced]
    rows.extend(incoming)

    with tempfile.NamedTemporaryFile(
        "w", newline="", dir=combined_path.parent, delete=False
    ) as f:
        temporary_path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, combined_path)


if __name__ == "__main__":
    main()
