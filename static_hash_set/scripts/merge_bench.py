#!/usr/bin/env python3
"""Replace one method's rows in a combined benchmark CSV with a fresh run."""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("combined", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("suffix")
    parser.add_argument("--rename", nargs=2, action="append", default=[])
    args = parser.parse_args()
    combined_path, new_path = args.combined, args.new
    label_suffix = args.suffix
    renames = dict(args.rename)
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
        row["h"] = renames.get(row["h"], row["h"]) + label_suffix
    # Preserve sizes, loads and thread/mode combinations absent from this run.
    def key(row):
        return tuple(row[column] for column in ("h", "alpha", "n", "threads", "metric"))
    replaced = {key(row) for row in incoming}
    rows = [row for row in existing if key(row) not in replaced]
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
