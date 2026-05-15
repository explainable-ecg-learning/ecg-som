"""
Merge the four per-GPU ablation CSVs into a single ablation_results_all.csv
and ablation_results_all.json, preserving all columns.

Usage:
    python ablation/merge_ablation_results.py
"""
import csv
import json
import glob
import os

# Resolve paths relative to this script so it works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))

INPUT_PATTERN = os.path.join(_HERE, "ablation_results_gpu*.csv")
RETRY_PATTERN = os.path.join(_HERE, "ablation_results_retry_gpu*.csv")
OUTPUT_CSV    = os.path.join(_HERE, "ablation_results_all.csv")
OUTPUT_JSON   = os.path.join(_HERE, "ablation_results_all.json")


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: str) -> None:
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


def main() -> None:
    files = sorted(glob.glob(INPUT_PATTERN))
    retry_files = sorted(glob.glob(RETRY_PATTERN))
    all_files = files + retry_files
    if not all_files:
        print(f"No files matching '{INPUT_PATTERN}' or '{RETRY_PATTERN}' found.")
        return

    all_rows: list[dict] = []
    for path in all_files:
        rows = read_csv(path)
        print(f"  {path}: {len(rows)} row(s)")
        all_rows.extend(rows)

    # Deduplicate: for each variant keep the last row that has a valid NMI,
    # falling back to the last row if none succeeded.
    seen_variants: dict[str, dict] = {}
    for row in all_rows:
        variant = row.get("variant", "")
        prev = seen_variants.get(variant)
        nmi = row.get("NMI", "")
        if prev is None:
            seen_variants[variant] = row
        elif nmi not in ("", None):
            # prefer successful rows
            seen_variants[variant] = row
    all_rows = list(seen_variants.values())

    write_csv(all_rows, OUTPUT_CSV)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"\nMerged {len(all_rows)} rows from {len(files)} file(s).")
    print(f"  CSV  → {OUTPUT_CSV}")
    print(f"  JSON → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
