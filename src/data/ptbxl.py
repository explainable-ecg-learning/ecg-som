"""PTB-XL dataset utilities and statistics."""
import numpy as np

from src.data.dataset import ECG_Dataset


def print_dataset_stats(ds, split="all"):
    """Print class distribution and demographic statistics for a split."""
    try:
        records = ds.get_data(split=split)
    except ValueError as e:
        print(f"Skipping stats for split '{split}': {e}")
        return

    print(f" DATASET STATISTICS | Split: {split.upper()} | Total: {len(records)}")

    y      = np.array([r.label for r in records])
    ages   = np.array([r.age   for r in records])
    sexes  = np.array([r.sex   for r in records])

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{'Class':<12} | {'Count':<8} | {'Percent':<8}")
    print("-" * 34)
    for label, count in zip(unique, counts):
        print(f"{label:<12} | {count:<8} | {count / len(records) * 100:.1f}%")

    multi = sum(1 for r in records if getattr(r, "extra_labels", []))
    print(f"\nRecords with >1 label: {multi} ({multi / len(records) * 100:.1f}%)")

    male_count   = int(np.sum(sexes == 0))
    female_count = int(np.sum(sexes == 1))
    print(f"\n--- Demographics ---")
    print(f"Age: {ages.mean():.1f} ± {ages.std():.1f} years  "
          f"(min={ages.min():.0f}, max={ages.max():.0f})")
    print(f"Sex: {male_count} Male ({male_count / len(records) * 100:.1f}%) / "
          f"{female_count} Female ({female_count / len(records) * 100:.1f}%)")


if __name__ == "__main__":
    fs = 100
    dataset_path = f"./data/ptbxl_{fs}_T-12ms.pkl"

    ds = ECG_Dataset(fs)
    ds.import_ptbxl(base_path="/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
    ds.save(dataset_path)

    ds = ECG_Dataset.load(dataset_path)
    for split in ("train", "val", "test"):
        print_dataset_stats(ds, split=split)
