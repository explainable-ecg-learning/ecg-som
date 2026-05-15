import numpy as np
from ECG_Dataset import ECG_Dataset
from draw_utils import draw_signal, draw_record


def print_dataset_stats(ds, split="all"):
    """
    Calculates and prints class distribution and demographic statistics
    for a specific split of the dataset.
    """
    # 1. Retrieve data for the requested split
    try:
        records = ds.get_data(split=split)
    except ValueError as e:
        print(f"Skipping stats for split '{split}': {e}")
        return

    # 2. Header
    print(f" DATASET STATISTICS | Split: {split.upper()} | Total: {len(records)}")

    # 3. Class Distribution
    # Transform integer labels back to strings (e.g., 0 -> 'NORM')
    y = np.array([record.label for record in records])
    ages = np.array([record.age for record in records])
    sexes = np.array([record.sex for record in records])

    # Transform integer labels to strings if label_encoder is fitted
    if hasattr(ds.label_encoder, 'classes_') and len(ds.label_encoder.classes_) > 0:
        y_encoded = ds.label_encoder.transform(y)
        y_labels = y
    else:
        y_labels = y

    unique, counts = np.unique(y_labels, return_counts=True)

    print(f"\n{'Class':<12} | {'Count':<8} | {'Percent':<8}")
    print("-" * 34)
    for label, count in zip(unique, counts):
        percent = (count / len(records)) * 100
        print(f"{label:<12} | {count:<8} | {percent:.1f}%")

    multi_label_count = sum(1 for record in records if getattr(record, "extra_labels", []))
    if len(records) > 0:
        multi_label_pct = (multi_label_count / len(records)) * 100
    else:
        multi_label_pct = 0.0
    print(f"\nRecords with >1 label: {multi_label_count} ({multi_label_pct:.1f}%)")

    # 4. Demographics
    print("\n--- Demographics ---")
    min_age = np.min(ages)
    max_age = np.max(ages)
    mean_age = np.mean(ages)
    std_age = np.std(ages)

    # 0 = Male, 1 = Female (based on ECG_Record logic)
    male_count = np.sum(sexes == 0)
    female_count = np.sum(sexes == 1)

    print(f"Min age: {min_age}")
    print(f"Max age: {max_age}")
    print(f"Age: {mean_age:.1f} ± {std_age:.1f} years")
    print(f"Sex: {male_count} Male ({male_count / len(records) * 100:.1f}%) / "
          f"{female_count} Female ({female_count / len(records) * 100:.1f}%)")


if __name__ == "__main__":
    fs = 100
    dataset_filename = f"ptbxl_{fs}_T-12ms.pkl"
    # 1. Load Dataset
    dataset = ECG_Dataset(fs)
    dataset.import_ptbxl(base_path="/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
    dataset.save("./data/" + dataset_filename)
    # 2. Read
    ds = ECG_Dataset.load("./data/" + dataset_filename)

    records = ds.get_data()

    print_dataset_stats(ds, split="train")
    print_dataset_stats(ds, split="val")
    print_dataset_stats(ds, split="test")

    # 3. Visualize one sample
    debug_samples = []
    idx = np.random.randint(0, len(ds))
    debug_samples.append(idx)
    for sample_id in debug_samples:
        record = ds.get_record_by_id(sample_id)
        draw_record(record)
        for beat in record.beats:
            draw_signal(beat, fs, '')
