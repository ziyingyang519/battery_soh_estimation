import csv
from collections import Counter
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_features_4p0_to_4p2_audit.csv"
)


with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))


sample_count = len(rows)
train_end = int(sample_count * 0.60)
validation_end = int(sample_count * 0.80)

splits = {
    "Train": rows[:train_end],
    "Validation": rows[train_end:validation_end],
    "Test": rows[validation_end:],
}


def interval_ids(split_rows):
    return {
        int(row["reference_interval_id"])
        for row in split_rows
    }


def print_split_summary(name, split_rows):
    ids = sorted(interval_ids(split_rows))
    print(f"{name}: {len(split_rows)} samples")
    print(
        f"  Reference intervals: {ids[0]} to {ids[-1]} "
        f"({len(ids)} unique intervals)"
    )


def print_boundary(left_name, left_rows, right_name, right_rows):
    left_last = left_rows[-1]
    right_first = right_rows[0]

    print(f"\n{left_name} -> {right_name} boundary")
    print(
        f"  Last {left_name} row: "
        f"order={left_last['random_order']}, "
        f"interval={left_last['reference_interval_id']}, "
        f"fraction={float(left_last['interpolation_fraction']):.3f}"
    )
    print(
        f"  First {right_name} row: "
        f"order={right_first['random_order']}, "
        f"interval={right_first['reference_interval_id']}, "
        f"fraction={float(right_first['interpolation_fraction']):.3f}"
    )


def find_group_aware_split(rows):
    counts = Counter(
        int(row["reference_interval_id"])
        for row in rows
    )
    ordered_ids = sorted(counts)
    target_counts = (
        sample_count * 0.60,
        sample_count * 0.20,
        sample_count * 0.20,
    )
    best_split = None

    for train_last_index in range(len(ordered_ids) - 2):
        for validation_last_index in range(
            train_last_index + 1,
            len(ordered_ids) - 1,
        ):
            train_ids = ordered_ids[:train_last_index + 1]
            validation_ids = ordered_ids[
                train_last_index + 1:validation_last_index + 1
            ]
            test_ids = ordered_ids[validation_last_index + 1:]
            split_counts = (
                sum(counts[interval_id] for interval_id in train_ids),
                sum(counts[interval_id] for interval_id in validation_ids),
                sum(counts[interval_id] for interval_id in test_ids),
            )
            score = sum(
                (actual - target) ** 2
                for actual, target in zip(split_counts, target_counts)
            )

            if best_split is None or score < best_split["score"]:
                best_split = {
                    "train_ids": train_ids,
                    "validation_ids": validation_ids,
                    "test_ids": test_ids,
                    "counts": split_counts,
                    "score": score,
                }

    return counts, best_split


print("RW3 reference-interval split audit")
print(f"Total samples: {sample_count}")
print(f"Train / Validation / Test: {train_end} / "
      f"{validation_end - train_end} / {sample_count - validation_end}")

for split_name, split_rows in splits.items():
    print_split_summary(split_name, split_rows)


print("\nShared reference intervals")
split_names = list(splits)

for index, left_name in enumerate(split_names):
    for right_name in split_names[index + 1:]:
        shared = sorted(
            interval_ids(splits[left_name])
            & interval_ids(splits[right_name])
        )
        print(
            f"{left_name} / {right_name}: "
            f"{shared if shared else 'none'}"
        )


print_boundary("Train", splits["Train"], "Validation", splits["Validation"])
print_boundary(
    "Validation",
    splits["Validation"],
    "Test",
    splits["Test"],
)


interval_counts, recommended_split = find_group_aware_split(rows)

print("\nSamples per reference interval")
for interval_id in sorted(interval_counts):
    print(f"Interval {interval_id}: {interval_counts[interval_id]} samples")


print("\nRecommended group-aware chronological split")
for name, ids, count in (
    ("Train", recommended_split["train_ids"], recommended_split["counts"][0]),
    (
        "Validation",
        recommended_split["validation_ids"],
        recommended_split["counts"][1],
    ),
    ("Test", recommended_split["test_ids"], recommended_split["counts"][2]),
):
    print(
        f"{name}: intervals {ids[0]} to {ids[-1]}, "
        f"{count} samples ({count / sample_count * 100:.1f}%)"
    )
