import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


BATTERY_ID = "RW3"
WINDOW_SECONDS = 5.0
TOLERANCE = 1e-9
VOLTAGE_WINDOWS = {
    "3.6-3.8 V": (3.6, 3.8),
    "3.8-4.0 V": (3.8, 4.0),
    "4.0-4.2 V": (4.0, 4.2),
}

project_root = Path(__file__).resolve().parents[1]
mat_path = project_root / "data" / "raw" / f"{BATTERY_ID}.mat"
processed_path = project_root / "data" / "processed"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {detail}")
    return passed


mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
steps = np.atleast_1d(mat["data"].step)

# Rebuild capacity-based SOH directly from the raw reference discharges.
references = []
for step_index, step in enumerate(steps):
    if str(step.comment) != "reference discharge":
        continue

    relative_time = as_float_array(step.relativeTime)
    current = as_float_array(step.current)
    global_time = as_float_array(step.time)
    capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600
    references.append(
        {
            "step_index": step_index,
            "time_s": float(global_time[0]),
            "capacity_ah": float(capacity_ah),
        }
    )

references.sort(key=lambda reference: reference["time_s"])
initial_capacity_ah = references[0]["capacity_ah"]
for order, reference in enumerate(references, start=1):
    reference["reference_order"] = order
    reference["measured_soh_percent"] = (
        reference["capacity_ah"] / initial_capacity_ah * 100
    )

reference_times = np.array([reference["time_s"] for reference in references])


def rebuild_closest_features(voltage_low, voltage_high):
    """Return the closest eligible feature before each following reference."""
    closest_features = {}

    for step_index, step in enumerate(steps):
        if not (
            str(step.type) == "D"
            and str(step.comment) == "discharge (random walk)"
        ):
            continue

        relative_time = as_float_array(step.relativeTime)
        global_time = as_float_array(step.time)
        voltage = as_float_array(step.voltage)
        current = as_float_array(step.current)
        mask = relative_time < WINDOW_SECONDS

        if np.count_nonzero(mask) != 5:
            continue

        window_voltage = voltage[mask]
        window_current = current[mask]
        start_voltage_v = float(window_voltage[0])

        if not voltage_low <= start_voltage_v <= voltage_high:
            continue

        feature_time_s = float(global_time[0])
        next_reference_index = int(
            np.searchsorted(reference_times, feature_time_s, side="right")
        )

        # There must be both a preceding and a following reference discharge.
        if next_reference_index == 0 or next_reference_index >= len(references):
            continue

        target_order = next_reference_index + 1
        feature = {
            "step_index": step_index,
            "time_s": feature_time_s,
            "start_voltage_v": start_voltage_v,
            "irwst_ohm": float(
                (window_voltage[0] - window_voltage[-1]) / window_current[-1]
            ),
            "previous_reference_order": next_reference_index,
            "target_reference_order": target_order,
        }

        current_best = closest_features.get(target_order)
        if current_best is None or feature_time_s > current_best["time_s"]:
            closest_features[target_order] = feature

    return closest_features


closest_by_window = {
    name: rebuild_closest_features(*voltage_range)
    for name, voltage_range in VOLTAGE_WINDOWS.items()
}


def read_csv(filename):
    with (processed_path / filename).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def check_pair_rows(rows, name, default_window=None):
    all_passed = True
    groups = {}

    for row in rows:
        window_name = row.get("window", default_window)
        groups.setdefault(window_name, []).append(row)

    for window_name, group_rows in groups.items():
        if window_name not in closest_by_window:
            raise ValueError(f"Unknown voltage window: {window_name}")

        expected_features = closest_by_window[window_name]
        orders = [int(row["reference_order"]) for row in group_rows]
        all_passed &= report(
            f"{name} / {window_name} unique target references",
            len(orders) == len(set(orders)),
            f"{len(orders)} rows, {len(set(orders))} unique targets",
        )

        for row in group_rows:
            target_order = int(row["reference_order"])
            target_reference = references[target_order - 1]
            previous_reference = references[target_order - 2]
            feature_step_index = int(row["feature_step_index"])
            feature_step = steps[feature_step_index]
            feature_time_s = float(as_float_array(feature_step.time)[0])
            expected_feature = expected_features.get(target_order)

            all_passed &= report(
                f"{name} / {window_name} / reference {target_order}: raw step type",
                (
                    str(feature_step.type) == "D"
                    and str(feature_step.comment) == "discharge (random walk)"
                ),
                f"feature step {feature_step_index}",
            )

            all_passed &= report(
                f"{name} / {window_name} / reference {target_order}: interval membership",
                previous_reference["time_s"] < feature_time_s < target_reference["time_s"],
                (
                    f"previous ref {previous_reference['reference_order']} < "
                    f"feature < target ref {target_order}"
                ),
            )

            all_passed &= report(
                f"{name} / {window_name} / reference {target_order}: closest feature",
                expected_feature is not None
                and feature_step_index == expected_feature["step_index"],
                (
                    f"stored step={feature_step_index}, expected step="
                    f"{None if expected_feature is None else expected_feature['step_index']}"
                ),
            )

            stored_gap_hours = float(row["gap_hours"])
            raw_gap_hours = (
                target_reference["time_s"] - feature_time_s
            ) / 3600
            all_passed &= report(
                f"{name} / {window_name} / reference {target_order}: time gap",
                stored_gap_hours >= 0
                and np.isclose(stored_gap_hours, raw_gap_hours, atol=TOLERANCE),
                f"stored={stored_gap_hours:.3f} h, raw={raw_gap_hours:.3f} h",
            )

            stored_soh = float(
                row["measured_reference_soh_percent"]
                if "measured_reference_soh_percent" in row
                else row["measured_soh_percent"]
            )
            all_passed &= report(
                f"{name} / {window_name} / reference {target_order}: measured SOH",
                np.isclose(
                    stored_soh,
                    target_reference["measured_soh_percent"],
                    atol=TOLERANCE,
                ),
                (
                    f"stored={stored_soh:.3f}%, raw="
                    f"{target_reference['measured_soh_percent']:.3f}%"
                ),
            )

            if "reference_step_index" in row:
                all_passed &= report(
                    f"{name} / {window_name} / reference {target_order}: reference step",
                    int(row["reference_step_index"]) == target_reference["step_index"],
                    (
                        f"stored={row['reference_step_index']}, raw="
                        f"{target_reference['step_index']}"
                    ),
                )

    return all_passed


all_passed = True

split_rows = read_csv("rw3_4p0_to_4p2_measured_reference_split.csv")
all_passed &= check_pair_rows(
    split_rows,
    "Measured-reference-only split",
    default_window="4.0-4.2 V",
)

split_order = {"train": 0, "validation": 1, "test": 2}
split_values = [split_order[row["split"]] for row in split_rows]
all_passed &= report(
    "Measured-reference-only split chronology",
    split_values == sorted(split_values),
    "all train rows precede validation rows, which precede test rows",
)

orders_by_split = {
    split: [
        int(row["reference_order"])
        for row in split_rows
        if row["split"] == split
    ]
    for split in split_order
}
all_passed &= report(
    "Measured-reference-only target-reference separation",
    not (
        set(orders_by_split["train"])
        & set(orders_by_split["validation"])
        or set(orders_by_split["train"]) & set(orders_by_split["test"])
        or set(orders_by_split["validation"]) & set(orders_by_split["test"])
    ),
    "; ".join(
        f"{split}={orders}" for split, orders in orders_by_split.items()
    ),
)

threshold_summary_rows = read_csv("rw3_measured_reference_gap_threshold_summary.csv")
for summary_row in threshold_summary_rows:
    rule = summary_row["gap_rule"]
    threshold_hours = {
        "Gap <= 6 h": 6.0,
        "Gap <= 12 h": 12.0,
        "No gap limit": np.inf,
    }[rule]
    filtered = [row for row in split_rows if float(row["gap_hours"]) <= threshold_hours]
    expected_counts = Counter(row["split"] for row in filtered)
    stored_counts = {
        "train": int(summary_row["train_samples"]),
        "validation": int(summary_row["validation_samples"]),
        "test": int(summary_row["test_samples"]),
    }
    all_passed &= report(
        f"{rule}: filtered split counts",
        dict(expected_counts) == stored_counts,
        f"computed={dict(expected_counts)}, stored={stored_counts}",
    )
    all_passed &= report(
        f"{rule}: threshold compliance",
        all(float(row["gap_hours"]) <= threshold_hours for row in filtered),
        f"maximum retained gap={max(float(row['gap_hours']) for row in filtered):.3f} h",
    )

window_rows = read_csv("rw3_measured_reference_window_pairs.csv")
all_passed &= check_pair_rows(
    window_rows,
    "Voltage-window comparison",
)

print()
if not all_passed:
    raise SystemExit("Measured-reference pair integrity check failed.")

print("All measured-reference pair integrity checks passed.")
