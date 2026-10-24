import csv
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


parser = argparse.ArgumentParser(
    description="Build an audited five-second SOH feature table."
)
parser.add_argument(
    "battery_id",
    nargs="?",
    default="RW3",
    help="Battery file ID in data/raw, for example RW3 or RW4.",
)
args = parser.parse_args()
BATTERY_ID = args.battery_id.upper()

WINDOW_SECONDS = 5.0
VOLTAGE_WINDOW_LOW = 4.0
VOLTAGE_WINDOW_HIGH = 4.2
WINDOW_LABEL = "4p0_to_4p2"
OUTPUT_SUFFIX = "_audit"

project_root = Path(__file__).resolve().parents[1]
mat_path = (
    project_root
    / "data"
    / "raw"
    / f"{BATTERY_ID}.mat"
)


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
data = mat["data"]
steps = np.atleast_1d(data.step)

print(f"Loaded {BATTERY_ID} with {len(steps)} steps.")


reference_records = []

for step_index, step in enumerate(steps):
    if str(step.comment) != "reference discharge":
        continue

    relative_time = as_float_array(step.relativeTime)
    current = as_float_array(step.current)
    global_time = as_float_array(step.time)

    capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600

    reference_records.append(
        {
            "step_index": step_index,
            "time_s": float(global_time[0]),
            "capacity_ah": float(capacity_ah),
        }
    )

reference_records.sort(key=lambda record: record["time_s"])

initial_capacity_ah = reference_records[0]["capacity_ah"]

for record in reference_records:
    record["soh_percent"] = (
        record["capacity_ah"] / initial_capacity_ah * 100
    )

print(f"Built {len(reference_records)} SOH reference labels.")
print(f"Initial capacity: {initial_capacity_ah:.4f} Ah")
print(f"First SOH: {reference_records[0]['soh_percent']:.2f}%")
print(f"Last SOH: {reference_records[-1]['soh_percent']:.2f}%")


reference_times = np.array(
    [record["time_s"] for record in reference_records]
)
reference_soh = np.array(
    [record["soh_percent"] for record in reference_records]
)

feature_rows = []

for step_index, step in enumerate(steps):
    is_random_discharge = (
        str(step.type) == "D"
        and str(step.comment) == "discharge (random walk)"
    )

    if not is_random_discharge:
        continue

    relative_time = as_float_array(step.relativeTime)
    global_time = as_float_array(step.time)
    voltage = as_float_array(step.voltage)
    current = as_float_array(step.current)

    # Keep the first five 1 Hz samples: 0 <= t < 5 seconds.
    five_second_mask = relative_time < WINDOW_SECONDS

    if np.count_nonzero(five_second_mask) < 2:
        continue

    step_start_time_s = float(global_time[0])

    if (
        step_start_time_s < reference_times[0]
        or step_start_time_s > reference_times[-1]
    ):
        continue

    window_voltage = voltage[five_second_mask]
    window_current = current[five_second_mask]
    window_relative_time = relative_time[five_second_mask]
    start_voltage_v = float(window_voltage[0])
    end_voltage_v = float(window_voltage[-1])
    end_current_a = float(window_current[-1])

    if not (
        VOLTAGE_WINDOW_LOW
        <= start_voltage_v
        <= VOLTAGE_WINDOW_HIGH
    ):
        continue

    irwst_ohm = float(
        (start_voltage_v - end_voltage_v)
        / end_current_a
    )

    vsdwst_v = float(np.std(window_voltage))

    next_reference_index = int(
        np.searchsorted(reference_times, step_start_time_s, side="right")
    )
    previous_reference_index = next_reference_index - 1

    if next_reference_index >= len(reference_records):
        continue

    previous_reference = reference_records[previous_reference_index]
    next_reference = reference_records[next_reference_index]
    interpolation_fraction = (
        (step_start_time_s - previous_reference["time_s"])
        / (next_reference["time_s"] - previous_reference["time_s"])
    )
    soh_percent = float(
        np.interp(step_start_time_s, reference_times, reference_soh)
    )

    feature_rows.append(
        {
            "random_order": len(feature_rows) + 1,
            "step_index": step_index,
            "experiment_time_s": step_start_time_s,
            "window_sample_count": int(len(window_relative_time)),
            "window_start_time_s": float(window_relative_time[0]),
            "window_end_time_s": float(window_relative_time[-1]),
            "start_voltage_v": start_voltage_v,
            "end_voltage_v": end_voltage_v,
            "end_current_a": end_current_a,
            "reference_interval_id": previous_reference_index + 1,
            "previous_reference_step_index": previous_reference["step_index"],
            "next_reference_step_index": next_reference["step_index"],
            "previous_reference_soh_percent": previous_reference["soh_percent"],
            "next_reference_soh_percent": next_reference["soh_percent"],
            "interpolation_fraction": interpolation_fraction,
            "soh_percent": soh_percent,
            "irwst_ohm": irwst_ohm,
            "vsdwst_v": vsdwst_v,
        }
    )

print(f"Built {len(feature_rows)} five-second feature rows.")
print("\nFirst feature row:")
print(feature_rows[0])
print("\nLast feature row:")
print(feature_rows[-1])


output_path = (
    project_root
    / "data"
    / "processed"
    / f"{BATTERY_ID.lower()}_features_{WINDOW_LABEL}{OUTPUT_SUFFIX}.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "random_order",
    "step_index",
    "experiment_time_s",
    "window_sample_count",
    "window_start_time_s",
    "window_end_time_s",
    "start_voltage_v",
    "end_voltage_v",
    "end_current_a",
    "reference_interval_id",
    "previous_reference_step_index",
    "next_reference_step_index",
    "previous_reference_soh_percent",
    "next_reference_soh_percent",
    "interpolation_fraction",
    "soh_percent",
    "irwst_ohm",
    "vsdwst_v",
]

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(feature_rows)

print(f"\nSaved feature table to: {output_path}")
