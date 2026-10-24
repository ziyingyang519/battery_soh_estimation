from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


BATTERY_IDS = ["RW3", "RW4", "RW5", "RW6"]

project_root = Path(__file__).resolve().parents[1]


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def load_reference_capacity_and_soh(battery_id):
    mat_path = project_root / "data" / "raw" / f"{battery_id}.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(mat["data"].step)

    measurements = []
    for step in steps:
        if str(step.comment) != "reference discharge":
            continue

        relative_time = as_float_array(step.relativeTime)
        current = as_float_array(step.current)
        global_time = as_float_array(step.time)
        capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600
        measurements.append((float(global_time[0]), float(capacity_ah)))

    measurements.sort(key=lambda measurement: measurement[0])
    capacity_ah = np.array([measurement[1] for measurement in measurements])
    soh_percent = capacity_ah / capacity_ah[0] * 100

    return capacity_ah, soh_percent


figure, (capacity_axis, soh_axis) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
)

for battery_id in BATTERY_IDS:
    capacity_ah, soh_percent = load_reference_capacity_and_soh(battery_id)
    reference_order = np.arange(1, len(capacity_ah) + 1)

    capacity_axis.plot(
        reference_order,
        capacity_ah,
        marker="o",
        markersize=3,
        label=battery_id,
    )
    soh_axis.plot(
        reference_order,
        soh_percent,
        marker="o",
        markersize=3,
        label=battery_id,
    )

    print(
        f"{battery_id}: references={len(capacity_ah)}, "
        f"initial={capacity_ah[0]:.4f} Ah, "
        f"final={capacity_ah[-1]:.4f} Ah, "
        f"final SOH={soh_percent[-1]:.2f}%"
    )

capacity_axis.set_title("Reference-discharge capacity degradation")
capacity_axis.set_xlabel("Reference discharge order")
capacity_axis.set_ylabel("Capacity (Ah)")
capacity_axis.grid(alpha=0.3)
capacity_axis.legend()

soh_axis.set_title("SOH labels calculated from reference-discharge capacity")
soh_axis.set_xlabel("Reference discharge order")
soh_axis.set_ylabel("SOH (%)")
soh_axis.grid(alpha=0.3)
soh_axis.legend()

figure.suptitle("NASA Randomized Battery Usage 2: Reference Capacity and SOH")
figure.tight_layout()

output_path = project_root / "outputs" / "reference_capacity_soh_rw3_to_rw6.png"
figure.savefig(output_path, dpi=180)
plt.close(figure)

print(f"Saved plot to: {output_path}")
