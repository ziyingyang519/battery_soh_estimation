# 目標：只看 RW3 每段 random-walk 放電開始的前 5 秒，能不能預測這顆電池目前的健康程度
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.io import loadmat

project_root = Path(__file__).resolve().parents[1]
mat_path = project_root / "data" / "raw" / "RW3.mat"

mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

print("MATLAB variables:", list(mat.keys()))

data = mat["data"]
steps = np.atleast_1d(data.step)

print("Number of steps:", len(steps))
print("First step fields:", steps[0]._fieldnames)
print("First step type:", steps[0].type)
print("First step comment:", steps[0].comment)
print("First step sample count:", len(steps[0].time))

reference_steps = [
    (index, step)
    for index, step in enumerate(steps)
    if str(step.comment) == "reference discharge"
]

reference_records = []

for index, step in reference_steps:
    current = np.atleast_1d(step.current)
    time = np.atleast_1d(step.relativeTime)

    capacity_ah = np.trapezoid(np.abs(current), time) / 3600

    reference_records.append(
        {
            "step_index": index,
            "capacity_ah": capacity_ah,
        }
    )

initial_capacity_ah = reference_records[0]["capacity_ah"]

for record in reference_records:
    record["soh_percent"] = (
        record["capacity_ah"] / initial_capacity_ah * 100
    )

print("Reference discharge count:", len(reference_records))
print(f"Initial capacity: {initial_capacity_ah:.4f} Ah")

print("\nFirst five SOH labels:")
for record in reference_records[:5]:
    print(
        f"Step {record['step_index']}: "
        f"capacity={record['capacity_ah']:.4f} Ah, "
        f"SOH={record['soh_percent']:.2f}%"
    )

print("\nLast five SOH labels:")
for record in reference_records[-5:]:
    print(
        f"Step {record['step_index']}: "
        f"capacity={record['capacity_ah']:.4f} Ah, "
        f"SOH={record['soh_percent']:.2f}%"
    )

output_path = project_root / "data" / "processed" / "rw3_reference_soh.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "reference_order",
            "step_index",
            "capacity_ah",
            "soh_percent",
        ],
    )
    writer.writeheader()

    for reference_order, record in enumerate(reference_records, start=1):
        writer.writerow(
            {
                "reference_order": reference_order,
                "step_index": record["step_index"],
                "capacity_ah": f"{record['capacity_ah']:.6f}",
                "soh_percent": f"{record['soh_percent']:.6f}",
            }
        )

print(f"\nSaved SOH labels to: {output_path}")

step_categories = Counter(
    (str(step.type), str(step.comment))
    for step in steps
)

print("\nStep categories:")
for (step_type, comment), count in step_categories.most_common():
    print(f"type={step_type}, count={count}, comment={comment}")


random_discharge_steps = [
    (index, step)
    for index, step in enumerate(steps)
    if (
        str(step.type) == "D"
        and str(step.comment) == "discharge (random walk)"
    )
]

print("\nRandom-walk discharge step count:", len(random_discharge_steps))
print("\nFirst three random-walk discharge steps:")

for index, step in random_discharge_steps[:3]:
    time = np.atleast_1d(step.relativeTime)
    voltage = np.atleast_1d(step.voltage)
    current = np.atleast_1d(step.current)

    sample_interval_s = np.median(np.diff(time))
    duration_s = time[-1] - time[0]

    print(
        f"Step {index}: "
        f"samples={len(time)}, "
        f"sample interval={sample_interval_s:.3f} s, "
        f"duration={duration_s:.1f} s, "
        f"current={np.min(current):.3f}-{np.max(current):.3f} A, "
        f"voltage={np.min(voltage):.3f}-{np.max(voltage):.3f} V"
    )