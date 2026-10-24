import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BATTERIES = ["RW3", "RW4", "RW5", "RW6"]
WINDOW_LABEL = "4p0_to_4p2"

project_root = Path(__file__).resolve().parents[1]


def load_feature_table(battery_id):
    data_path = (
        project_root
        / "data"
        / "processed"
        / f"{battery_id.lower()}_features_{WINDOW_LABEL}.csv"
    )

    with data_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    irwst = np.array(
        [float(row["irwst_ohm"]) for row in rows],
        dtype=float,
    )
    soh = np.array(
        [float(row["soh_percent"]) for row in rows],
        dtype=float,
    )

    return irwst, soh


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

for axis, battery_id in zip(axes, BATTERIES):
    irwst, soh = load_feature_table(battery_id)

    correlation = np.corrcoef(irwst, soh)[0, 1]
    slope, intercept = np.polyfit(irwst, soh, 1)

    x_line = np.linspace(-0.015, 0.065, 100)
    y_line = slope * x_line + intercept

    axis.scatter(
        irwst,
        soh,
        alpha=0.55,
        s=18,
    )
    axis.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
    )

    axis.set_xlim(-0.015, 0.065)
    axis.set_ylim(50, 102)
    axis.set_xlabel("IRWST (ohm)")
    axis.set_ylabel("SOH (%)")
    axis.set_title(
        f"{battery_id}: r={correlation:.3f}, "
        f"slope={slope:.1f}"
    )
    axis.grid(alpha=0.3)

    print(
        f"{battery_id}: "
        f"samples={len(soh)}, "
        f"SOH={soh.min():.2f}-{soh.max():.2f}%, "
        f"IRWST={irwst.min():.4f}-{irwst.max():.4f} ohm, "
        f"Pearson r={correlation:.3f}, "
        f"slope={slope:.1f}"
    )

plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "irwst_soh_relationships_rw3_to_rw6.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")