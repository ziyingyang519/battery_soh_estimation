import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WINDOW_LABEL = "4p0_to_4p2"

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"rw3_5s_features_{WINDOW_LABEL}.csv"
)

with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

soh_percent = np.array(
    [float(row["soh_percent"]) for row in rows],
    dtype=float,
)

features = {
    "IRWST (ohm)": np.array(
        [float(row["irwst_ohm"]) for row in rows],
        dtype=float,
    ),
    "VSDWST (V)": np.array(
        [float(row["vsdwst_v"]) for row in rows],
        dtype=float,
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for axis, (feature_label, feature_values) in zip(
    axes,
    features.items(),
):
    correlation = np.corrcoef(feature_values, soh_percent)[0, 1]
    slope, intercept = np.polyfit(feature_values, soh_percent, 1)

    x_line = np.linspace(
        feature_values.min(),
        feature_values.max(),
        100,
    )
    y_line = slope * x_line + intercept

    axis.scatter(
        feature_values,
        soh_percent,
        alpha=0.55,
        s=20,
    )
    axis.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
    )

    axis.set_xlabel(feature_label)
    axis.set_ylabel("SOH (%)")
    axis.set_title(f"{feature_label} vs SOH\nPearson r = {correlation:.3f}")
    axis.grid(alpha=0.3)

    print(f"{feature_label} Pearson r: {correlation:.3f}")

fig.suptitle("RW3: 4.0-4.2 V starting-voltage window")
fig.tight_layout()

output_path = (
    project_root
    / "outputs"
    / f"feature_soh_relationships_{WINDOW_LABEL}.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved plot to: {output_path}")