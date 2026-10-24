import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"

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

    X = np.array(
        [[float(row[FEATURE_NAME])] for row in rows],
        dtype=float,
    )
    y = np.array(
        [float(row["soh_percent"]) for row in rows],
        dtype=float,
    )

    return rows, X, y


def print_error_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"{name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")


rw3_rows, X_rw3, y_rw3 = load_feature_table("RW3")
rw4_rows, X_rw4, y_rw4 = load_feature_table("RW4")

scaler = StandardScaler()
X_rw3_scaled = scaler.fit_transform(X_rw3)
X_rw4_scaled = scaler.transform(X_rw4)

model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
model.fit(X_rw3_scaled, y_rw3)

rw4_prediction = model.predict(X_rw4_scaled)
residual = rw4_prediction - y_rw4
absolute_error = np.abs(residual)
irwst = X_rw4[:, 0]

negative_mask = irwst < 0
nonnegative_mask = ~negative_mask

print(f"RW4 samples: {len(y_rw4)}")
print(
    f"Negative IRWST samples: {negative_mask.sum()} "
    f"({negative_mask.mean() * 100:.2f}%)"
)

print("\nError by IRWST sign")
print_error_metrics(
    "IRWST < 0",
    y_rw4[negative_mask],
    rw4_prediction[negative_mask],
)
print_error_metrics(
    "IRWST >= 0",
    y_rw4[nonnegative_mask],
    rw4_prediction[nonnegative_mask],
)

print("\nTop 10 largest absolute errors")
top_indices = np.argsort(absolute_error)[-10:][::-1]

for index in top_indices:
    print(
        f"Sample {index + 1}: "
        f"IRWST={irwst[index]:.4f} ohm, "
        f"true={y_rw4[index]:.2f}%, "
        f"predicted={rw4_prediction[index]:.2f}%, "
        f"error={residual[index]:+.2f}%"
    )

sample_order = np.arange(1, len(y_rw4) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    irwst[nonnegative_mask],
    residual[nonnegative_mask],
    alpha=0.6,
    label="IRWST >= 0",
)
axes[0].scatter(
    irwst[negative_mask],
    residual[negative_mask],
    color="red",
    alpha=0.8,
    label="IRWST < 0",
)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_xlabel("IRWST (ohm)")
axes[0].set_ylabel("Prediction error (%)")
axes[0].set_title("RW4 prediction error vs IRWST")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(
    sample_order,
    residual,
    color="tab:blue",
    linewidth=1,
)
axes[1].scatter(
    sample_order[negative_mask],
    residual[negative_mask],
    color="red",
    label="IRWST < 0",
)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xlabel("RW4 sample order")
axes[1].set_ylabel("Prediction error (%)")
axes[1].set_title("RW4 prediction error over time")
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "rw4_irwst_error_analysis.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")