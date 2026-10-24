import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOWS = [1, 3, 5, 9, 15]

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"rw3_features_{WINDOW_LABEL}.csv"
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

sample_count = len(y)
train_end = int(sample_count * 0.60)
validation_end = int(sample_count * 0.80)

X_train = X[:train_end]
X_validation = X[train_end:validation_end]

y_train = y[:train_end]
y_validation = y[train_end:validation_end]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

validation_prediction = model.predict(X_validation_scaled)
validation_prediction = np.clip(
    validation_prediction,
    0,
    100,
)


def causal_moving_median(values, window_size):
    smoothed = np.empty_like(values)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = np.median(
            values[start_index:index + 1]
        )

    return smoothed


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


results = {}
best_window = None
best_rmse = float("inf")

print("RW3 validation: causal moving-median comparison")

for window_size in SMOOTHING_WINDOWS:
    smoothed_prediction = causal_moving_median(
        validation_prediction,
        window_size,
    )
    mae, rmse = calculate_metrics(
        y_validation,
        smoothed_prediction,
    )

    results[window_size] = smoothed_prediction

    print(
        f"Window {window_size}: "
        f"MAE={mae:.3f}%, RMSE={rmse:.3f}%"
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_window = window_size

print(
    f"\nSelected window from validation: "
    f"{best_window} samples"
)

validation_order = np.arange(1, len(y_validation) + 1)

plt.figure(figsize=(10, 5))
plt.plot(
    validation_order,
    y_validation,
    label="True SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    validation_order,
    validation_prediction,
    label="Raw clipped prediction",
    alpha=0.7,
)
plt.plot(
    validation_order,
    results[best_window],
    label=f"Causal median (window={best_window})",
    linewidth=2,
)

plt.xlabel("RW3 validation sample order")
plt.ylabel("SOH (%)")
plt.title("RW3 validation: smoothing-window selection")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "rw3_validation_smoothing_selection.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved plot to: {output_path}")