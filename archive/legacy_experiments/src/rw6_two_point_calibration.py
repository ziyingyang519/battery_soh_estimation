import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOW = 15
ANCHOR_SIZE = 10
MID_ANCHOR_FRACTION = 0.55
FINAL_TEST_FRACTION = 0.20

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

    return X, y


def causal_moving_median(values, window_size):
    smoothed = np.empty_like(values)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = np.median(
            values[start_index:index + 1]
        )

    return smoothed


def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"{name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")


X_rw3, y_rw3 = load_feature_table("RW3")
X_rw6, y_rw6 = load_feature_table("RW6")

final_test_start = int(
    len(y_rw6) * (1 - FINAL_TEST_FRACTION)
)
early_anchor_start = 0
mid_anchor_start = int(len(y_rw6) * MID_ANCHOR_FRACTION)

early_anchor_slice = slice(
    early_anchor_start,
    early_anchor_start + ANCHOR_SIZE,
)
mid_anchor_slice = slice(
    mid_anchor_start,
    mid_anchor_start + ANCHOR_SIZE,
)

if mid_anchor_start + ANCHOR_SIZE > final_test_start:
    raise ValueError("Midlife anchor overlaps the final-test region.")

scaler = StandardScaler()
X_rw3_scaled = scaler.fit_transform(X_rw3)
X_rw6_scaled = scaler.transform(X_rw6)

base_model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
base_model.fit(X_rw3_scaled, y_rw3)

base_prediction = base_model.predict(X_rw6_scaled)
base_prediction = np.clip(base_prediction, 0, 100)
base_prediction = causal_moving_median(
    base_prediction,
    SMOOTHING_WINDOW,
)

anchor_base_prediction = np.array(
    [
        base_prediction[early_anchor_slice].mean(),
        base_prediction[mid_anchor_slice].mean(),
    ]
).reshape(-1, 1)

anchor_true_soh = np.array(
    [
        y_rw6[early_anchor_slice].mean(),
        y_rw6[mid_anchor_slice].mean(),
    ]
)

calibration_model = LinearRegression()
calibration_model.fit(
    anchor_base_prediction,
    anchor_true_soh,
)

calibrated_prediction = calibration_model.predict(
    base_prediction.reshape(-1, 1)
)
calibrated_prediction = np.clip(
    calibrated_prediction,
    0,
    100,
)

print(f"RW6 total samples: {len(y_rw6)}")
print(
    f"Early anchor: samples "
    f"{early_anchor_start + 1}-{early_anchor_start + ANCHOR_SIZE}, "
    f"true SOH mean={anchor_true_soh[0]:.2f}%"
)
print(
    f"Midlife anchor: samples "
    f"{mid_anchor_start + 1}-{mid_anchor_start + ANCHOR_SIZE}, "
    f"true SOH mean={anchor_true_soh[1]:.2f}%"
)
print(
    "Calibration equation: "
    f"SOH_calibrated = "
    f"{calibration_model.coef_[0]:.3f} * "
    f"SOH_base + {calibration_model.intercept_:.3f}"
)

print(
    f"Final-test samples: "
    f"{len(y_rw6[final_test_start:])}"
)
print(
    f"Final-test SOH range: "
    f"{y_rw6[final_test_start:].min():.2f}% - "
    f"{y_rw6[final_test_start:].max():.2f}%"
)

print("\nRW6 final-test performance")
print_metrics(
    "RW3 base model",
    y_rw6[final_test_start:],
    base_prediction[final_test_start:],
)
print_metrics(
    "Two-point calibrated model",
    y_rw6[final_test_start:],
    calibrated_prediction[final_test_start:],
)

sample_order = np.arange(1, len(y_rw6) + 1)

plt.figure(figsize=(10, 5))
plt.plot(
    sample_order,
    y_rw6,
    label="RW6 true SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    sample_order,
    base_prediction,
    label="RW3 base model",
)
plt.plot(
    sample_order,
    calibrated_prediction,
    label="Two-point calibrated model",
)

plt.axvspan(
    early_anchor_start + 1,
    early_anchor_start + ANCHOR_SIZE,
    color="green",
    alpha=0.15,
    label="Early calibration anchor",
)
plt.axvspan(
    mid_anchor_start + 1,
    mid_anchor_start + ANCHOR_SIZE,
    color="steelblue",
    alpha=0.15,
    label="Midlife calibration anchor",
)
plt.axvspan(
    final_test_start + 1,
    len(y_rw6),
    color="orange",
    alpha=0.12,
    label="Final-test region",
)

plt.xlabel("RW6 sample order")
plt.ylabel("SOH (%)")
plt.title("RW6: two-point periodic calibration")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "rw6_two_point_calibration.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")