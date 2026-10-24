import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOW = 15

parser = argparse.ArgumentParser(
    description="Train on RW3 and evaluate an external target battery."
)
parser.add_argument(
    "target_battery",
    nargs="?",
    default="RW4",
    help="External target battery ID, for example RW4 or RW5.",
)
args = parser.parse_args()
TARGET_BATTERY = args.target_battery.upper()

project_root = Path(__file__).resolve().parents[1]


def load_audit_feature_table(battery_id):
    data_path = (
        project_root
        / "data"
        / "processed"
        / f"{battery_id.lower()}_features_{WINDOW_LABEL}_audit.csv"
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


def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")


def causal_moving_median(values, window_size):
    """Smooth each prediction using only the current and earlier RW4 predictions."""
    smoothed = np.empty_like(values)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = np.median(values[start_index:index + 1])

    return smoothed


X_rw3, y_rw3 = load_audit_feature_table(SOURCE_BATTERY)
X_rw4, y_rw4 = load_audit_feature_table(TARGET_BATTERY)

print(f"Source training battery: {SOURCE_BATTERY}")
print(f"Source training samples: {len(X_rw3)}")
print(f"Target external-test battery: {TARGET_BATTERY}")
print(f"Target external-test samples: {len(X_rw4)}")
print(
    f"{SOURCE_BATTERY} IRWST range: "
    f"{X_rw3.min():.4f} - {X_rw3.max():.4f} ohm"
)
print(
    f"{TARGET_BATTERY} IRWST range: "
    f"{X_rw4.min():.4f} - {X_rw4.max():.4f} ohm"
)

# RW4 is never used to fit the scaler or the model.
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

rw3_prediction = model.predict(X_rw3_scaled)
rw4_prediction = model.predict(X_rw4_scaled)
rw4_prediction_clipped = np.clip(rw4_prediction, 0, 100)
rw4_prediction_smoothed = causal_moving_median(
    rw4_prediction_clipped,
    SMOOTHING_WINDOW,
)
clipped_count = np.count_nonzero(rw4_prediction != rw4_prediction_clipped)

print(
    f"\n========== Linear SVR: {SOURCE_BATTERY} to "
    f"{TARGET_BATTERY} external test =========="
)
print_metrics(f"{SOURCE_BATTERY} training", y_rw3, rw3_prediction)
print_metrics(f"{TARGET_BATTERY} external test (raw)", y_rw4, rw4_prediction)
print_metrics(
    f"{TARGET_BATTERY} external test (clipped)",
    y_rw4,
    rw4_prediction_clipped,
)
print_metrics(
    f"{TARGET_BATTERY} external test "
    f"(clipped + causal median, window={SMOOTHING_WINDOW})",
    y_rw4,
    rw4_prediction_smoothed,
)
print(
    f"{TARGET_BATTERY} prediction range: {rw4_prediction.min():.2f}% - "
    f"{rw4_prediction.max():.2f}%"
)
print(
    f"{TARGET_BATTERY} predictions outside physical 0-100% range: "
    f"{clipped_count}"
)

sample_order = np.arange(1, len(y_rw4) + 1)
plt.figure(figsize=(10, 5))
plt.plot(
    sample_order,
    y_rw4,
    label=f"{TARGET_BATTERY} interpolated SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    sample_order,
    rw4_prediction,
    label="Linear SVR raw",
    alpha=0.5,
)
plt.plot(
    sample_order,
    rw4_prediction_smoothed,
    label=(
        "Linear SVR clipped + "
        f"causal median (window={SMOOTHING_WINDOW})"
    ),
)
plt.xlabel(f"{TARGET_BATTERY} random-walk sample order")
plt.ylabel("SOH (%)")
plt.title(
    f"External test: train on {SOURCE_BATTERY}, "
    f"predict {TARGET_BATTERY}"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / (
        f"{SOURCE_BATTERY.lower()}_to_{TARGET_BATTERY.lower()}_"
        "external_linear_svr_clipped_smoothed.png"
    )
)
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved plot to: {output_path}")
