import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
TARGET_BATTERIES = ["RW4", "RW5", "RW6"]
WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOW = 15

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


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


X_source, y_source = load_feature_table(SOURCE_BATTERY)

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)

model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
model.fit(X_source_scaled, y_source)

print(f"Source battery: {SOURCE_BATTERY}")
print(f"Source training samples: {len(y_source)}")

fig, axes = plt.subplots(
    len(TARGET_BATTERIES),
    1,
    figsize=(10, 12),
)

summary_rows = []

for axis, target_battery in zip(axes, TARGET_BATTERIES):
    X_target, y_target = load_feature_table(target_battery)
    X_target_scaled = scaler.transform(X_target)

    raw_prediction = model.predict(X_target_scaled)
    clipped_prediction = np.clip(raw_prediction, 0, 100)
    smoothed_prediction = causal_moving_median(
        clipped_prediction,
        SMOOTHING_WINDOW,
    )

    raw_mae, raw_rmse = calculate_metrics(
        y_target,
        raw_prediction,
    )
    clipped_mae, clipped_rmse = calculate_metrics(
        y_target,
        clipped_prediction,
    )
    smoothed_mae, smoothed_rmse = calculate_metrics(
        y_target,
        smoothed_prediction,
    )

    summary_rows.append(
        {
            "battery": target_battery,
            "samples": len(y_target),
            "soh_range": (
                f"{y_target.min():.2f}% - "
                f"{y_target.max():.2f}%"
            ),
            "raw_mae": raw_mae,
            "raw_rmse": raw_rmse,
            "clipped_mae": clipped_mae,
            "clipped_rmse": clipped_rmse,
            "smoothed_mae": smoothed_mae,
            "smoothed_rmse": smoothed_rmse,
        }
    )

    print(f"\n========== {target_battery} external test ==========")
    print(f"Samples: {len(y_target)}")
    print(
        f"SOH range: "
        f"{y_target.min():.2f}% - {y_target.max():.2f}%"
    )
    print(f"Raw: MAE={raw_mae:.3f}%, RMSE={raw_rmse:.3f}%")
    print(
        f"Clipped: "
        f"MAE={clipped_mae:.3f}%, "
        f"RMSE={clipped_rmse:.3f}%"
    )
    print(
        f"Clipped + causal median: "
        f"MAE={smoothed_mae:.3f}%, "
        f"RMSE={smoothed_rmse:.3f}%"
    )

    sample_order = np.arange(1, len(y_target) + 1)

    axis.plot(
        sample_order,
        y_target,
        label=f"{target_battery} true SOH",
        color="black",
        linewidth=2,
    )
    axis.plot(
        sample_order,
        raw_prediction,
        label="Raw prediction",
        alpha=0.35,
    )
    axis.plot(
        sample_order,
        smoothed_prediction,
        label="Clipped + causal median",
        linewidth=2,
    )
    axis.set_title(
        f"Train on RW3, test on {target_battery}"
    )
    axis.set_xlabel(f"{target_battery} sample order")
    axis.set_ylabel("SOH (%)")
    axis.grid(alpha=0.3)
    axis.legend()

print("\n========== Summary ==========")
for row in summary_rows:
    print(
        f"{row['battery']}: "
        f"raw RMSE={row['raw_rmse']:.3f}%, "
        f"smoothed RMSE={row['smoothed_rmse']:.3f}%"
    )

plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "rw3_to_rw4_rw5_rw6_external_tests.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")