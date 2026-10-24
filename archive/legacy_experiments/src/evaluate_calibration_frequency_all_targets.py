import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
TARGET_BATTERIES = ["RW4", "RW5", "RW6"]
WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOW = 15
ANCHOR_SIZE = 10
MID_ANCHOR_FRACTION = 0.55
LATE_ANCHOR_FRACTION = 0.70
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


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def affine_calibration(base_values, true_values, prediction):
    calibrator = LinearRegression()
    calibrator.fit(
        base_values.reshape(-1, 1),
        true_values,
    )

    calibrated = calibrator.predict(
        prediction.reshape(-1, 1)
    )

    return np.clip(calibrated, 0, 100)


X_source, y_source = load_feature_table(SOURCE_BATTERY)

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)

base_model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
base_model.fit(X_source_scaled, y_source)

all_results = {}

for target_battery in TARGET_BATTERIES:
    X_target, y_target = load_feature_table(target_battery)
    X_target_scaled = scaler.transform(X_target)

    final_test_start = int(
        len(y_target) * (1 - FINAL_TEST_FRACTION)
    )

    anchor_starts = {
        "Early": 0,
        "Midlife": int(
            len(y_target) * MID_ANCHOR_FRACTION
        ),
        "Late": int(
            len(y_target) * LATE_ANCHOR_FRACTION
        ),
    }

    anchor_slices = {
        name: slice(start, start + ANCHOR_SIZE)
        for name, start in anchor_starts.items()
    }

    for name, anchor_slice in anchor_slices.items():
        if anchor_slice.stop > final_test_start:
            raise ValueError(
                f"{target_battery} {name} anchor overlaps final test."
            )

    base_prediction = base_model.predict(X_target_scaled)
    base_prediction = np.clip(base_prediction, 0, 100)
    base_prediction = causal_moving_median(
        base_prediction,
        SMOOTHING_WINDOW,
    )

    anchor_base_means = {
        name: base_prediction[anchor_slice].mean()
        for name, anchor_slice in anchor_slices.items()
    }
    anchor_true_means = {
        name: y_target[anchor_slice].mean()
        for name, anchor_slice in anchor_slices.items()
    }

    zero_anchor_prediction = base_prediction.copy()

    one_anchor_offset = (
        anchor_true_means["Early"]
        - anchor_base_means["Early"]
    )
    one_anchor_prediction = np.clip(
        base_prediction + one_anchor_offset,
        0,
        100,
    )

    two_anchor_prediction = affine_calibration(
        np.array(
            [
                anchor_base_means["Early"],
                anchor_base_means["Midlife"],
            ]
        ),
        np.array(
            [
                anchor_true_means["Early"],
                anchor_true_means["Midlife"],
            ]
        ),
        base_prediction,
    )

    three_anchor_prediction = affine_calibration(
        np.array(
            [
                anchor_base_means["Early"],
                anchor_base_means["Midlife"],
                anchor_base_means["Late"],
            ]
        ),
        np.array(
            [
                anchor_true_means["Early"],
                anchor_true_means["Midlife"],
                anchor_true_means["Late"],
            ]
        ),
        base_prediction,
    )

    scenarios = {
        "0 anchors": zero_anchor_prediction,
        "1 anchor": one_anchor_prediction,
        "2 anchors": two_anchor_prediction,
        "3 anchors": three_anchor_prediction,
    }

    print(f"\n========== {target_battery} ==========")
    print(
        f"Final-test SOH range: "
        f"{y_target[final_test_start:].min():.2f}% - "
        f"{y_target[final_test_start:].max():.2f}%"
    )

    target_results = {}

    for name, prediction in scenarios.items():
        mae, rmse = calculate_metrics(
            y_target[final_test_start:],
            prediction[final_test_start:],
        )
        target_results[name] = {
            "mae": mae,
            "rmse": rmse,
        }

        print(
            f"{name}: "
            f"MAE={mae:.3f}%, "
            f"RMSE={rmse:.3f}%"
        )

    all_results[target_battery] = target_results

print("\n========== RMSE summary ==========")
for target_battery, target_results in all_results.items():
    print(
        f"{target_battery}: "
        + ", ".join(
            f"{name}={values['rmse']:.3f}%"
            for name, values in target_results.items()
        )
    )

scenario_names = [
    "0 anchors",
    "1 anchor",
    "2 anchors",
    "3 anchors",
]

x_positions = np.arange(len(TARGET_BATTERIES))
bar_width = 0.18

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for index, scenario_name in enumerate(scenario_names):
    rmse_values = [
        all_results[target][scenario_name]["rmse"]
        for target in TARGET_BATTERIES
    ]
    mae_values = [
        all_results[target][scenario_name]["mae"]
        for target in TARGET_BATTERIES
    ]

    offset = (
        index - (len(scenario_names) - 1) / 2
    ) * bar_width

    axes[0].bar(
        x_positions + offset,
        rmse_values,
        width=bar_width,
        label=scenario_name,
    )
    axes[1].bar(
        x_positions + offset,
        mae_values,
        width=bar_width,
        label=scenario_name,
    )

axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(TARGET_BATTERIES)
axes[0].set_ylabel("Final-test RMSE (%)")
axes[0].set_title("Periodic calibration: RMSE comparison")
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend()

axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(TARGET_BATTERIES)
axes[1].set_ylabel("Final-test MAE (%)")
axes[1].set_title("Periodic calibration: MAE comparison")
axes[1].grid(axis="y", alpha=0.3)
axes[1].legend()

plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "calibration_frequency_rw4_rw5_rw6.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")