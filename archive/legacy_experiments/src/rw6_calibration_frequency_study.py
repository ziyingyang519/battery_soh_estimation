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
    calibrated = np.clip(calibrated, 0, 100)

    equation = (
        f"{calibrator.coef_[0]:.3f} * "
        f"SOH_base + {calibrator.intercept_:.3f}"
    )

    return calibrated, equation


X_rw3, y_rw3 = load_feature_table("RW3")
X_rw6, y_rw6 = load_feature_table("RW6")

final_test_start = int(
    len(y_rw6) * (1 - FINAL_TEST_FRACTION)
)

anchor_starts = {
    "Early": 0,
    "Midlife": int(len(y_rw6) * MID_ANCHOR_FRACTION),
    "Late": int(len(y_rw6) * LATE_ANCHOR_FRACTION),
}

anchor_slices = {
    name: slice(start, start + ANCHOR_SIZE)
    for name, start in anchor_starts.items()
}

for name, anchor_slice in anchor_slices.items():
    if anchor_slice.stop > final_test_start:
        raise ValueError(
            f"{name} anchor overlaps final-test region."
        )

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

anchor_base_means = {}
anchor_true_means = {}

print("RW6 calibration anchors")
for name, anchor_slice in anchor_slices.items():
    anchor_base_means[name] = base_prediction[
        anchor_slice
    ].mean()
    anchor_true_means[name] = y_rw6[anchor_slice].mean()

    print(
        f"{name}: samples "
        f"{anchor_slice.start + 1}-{anchor_slice.stop}, "
        f"base mean={anchor_base_means[name]:.2f}%, "
        f"true mean={anchor_true_means[name]:.2f}%"
    )

# 0 anchors: no target-cell calibration.
zero_anchor_prediction = base_prediction.copy()

# 1 anchor: only an offset can be identified from one point.
one_anchor_offset = (
    anchor_true_means["Early"]
    - anchor_base_means["Early"]
)
one_anchor_prediction = np.clip(
    base_prediction + one_anchor_offset,
    0,
    100,
)

# 2 anchors: estimate an affine slope and offset.
two_base_values = np.array(
    [
        anchor_base_means["Early"],
        anchor_base_means["Midlife"],
    ]
)
two_true_values = np.array(
    [
        anchor_true_means["Early"],
        anchor_true_means["Midlife"],
    ]
)
two_anchor_prediction, two_equation = affine_calibration(
    two_base_values,
    two_true_values,
    base_prediction,
)

# 3 anchors: fit the affine correction by least squares.
three_base_values = np.array(
    [
        anchor_base_means["Early"],
        anchor_base_means["Midlife"],
        anchor_base_means["Late"],
    ]
)
three_true_values = np.array(
    [
        anchor_true_means["Early"],
        anchor_true_means["Midlife"],
        anchor_true_means["Late"],
    ]
)
three_anchor_prediction, three_equation = affine_calibration(
    three_base_values,
    three_true_values,
    base_prediction,
)

print(
    f"\n1-anchor offset correction: "
    f"{one_anchor_offset:+.3f}%"
)
print(
    f"2-anchor equation: "
    f"SOH_calibrated = {two_equation}"
)
print(
    f"3-anchor equation: "
    f"SOH_calibrated = {three_equation}"
)

scenarios = {
    "0 anchors": zero_anchor_prediction,
    "1 anchor": one_anchor_prediction,
    "2 anchors": two_anchor_prediction,
    "3 anchors": three_anchor_prediction,
}

print("\nRW6 final-test performance")
print(
    f"Final-test samples: "
    f"{len(y_rw6[final_test_start:])}"
)
print(
    f"Final-test SOH range: "
    f"{y_rw6[final_test_start:].min():.2f}% - "
    f"{y_rw6[final_test_start:].max():.2f}%"
)

metrics = {}

for name, prediction in scenarios.items():
    mae, rmse = calculate_metrics(
        y_rw6[final_test_start:],
        prediction[final_test_start:],
    )
    metrics[name] = {"mae": mae, "rmse": rmse}

    print(
        f"{name}: "
        f"MAE={mae:.3f}%, "
        f"RMSE={rmse:.3f}%"
    )

sample_order = np.arange(1, len(y_rw6) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    sample_order,
    y_rw6,
    label="RW6 true SOH",
    color="black",
    linewidth=2,
)
axes[0].plot(
    sample_order,
    zero_anchor_prediction,
    label="0 anchors",
    alpha=0.75,
)
axes[0].plot(
    sample_order,
    two_anchor_prediction,
    label="2 anchors",
    linewidth=2,
)
axes[0].plot(
    sample_order,
    three_anchor_prediction,
    label="3 anchors",
    linewidth=2,
)

anchor_colors = {
    "Early": "green",
    "Midlife": "steelblue",
    "Late": "purple",
}

for name, anchor_slice in anchor_slices.items():
    axes[0].axvspan(
        anchor_slice.start + 1,
        anchor_slice.stop,
        color=anchor_colors[name],
        alpha=0.12,
        label=f"{name} anchor",
    )

axes[0].axvspan(
    final_test_start + 1,
    len(y_rw6),
    color="orange",
    alpha=0.12,
    label="Final test",
)

axes[0].set_xlabel("RW6 sample order")
axes[0].set_ylabel("SOH (%)")
axes[0].set_title("RW6 periodic calibration")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)

scenario_names = list(metrics.keys())
rmse_values = [
    metrics[name]["rmse"]
    for name in scenario_names
]

bars = axes[1].bar(
    scenario_names,
    rmse_values,
    color=["gray", "skyblue", "steelblue", "purple"],
)

axes[1].set_ylabel("Final-test RMSE (%)")
axes[1].set_title("RW6 calibration-frequency comparison")
axes[1].grid(axis="y", alpha=0.3)

for bar, value in zip(bars, rmse_values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "rw6_calibration_frequency_study.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")