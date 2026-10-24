import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
TARGET_BATTERIES = ["RW4", "RW5", "RW6"]
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
MAX_GAP_HOURS = 12.0
TRAIN_VALIDATION_FRACTION = 0.80
SVR_C = 10.0
SVR_EPSILON = 0.1

project_root = Path(__file__).resolve().parents[1]
processed_path = project_root / "data" / "processed"
output_path = project_root / "outputs"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def build_measured_reference_pairs(battery_id):
    mat_path = project_root / "data" / "raw" / f"{battery_id}.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(mat["data"].step)

    references = []
    for step_index, step in enumerate(steps):
        if str(step.comment) != "reference discharge":
            continue

        relative_time = as_float_array(step.relativeTime)
        current = as_float_array(step.current)
        global_time = as_float_array(step.time)
        capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600
        references.append(
            {
                "step_index": step_index,
                "time_s": float(global_time[0]),
                "capacity_ah": float(capacity_ah),
            }
        )

    references.sort(key=lambda reference: reference["time_s"])
    initial_capacity_ah = references[0]["capacity_ah"]
    for order, reference in enumerate(references, start=1):
        reference["reference_order"] = order
        reference["measured_soh_percent"] = (
            reference["capacity_ah"] / initial_capacity_ah * 100
        )

    reference_times = np.array([reference["time_s"] for reference in references])
    closest_feature_by_target = {}

    for step_index, step in enumerate(steps):
        if not (
            str(step.type) == "D"
            and str(step.comment) == "discharge (random walk)"
        ):
            continue

        relative_time = as_float_array(step.relativeTime)
        global_time = as_float_array(step.time)
        voltage = as_float_array(step.voltage)
        current = as_float_array(step.current)
        five_second_mask = relative_time < WINDOW_SECONDS

        if np.count_nonzero(five_second_mask) != 5:
            continue

        window_voltage = voltage[five_second_mask]
        window_current = current[five_second_mask]
        if not START_VOLTAGE_LOW <= window_voltage[0] <= START_VOLTAGE_HIGH:
            continue

        feature_time_s = float(global_time[0])
        next_reference_index = int(
            np.searchsorted(reference_times, feature_time_s, side="right")
        )
        if next_reference_index == 0 or next_reference_index >= len(references):
            continue

        target_reference = references[next_reference_index]
        target_order = target_reference["reference_order"]
        candidate = {
            "reference_order": target_order,
            "feature_step_index": step_index,
            "feature_time_s": feature_time_s,
            "gap_hours": (target_reference["time_s"] - feature_time_s) / 3600,
            "irwst_ohm": float(
                (window_voltage[0] - window_voltage[-1]) / window_current[-1]
            ),
            "measured_soh_percent": target_reference["measured_soh_percent"],
        }

        current_best = closest_feature_by_target.get(target_order)
        if current_best is None or feature_time_s > current_best["feature_time_s"]:
            closest_feature_by_target[target_order] = candidate

    return [
        closest_feature_by_target[target_order]
        for target_order in sorted(closest_feature_by_target)
        if closest_feature_by_target[target_order]["gap_hours"] <= MAX_GAP_HOURS
    ]


def rows_to_arrays(rows):
    X = np.array([[row["irwst_ohm"]] for row in rows], dtype=float)
    y = np.array([row["measured_soh_percent"] for row in rows], dtype=float)
    return X, y


source_rows = build_measured_reference_pairs(SOURCE_BATTERY)
source_train_count = int(len(source_rows) * TRAIN_VALIDATION_FRACTION)
source_train_rows = source_rows[:source_train_count]
X_source_train, y_source_train = rows_to_arrays(source_train_rows)

scaler = StandardScaler()
X_source_train_scaled = scaler.fit_transform(X_source_train)

dummy = DummyRegressor(strategy="mean").fit(X_source_train, y_source_train)
svr = LinearSVR(
    C=SVR_C,
    epsilon=SVR_EPSILON,
    max_iter=20000,
    random_state=42,
).fit(X_source_train_scaled, y_source_train)

print("External evaluation of the measured-reference main baseline")
print(
    f"Source training data: {SOURCE_BATTERY} train + validation only, "
    f"{len(source_train_rows)} pairs"
)
print(
    f"Model: Linear SVR, C={SVR_C}, epsilon={SVR_EPSILON}; "
    f"IRWST from first {WINDOW_SECONDS:.0f} s at {START_VOLTAGE_LOW:.1f}-"
    f"{START_VOLTAGE_HIGH:.1f} V; gap <= {MAX_GAP_HOURS:.0f} h"
)
print("RW3 internal final-test pairs are not used for external-model training.")

external_results = {}
summary_rows = []
for target_battery in TARGET_BATTERIES:
    target_rows = build_measured_reference_pairs(target_battery)
    X_target, y_target = rows_to_arrays(target_rows)
    X_target_scaled = scaler.transform(X_target)

    dummy_prediction = dummy.predict(X_target)
    svr_prediction = svr.predict(X_target_scaled)
    dummy_mae, dummy_rmse = calculate_metrics(y_target, dummy_prediction)
    svr_mae, svr_rmse = calculate_metrics(y_target, svr_prediction)

    external_results[target_battery] = {
        "rows": target_rows,
        "true_soh": y_target,
        "dummy_prediction": dummy_prediction,
        "svr_prediction": svr_prediction,
        "dummy_mae": dummy_mae,
        "dummy_rmse": dummy_rmse,
        "svr_mae": svr_mae,
        "svr_rmse": svr_rmse,
    }

    print(f"\n========== RW3 -> {target_battery} external test ==========")
    print(
        f"Pairs: {len(target_rows)}, SOH range: "
        f"{y_target.min():.2f}-{y_target.max():.2f}%"
    )
    print(f"Train-mean dummy: MAE={dummy_mae:.3f}%, RMSE={dummy_rmse:.3f}%")
    print(f"Linear SVR baseline: MAE={svr_mae:.3f}%, RMSE={svr_rmse:.3f}%")

    for model_name, mae, rmse in [
        ("Train-mean dummy", dummy_mae, dummy_rmse),
        ("IRWST Linear SVR main baseline", svr_mae, svr_rmse),
    ]:
        summary_rows.append(
            {
                "source_battery": SOURCE_BATTERY,
                "source_train_pairs": len(source_train_rows),
                "target_battery": target_battery,
                "target_pairs": len(target_rows),
                "model": model_name,
                "mae_percent": mae,
                "rmse_percent": rmse,
            }
        )

summary_file = processed_path / "main_baseline_external_rw4_rw5_rw6_summary.csv"
with summary_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, target_battery in zip(axes, TARGET_BATTERIES):
    result = external_results[target_battery]
    reference_orders = [row["reference_order"] for row in result["rows"]]

    axis.plot(
        reference_orders,
        result["true_soh"],
        color="black",
        marker="o",
        linewidth=2.5,
        label=f"{target_battery} measured reference SOH",
    )
    axis.plot(
        reference_orders,
        result["dummy_prediction"],
        marker="o",
        label="Train-mean dummy",
    )
    axis.plot(
        reference_orders,
        result["svr_prediction"],
        marker="o",
        label="RW3-trained Linear SVR",
    )
    axis.set_title(
        f"RW3 -> {target_battery}\nSVR RMSE={result['svr_rmse']:.3f}%"
    )
    axis.set_xlabel(f"{target_battery} reference discharge order")
    axis.set_ylabel("SOH (%)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

figure.suptitle("External test: measured-reference Linear SVR main baseline", fontsize=15)
figure.tight_layout()
plot_file = output_path / "main_baseline_external_rw4_rw5_rw6.png"
figure.savefig(plot_file, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_file}")
print(f"Saved plot to: {plot_file}")
