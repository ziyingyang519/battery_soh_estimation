import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


SOURCE_BATTERY = "RW3"
TARGET_BATTERIES = ["RW4", "RW5", "RW6"]
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
MAX_GAP_HOURS = 12.0

project_root = Path(__file__).resolve().parents[1]
processed_path = project_root / "data" / "processed"


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
        mask = relative_time < WINDOW_SECONDS

        if np.count_nonzero(mask) != 5:
            continue

        window_voltage = voltage[mask]
        window_current = current[mask]
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
            "feature_time_s": feature_time_s,
            "elapsed_experiment_hours": (
                feature_time_s - references[0]["time_s"]
            ) / 3600,
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


def arrays(rows, feature_columns):
    X = np.array(
        [[row[column] for column in feature_columns] for row in rows],
        dtype=float,
    )
    y = np.array([row["measured_soh_percent"] for row in rows], dtype=float)
    return X, y


def evaluate_feature_sets(train_rows, test_rows):
    feature_sets = {
        "Time only": ["elapsed_experiment_hours"],
        "IRWST only": ["irwst_ohm"],
        "Time + IRWST": ["elapsed_experiment_hours", "irwst_ohm"],
    }
    results = {}

    for name, columns in feature_sets.items():
        X_train, y_train = arrays(train_rows, columns)
        X_test, y_test = arrays(test_rows, columns)
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mae, rmse = calculate_metrics(y_test, prediction)
        results[name] = {
            "prediction": prediction,
            "mae": mae,
            "rmse": rmse,
        }

    return results


source_rows = build_measured_reference_pairs(SOURCE_BATTERY)
external_results = {}
summary_rows = []

print("Cross-cell measured-reference comparison: time + IRWST")
print(f"Pairing rule: closest 4.0-4.2 V feature with gap <= {MAX_GAP_HOURS:.0f} h")
print(f"RW3 source training pairs: {len(source_rows)}")

for target_battery in TARGET_BATTERIES:
    target_rows = build_measured_reference_pairs(target_battery)
    results = evaluate_feature_sets(source_rows, target_rows)
    external_results[target_battery] = {
        "rows": target_rows,
        "results": results,
    }

    print(f"\n========== RW3 train -> {target_battery} external test ==========")
    print(
        f"{target_battery} pairs: {len(target_rows)}, SOH range="
        f"{min(row['measured_soh_percent'] for row in target_rows):.2f}-"
        f"{max(row['measured_soh_percent'] for row in target_rows):.2f}%"
    )
    for model_name, result in results.items():
        print(
            f"{model_name}: MAE={result['mae']:.3f}%, "
            f"RMSE={result['rmse']:.3f}%"
        )
        summary_rows.append(
            {
                "source_battery": SOURCE_BATTERY,
                "target_battery": target_battery,
                "model": model_name,
                "target_samples": len(target_rows),
                "mae_percent": result["mae"],
                "rmse_percent": result["rmse"],
            }
        )

summary_path = (
    processed_path / "rw3_to_rw4_rw5_rw6_time_plus_irwst_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, axes = plt.subplots(2, 3, figsize=(19, 9))
for column, target_battery in enumerate(TARGET_BATTERIES):
    target_data = external_results[target_battery]
    rows = target_data["rows"]
    results = target_data["results"]
    orders = [row["reference_order"] for row in rows]
    measured_soh = [row["measured_soh_percent"] for row in rows]

    curve_axis = axes[0, column]
    curve_axis.plot(
        orders,
        measured_soh,
        color="black",
        marker="o",
        linewidth=2.5,
        label=f"{target_battery} measured reference SOH",
    )
    for model_name, result in results.items():
        curve_axis.plot(
            orders,
            result["prediction"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )
    curve_axis.set_title(f"RW3 train -> {target_battery} external test")
    curve_axis.set_xlabel(f"{target_battery} reference discharge order")
    curve_axis.set_ylabel("SOH (%)")
    curve_axis.grid(alpha=0.3)
    curve_axis.legend(fontsize=8)

    metric_axis = axes[1, column]
    model_names = list(results)
    x_positions = np.arange(len(model_names))
    metric_axis.bar(
        x_positions - 0.18,
        [results[name]["mae"] for name in model_names],
        width=0.36,
        label="MAE",
    )
    metric_axis.bar(
        x_positions + 0.18,
        [results[name]["rmse"] for name in model_names],
        width=0.36,
        label="RMSE",
    )
    metric_axis.set_title(f"RW3 -> {target_battery}: lower is better")
    metric_axis.set_ylabel("Error (SOH percentage points)")
    metric_axis.set_xticks(x_positions, model_names)
    metric_axis.grid(axis="y", alpha=0.3)
    metric_axis.legend()

figure.suptitle("Does IRWST add value beyond elapsed time across cells?", fontsize=15)
figure.tight_layout()
plot_path = (
    project_root
    / "outputs"
    / "rw3_to_rw4_rw5_rw6_time_plus_irwst_comparison.png"
)
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
