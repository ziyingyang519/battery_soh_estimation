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


def pearson_r(x_values, y_values):
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def build_measured_reference_pairs(battery_id):
    mat_path = project_root / "data" / "raw" / f"{battery_id}.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(mat["data"].step)

    references = []
    for step in steps:
        if str(step.comment) != "reference discharge":
            continue

        relative_time = as_float_array(step.relativeTime)
        current = as_float_array(step.current)
        global_time = as_float_array(step.time)
        capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600
        references.append(
            {
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

    for step in steps:
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
            "gap_hours": (target_reference["time_s"] - feature_time_s) / 3600,
            "current_mean_a": float(np.mean(np.abs(window_current))),
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
        "Current only": ["current_mean_a"],
        "IRWST only": ["irwst_ohm"],
        "Current + IRWST": ["current_mean_a", "irwst_ohm"],
    }
    results = {}

    for name, columns in feature_sets.items():
        X_train, y_train = arrays(train_rows, columns)
        X_test, y_test = arrays(test_rows, columns)
        model = LinearRegression().fit(X_train, y_train)
        prediction = model.predict(X_test)
        mae, rmse = calculate_metrics(y_test, prediction)
        results[name] = {"mae": mae, "rmse": rmse}

    return results


all_rows = {
    battery_id: build_measured_reference_pairs(battery_id)
    for battery_id in [SOURCE_BATTERY, *TARGET_BATTERIES]
}
source_rows = all_rows[SOURCE_BATTERY]
summary_rows = []

print("Current-confounding check for IRWST")
print(f"Pairing rule: closest 4.0-4.2 V feature with gap <= {MAX_GAP_HOURS:.0f} h")
print(f"RW3 source training pairs: {len(source_rows)}")

for battery_id, rows in all_rows.items():
    current_values = np.array([row["current_mean_a"] for row in rows])
    irwst_values = np.array([row["irwst_ohm"] for row in rows])
    soh_values = np.array([row["measured_soh_percent"] for row in rows])
    print(
        f"{battery_id}: current={current_values.min():.3f}-"
        f"{current_values.max():.3f} A, median={np.median(current_values):.3f} A; "
        f"r(current, IRWST)={pearson_r(current_values, irwst_values):.3f}; "
        f"r(current, SOH)={pearson_r(current_values, soh_values):.3f}"
    )

external_results = {}
for target_battery in TARGET_BATTERIES:
    target_rows = all_rows[target_battery]
    results = evaluate_feature_sets(source_rows, target_rows)
    external_results[target_battery] = results

    print(f"\n========== RW3 train -> {target_battery} external test ==========")
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
                "target_pairs": len(target_rows),
                "mae_percent": result["mae"],
                "rmse_percent": result["rmse"],
            }
        )

summary_path = processed_path / "current_confounding_irwst_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, axes = plt.subplots(2, 4, figsize=(23, 9))
for column, battery_id in enumerate([SOURCE_BATTERY, *TARGET_BATTERIES]):
    rows = all_rows[battery_id]
    current_values = [row["current_mean_a"] for row in rows]
    irwst_values = [row["irwst_ohm"] for row in rows]
    soh_values = [row["measured_soh_percent"] for row in rows]

    axis = axes[0, column]
    points = axis.scatter(
        current_values,
        irwst_values,
        c=soh_values,
        cmap="viridis",
        s=55,
    )
    axis.set_title(f"{battery_id}: current vs IRWST")
    axis.set_xlabel("Mean current in first 5 s (A)")
    axis.set_ylabel("IRWST (ohm)")
    axis.grid(alpha=0.3)
    figure.colorbar(points, ax=axis, label="Measured SOH (%)")

for column, target_battery in enumerate(TARGET_BATTERIES):
    results = external_results[target_battery]
    model_names = list(results)
    positions = np.arange(len(model_names))
    axis = axes[1, column]
    axis.bar(
        positions - 0.18,
        [results[name]["mae"] for name in model_names],
        width=0.36,
        label="MAE",
    )
    axis.bar(
        positions + 0.18,
        [results[name]["rmse"] for name in model_names],
        width=0.36,
        label="RMSE",
    )
    axis.set_title(f"RW3 -> {target_battery}: lower is better")
    axis.set_ylabel("Error (SOH percentage points)")
    axis.set_xticks(positions, model_names)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

axes[1, 3].axis("off")

figure.suptitle("Sanity check: is IRWST strongly confounded by current?", fontsize=15)
figure.tight_layout()
plot_path = project_root / "outputs" / "current_confounding_irwst.png"
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
