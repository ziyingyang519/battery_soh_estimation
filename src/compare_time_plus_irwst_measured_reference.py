import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def build_raw_measured_pairs(battery_id):
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


def evaluate_models(train_rows, evaluation_rows):
    feature_sets = {
        "Time only": ["elapsed_experiment_hours"],
        "IRWST only": ["irwst_ohm"],
        "Time + IRWST": ["elapsed_experiment_hours", "irwst_ohm"],
    }
    results = {}

    for model_name, columns in feature_sets.items():
        X_train, y_train = arrays(train_rows, columns)
        X_evaluation, y_evaluation = arrays(evaluation_rows, columns)
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_evaluation)
        mae, rmse = calculate_metrics(y_evaluation, prediction)
        results[model_name] = {
            "prediction": prediction,
            "mae": mae,
            "rmse": rmse,
        }

    return results


# Internal RW3 comparison: keep the existing chronological boundaries and apply
# the predeclared 12-hour quality rule inside each split.
with (
    processed_path / "rw3_4p0_to_4p2_measured_reference_split.csv"
).open(encoding="utf-8") as file:
    rw3_split_rows = list(csv.DictReader(file))

first_rw3_feature_time_s = min(
    float(row["feature_time_s"]) for row in rw3_split_rows
)
rw3_internal_rows = []
for row in rw3_split_rows:
    if float(row["gap_hours"]) > MAX_GAP_HOURS:
        continue
    rw3_internal_rows.append(
        {
            "reference_order": int(row["reference_order"]),
            "elapsed_experiment_hours": (
                float(row["feature_time_s"]) - first_rw3_feature_time_s
            ) / 3600,
            "irwst_ohm": float(row["irwst_ohm"]),
            "measured_soh_percent": float(row["measured_soh_percent"]),
            "split": row["split"],
        }
    )

rw3_train = [row for row in rw3_internal_rows if row["split"] == "train"]
rw3_validation = [
    row for row in rw3_internal_rows if row["split"] == "validation"
]
rw3_test = [row for row in rw3_internal_rows if row["split"] == "test"]

# External test: use all quality-filtered RW3 pairs to train, and no RW4 pair
# during fitting.
rw3_external_train = build_raw_measured_pairs("RW3")
rw4_external_test = build_raw_measured_pairs("RW4")

rw3_internal_results = evaluate_models(rw3_train, rw3_test)
rw3_external_results = evaluate_models(rw3_external_train, rw4_external_test)

print("Time + IRWST measured-reference comparison")
print(f"Pairing rule: closest 4.0-4.2 V feature with gap <= {MAX_GAP_HOURS:.0f} h")
print(
    "RW3 internal samples: "
    f"train={len(rw3_train)}, validation={len(rw3_validation)}, test={len(rw3_test)}"
)
print(
    "RW3-to-RW4 external samples: "
    f"train={len(rw3_external_train)}, test={len(rw4_external_test)}"
)

summary_rows = []
for scenario_name, results in (
    ("RW3 chronological test", rw3_internal_results),
    ("RW3 train to RW4 external test", rw3_external_results),
):
    print(f"\n========== {scenario_name} ==========")
    for model_name, result in results.items():
        print(
            f"{model_name}: MAE={result['mae']:.3f}%, "
            f"RMSE={result['rmse']:.3f}%"
        )
        summary_rows.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "mae_percent": result["mae"],
                "rmse_percent": result["rmse"],
            }
        )

summary_path = (
    processed_path / "time_plus_irwst_measured_reference_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, axes = plt.subplots(2, 2, figsize=(13, 9))

for axis, rows, results, title in (
    (
        axes[0, 0],
        rw3_test,
        rw3_internal_results,
        "RW3 chronological measured-reference test",
    ),
    (
        axes[0, 1],
        rw4_external_test,
        rw3_external_results,
        "Train on RW3, external measured-reference test on RW4",
    ),
):
    orders = [row["reference_order"] for row in rows]
    measured_soh = [row["measured_soh_percent"] for row in rows]
    axis.plot(
        orders,
        measured_soh,
        color="black",
        marker="o",
        linewidth=2.5,
        label="Measured reference SOH",
    )
    for model_name, result in results.items():
        axis.plot(
            orders,
            result["prediction"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )
    axis.set_title(title)
    axis.set_xlabel("Reference discharge order")
    axis.set_ylabel("SOH (%)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

for axis, scenario_name, results in (
    (axes[1, 0], "RW3 chronological test", rw3_internal_results),
    (
        axes[1, 1],
        "RW3 to RW4 external test",
        rw3_external_results,
    ),
):
    names = list(results)
    x_positions = np.arange(len(names))
    axis.bar(
        x_positions - 0.18,
        [results[name]["mae"] for name in names],
        width=0.36,
        label="MAE",
    )
    axis.bar(
        x_positions + 0.18,
        [results[name]["rmse"] for name in names],
        width=0.36,
        label="RMSE",
    )
    axis.set_title(scenario_name)
    axis.set_ylabel("Error (SOH percentage points)")
    axis.set_xticks(x_positions, names)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

figure.suptitle("Does IRWST add value beyond elapsed time?", fontsize=15)
figure.tight_layout()
plot_path = (
    project_root
    / "outputs"
    / "time_plus_irwst_measured_reference_comparison.png"
)
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
