from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
TARGET_BATTERY = "RW4"
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
MAX_GAP_HOURS = 12.0
IRWST_SVR_C = 100.0

project_root = Path(__file__).resolve().parents[1]


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
        start_voltage_v = float(window_voltage[0])
        if not START_VOLTAGE_LOW <= start_voltage_v <= START_VOLTAGE_HIGH:
            continue

        feature_time_s = float(global_time[0])
        next_reference_index = int(
            np.searchsorted(reference_times, feature_time_s, side="right")
        )
        if next_reference_index == 0 or next_reference_index >= len(references):
            continue

        target_reference = references[next_reference_index]
        gap_hours = (target_reference["time_s"] - feature_time_s) / 3600
        target_order = target_reference["reference_order"]
        feature = {
            "feature_step_index": step_index,
            "feature_time_s": feature_time_s,
            "gap_hours": gap_hours,
            "irwst_ohm": float(
                (window_voltage[0] - window_voltage[-1]) / window_current[-1]
            ),
        }

        current_best = closest_feature_by_target.get(target_order)
        if current_best is None or feature_time_s > current_best["feature_time_s"]:
            closest_feature_by_target[target_order] = feature

    pairs = []
    for target_order in sorted(closest_feature_by_target):
        feature = closest_feature_by_target[target_order]
        target_reference = references[target_order - 1]
        pairs.append(
            {
                "battery": battery_id,
                "reference_order": target_order,
                "previous_reference_order": target_order - 1,
                "feature_step_index": feature["feature_step_index"],
                "feature_time_s": feature["feature_time_s"],
                "elapsed_experiment_hours": (
                    feature["feature_time_s"] - references[0]["time_s"]
                ) / 3600,
                "gap_hours": feature["gap_hours"],
                "irwst_ohm": feature["irwst_ohm"],
                "measured_soh_percent": target_reference["measured_soh_percent"],
            }
        )

    return pairs


def feature_array(rows, feature_name):
    return np.array([[row[feature_name]] for row in rows], dtype=float)


def labels(rows):
    return np.array([row["measured_soh_percent"] for row in rows], dtype=float)


rw3_all_pairs = build_measured_reference_pairs(SOURCE_BATTERY)
rw4_all_pairs = build_measured_reference_pairs(TARGET_BATTERY)
rw3_pairs = [row for row in rw3_all_pairs if row["gap_hours"] <= MAX_GAP_HOURS]
rw4_pairs = [row for row in rw4_all_pairs if row["gap_hours"] <= MAX_GAP_HOURS]

y_rw3 = labels(rw3_pairs)
y_rw4 = labels(rw4_pairs)

models = {
    "Train-mean dummy": {
        "feature": "irwst_ohm",
        "model": DummyRegressor(strategy="mean"),
        "meaning": "No feature.",
    },
    "IRWST linear regression": {
        "feature": "irwst_ohm",
        "model": LinearRegression(),
        "meaning": "Electrical health feature.",
    },
    "Elapsed-time linear regression": {
        "feature": "elapsed_experiment_hours",
        "model": LinearRegression(),
        "meaning": "Elapsed experimental time only.",
    },
    "Reference-order linear regression": {
        "feature": "previous_reference_order",
        "model": LinearRegression(),
        "meaning": "Idealized scheduled-aging-progress proxy.",
    },
    "IRWST Linear SVR": {
        "feature": "irwst_ohm",
        "model": make_pipeline(
            StandardScaler(),
            LinearSVR(
                C=IRWST_SVR_C,
                epsilon=0.1,
                max_iter=20000,
                random_state=42,
            ),
        ),
        "meaning": "Current IRWST model with C fixed from RW3 validation.",
    },
}

print("Cross-cell measured-reference age-baseline comparison")
print(f"Pairing rule: closest 4.0-4.2 V feature with gap <= {MAX_GAP_HOURS:.0f} h")
print(
    f"RW3 source pairs: {len(rw3_pairs)} of {len(rw3_all_pairs)} available "
    f"(SOH {y_rw3.min():.2f}-{y_rw3.max():.2f}%)"
)
print(
    f"RW4 external-test pairs: {len(rw4_pairs)} of {len(rw4_all_pairs)} available "
    f"(SOH {y_rw4.min():.2f}-{y_rw4.max():.2f}%)"
)

results = []
predictions = {}

for model_name, specification in models.items():
    feature_name = specification["feature"]
    model = specification["model"]
    X_rw3 = feature_array(rw3_pairs, feature_name)
    X_rw4 = feature_array(rw4_pairs, feature_name)

    model.fit(X_rw3, y_rw3)
    train_prediction = model.predict(X_rw3)
    external_prediction = model.predict(X_rw4)
    train_mae, train_rmse = calculate_metrics(y_rw3, train_prediction)
    external_mae, external_rmse = calculate_metrics(y_rw4, external_prediction)

    results.append(
        {
            "model": model_name,
            "feature": feature_name,
            "meaning": specification["meaning"],
            "rw3_train_mae_percent": train_mae,
            "rw3_train_rmse_percent": train_rmse,
            "rw4_external_mae_percent": external_mae,
            "rw4_external_rmse_percent": external_rmse,
        }
    )
    predictions[model_name] = external_prediction

    print(f"\n========== {model_name} ==========")
    print(f"Feature: {feature_name}")
    print(f"RW3 train: MAE={train_mae:.3f}%, RMSE={train_rmse:.3f}%")
    print(f"RW4 external test: MAE={external_mae:.3f}%, RMSE={external_rmse:.3f}%")

summary_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_to_rw4_measured_reference_age_baseline_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as file:
    import csv

    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

test_orders = np.array([row["reference_order"] for row in rw4_pairs], dtype=int)
figure, (curve_axis, metric_axis) = plt.subplots(2, 1, figsize=(11, 9))

curve_axis.plot(
    test_orders,
    y_rw4,
    color="black",
    linewidth=2.5,
    marker="o",
    label="RW4 measured reference SOH",
)
for model_name, prediction in predictions.items():
    curve_axis.plot(
        test_orders,
        prediction,
        linewidth=1.6,
        label=model_name,
    )
curve_axis.set_title("Train on RW3, external measured-reference test on RW4")
curve_axis.set_xlabel("RW4 reference discharge order")
curve_axis.set_ylabel("SOH (%)")
curve_axis.grid(alpha=0.3)
curve_axis.legend(fontsize=9, ncol=2)

model_names = [row["model"] for row in results]
x_positions = np.arange(len(model_names))
metric_axis.bar(
    x_positions - 0.18,
    [row["rw4_external_mae_percent"] for row in results],
    width=0.36,
    label="RW4 external MAE",
)
metric_axis.bar(
    x_positions + 0.18,
    [row["rw4_external_rmse_percent"] for row in results],
    width=0.36,
    label="RW4 external RMSE",
)
metric_axis.set_title("External-test comparison: lower is better")
metric_axis.set_ylabel("Error (SOH percentage points)")
metric_axis.set_xticks(x_positions, model_names, rotation=18, ha="right")
metric_axis.grid(axis="y", alpha=0.3)
metric_axis.legend()

figure.tight_layout()
plot_path = (
    project_root
    / "outputs"
    / "rw3_to_rw4_measured_reference_age_baseline_comparison.png"
)
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
