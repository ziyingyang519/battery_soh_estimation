import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


BATTERY_ID = "RW3"
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]

project_root = Path(__file__).resolve().parents[1]
mat_path = project_root / "data" / "raw" / f"{BATTERY_ID}.mat"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def print_metrics(name, y_true, y_pred):
    mae, rmse = calculate_metrics(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")
    return mae, rmse


mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
steps = np.atleast_1d(mat["data"].step)

# First, rebuild capacity-based SOH from the raw reference discharges.
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

for reference_order, reference in enumerate(references, start=1):
    reference["reference_order"] = reference_order
    reference["measured_soh_percent"] = (
        reference["capacity_ah"] / initial_capacity_ah * 100
    )

reference_times = np.array([reference["time_s"] for reference in references])

# Keep eligible five-second IRWST features. No interpolated SOH is created here.
eligible_rows = []
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
    start_voltage_v = float(window_voltage[0])

    if not START_VOLTAGE_LOW <= start_voltage_v <= START_VOLTAGE_HIGH:
        continue

    feature_time_s = float(global_time[0])
    next_reference_index = int(
        np.searchsorted(reference_times, feature_time_s, side="right")
    )

    # A feature must be between two references so its future target is known.
    if next_reference_index == 0 or next_reference_index >= len(references):
        continue

    eligible_rows.append(
        {
            "feature_step_index": step_index,
            "feature_time_s": feature_time_s,
            "irwst_ohm": float(
                (window_voltage[0] - window_voltage[-1]) / window_current[-1]
            ),
            "next_reference_index": next_reference_index,
        }
    )

# For each measured reference discharge, retain only the closest preceding feature
# in its own reference-to-reference interval. Each label is directly measured.
closest_feature_by_reference = {}
for row in eligible_rows:
    reference_index = row["next_reference_index"]
    current_best = closest_feature_by_reference.get(reference_index)

    if (
        current_best is None
        or row["feature_time_s"] > current_best["feature_time_s"]
    ):
        closest_feature_by_reference[reference_index] = row

samples = []
for reference_index in sorted(closest_feature_by_reference):
    feature = closest_feature_by_reference[reference_index]
    reference = references[reference_index]
    samples.append(
        {
            "reference_order": reference["reference_order"],
            "reference_step_index": reference["step_index"],
            "feature_step_index": feature["feature_step_index"],
            "feature_time_s": feature["feature_time_s"],
            "reference_time_s": reference["time_s"],
            "gap_hours": (
                reference["time_s"] - feature["feature_time_s"]
            ) / 3600,
            "irwst_ohm": feature["irwst_ohm"],
            "measured_capacity_ah": reference["capacity_ah"],
            "measured_soh_percent": reference["measured_soh_percent"],
        }
    )

sample_count = len(samples)
train_end = int(sample_count * 0.60)
validation_end = int(sample_count * 0.80)

train_samples = samples[:train_end]
validation_samples = samples[train_end:validation_end]
test_samples = samples[validation_end:]


def samples_to_arrays(sample_rows):
    X = np.array([[row["irwst_ohm"]] for row in sample_rows], dtype=float)
    y = np.array([row["measured_soh_percent"] for row in sample_rows], dtype=float)
    return X, y


X_train, y_train = samples_to_arrays(train_samples)
X_validation, y_validation = samples_to_arrays(validation_samples)
X_test, y_test = samples_to_arrays(test_samples)

print("RW3 measured-reference-only experiment: 4.0-4.2 V")
print(f"Eligible random-walk features: {len(eligible_rows)}")
print(f"Measured-reference samples: {sample_count}")
print(
    "Chronological split: "
    f"train={len(train_samples)}, "
    f"validation={len(validation_samples)}, "
    f"test={len(test_samples)}"
)
print(
    "Reference orders: "
    f"train {train_samples[0]['reference_order']}-{train_samples[-1]['reference_order']}, "
    f"validation {validation_samples[0]['reference_order']}-"
    f"{validation_samples[-1]['reference_order']}, "
    f"test {test_samples[0]['reference_order']}-{test_samples[-1]['reference_order']}"
)

# The scaler and candidate models learn only the training set during selection.
selection_scaler = StandardScaler()
X_train_scaled = selection_scaler.fit_transform(X_train)
X_validation_scaled = selection_scaler.transform(X_validation)

validation_results = []
for C_value in C_CANDIDATES:
    model = LinearSVR(
        C=C_value,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    validation_prediction = model.predict(X_validation_scaled)
    validation_mae, validation_rmse = calculate_metrics(
        y_validation,
        validation_prediction,
    )
    validation_results.append(
        {
            "C": C_value,
            "mae": validation_mae,
            "rmse": validation_rmse,
        }
    )
    print(
        f"Validation candidate C={C_value}: "
        f"MAE={validation_mae:.3f}%, RMSE={validation_rmse:.3f}%"
    )

best_result = min(validation_results, key=lambda result: result["rmse"])
best_C = best_result["C"]
print(f"Selected C from validation: {best_C}")

# Report train/validation performance for the selected model before final test use.
selected_model = LinearSVR(
    C=best_C,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
selected_model.fit(X_train_scaled, y_train)
train_prediction = selected_model.predict(X_train_scaled)
validation_prediction = selected_model.predict(X_validation_scaled)

print("\nSelected-model performance before final test")
print_metrics("Train", y_train, train_prediction)
print_metrics("Validation", y_validation, validation_prediction)

# After selection, retrain on all observed measured-reference samples before test.
X_train_validation = np.vstack([X_train, X_validation])
y_train_validation = np.concatenate([y_train, y_validation])
final_scaler = StandardScaler()
X_train_validation_scaled = final_scaler.fit_transform(X_train_validation)
X_test_scaled = final_scaler.transform(X_test)

final_model = LinearSVR(
    C=best_C,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
final_model.fit(X_train_validation_scaled, y_train_validation)
test_prediction_raw = final_model.predict(X_test_scaled)
test_prediction_clipped = np.clip(test_prediction_raw, 0, 100)

print("\nFinal unseen test performance")
print_metrics("Test (raw)", y_test, test_prediction_raw)
print_metrics("Test (clipped)", y_test, test_prediction_clipped)

for sample, raw_prediction, clipped_prediction in zip(
    test_samples,
    test_prediction_raw,
    test_prediction_clipped,
):
    sample["predicted_soh_raw_percent"] = raw_prediction
    sample["predicted_soh_clipped_percent"] = clipped_prediction

for sample in train_samples:
    sample["split"] = "train"
for sample in validation_samples:
    sample["split"] = "validation"
for sample in test_samples:
    sample["split"] = "test"

pair_output_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_4p0_to_4p2_measured_reference_split.csv"
)
fieldnames = list(samples[0].keys()) + [
    "predicted_soh_raw_percent",
    "predicted_soh_clipped_percent",
]
with pair_output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(samples)

summary_output_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_4p0_to_4p2_measured_reference_summary.csv"
)
summary_rows = [
    {
        "split": "train",
        "samples": len(train_samples),
        "reference_order_range": (
            f"{train_samples[0]['reference_order']}-"
            f"{train_samples[-1]['reference_order']}"
        ),
        "gap_hours_min": min(row["gap_hours"] for row in train_samples),
        "gap_hours_median": float(
            np.median([row["gap_hours"] for row in train_samples])
        ),
        "gap_hours_max": max(row["gap_hours"] for row in train_samples),
    },
    {
        "split": "validation",
        "samples": len(validation_samples),
        "reference_order_range": (
            f"{validation_samples[0]['reference_order']}-"
            f"{validation_samples[-1]['reference_order']}"
        ),
        "gap_hours_min": min(row["gap_hours"] for row in validation_samples),
        "gap_hours_median": float(
            np.median([row["gap_hours"] for row in validation_samples])
        ),
        "gap_hours_max": max(row["gap_hours"] for row in validation_samples),
    },
    {
        "split": "test",
        "samples": len(test_samples),
        "reference_order_range": (
            f"{test_samples[0]['reference_order']}-"
            f"{test_samples[-1]['reference_order']}"
        ),
        "gap_hours_min": min(row["gap_hours"] for row in test_samples),
        "gap_hours_median": float(
            np.median([row["gap_hours"] for row in test_samples])
        ),
        "gap_hours_max": max(row["gap_hours"] for row in test_samples),
    },
]
with summary_output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, (curve_axis, gap_axis) = plt.subplots(2, 1, figsize=(10, 8))
all_reference_orders = np.array([row["reference_order"] for row in samples])
all_measured_soh = np.array([row["measured_soh_percent"] for row in samples])

curve_axis.plot(
    all_reference_orders,
    all_measured_soh,
    color="black",
    linewidth=2,
    marker="o",
    label="Measured reference SOH",
)
curve_axis.scatter(
    [row["reference_order"] for row in test_samples],
    test_prediction_clipped,
    color="tab:red",
    s=60,
    zorder=3,
    label="Final test prediction",
)
curve_axis.axvspan(
    train_samples[0]["reference_order"] - 0.5,
    train_samples[-1]["reference_order"] + 0.5,
    alpha=0.10,
    color="tab:blue",
    label="Train",
)
curve_axis.axvspan(
    validation_samples[0]["reference_order"] - 0.5,
    validation_samples[-1]["reference_order"] + 0.5,
    alpha=0.10,
    color="tab:orange",
    label="Validation",
)
curve_axis.axvspan(
    test_samples[0]["reference_order"] - 0.5,
    test_samples[-1]["reference_order"] + 0.5,
    alpha=0.10,
    color="tab:red",
    label="Test",
)
curve_axis.set_title("RW3 4.0-4.2 V: measured-reference-only split")
curve_axis.set_xlabel("Reference discharge order")
curve_axis.set_ylabel("SOH (%)")
curve_axis.grid(alpha=0.3)
curve_axis.legend(ncol=2)

gap_axis.bar(
    all_reference_orders,
    [row["gap_hours"] for row in samples],
    color="tab:gray",
)
gap_axis.axvline(train_samples[-1]["reference_order"] + 0.5, color="tab:blue")
gap_axis.axvline(validation_samples[-1]["reference_order"] + 0.5, color="tab:orange")
gap_axis.set_title("Feature-to-reference time gap for every directly measured label")
gap_axis.set_xlabel("Reference discharge order")
gap_axis.set_ylabel("Time gap (hours)")
gap_axis.grid(axis="y", alpha=0.3)

figure.tight_layout()
plot_output_path = (
    project_root
    / "outputs"
    / "rw3_4p0_to_4p2_measured_reference_split.png"
)
figure.savefig(plot_output_path, dpi=180)
plt.close(figure)

print(f"\nSaved measured-reference samples to: {pair_output_path}")
print(f"Saved split summary to: {summary_output_path}")
print(f"Saved plot to: {plot_output_path}")
