import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


BATTERY_ID = "RW3"
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
MAX_GAP_HOURS = 12.0
C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

project_root = Path(__file__).resolve().parents[1]
mat_path = project_root / "data" / "raw" / f"{BATTERY_ID}.mat"
processed_path = project_root / "data" / "processed"
output_path = project_root / "outputs"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def print_metrics(split_name, y_true, y_pred):
    mae, rmse = calculate_metrics(y_true, y_pred)
    print(f"{split_name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")
    return mae, rmse


mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
steps = np.atleast_1d(mat["data"].step)

# Build capacity-based SOH labels from real reference-discharge measurements.
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

samples = [
    closest_feature_by_target[target_order]
    for target_order in sorted(closest_feature_by_target)
    if closest_feature_by_target[target_order]["gap_hours"] <= MAX_GAP_HOURS
]

train_end = int(len(samples) * 0.60)
validation_end = int(len(samples) * 0.80)
train_samples = samples[:train_end]
validation_samples = samples[train_end:validation_end]
test_samples = samples[validation_end:]


def samples_to_arrays(sample_rows):
    X = np.array([[row["irwst_ohm"]] for row in sample_rows], dtype=float)
    y = np.array(
        [row["measured_soh_percent"] for row in sample_rows],
        dtype=float,
    )
    return X, y


X_train, y_train = samples_to_arrays(train_samples)
X_validation, y_validation = samples_to_arrays(validation_samples)
X_test, y_test = samples_to_arrays(test_samples)

print("Main measured-reference baseline: RW3, IRWST only")
print("Target: capacity-based SOH measured by the next reference discharge")
print(
    f"Feature rule: first {WINDOW_SECONDS:.0f} s, start voltage "
    f"{START_VOLTAGE_LOW:.1f}-{START_VOLTAGE_HIGH:.1f} V"
)
print(f"Maximum feature-to-reference gap: {MAX_GAP_HOURS:.0f} h")
print(
    f"Chronological split: train={len(train_samples)}, "
    f"validation={len(validation_samples)}, test={len(test_samples)}"
)
print(
    f"Reference orders: train {train_samples[0]['reference_order']}-"
    f"{train_samples[-1]['reference_order']}, validation "
    f"{validation_samples[0]['reference_order']}-"
    f"{validation_samples[-1]['reference_order']}, test "
    f"{test_samples[0]['reference_order']}-"
    f"{test_samples[-1]['reference_order']}"
)

# The dummy and linear regression provide simple reference comparators.
models = {
    "Train-mean dummy": DummyRegressor(strategy="mean"),
    "IRWST linear regression (interpretability comparator)": LinearRegression(),
}

predictions = {}
summary_rows = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    split_data = {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
    }
    predictions[model_name] = {}

    print(f"\n========== {model_name} ==========")
    for split_name, (X_split, y_split) in split_data.items():
        prediction = model.predict(X_split)
        mae, rmse = print_metrics(split_name.title(), y_split, prediction)
        predictions[model_name][split_name] = prediction
        summary_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "samples": len(y_split),
                "mae_percent": mae,
                "rmse_percent": rmse,
            }
        )

# Select the primary Linear SVR baseline using validation data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

svr_candidates = []
for C_value in C_CANDIDATES:
    candidate = LinearSVR(
        C=C_value,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    candidate.fit(X_train_scaled, y_train)
    validation_prediction = candidate.predict(X_validation_scaled)
    _, validation_rmse = calculate_metrics(y_validation, validation_prediction)
    svr_candidates.append({"C": C_value, "validation_rmse": validation_rmse})
    print(
        f"Linear SVR candidate C={C_value}: "
        f"validation RMSE={validation_rmse:.3f}%"
    )

best_svr = min(svr_candidates, key=lambda result: result["validation_rmse"])
best_C = best_svr["C"]
print(f"Selected Linear SVR C from validation: {best_C}")

svr = LinearSVR(C=best_C, epsilon=0.1, max_iter=20000, random_state=42)
svr.fit(X_train_scaled, y_train)

svr_name = "IRWST Linear SVR (main baseline)"
print(f"\n========== {svr_name} ==========")
for split_name, X_split, y_split in [
    ("train", X_train_scaled, y_train),
    ("validation", X_validation_scaled, y_validation),
    ("test", X_test_scaled, y_test),
]:
    prediction = svr.predict(X_split)
    mae, rmse = print_metrics(split_name.title(), y_split, prediction)
    predictions[svr_name] = {
        **predictions.get(svr_name, {}),
        split_name: prediction,
    }
    summary_rows.append(
        {
            "model": svr_name,
            "split": split_name,
            "samples": len(y_split),
            "mae_percent": mae,
            "rmse_percent": rmse,
        }
    )

for split_name, split_samples in [
    ("train", train_samples),
    ("validation", validation_samples),
    ("test", test_samples),
]:
    for sample in split_samples:
        sample["split"] = split_name

summary_file = processed_path / "rw3_measured_reference_main_baseline_summary.csv"
with summary_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

pair_file = processed_path / "rw3_measured_reference_main_baseline_pairs.csv"
with pair_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=samples[0].keys())
    writer.writeheader()
    writer.writerows(samples)

test_orders = [row["reference_order"] for row in test_samples]
plt.figure(figsize=(9, 5))
plt.plot(
    test_orders,
    y_test,
    color="black",
    marker="o",
    linewidth=2.5,
    label="Measured reference SOH",
)
plt.plot(
    test_orders,
    predictions["Train-mean dummy"]["test"],
    marker="o",
    label="Train-mean dummy",
)
plt.plot(
    test_orders,
    predictions["IRWST linear regression (interpretability comparator)"]["test"],
    marker="o",
    label="IRWST linear regression",
)
plt.plot(
    test_orders,
    predictions[svr_name]["test"],
    marker="o",
    label="IRWST Linear SVR",
)
plt.xlabel("RW3 reference discharge order")
plt.ylabel("SOH (%)")
plt.title("RW3 main measured-reference baseline: chronological test")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plot_file = output_path / "rw3_measured_reference_main_baseline.png"
plt.savefig(plot_file, dpi=180)
plt.close()

print(f"\nSaved summary to: {summary_file}")
print(f"Saved pairs to: {pair_file}")
print(f"Saved plot to: {plot_file}")
