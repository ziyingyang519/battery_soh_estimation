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
TRAIN_LAST_INTERVAL = 30
VOLTAGE_WINDOWS = {
    "3.6-3.8 V": (3.6, 3.8),
    "3.8-4.0 V": (3.8, 4.0),
    "4.0-4.2 V": (4.0, 4.2),
}

project_root = Path(__file__).resolve().parents[1]
mat_path = project_root / "data" / "raw" / f"{BATTERY_ID}.mat"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def calculate_metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
    )


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

for reference_order, reference in enumerate(references, start=1):
    reference["reference_order"] = reference_order
    reference["measured_soh_percent"] = (
        reference["capacity_ah"] / initial_capacity_ah * 100
    )

reference_times = np.array([reference["time_s"] for reference in references])
reference_soh = np.array(
    [reference["measured_soh_percent"] for reference in references]
)


def build_window_rows(voltage_low, voltage_high):
    rows = []

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

        feature_time_s = float(global_time[0])
        next_reference_index = int(
            np.searchsorted(reference_times, feature_time_s, side="right")
        )

        if next_reference_index == 0 or next_reference_index >= len(references):
            continue

        window_voltage = voltage[five_second_mask]
        window_current = current[five_second_mask]
        start_voltage_v = float(window_voltage[0])

        if not voltage_low <= start_voltage_v <= voltage_high:
            continue

        previous_reference = references[next_reference_index - 1]
        next_reference = references[next_reference_index]
        interpolation_fraction = (
            (feature_time_s - previous_reference["time_s"])
            / (next_reference["time_s"] - previous_reference["time_s"])
        )
        interpolated_soh = float(
            np.interp(feature_time_s, reference_times, reference_soh)
        )
        irwst_ohm = float(
            (window_voltage[0] - window_voltage[-1]) / window_current[-1]
        )

        rows.append(
            {
                "step_index": step_index,
                "feature_time_s": feature_time_s,
                "irwst_ohm": irwst_ohm,
                "interpolated_soh_percent": interpolated_soh,
                "reference_interval_id": previous_reference["reference_order"],
                "next_reference_order": next_reference["reference_order"],
                "next_reference_time_s": next_reference["time_s"],
                "next_reference_measured_soh_percent": next_reference[
                    "measured_soh_percent"
                ],
            }
        )

    return rows


all_results = {}

for window_name, (voltage_low, voltage_high) in VOLTAGE_WINDOWS.items():
    rows = build_window_rows(voltage_low, voltage_high)
    training_rows = [
        row
        for row in rows
        if row["reference_interval_id"] <= TRAIN_LAST_INTERVAL
    ]
    X_train = np.array(
        [[row["irwst_ohm"]] for row in training_rows],
        dtype=float,
    )
    y_train = np.array(
        [row["interpolated_soh_percent"] for row in training_rows],
        dtype=float,
    )

    scaler = StandardScaler()
    model = LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    model.fit(scaler.fit_transform(X_train), y_train)

    # A target reference must occur after every reference anchor used by training labels.
    candidate_rows = [
        row
        for row in rows
        if row["next_reference_order"] > TRAIN_LAST_INTERVAL + 1
    ]
    closest_rows = {}

    for row in candidate_rows:
        target_order = row["next_reference_order"]
        current_best = closest_rows.get(target_order)

        if (
            current_best is None
            or row["feature_time_s"] > current_best["feature_time_s"]
        ):
            closest_rows[target_order] = row

    pairs = []
    for target_order in sorted(closest_rows):
        row = closest_rows[target_order]
        prediction = float(
            model.predict(scaler.transform([[row["irwst_ohm"]]]))[0]
        )
        pairs.append(
            {
                "window": window_name,
                "reference_order": target_order,
                "feature_step_index": row["step_index"],
                "gap_hours": (
                    row["next_reference_time_s"] - row["feature_time_s"]
                ) / 3600,
                "irwst_ohm": row["irwst_ohm"],
                "measured_reference_soh_percent": row[
                    "next_reference_measured_soh_percent"
                ],
                "predicted_soh_percent": prediction,
            }
        )

    y_true = np.array(
        [pair["measured_reference_soh_percent"] for pair in pairs],
        dtype=float,
    )
    y_pred = np.array(
        [pair["predicted_soh_percent"] for pair in pairs],
        dtype=float,
    )
    mae, rmse = calculate_metrics(y_true, y_pred)

    all_results[window_name] = {
        "rows": rows,
        "training_rows": training_rows,
        "pairs": pairs,
        "all_pairs_mae": mae,
        "all_pairs_rmse": rmse,
    }

    print(f"\n========== {window_name} ==========")
    print(f"All eligible feature rows: {len(rows)}")
    print(f"Training rows: {len(training_rows)}")
    print(f"Measured-reference pairs: {len(pairs)}")
    print(f"All-pair MAE={mae:.3f}%, RMSE={rmse:.3f}%")
    print(
        "Time gap: "
        f"{min(pair['gap_hours'] for pair in pairs):.2f}-"
        f"{max(pair['gap_hours'] for pair in pairs):.2f} hours"
    )


common_reference_orders = set.intersection(
    *[
        {pair["reference_order"] for pair in result["pairs"]}
        for result in all_results.values()
    ]
)
common_reference_orders = sorted(common_reference_orders)

print("\n========== Fair comparison on shared reference anchors ==========")
print(f"Shared reference orders: {common_reference_orders}")

summary_rows = []
for window_name, result in all_results.items():
    common_pairs = [
        pair
        for pair in result["pairs"]
        if pair["reference_order"] in common_reference_orders
    ]
    y_true = np.array(
        [pair["measured_reference_soh_percent"] for pair in common_pairs],
        dtype=float,
    )
    y_pred = np.array(
        [pair["predicted_soh_percent"] for pair in common_pairs],
        dtype=float,
    )
    mae, rmse = calculate_metrics(y_true, y_pred)
    median_gap_hours = float(np.median([pair["gap_hours"] for pair in common_pairs]))

    result["common_pairs"] = common_pairs
    result["common_mae"] = mae
    result["common_rmse"] = rmse
    result["common_median_gap_hours"] = median_gap_hours

    summary_rows.append(
        {
            "window": window_name,
            "all_feature_rows": len(result["rows"]),
            "training_rows": len(result["training_rows"]),
            "all_reference_pairs": len(result["pairs"]),
            "all_pairs_mae_percent": result["all_pairs_mae"],
            "all_pairs_rmse_percent": result["all_pairs_rmse"],
            "shared_reference_pairs": len(common_pairs),
            "shared_pairs_mae_percent": mae,
            "shared_pairs_rmse_percent": rmse,
            "shared_pairs_median_gap_hours": median_gap_hours,
        }
    )

    print(
        f"{window_name}: MAE={mae:.3f}%, RMSE={rmse:.3f}%, "
        f"median gap={median_gap_hours:.2f} h"
    )


pair_rows = [
    pair
    for result in all_results.values()
    for pair in result["pairs"]
]
pair_output_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_measured_reference_window_pairs.csv"
)
summary_output_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_measured_reference_window_summary.csv"
)

with pair_output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=pair_rows[0].keys())
    writer.writeheader()
    writer.writerows(pair_rows)

with summary_output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)


window_names = list(VOLTAGE_WINDOWS)
common_pairs = all_results[window_names[0]]["common_pairs"]
reference_order = np.array(
    [pair["reference_order"] for pair in common_pairs],
    dtype=int,
)
measured_soh = np.array(
    [pair["measured_reference_soh_percent"] for pair in common_pairs],
    dtype=float,
)

figure, (prediction_axis, metric_axis) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
)

prediction_axis.plot(
    reference_order,
    measured_soh,
    color="black",
    marker="o",
    linewidth=2,
    label="Measured reference SOH",
)

for window_name in window_names:
    predictions = np.array(
        [pair["predicted_soh_percent"] for pair in all_results[window_name]["common_pairs"]],
        dtype=float,
    )
    prediction_axis.plot(
        reference_order,
        predictions,
        marker="o",
        label=f"{window_name} prediction",
    )

prediction_axis.set_title("RW3: measured-reference comparison on shared anchors")
prediction_axis.set_xlabel("Held-out reference discharge order")
prediction_axis.set_ylabel("SOH (%)")
prediction_axis.grid(alpha=0.3)
prediction_axis.legend()

positions = np.arange(len(window_names))
width = 0.35
metric_axis.bar(
    positions - width / 2,
    [all_results[name]["common_mae"] for name in window_names],
    width,
    label="MAE",
)
metric_axis.bar(
    positions + width / 2,
    [all_results[name]["common_rmse"] for name in window_names],
    width,
    label="RMSE",
)
metric_axis.set_title("Prediction error on the same measured-reference anchors")
metric_axis.set_xticks(positions, window_names)
metric_axis.set_ylabel("Error (SOH percentage points)")
metric_axis.grid(axis="y", alpha=0.3)
metric_axis.legend()

figure.tight_layout()

plot_output_path = (
    project_root
    / "outputs"
    / "rw3_measured_reference_window_comparison.png"
)
figure.savefig(plot_output_path, dpi=180)
plt.close(figure)

print(f"Saved pair details to: {pair_output_path}")
print(f"Saved summary to: {summary_output_path}")
print(f"Saved plot to: {plot_output_path}")
