import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


FIXED_C = 100.0
GAP_THRESHOLDS_HOURS = [6.0, 12.0, np.inf]

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_4p0_to_4p2_measured_reference_split.csv"
)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def samples_to_arrays(rows):
    X = np.array([[row["irwst_ohm"]] for row in rows], dtype=float)
    y = np.array([row["measured_soh_percent"] for row in rows], dtype=float)
    return X, y


with data_path.open(encoding="utf-8") as file:
    raw_rows = list(csv.DictReader(file))

samples = [
    {
        "reference_order": int(row["reference_order"]),
        "gap_hours": float(row["gap_hours"]),
        "irwst_ohm": float(row["irwst_ohm"]),
        "measured_soh_percent": float(row["measured_soh_percent"]),
        "split": row["split"],
    }
    for row in raw_rows
]

results = {}

for threshold_hours in GAP_THRESHOLDS_HOURS:
    threshold_label = (
        "No gap limit"
        if np.isinf(threshold_hours)
        else f"Gap <= {threshold_hours:.0f} h"
    )
    filtered_samples = [
        row for row in samples if row["gap_hours"] <= threshold_hours
    ]
    train_rows = [row for row in filtered_samples if row["split"] == "train"]
    validation_rows = [
        row for row in filtered_samples if row["split"] == "validation"
    ]
    test_rows = [row for row in filtered_samples if row["split"] == "test"]

    if min(len(train_rows), len(validation_rows), len(test_rows)) == 0:
        raise ValueError(f"{threshold_label} leaves an empty chronological split.")

    X_train, y_train = samples_to_arrays(train_rows)
    X_validation, y_validation = samples_to_arrays(validation_rows)
    X_test, y_test = samples_to_arrays(test_rows)

    # C=100 was selected once using the original measured-reference validation.
    # It stays fixed here because the strict 6-hour validation set has only two rows.
    selection_scaler = StandardScaler()
    X_train_scaled = selection_scaler.fit_transform(X_train)
    X_validation_scaled = selection_scaler.transform(X_validation)
    selection_model = LinearSVR(
        C=FIXED_C,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    selection_model.fit(X_train_scaled, y_train)
    validation_prediction = selection_model.predict(X_validation_scaled)
    validation_mae, validation_rmse = calculate_metrics(
        y_validation,
        validation_prediction,
    )

    # Refit with all available train and validation labels before one final test.
    X_train_validation = np.vstack([X_train, X_validation])
    y_train_validation = np.concatenate([y_train, y_validation])
    final_scaler = StandardScaler()
    X_train_validation_scaled = final_scaler.fit_transform(X_train_validation)
    X_test_scaled = final_scaler.transform(X_test)
    final_model = LinearSVR(
        C=FIXED_C,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    final_model.fit(X_train_validation_scaled, y_train_validation)
    test_prediction = np.clip(final_model.predict(X_test_scaled), 0, 100)
    test_mae, test_rmse = calculate_metrics(y_test, test_prediction)

    prediction_by_order = {
        row["reference_order"]: prediction
        for row, prediction in zip(test_rows, test_prediction)
    }
    measured_by_order = {
        row["reference_order"]: row["measured_soh_percent"]
        for row in test_rows
    }

    results[threshold_label] = {
        "threshold_hours": threshold_hours,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "prediction_by_order": prediction_by_order,
        "measured_by_order": measured_by_order,
    }

    print(f"\n========== {threshold_label} ==========")
    print(
        "Samples after gap filter: "
        f"train={len(train_rows)}, "
        f"validation={len(validation_rows)}, "
        f"test={len(test_rows)}"
    )
    print(
        "Test reference orders: "
        f"{[row['reference_order'] for row in test_rows]}"
    )
    print(
        f"Validation: MAE={validation_mae:.3f}%, "
        f"RMSE={validation_rmse:.3f}%"
    )
    print(f"Final test: MAE={test_mae:.3f}%, RMSE={test_rmse:.3f}%")

shared_test_orders = sorted(
    set.intersection(
        *[
            set(result["prediction_by_order"])
            for result in results.values()
        ]
    )
)

print("\n========== Fair test comparison on shared reference orders ==========")
print(f"Shared orders: {shared_test_orders}")

summary_rows = []
for threshold_label, result in results.items():
    shared_y_true = np.array(
        [result["measured_by_order"][order] for order in shared_test_orders],
        dtype=float,
    )
    shared_y_pred = np.array(
        [result["prediction_by_order"][order] for order in shared_test_orders],
        dtype=float,
    )
    shared_mae, shared_rmse = calculate_metrics(shared_y_true, shared_y_pred)
    result["shared_mae"] = shared_mae
    result["shared_rmse"] = shared_rmse

    print(
        f"{threshold_label}: shared-test "
        f"MAE={shared_mae:.3f}%, RMSE={shared_rmse:.3f}%"
    )

    summary_rows.append(
        {
            "gap_rule": threshold_label,
            "train_samples": len(result["train_rows"]),
            "validation_samples": len(result["validation_rows"]),
            "test_samples": len(result["test_rows"]),
            "validation_mae_percent": result["validation_mae"],
            "validation_rmse_percent": result["validation_rmse"],
            "test_mae_percent": result["test_mae"],
            "test_rmse_percent": result["test_rmse"],
            "shared_test_samples": len(shared_test_orders),
            "shared_test_mae_percent": shared_mae,
            "shared_test_rmse_percent": shared_rmse,
        }
    )

summary_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_measured_reference_gap_threshold_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

figure, (curve_axis, metric_axis) = plt.subplots(2, 1, figsize=(10, 9))
colors = ["tab:green", "tab:orange", "tab:blue"]

for color, (threshold_label, result) in zip(colors, results.items()):
    test_orders = [row["reference_order"] for row in result["test_rows"]]
    measured_soh = [row["measured_soh_percent"] for row in result["test_rows"]]
    predicted_soh = [result["prediction_by_order"][order] for order in test_orders]
    curve_axis.plot(
        test_orders,
        measured_soh,
        color="black",
        linewidth=2,
        marker="o",
    )
    curve_axis.plot(
        test_orders,
        predicted_soh,
        color=color,
        linewidth=2,
        marker="o",
        label=threshold_label,
    )

curve_axis.set_title("RW3 measured-reference test predictions under time-gap rules")
curve_axis.set_xlabel("Reference discharge order")
curve_axis.set_ylabel("SOH (%)")
curve_axis.grid(alpha=0.3)
curve_axis.legend(title="Prediction rule")

labels = [row["gap_rule"] for row in summary_rows]
x_positions = np.arange(len(labels))
bar_width = 0.35
metric_axis.bar(
    x_positions - bar_width / 2,
    [row["shared_test_mae_percent"] for row in summary_rows],
    width=bar_width,
    label="Shared-test MAE",
)
metric_axis.bar(
    x_positions + bar_width / 2,
    [row["shared_test_rmse_percent"] for row in summary_rows],
    width=bar_width,
    label="Shared-test RMSE",
)
metric_axis.set_title(
    f"Fair comparison on shared test references {shared_test_orders}"
)
metric_axis.set_ylabel("Error (SOH percentage points)")
metric_axis.set_xticks(x_positions, labels)
metric_axis.grid(axis="y", alpha=0.3)
metric_axis.legend()

figure.tight_layout()
plot_path = (
    project_root
    / "outputs"
    / "rw3_measured_reference_gap_threshold_sensitivity.png"
)
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
