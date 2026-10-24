import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


IRWST_SVR_C = 100.0

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


with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

rows.sort(key=lambda row: int(row["reference_order"]))
first_feature_time_s = min(float(row["feature_time_s"]) for row in rows)

for row in rows:
    # This is known when the random-walk feature is observed.
    row["elapsed_experiment_hours"] = (
        float(row["feature_time_s"]) - first_feature_time_s
    ) / 3600
    # The previous interval is known in this scheduled experimental protocol.
    # It is intentionally an optimistic age-progress baseline, not a BMS feature.
    row["previous_reference_order"] = int(row["reference_order"]) - 1


def split_rows(name):
    return [row for row in rows if row["split"] == name]


train_rows = split_rows("train")
validation_rows = split_rows("validation")
test_rows = split_rows("test")


def feature_array(split_rows, column):
    return np.array([[float(row[column])] for row in split_rows], dtype=float)


def labels(split_rows):
    return np.array(
        [float(row["measured_soh_percent"]) for row in split_rows],
        dtype=float,
    )


y_train = labels(train_rows)
y_validation = labels(validation_rows)
y_test = labels(test_rows)

models = {
    "Train-mean dummy": {
        "feature": "irwst_ohm",
        "model": DummyRegressor(strategy="mean"),
        "interpretation": "No feature; predicts the training SOH mean.",
    },
    "IRWST linear regression": {
        "feature": "irwst_ohm",
        "model": LinearRegression(),
        "interpretation": "Health-feature baseline using the same direct labels.",
    },
    "Elapsed-time linear regression": {
        "feature": "elapsed_experiment_hours",
        "model": LinearRegression(),
        "interpretation": "Uses only elapsed experiment time, not electrical response.",
    },
    "Reference-order linear regression": {
        "feature": "previous_reference_order",
        "model": LinearRegression(),
        "interpretation": "Optimistic aging-progress proxy available only in this scheduled test protocol.",
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
        "interpretation": "Current measured-reference IRWST model.",
    },
}

results = []
test_predictions = {}

print("RW3 measured-reference age-baseline comparison")
print(
    f"Samples: train={len(train_rows)}, validation={len(validation_rows)}, "
    f"test={len(test_rows)}"
)
print(
    "Test reference orders: "
    f"{[int(row['reference_order']) for row in test_rows]}"
)

for model_name, specification in models.items():
    feature_name = specification["feature"]
    model = specification["model"]
    X_train = feature_array(train_rows, feature_name)
    X_validation = feature_array(validation_rows, feature_name)
    X_test = feature_array(test_rows, feature_name)

    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    validation_prediction = model.predict(X_validation)
    test_prediction = model.predict(X_test)

    train_mae, train_rmse = calculate_metrics(y_train, train_prediction)
    validation_mae, validation_rmse = calculate_metrics(
        y_validation,
        validation_prediction,
    )
    test_mae, test_rmse = calculate_metrics(y_test, test_prediction)

    results.append(
        {
            "model": model_name,
            "feature": feature_name,
            "interpretation": specification["interpretation"],
            "train_mae_percent": train_mae,
            "train_rmse_percent": train_rmse,
            "validation_mae_percent": validation_mae,
            "validation_rmse_percent": validation_rmse,
            "test_mae_percent": test_mae,
            "test_rmse_percent": test_rmse,
        }
    )
    test_predictions[model_name] = test_prediction

    print(f"\n========== {model_name} ==========")
    print(f"Feature: {feature_name}")
    print(f"Train: MAE={train_mae:.3f}%, RMSE={train_rmse:.3f}%")
    print(
        f"Validation: MAE={validation_mae:.3f}%, "
        f"RMSE={validation_rmse:.3f}%"
    )
    print(f"Test: MAE={test_mae:.3f}%, RMSE={test_rmse:.3f}%")

summary_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_measured_reference_age_baseline_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

test_orders = np.array(
    [int(row["reference_order"]) for row in test_rows],
    dtype=int,
)
figure, (curve_axis, metric_axis) = plt.subplots(2, 1, figsize=(11, 9))

curve_axis.plot(
    test_orders,
    y_test,
    marker="o",
    color="black",
    linewidth=2.5,
    label="Measured reference SOH",
)
for model_name, prediction in test_predictions.items():
    curve_axis.plot(
        test_orders,
        prediction,
        marker="o",
        linewidth=1.8,
        label=model_name,
    )

curve_axis.set_title("RW3 measured-reference test: feature versus aging-progress baselines")
curve_axis.set_xlabel("Reference discharge order")
curve_axis.set_ylabel("SOH (%)")
curve_axis.grid(alpha=0.3)
curve_axis.legend(fontsize=9)

model_names = [result["model"] for result in results]
x_positions = np.arange(len(model_names))
metric_axis.bar(
    x_positions - 0.18,
    [result["test_mae_percent"] for result in results],
    width=0.36,
    label="Test MAE",
)
metric_axis.bar(
    x_positions + 0.18,
    [result["test_rmse_percent"] for result in results],
    width=0.36,
    label="Test RMSE",
)
metric_axis.set_title("Lower error means a more useful predictor on this split")
metric_axis.set_ylabel("Error (SOH percentage points)")
metric_axis.set_xticks(x_positions, model_names, rotation=18, ha="right")
metric_axis.grid(axis="y", alpha=0.3)
metric_axis.legend()

figure.tight_layout()
plot_path = (
    project_root
    / "outputs"
    / "rw3_measured_reference_age_baseline_comparison.png"
)
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
