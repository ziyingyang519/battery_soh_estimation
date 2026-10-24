import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"rw3_5s_features_{WINDOW_LABEL}.csv"
)

with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

X = np.array(
    [[float(row[FEATURE_NAME])] for row in rows],
    dtype=float,
)

y = np.array(
    [float(row["soh_percent"]) for row in rows],
    dtype=float,
)

sample_count = len(rows)
train_end = int(sample_count * 0.60)
validation_end = int(sample_count * 0.80)

X_train = X[:train_end]
X_validation = X[train_end:validation_end]
X_test = X[validation_end:]

y_train = y[:train_end]
y_validation = y[train_end:validation_end]
y_test = y[validation_end:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Linear SVR": LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    ),
}


def report_metrics(split_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(
        f"{split_name}: "
        f"MAE={mae:.3f}%, "
        f"RMSE={rmse:.3f}%"
    )


test_predictions = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)

    train_prediction = model.predict(X_train_scaled)
    validation_prediction = model.predict(X_validation_scaled)
    test_prediction = model.predict(X_test_scaled)

    print(f"\n{'=' * 10} {model_name} {'=' * 10}")
    report_metrics("Train", y_train, train_prediction)
    report_metrics("Validation", y_validation, validation_prediction)
    report_metrics("Test", y_test, test_prediction)

    test_predictions[model_name] = test_prediction

test_order = np.arange(1, len(y_test) + 1)

plt.figure(figsize=(10, 5))
plt.plot(
    test_order,
    y_test,
    label="True SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    test_order,
    test_predictions["Linear Regression"],
    label="Linear Regression",
)
plt.plot(
    test_order,
    test_predictions["Linear SVR"],
    label="Linear SVR",
)

plt.xlabel("Test sample order")
plt.ylabel("SOH (%)")
plt.title("IRWST only: linear-model comparison")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "linear_model_comparison_irwst_4p0_to_4p2.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")