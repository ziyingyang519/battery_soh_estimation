import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SOURCE_BATTERY = "RW3"
TARGET_BATTERY = "RW4"
WINDOW_LABEL = "4p0_to_4p2"
FEATURE_NAME = "irwst_ohm"
SMOOTHING_WINDOW = 15

project_root = Path(__file__).resolve().parents[1]


def load_feature_table(battery_id):
    data_path = (
        project_root
        / "data"
        / "processed"
        / f"{battery_id.lower()}_features_{WINDOW_LABEL}.csv"
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

    return X, y

def causal_moving_median(values, window_size):
    smoothed = np.empty_like(values)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = np.median(
            values[start_index:index + 1]
        )

    return smoothed

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(
        f"{name}: "
        f"MAE={mae:.3f}%, "
        f"RMSE={rmse:.3f}%"
    )


X_rw3, y_rw3 = load_feature_table(SOURCE_BATTERY)
X_rw4, y_rw4 = load_feature_table(TARGET_BATTERY)

print(f"RW3 training samples: {len(X_rw3)}")
print(f"RW4 external-test samples: {len(X_rw4)}")
print(
    f"RW3 IRWST range: "
    f"{X_rw3.min():.4f} - {X_rw3.max():.4f} ohm"
)
print(
    f"RW4 IRWST range: "
    f"{X_rw4.min():.4f} - {X_rw4.max():.4f} ohm"
)

# The scaler learns only RW3 statistics.
scaler = StandardScaler()
X_rw3_scaled = scaler.fit_transform(X_rw3)
X_rw4_scaled = scaler.transform(X_rw4)

models = {
    "Linear Regression": LinearRegression(),
    "Linear SVR": LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    ),
}

rw4_predictions = {}

for model_name, model in models.items():
    model.fit(X_rw3_scaled, y_rw3)

    rw3_prediction = model.predict(X_rw3_scaled)
    rw4_prediction = model.predict(X_rw4_scaled)
    rw4_prediction_clipped = np.clip(
        rw4_prediction,
        0,
        100,
    )
    rw4_prediction_smoothed = causal_moving_median(
        rw4_prediction_clipped,
        SMOOTHING_WINDOW,
    )

    clipped_count = np.count_nonzero(
        rw4_prediction != rw4_prediction_clipped
    )

    print(f"\n{'=' * 10} {model_name} {'=' * 10}")
    print_metrics("RW3 train", y_rw3, rw3_prediction)
    print_metrics(
        "RW4 external test (raw)",
        y_rw4,
        rw4_prediction,
    )
    print_metrics(
        "RW4 external test (clipped)",
        y_rw4,
        rw4_prediction_clipped,
    )
    print(f"Predictions clipped to 0-100%: {clipped_count}")
    print_metrics(
        "RW4 external test (clipped + causal median)",
        y_rw4,
        rw4_prediction_smoothed,
    )

    rw4_predictions[model_name] = rw4_prediction_smoothed

test_order = np.arange(1, len(y_rw4) + 1)

plt.figure(figsize=(10, 5))
plt.plot(
    test_order,
    y_rw4,
    label="RW4 true SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    test_order,
    rw4_predictions["Linear Regression"],
    label="Linear Regression + causal median"
)
plt.plot(
    test_order,
    rw4_predictions["Linear SVR"],
    label="Linear SVR + causal median"
)

plt.xlabel("RW4 sample order")
plt.ylabel("SOH (%)")
plt.title("Cross-cell test: train on RW3, test on RW4")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / "cross_cell_rw3_to_rw4_irwst_clipped_smoothed.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")
