import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


WINDOW_LABEL = "4p0_to_4p2"

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"rw3_5s_features_{WINDOW_LABEL}.csv"
)

feature_sets = {
    "IRWST only": ["irwst_ohm"],
    "VSDWST only": ["vsdwst_v"],
    "IRWST + VSDWST": ["irwst_ohm", "vsdwst_v"],
}

with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

y = np.array(
    [float(row["soh_percent"]) for row in rows],
    dtype=float,
)

sample_count = len(rows)
train_end = int(sample_count * 0.60)
validation_end = int(sample_count * 0.80)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


for set_name, feature_names in feature_sets.items():
    X = np.array(
        [
            [float(row[feature]) for feature in feature_names]
            for row in rows
        ],
        dtype=float,
    )

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

    model = LinearSVR(
        C=1.0,
        epsilon=0.1,
        max_iter=20000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    train_prediction = model.predict(X_train_scaled)
    validation_prediction = model.predict(X_validation_scaled)
    test_prediction = model.predict(X_test_scaled)

    print(f"\n{'=' * 10} {set_name} {'=' * 10}")
    print(
        "Train: "
        f"MAE={mean_absolute_error(y_train, train_prediction):.3f}%, "
        f"RMSE={calculate_rmse(y_train, train_prediction):.3f}%"
    )
    print(
        "Validation: "
        f"MAE={mean_absolute_error(y_validation, validation_prediction):.3f}%, "
        f"RMSE={calculate_rmse(y_validation, validation_prediction):.3f}%"
    )
    print(
        "Test: "
        f"MAE={mean_absolute_error(y_test, test_prediction):.3f}%, "
        f"RMSE={calculate_rmse(y_test, test_prediction):.3f}%"
    )