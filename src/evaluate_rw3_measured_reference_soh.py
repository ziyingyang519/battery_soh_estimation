import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


BATTERY_ID = "RW3"
WINDOW_LABEL = "4p0_to_4p2"
DATA_SUFFIX = "_audit"
TRAIN_LAST_INTERVAL = 30

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"{BATTERY_ID.lower()}_features_{WINDOW_LABEL}{DATA_SUFFIX}.csv"
)
mat_path = project_root / "data" / "raw" / f"{BATTERY_ID}.mat"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def load_measured_references():
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

    return {
        reference["step_index"]: reference
        for reference in references
    }


def rows_to_arrays(rows):
    features = np.array(
        [[float(row["irwst_ohm"])] for row in rows],
        dtype=float,
    )
    labels = np.array(
        [float(row["soh_percent"]) for row in rows],
        dtype=float,
    )
    return features, labels


def print_metrics(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")


with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

references_by_step = load_measured_references()

# The final model may use training and validation intervals, but not later anchors.
training_rows = [
    row
    for row in rows
    if int(row["reference_interval_id"]) <= TRAIN_LAST_INTERVAL
]
training_reference_steps = {
    int(row["previous_reference_step_index"])
    for row in training_rows
} | {
    int(row["next_reference_step_index"])
    for row in training_rows
}

X_train, y_train = rows_to_arrays(training_rows)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

print("RW3 measured-reference validation")
print(f"Training intervals: 1-{TRAIN_LAST_INTERVAL}")
print(f"Training samples: {len(training_rows)}")
print(f"Reference anchors used by training labels: {len(training_reference_steps)}")

# Keep one closest eligible random-walk input before each unseen reference discharge.
closest_rows_by_reference = {}

for row in rows:
    next_reference_step = int(row["next_reference_step_index"])

    if next_reference_step in training_reference_steps:
        continue

    reference = references_by_step[next_reference_step]
    feature_time_s = float(row["experiment_time_s"])

    if feature_time_s >= reference["time_s"]:
        continue

    previous_row = closest_rows_by_reference.get(next_reference_step)
    if (
        previous_row is None
        or feature_time_s > float(previous_row["experiment_time_s"])
    ):
        closest_rows_by_reference[next_reference_step] = row

selected_rows = [
    closest_rows_by_reference[step_index]
    for step_index in sorted(
        closest_rows_by_reference,
        key=lambda step_index: references_by_step[step_index]["reference_order"],
    )
]

X_reference_test = np.array(
    [[float(row["irwst_ohm"])] for row in selected_rows],
    dtype=float,
)
y_measured_reference = np.array(
    [
        references_by_step[int(row["next_reference_step_index"])][
            "measured_soh_percent"
        ]
        for row in selected_rows
    ],
    dtype=float,
)

y_pred_raw = model.predict(scaler.transform(X_reference_test))
y_pred_clipped = np.clip(y_pred_raw, 0, 100)

print(f"Unseen measured-reference targets: {len(selected_rows)}")
print_metrics("Measured-reference test (raw)", y_measured_reference, y_pred_raw)
print_metrics(
    "Measured-reference test (clipped)",
    y_measured_reference,
    y_pred_clipped,
)

pair_rows = []
for row, true_soh, raw_prediction, clipped_prediction in zip(
    selected_rows,
    y_measured_reference,
    y_pred_raw,
    y_pred_clipped,
):
    reference = references_by_step[int(row["next_reference_step_index"])]
    gap_hours = (
        reference["time_s"] - float(row["experiment_time_s"])
    ) / 3600
    pair_rows.append(
        {
            "reference_order": reference["reference_order"],
            "reference_step_index": reference["step_index"],
            "closest_feature_step_index": row["step_index"],
            "feature_time_s": row["experiment_time_s"],
            "reference_time_s": reference["time_s"],
            "gap_hours": gap_hours,
            "irwst_ohm": row["irwst_ohm"],
            "measured_capacity_ah": reference["capacity_ah"],
            "measured_reference_soh_percent": true_soh,
            "predicted_soh_raw_percent": raw_prediction,
            "predicted_soh_clipped_percent": clipped_prediction,
        }
    )

print(
    "Feature-to-reference time gap: "
    f"{min(row['gap_hours'] for row in pair_rows):.2f}-"
    f"{max(row['gap_hours'] for row in pair_rows):.2f} hours"
)

pair_output_path = (
    project_root
    / "data"
    / "processed"
    / "rw3_measured_reference_test_pairs.csv"
)

with pair_output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=pair_rows[0].keys())
    writer.writeheader()
    writer.writerows(pair_rows)

reference_order = np.array(
    [row["reference_order"] for row in pair_rows],
    dtype=int,
)
gap_hours = np.array([row["gap_hours"] for row in pair_rows], dtype=float)

figure, (soh_axis, gap_axis) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
)

soh_axis.plot(
    reference_order,
    y_measured_reference,
    marker="o",
    color="black",
    linewidth=2,
    label="Measured reference SOH",
)
soh_axis.plot(
    reference_order,
    y_pred_raw,
    marker="o",
    label="Linear SVR prediction from closest random-walk segment",
)
soh_axis.set_title("RW3: prediction compared with unseen measured reference SOH")
soh_axis.set_xlabel("Held-out reference discharge order")
soh_axis.set_ylabel("SOH (%)")
soh_axis.grid(alpha=0.3)
soh_axis.legend()

gap_axis.bar(reference_order, gap_hours)
gap_axis.set_title("Time from selected random-walk input to reference discharge")
gap_axis.set_xlabel("Held-out reference discharge order")
gap_axis.set_ylabel("Time gap (hours)")
gap_axis.grid(axis="y", alpha=0.3)

figure.tight_layout()

plot_output_path = (
    project_root
    / "outputs"
    / "rw3_measured_reference_validation.png"
)
figure.savefig(plot_output_path, dpi=180)
plt.close(figure)

print(f"Saved pairs to: {pair_output_path}")
print(f"Saved plot to: {plot_output_path}")
