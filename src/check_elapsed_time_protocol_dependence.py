import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


BATTERIES = ["RW3", "RW4", "RW5", "RW6"]
WINDOW_SECONDS = 5.0
START_VOLTAGE_LOW = 4.0
START_VOLTAGE_HIGH = 4.2
MAX_GAP_HOURS = 12.0

project_root = Path(__file__).resolve().parents[1]
processed_path = project_root / "data" / "processed"


def as_float_array(values):
    return np.asarray(np.atleast_1d(values), dtype=float)


def build_measured_reference_pairs(battery_id):
    mat_path = project_root / "data" / "raw" / f"{battery_id}.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    steps = np.atleast_1d(mat["data"].step)

    references = []
    for step in steps:
        if str(step.comment) != "reference discharge":
            continue

        relative_time = as_float_array(step.relativeTime)
        current = as_float_array(step.current)
        global_time = as_float_array(step.time)
        capacity_ah = np.trapezoid(np.abs(current), relative_time) / 3600
        references.append(
            {
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

    for step in steps:
        if not (
            str(step.type) == "D"
            and str(step.comment) == "discharge (random walk)"
        ):
            continue

        relative_time = as_float_array(step.relativeTime)
        global_time = as_float_array(step.time)
        voltage = as_float_array(step.voltage)
        mask = relative_time < WINDOW_SECONDS

        if np.count_nonzero(mask) != 5:
            continue
        if not START_VOLTAGE_LOW <= voltage[mask][0] <= START_VOLTAGE_HIGH:
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
            "feature_time_s": feature_time_s,
            "elapsed_experiment_hours": (
                feature_time_s - references[0]["time_s"]
            ) / 3600,
            "gap_hours": (target_reference["time_s"] - feature_time_s) / 3600,
            "measured_soh_percent": target_reference["measured_soh_percent"],
        }

        current_best = closest_feature_by_target.get(target_order)
        if current_best is None or feature_time_s > current_best["feature_time_s"]:
            closest_feature_by_target[target_order] = candidate

    return [
        closest_feature_by_target[target_order]
        for target_order in sorted(closest_feature_by_target)
        if closest_feature_by_target[target_order]["gap_hours"] <= MAX_GAP_HOURS
    ]


all_rows = {
    battery_id: build_measured_reference_pairs(battery_id)
    for battery_id in BATTERIES
}
statistics = []
fits = {}

print("Protocol-dependence check: measured SOH versus elapsed experiment time")
print(f"Pairing rule: closest 4.0-4.2 V feature with gap <= {MAX_GAP_HOURS:.0f} h")

for battery_id, rows in all_rows.items():
    time_hours = np.array(
        [row["elapsed_experiment_hours"] for row in rows], dtype=float
    )
    measured_soh = np.array(
        [row["measured_soh_percent"] for row in rows], dtype=float
    )
    model = LinearRegression().fit(time_hours.reshape(-1, 1), measured_soh)
    fitted_soh = model.predict(time_hours.reshape(-1, 1))
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = r2_score(measured_soh, fitted_soh)

    fits[battery_id] = {
        "model": model,
        "time_hours": time_hours,
        "measured_soh": measured_soh,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }
    statistics.append(
        {
            "battery": battery_id,
            "eligible_pairs": len(rows),
            "time_min_hours": time_hours.min(),
            "time_max_hours": time_hours.max(),
            "soh_min_percent": measured_soh.min(),
            "soh_max_percent": measured_soh.max(),
            "slope_soh_percent_per_hour": slope,
            "intercept_percent": intercept,
            "within_cell_r_squared": r_squared,
        }
    )

    print(
        f"{battery_id}: pairs={len(rows)}, time={time_hours.min():.1f}-"
        f"{time_hours.max():.1f} h, slope={slope:.5f} SOH%/h, "
        f"within-cell R^2={r_squared:.3f}"
    )

summary_path = processed_path / "elapsed_time_protocol_dependence_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=statistics[0].keys())
    writer.writeheader()
    writer.writerows(statistics)

figure, axes = plt.subplots(1, 4, figsize=(21, 5))
for axis, battery_id in zip(axes, BATTERIES):
    fit = fits[battery_id]
    time_hours = fit["time_hours"]
    measured_soh = fit["measured_soh"]
    line_time = np.linspace(time_hours.min(), time_hours.max(), 100)
    line_soh = fit["model"].predict(line_time.reshape(-1, 1))

    axis.scatter(
        time_hours,
        measured_soh,
        color="#1f77b4",
        s=55,
        label="Measured-reference SOH",
    )
    axis.plot(
        line_time,
        line_soh,
        color="#d62728",
        linewidth=2,
        label="Cell-specific linear fit",
    )
    axis.set_title(
        f"{battery_id}: slope={fit['slope']:.5f} SOH%/h\n"
        f"within-cell R^2={fit['r_squared']:.3f}"
    )
    axis.set_xlabel("Elapsed experiment time (h)")
    axis.set_ylabel("Measured SOH (%)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

figure.suptitle(
    "Protocol-dependence check: each cell's measured SOH versus elapsed time",
    fontsize=15,
)
figure.tight_layout()
plot_path = project_root / "outputs" / "elapsed_time_protocol_dependence.png"
figure.savefig(plot_path, dpi=180)
plt.close(figure)

print(f"\nSaved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
