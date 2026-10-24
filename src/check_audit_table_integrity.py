import csv
import argparse
from pathlib import Path

import numpy as np


WINDOW_SECONDS = 5.0
VOLTAGE_WINDOW_LOW = 4.0
VOLTAGE_WINDOW_HIGH = 4.2
TOLERANCE = 1e-9

parser = argparse.ArgumentParser(
    description="Check an audited five-second SOH feature table."
)
parser.add_argument(
    "battery_id",
    nargs="?",
    default="RW3",
    help="Battery feature-table ID, for example RW3 or RW4.",
)
args = parser.parse_args()
BATTERY_ID = args.battery_id.upper()

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / f"{BATTERY_ID.lower()}_features_4p0_to_4p2_audit.csv"
)

required_columns = [
    "random_order",
    "experiment_time_s",
    "window_sample_count",
    "window_start_time_s",
    "window_end_time_s",
    "start_voltage_v",
    "end_voltage_v",
    "end_current_a",
    "reference_interval_id",
    "previous_reference_soh_percent",
    "next_reference_soh_percent",
    "interpolation_fraction",
    "soh_percent",
    "irwst_ohm",
    "vsdwst_v",
]


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {detail}")
    return passed


with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print(f"Checking {BATTERY_ID}: {len(rows)} feature rows")

missing_columns = [
    column for column in required_columns if column not in rows[0]
]

all_passed = report(
    "Required audit columns",
    not missing_columns,
    "all present" if not missing_columns else f"missing: {missing_columns}",
)

if missing_columns:
    raise SystemExit("Rebuild the audit feature table before running this check.")

numeric_columns = required_columns
values = {
    column: np.array([float(row[column]) for row in rows], dtype=float)
    for column in numeric_columns
}

all_passed &= report(
    "Finite numeric values",
    all(np.isfinite(column_values).all() for column_values in values.values()),
    f"checked {len(rows)} rows",
)

expected_order = np.arange(1, len(rows) + 1)
all_passed &= report(
    "Random-walk order",
    np.array_equal(values["random_order"], expected_order),
    "consecutive from 1 to sample count",
)

all_passed &= report(
    "Chronological experiment time",
    np.all(np.diff(values["experiment_time_s"]) > 0),
    "strictly increasing",
)

all_passed &= report(
    "First-five-second window",
    np.all(values["window_start_time_s"] >= -TOLERANCE)
    and np.all(values["window_end_time_s"] < WINDOW_SECONDS),
    (
        f"start={values['window_start_time_s'].min():.3f}-"
        f"{values['window_start_time_s'].max():.3f} s, "
        f"end={values['window_end_time_s'].min():.3f}-"
        f"{values['window_end_time_s'].max():.3f} s"
    ),
)

all_passed &= report(
    "Five samples per window",
    np.all(values["window_sample_count"] == 5),
    f"observed counts: {sorted(set(values['window_sample_count'].astype(int)))}",
)

all_passed &= report(
    "4.0-4.2 V start selection",
    np.all(values["start_voltage_v"] >= VOLTAGE_WINDOW_LOW - TOLERANCE)
    and np.all(values["start_voltage_v"] <= VOLTAGE_WINDOW_HIGH + TOLERANCE),
    (
        f"observed: {values['start_voltage_v'].min():.3f}-"
        f"{values['start_voltage_v'].max():.3f} V"
    ),
)

all_passed &= report(
    "Positive end current",
    np.all(values["end_current_a"] > 0),
    (
        f"observed: {values['end_current_a'].min():.3f}-"
        f"{values['end_current_a'].max():.3f} A"
    ),
)

recomputed_irwst = (
    (values["start_voltage_v"] - values["end_voltage_v"])
    / values["end_current_a"]
)
all_passed &= report(
    "IRWST calculation",
    np.allclose(values["irwst_ohm"], recomputed_irwst, atol=TOLERANCE),
    "matches (V_start - V_end) / I_end",
)

lower_soh = np.minimum(
    values["previous_reference_soh_percent"],
    values["next_reference_soh_percent"],
)
upper_soh = np.maximum(
    values["previous_reference_soh_percent"],
    values["next_reference_soh_percent"],
)
all_passed &= report(
    "Interpolation fraction",
    np.all(values["interpolation_fraction"] >= -TOLERANCE)
    and np.all(values["interpolation_fraction"] <= 1 + TOLERANCE),
    (
        f"observed: {values['interpolation_fraction'].min():.3f}-"
        f"{values['interpolation_fraction'].max():.3f}"
    ),
)

all_passed &= report(
    "Interpolated SOH bounds",
    np.all(values["soh_percent"] >= lower_soh - TOLERANCE)
    and np.all(values["soh_percent"] <= upper_soh + TOLERANCE),
    "every label lies between its two reference SOH values",
)

if not all_passed:
    raise SystemExit("\nIntegrity check failed. Do not train a model until the failed check is understood.")

print("\nAll audit-table integrity checks passed.")
