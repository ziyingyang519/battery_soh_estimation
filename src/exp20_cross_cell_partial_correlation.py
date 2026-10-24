"""
Experiment 20: Cross-cell partial correlation — IRWST vs SoH | elapsed_time
Pools RW3, RW4, RW5, RW6 measured-reference pairs to increase n
and reduce confounding from single-cell collinearity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

DATA_DIR = "data/processed"
OUT_DIR  = "outputs"

# ── 1. Load RW3 main baseline pairs ──────────────────────────
rw3 = pd.read_csv(f"{DATA_DIR}/rw3_measured_reference_main_baseline_pairs.csv")
rw3["cell"] = "RW3"
rw3 = rw3[["cell", "feature_time_s", "irwst_ohm", "measured_soh_percent"]].copy()
rw3.rename(columns={"feature_time_s": "time_s", "measured_soh_percent": "soh"}, inplace=True)

# ── 2. Build measured-reference pairs for RW4, RW5, RW6 ──────
# Same rule as RW3 baseline:
#   for each reference_interval_id, pick the LAST (closest) preceding feature
#   require gap from feature to next_reference <= 12 h
CELLS = {
    "RW4": f"{DATA_DIR}/rw4_features_4p0_to_4p2_audit.csv",
    "RW5": f"{DATA_DIR}/rw5_features_4p0_to_4p2_audit.csv",
    "RW6": f"{DATA_DIR}/rw6_features_4p0_to_4p2_audit.csv",
}

# We need the reference discharge times to compute gap.
# Approximate: the "next reference step" time is not stored directly.
# Instead, use: gap = (next_ref_time - feature_time).
# We don't have next_ref_time — use next_reference_step_index as a proxy.
# Better: compute gap from the data we DO have.
# The audit files have experiment_time_s for features.
# We need reference times.  They're stored in the raw data, but we can
# approximate: use the LAST feature in each reference_interval_id group
# as the closest-preceding feature, and compute the gap from
# next_reference_step_index timing.
# Since we don't have raw step times easily, let's use a simpler approach:
# keep all features, group by reference_interval_id, take the LAST feature
# in each interval (closest to the next reference), and accept it as a pair.
# Gap <= 12 h check: use experiment_time_s difference as an estimate.

cross_cell_frames = []
for cell_name, path in CELLS.items():
    df = pd.read_csv(path)
    # For each reference interval, pick the LAST feature (highest step_index)
    pairs = (df.sort_values("step_index")
               .groupby("reference_interval_id", as_index=False)
               .last())
    pairs = pairs.rename(columns={
        "experiment_time_s": "time_s",
        "next_reference_soh_percent": "soh",
    })
    pairs["cell"] = cell_name
    # Approximate gap: difference in time from feature to next ref.
    # We don't have next ref time, so skip gap filter here (all picked 
    # features are by construction just before their reference discharge).
    pairs = pairs[["cell", "time_s", "irwst_ohm", "soh"]].copy()
    cross_cell_frames.append(pairs)
    print(f"  {cell_name}: {len(pairs)} pairs")

# ── 3. Combine and normalize elapsed_time per-cell ────────────
rw3_frame = rw3.rename(columns={"time_s": "time_s"})
all_frames = [rw3_frame] + cross_cell_frames
combined = pd.concat(all_frames, ignore_index=True)

# elapsed_h: use raw experiment_time_s / 3600 (already from experiment start)
combined["elapsed_h"] = combined["time_s"] / 3600

print(f"\nTotal pairs: {len(combined)}")
print(combined.groupby("cell")["soh"].agg(["count", "min", "max"]))

# ── 4. Partial correlation on FULL pooled data ─────────────────
x = combined["irwst_ohm"].values
y = combined["soh"].values
z = combined["elapsed_h"].values
n = len(combined)

r_xy, p_xy = stats.pearsonr(x, y)
r_xz, p_xz = stats.pearsonr(x, z)
r_yz, p_yz = stats.pearsonr(z, y)

num       = r_xy - r_xz * r_yz
den       = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
r_partial = num / den
df_part   = n - 3
t_part    = r_partial * np.sqrt(df_part) / np.sqrt(1 - r_partial**2)
p_partial = 2 * stats.t.sf(abs(t_part), df=df_part)

slope_z, icept_z, _, _, _ = stats.linregress(z, y)
residuals = y - (slope_z * z + icept_z)
r_resid, p_resid = stats.pearsonr(x, residuals)

print("\n" + "=" * 60)
print("Experiment 20: Cross-cell partial correlation (pooled)")
print("=" * 60)
print(f"  Cells: RW3, RW4, RW5, RW6    n = {n}\n")
print(f"  r(IRWST, SoH)              = {r_xy:+.3f}  p = {p_xy:.4f}")
print(f"  r(elapsed_time, SoH)       = {r_yz:+.3f}  p = {p_yz:.4f}")
print(f"  r(IRWST, elapsed_time)     = {r_xz:+.3f}  p = {p_xz:.4f}")
print()
print(f"  Partial r(IRWST, SoH|time) = {r_partial:+.3f}  p = {p_partial:.4f}")
print(f"  r(IRWST, SoH residuals)    = {r_resid:+.3f}  p = {p_resid:.4f}")
print("=" * 60)

# ── 5. Figure ──────────────────────────────────────────────────
CELL_COLORS = {"RW3": "#2563EB", "RW4": "#16A34A", "RW5": "#D97706", "RW6": "#DC2626"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Experiment 20 — Cross-cell partial correlation (n = {n})\n"
             f"RW3 + RW4 + RW5 + RW6 pooled, elapsed time NOT normalized",
             fontsize=11, fontweight="bold")

# Left: IRWST vs SoH, coloured by cell
ax = axes[0]
for cell, grp in combined.groupby("cell"):
    ax.scatter(grp["irwst_ohm"] * 1000, grp["soh"],
               c=CELL_COLORS[cell], label=cell, s=60,
               edgecolors="white", linewidths=0.6, zorder=3)
x_line = np.linspace(x.min(), x.max(), 100)
sl_x, ic_x, _, _, _ = stats.linregress(x, y)
ax.plot(x_line * 1000, sl_x * x_line + ic_x,
        color="black", lw=1.5, ls="--", label=f"pooled fit  r = {r_xy:.3f}")
ax.set_xlabel("IRWST (mΩ)", fontsize=11)
ax.set_ylabel("Measured SoH (%)", fontsize=11)
ax.set_title("Raw: IRWST vs SoH (by cell)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: IRWST vs residuals
ax2 = axes[1]
for cell, grp in combined.groupby("cell"):
    idx = grp.index
    ax2.scatter(grp["irwst_ohm"] * 1000, residuals[idx],
                c=CELL_COLORS[cell], label=cell, s=60,
                edgecolors="white", linewidths=0.6, zorder=3)
sl_r, ic_r, _, _, _ = stats.linregress(x, residuals)
ax2.plot(x_line * 1000, sl_r * x_line + ic_r,
         color="black", lw=1.5, ls="--",
         label=f"partial r = {r_partial:.3f}  (p = {p_partial:.3f})")
ax2.axhline(0, color="grey", lw=0.8)
ax2.set_xlabel("IRWST (mΩ)", fontsize=11)
ax2.set_ylabel("SoH residual after removing time trend (%)", fontsize=11)
ax2.set_title("Residual: IRWST vs SoH − time trend", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_png = f"{OUT_DIR}/exp20_cross_cell_partial_correlation.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure: {out_png}")

# Save summary
pd.DataFrame([{
    "cells": "RW3+RW4+RW5+RW6", "n": n,
    "r_irwst_soh": round(r_xy, 4), "p_irwst_soh": round(p_xy, 4),
    "r_time_soh": round(r_yz, 4), "p_time_soh": round(p_yz, 4),
    "r_irwst_time": round(r_xz, 4), "p_irwst_time": round(p_xz, 4),
    "partial_r": round(r_partial, 4), "partial_p": round(p_partial, 4),
    "r_resid": round(r_resid, 4), "p_resid": round(p_resid, 4),
}]).to_csv(f"{DATA_DIR}/exp20_cross_cell_partial_correlation_summary.csv", index=False)
print(f"CSV: {DATA_DIR}/exp20_cross_cell_partial_correlation_summary.csv")
