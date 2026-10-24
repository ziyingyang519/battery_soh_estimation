"""
Experiment 19: Does IRWST contain information beyond elapsed time?

Method:
  1. Partial correlation: r(IRWST, SoH | elapsed_time)
  2. Residual model: fit SoH ~ elapsed_time (OLS), correlate residuals with IRWST
  3. Two-panel figure:
       Left  — IRWST vs SoH, colour = elapsed time (shows raw relationship)
       Right — IRWST vs SoH residual (after removing time trend)

Uses ALL 16 pairs (exploratory diagnostic, not a predictive split).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# ── Load ──────────────────────────────────────────────────
DATA = "data/processed/rw3_measured_reference_main_baseline_pairs.csv"
df = pd.read_csv(DATA)

df["elapsed_h"] = df["feature_time_s"] / 3600   # seconds → hours

x = df["irwst_ohm"].values
y = df["measured_soh_percent"].values
z = df["elapsed_h"].values
n = len(df)

# ── 1. Simple correlations ─────────────────────────────────
r_xy, p_xy = stats.pearsonr(x, y)   # IRWST vs SoH
r_xz, p_xz = stats.pearsonr(x, z)   # IRWST vs elapsed_time
r_yz, p_yz = stats.pearsonr(z, y)   # elapsed_time vs SoH

# ── 2. Partial correlation r(IRWST, SoH | elapsed_time) ───
num  = r_xy - r_xz * r_yz
den  = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
r_partial = num / den

df_partial = n - 3
t_partial  = r_partial * np.sqrt(df_partial) / np.sqrt(1 - r_partial**2)
p_partial  = 2 * stats.t.sf(abs(t_partial), df=df_partial)

# ── 3. Residual model ──────────────────────────────────────
slope_z, intercept_z, _, _, _ = stats.linregress(z, y)
y_hat     = slope_z * z + intercept_z
residuals = y - y_hat

r_resid, p_resid = stats.pearsonr(x, residuals)

slope_x, intercept_x, _, _, _ = stats.linregress(x, y)

# ── 4. Print results ───────────────────────────────────────
print("=" * 55)
print("Experiment 19: IRWST partial correlation analysis")
print("=" * 55)
print(f"  n = {n} pairs (all measured-reference, RW3)\n")
print(f"  r(IRWST,  SoH)           = {r_xy:+.3f}  p = {p_xy:.4f}")
print(f"  r(time,   SoH)           = {r_yz:+.3f}  p = {p_yz:.4f}")
print(f"  r(IRWST,  time)          = {r_xz:+.3f}  p = {p_xz:.4f}")
print()
print(f"  Partial r(IRWST, SoH | time)  = {r_partial:+.3f}  p = {p_partial:.4f}")
print(f"  r(IRWST, SoH residuals)       = {r_resid:+.3f}  p = {p_resid:.4f}")
print()
if abs(r_partial) >= 0.5 and p_partial < 0.05:
    verdict = "IRWST carries INDEPENDENT health information beyond elapsed time."
elif abs(r_partial) >= 0.3 and p_partial < 0.10:
    verdict = "IRWST shows WEAK independent signal — suggestive but not conclusive."
else:
    verdict = "No significant independent signal: IRWST adds little beyond elapsed time."
print(f"  Verdict: {verdict}")
print("=" * 55)

# ── 5. Figure ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Experiment 19 — Does IRWST predict SoH beyond elapsed time?\n"
             f"(RW3, n = {n} measured-reference pairs)",
             fontsize=12, fontweight="bold")

# ── Left panel: IRWST vs SoH, coloured by elapsed time ───
ax = axes[0]
sc = ax.scatter(x * 1000, y, c=z, cmap="Blues_r",
                vmin=z.min(), vmax=z.max(),
                s=70, edgecolors="white", linewidths=0.8, zorder=3)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line * 1000, slope_x * x_line + intercept_x,
        color="#d62728", lw=1.5, ls="--", label=f"IRWST fit  r = {r_xy:.3f}")
ax.set_xlabel("IRWST (mΩ)", fontsize=11)
ax.set_ylabel("Measured SoH (%)", fontsize=11)
ax.set_title("Raw relationship\n(colour = elapsed experiment hours)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("Elapsed time (h)", fontsize=9)

# ── Right panel: IRWST vs SoH residuals ──────────────────
ax2 = axes[1]
sc2 = ax2.scatter(x * 1000, residuals, c=z, cmap="Blues_r",
                  vmin=z.min(), vmax=z.max(),
                  s=70, edgecolors="white", linewidths=0.8, zorder=3)

sl_r, ic_r, _, _, _ = stats.linregress(x, residuals)
ax2.plot(x_line * 1000, sl_r * x_line + ic_r,
         color="#d62728", lw=1.5, ls="--",
         label=f"partial r = {r_partial:.3f}  (p = {p_partial:.3f})")
ax2.axhline(0, color="grey", lw=0.8, ls="-")
ax2.set_xlabel("IRWST (mΩ)", fontsize=11)
ax2.set_ylabel("SoH residual after removing time trend (%)", fontsize=11)
ax2.set_title("Residual correlation\n(SoH with elapsed-time trend removed)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_png = "outputs/exp19_irwst_partial_correlation.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")
plt.close()

# ── 6. Save summary CSV ────────────────────────────────────
summary = pd.DataFrame([{
    "r_irwst_soh":  round(r_xy,      4),
    "p_irwst_soh":  round(p_xy,      4),
    "r_time_soh":   round(r_yz,      4),
    "p_time_soh":   round(p_yz,      4),
    "r_irwst_time": round(r_xz,      4),
    "p_irwst_time": round(p_xz,      4),
    "partial_r":    round(r_partial,  4),
    "partial_p":    round(p_partial,  4),
    "r_resid":      round(r_resid,   4),
    "p_resid":      round(p_resid,   4),
    "n":            n,
    "verdict":      verdict,
}])
csv_out = "data/processed/exp19_irwst_partial_correlation_summary.csv"
summary.to_csv(csv_out, index=False)
print(f"Summary saved: {csv_out}")
