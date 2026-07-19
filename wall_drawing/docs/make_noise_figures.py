"""Generate figures for docs/01_sensor_noise.md from the measured noise table.

Data below is the experimental record (2026-07-12, iPhone 15 Pro LiDAR via
Record3D USB, 300 static frames per capture, per-pixel temporal sigma,
material-homogeneous ROIs — see depth_frames_viewer.py for ROI definitions).

Usage:
    python wall_drawing/docs/make_noise_figures.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# (distance_m, sigma_median_mm, sigma_p95_mm) — distances are per-surface means.
WALL = np.array([(0.527, 0.32, 0.49), (1.046, 0.41, 0.60), (2.031, 0.73, 1.19)])
# NOTE: in all three CHALK captures the foil sat at the frame's right edge; the
# A/B control below showed most of its elevated sigma is a frame-position
# effect, so the CHALK fit describes edge-positioned dark foil specifically.
CHALK = np.array([(0.244, 0.46, 0.73), (0.779, 0.59, 1.02), (1.735, 1.02, 1.56)])
CHALK_CENTERED = (0.783, 0.32, 0.45)   # A/B control: same foil, frame center

# Okabe-Ito, colorblind-safe.
COLORS = {"wall": "#0072B2", "chalkboard foil (frame edge)": "#E69F00"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, data in (("wall", WALL), ("chalkboard foil (frame edge)", CHALK)):
        d, med, p95 = data.T
        c = COLORS[name]
        # Affine noise model sigma(d) = sigma0 + k*d, fit on medians.
        k, sigma0 = np.polyfit(d, med, 1)
        dd = np.linspace(0, 2.2, 50)
        ax.plot(dd, sigma0 + k * dd, color=c, linewidth=1.2, alpha=0.5, zorder=1)
        ax.plot(d, med, "o", color=c, markersize=7, zorder=3,
                label=f"{name} — median (fit: {sigma0:.2f} + {k:.2f}·d mm)")
        ax.plot(d, p95, "o", color=c, markersize=7, markerfacecolor="none",
                zorder=2, label=f"{name} — p95")

    d_c, med_c, p95_c = CHALK_CENTERED
    c = COLORS["chalkboard foil (frame edge)"]
    ax.plot(d_c, med_c, "D", color=c, markersize=8, zorder=4,
            label="chalk foil, frame CENTER — median (A/B control)")
    ax.plot(d_c, p95_c, "D", color=c, markersize=8, markerfacecolor="none",
            zorder=4, label="chalk foil, frame CENTER — p95")
    ax.annotate("position effect:\nsame foil, centered",
                xy=(d_c, med_c), xytext=(1.05, 0.18), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("distance to surface [m]")
    ax.set_ylabel("temporal depth sigma [mm]")
    ax.set_title("iPhone LiDAR depth repeatability vs distance and material\n"
                 "(300 static frames per capture, per-pixel sigma over time)")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    out = os.path.join(OUT_DIR, "sigma_vs_distance.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
