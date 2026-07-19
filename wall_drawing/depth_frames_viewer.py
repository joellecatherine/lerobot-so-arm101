"""
Viewer + noise statistics for captured LiDAR depth stacks.

Loads a (N, H, W) depth stack, computes the per-pixel temporal sigma map, and
reports median / p95 sigma and invalid rate per material ROI.

ROIs are (row0, row1, col0, col1) in the ROTATED frame — verify them against
the plotted rectangles before trusting any numbers.

Usage:
    python wall_drawing/depth_frames_viewer.py                     # default: noise_1m
    python wall_drawing/depth_frames_viewer.py --data data/lidar_noise/noise_2m.npz
    python wall_drawing/depth_frames_viewer.py --no-plots          # table only

New capture files need an entry in ROIS (verify rectangles on the plot first).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# (row0, row1, col0, col1) per capture, in the rotated frame.
# Set from the mean-depth image; keep a few px inside each surface's true edge.
ROIS = {
    "data/lidar_noise/noise_chalkcentered_1m.npz": {
        "wall": (0, 110, 40, 75),        # verified against plot, correct.
        "chalkboard": (22, 110, 92, 165), # verified against plot, correct.
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/lidar_noise/noise_chalkcentered_1m.npz")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    stack = np.rot90(np.load(args.data)["depth"], k=1, axes=(1, 2))

    # Mask invalid returns (Record3D encodes dropouts as 0.0) before any stats.
    n_zero = int((stack == 0.0).sum())
    n_nan = int(np.isnan(stack).sum())
    stack = np.where(stack == 0.0, np.nan, stack)
    print(f"stack {stack.shape}  invalids: {n_zero} zeros, {n_nan} NaNs "
          f"({100 * (n_zero + n_nan) / stack.size:.2f}% of all values)\n")

    mean_map = np.nanmean(stack, axis=0)          # (H, W) scene geometry
    sigma_map = np.nanstd(stack, axis=0)          # (H, W) temporal repeatability

    rois = ROIS.get(args.data, {})
    for name, (r0, r1, c0, c1) in rois.items():
        sig = sigma_map[r0:r1, c0:c1]
        invalid_rate = np.isnan(stack[:, r0:r1, c0:c1]).mean()
        print(f"{name:12s} mean depth {np.nanmean(mean_map[r0:r1, c0:c1]):.3f} m | "
              f"sigma median {1e3 * np.nanmedian(sig):.2f} mm, "
              f"p95 {1e3 * np.nanpercentile(sig, 95):.2f} mm | "
              f"invalid {100 * invalid_rate:.2f}%")

    if args.no_plots:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, img, title, kw in (
        (axes[0], mean_map, "mean depth [m]", dict(vmin=0.3, vmax=2.5)),
        (axes[1], 1e3 * sigma_map, "temporal sigma [mm]", dict(vmin=0.0, vmax=10.0)),
    ):
        im = ax.imshow(img, **kw)
        fig.colorbar(im, ax=ax, label=title)
        ax.set_title(f"{args.data}\n{title}")
        for name, (r0, r1, c0, c1) in rois.items():
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                   fill=False, edgecolor="red", linewidth=1.5))
            ax.text(c0, r0 - 3, name, color="red", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{args.data.replace('.npz', '')}_depth_stats.png", dpi=200)
    
    plt.show()



if __name__ == "__main__":
    main()
