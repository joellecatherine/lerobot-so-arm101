"""
Derive that noise distribution is Gaussian for captured LiDAR depth stacks.

Usage:
    python wall_drawing/depth_frames_Gaussian_noise.py   # default: noise_chalkcentered_1m

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

# 66,128
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/lidar_noise/noise_chalkcentered_1m.npz")
    # parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    stack = np.rot90(np.load(args.data)["depth"], k=1, axes=(1, 2))

    sensor_noise_wall = stack[:, 55, 57]  # numpy: (row, col)
    sensor_noise_chalkboard = stack[:, 66, 128]  # numpy: (row, col)

    m_wall = np.nanmean(sensor_noise_wall, axis=0)
    s_wall = np.nanstd(sensor_noise_wall, axis=0)

    k1_wall = (np.abs(sensor_noise_wall - m_wall) < 1*s_wall).mean()
    k2_wall = (np.abs(sensor_noise_wall - m_wall) < 2*s_wall).mean()
    k3_wall = (np.abs(sensor_noise_wall - m_wall) < 3*s_wall).mean()
    print(f"wall: mean {m_wall:.3f} m, std {s_wall*1000:.3f} mm, k1 {k1_wall:.3f}, k2 {k2_wall:.3f}, k3 {k3_wall:.3f}")

    m_chalk = np.nanmean(sensor_noise_chalkboard, axis=0)
    s_chalk = np.nanstd(sensor_noise_chalkboard, axis=0)

    k1_chalk = (np.abs(sensor_noise_chalkboard - m_chalk) < 1*s_chalk).mean()
    k2_chalk = (np.abs(sensor_noise_chalkboard - m_chalk) < 2*s_chalk).mean()
    k3_chalk = (np.abs(sensor_noise_chalkboard - m_chalk) < 3*s_chalk).mean()
    print(f"chalkboard: mean {m_chalk:.3f} m, std {s_chalk*1000:.3f} mm, k1 {k1_chalk:.3f}, k2 {k2_chalk:.3f}, k3 {k3_chalk:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(sensor_noise_wall, bins=30)
    axes[0].set_title(f"sensor noise wall, mean {m_wall:.2f} m, std {s_wall*1000:.2f} mm")
    axes[1].hist(sensor_noise_chalkboard, bins=30)
    axes[1].set_title(f"sensor noise chalkboard, mean {m_chalk:.2f} m, std {s_chalk*1000:.2f} mm")
    for ax, m, s, ks in (
        (axes[0], m_wall, s_wall, (k1_wall, k2_wall, k3_wall)),
        (axes[1], m_chalk, s_chalk, (k1_chalk, k2_chalk, k3_chalk)),
    ):
        for n, k, color in zip((1, 2, 3), ks, ("tab:green", "tab:orange", "tab:red")):
            ax.axvline(m - n * s, color=color, linestyle="--", linewidth=1)
            ax.axvline(m + n * s, color=color, linestyle="--", linewidth=1,
                       label=f"±{n}σ: k={k:.3f}")
        ax.legend()
        ax.set_xlabel("depth [m]")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{args.data.replace('.npz', '')}_depth_Gaussian_noise_1m.png", dpi=200)
    plt.show()


    u_c = np.unique(sensor_noise_chalkboard)
    print("chalkboard: ", np.diff(u_c) * 1000)   # steps in mm
    u_w = np.unique(sensor_noise_wall)
    print("wall: ", np.diff(u_w) * 1000)   # steps in mm

    mean_map = np.nanmean(stack, axis=0)          # (H, W) scene geometry
    sigma_map = np.nanstd(stack, axis=0)          # (H, W) temporal repeatability

    rois = ROIS.get(args.data, {})
    # for name, (r0, r1, c0, c1) in rois.items():
    #     sig = sigma_map[r0:r1, c0:c1]
    #     invalid_rate = np.isnan(stack[:, r0:r1, c0:c1]).mean()
    #     print(f"{name:12s} mean depth {np.nanmean(mean_map[r0:r1, c0:c1]):.3f} m | "
    #           f"sigma median {1e3 * np.nanmedian(sig):.2f} mm, "
    #           f"p95 {1e3 * np.nanpercentile(sig, 95):.2f} mm | "
    #           f"invalid {100 * invalid_rate:.2f}%")

    fig, axes = plt.subplots(1, 2,  figsize=(12, 5))
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
            ax.plot(128, 66, "o", color="yellow", markersize=5)  # matplotlib: (x, y) = (col, row)
            ax.text(128, 66 - 3, "sensor noise chalk", color="yellow", fontsize=9)
            ax.plot(57, 55, "o", color="yellow", markersize=5)  # matplotlib: (x, y) = (col, row)
            ax.text(57, 55 - 3, "sensor noise wall", color="yellow", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{args.data.replace('.npz', '')}_depth_stats.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    main()
