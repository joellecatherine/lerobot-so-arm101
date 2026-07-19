"""Capture iPhone frames of the board for offline analysis.

Live viewer with the current detection overlaid (green). Press 'S' in the window
to SAVE the current frame; close the window (or 'Q') to quit. Point at the board
when it's recognised correctly and save a few; also save some false-positive
scenes (floor / furniture wrongly green) for comparison.

Each save writes RGB, depth, intrinsics, pose, the leveled point cloud + colors,
the current detection result, and a .ply -- everything needed to analyze offline
what separates the board from the false positives.

Usage:
    python wall_drawing/capture_board.py
Saves to: outputs/board_captures/cap_*/
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import open3d as o3d

import board_tracker as bt
import wall_perception as wp
from iphone_camera import Record3DCamera

OUT = "outputs/board_captures"
MAX_BOARD_HEIGHT = 0.4


def leveled_cloud(frame):
    pts_cv, valid = wp.backproject(frame.depth, frame.intrinsics, frame.rgb.shape[:2])
    pts = wp.level_with_gravity(pts_cv, frame.pose)
    colors = wp.sample_colors(frame.rgb, frame.depth.shape, valid)
    return wp.downsample(pts, colors)


def detect(pts, colors):
    """Run the current board heuristic on the low region. Returns (plane|None, thr, mask)."""
    low_pts, low_cols = bt.filter_low(pts, colors, MAX_BOARD_HEIGHT)
    if len(low_pts) <= 50:
        return None, 1.0, np.zeros(len(pts), bool)
    thr = wp.adaptive_dark_threshold(low_cols)
    plane, _inl, _label, _ = wp.fit_board_plane(low_pts, low_cols, thr)
    if plane is None:
        return None, thr, np.zeros(len(pts), bool)
    mask = (pts[:, 1] <= MAX_BOARD_HEIGHT) & (np.abs(pts @ plane[:3] + plane[3]) < wp.DIST_THRESH * 2)
    return plane, thr, mask


def save_capture(frame, pts, colors, plane, thr, mask):
    d = os.path.join(OUT, f"cap_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/rgb.png", cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR))
    np.save(f"{d}/depth.npy", frame.depth)
    np.save(f"{d}/intrinsics.npy", frame.intrinsics)
    p = frame.pose
    np.save(f"{d}/pose.npy", np.array([p.qx, p.qy, p.qz, p.qw, p.tx, p.ty, p.tz]))
    np.save(f"{d}/points.npy", pts)
    np.save(f"{d}/colors.npy", colors)
    np.save(f"{d}/board_mask.npy", mask)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"{d}/cloud.ply", pc)
    meta = {
        "detected": plane is not None,
        "plane": plane.tolist() if plane is not None else None,
        "board_points": int(mask.sum()),
        "dark_threshold": float(thr),
        "total_points": int(len(pts)),
    }
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=2)
    print(f"\nSaved {d}  (detected={meta['detected']}, board_points={meta['board_points']})")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cam = Record3DCamera()
    cam.connect()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Capture board (press S to save, close window to quit)")
    pcd = o3d.geometry.PointCloud()
    added = False
    save_flag = [False]
    vis.register_key_callback(ord("S"), lambda v: save_flag.__setitem__(0, True))

    print("Point at the board; press S in the window to save. Close window to quit.")
    latest = {}
    try:
        while True:
            frame = cam.wait(timeout_s=2.0)
            if frame is None:
                if not vis.poll_events():
                    break
                continue
            pts, colors = leveled_cloud(frame)
            plane, thr, mask = detect(pts, colors)

            vis_colors = colors.copy()
            vis_colors[mask] = [0.1, 0.9, 0.2]
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            if not added:
                vis.add_geometry(pcd)
                added = True
            else:
                vis.update_geometry(pcd)

            latest = dict(frame=frame, pts=pts, colors=colors, plane=plane, thr=thr, mask=mask)
            print(f"\rdetected={plane is not None}  board_points={int(mask.sum()):>5}  "
                  f"(press S to save)   ", end="", flush=True)

            if save_flag[0]:
                save_flag[0] = False
                save_capture(**latest)

            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        vis.destroy_window()
        cam.disconnect()

    os._exit(0)


if __name__ == "__main__":
    main()
