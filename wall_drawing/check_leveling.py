"""Diagnose the gravity leveling / camera-convention.

Grabs one iPhone frame, fits the dominant plane, and prints its normal in the
gravity-leveled frame. Use it to confirm whether leveling is correct:

  * Point the phone at the FLOOR  -> leveled normal should be ~[0, +/-1, 0]
    (the vertical axis), because the floor is horizontal.
  * Point at a WALL               -> leveled normal should be ~horizontal
    (small y component).

If the FLOOR comes back with a non-vertical normal, the leveling/convention is
wrong (CV_TO_ARKIT or the quaternion handling) -- that would explain the floor
and furniture being mistaken for the board.

Usage:
    python wall_drawing/check_leveling.py     # point at the floor, then a wall
"""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d

import wall_perception as wp
from iphone_camera import Record3DCamera


def dominant_normal(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    model, inl = pcd.segment_plane(0.02, 3, 300)
    n = np.asarray(model[:3])
    return n / (np.linalg.norm(n) or 1.0), len(inl)


def main() -> None:
    cam = Record3DCamera()
    cam.connect()
    frame = cam.wait(timeout_s=10.0)
    cam.disconnect()
    if frame is None:
        raise TimeoutError("No iPhone frame. Is Record3D USB streaming on?")

    p = frame.pose
    pts_cv, _ = wp.backproject(frame.depth, frame.intrinsics, frame.rgb.shape[:2])
    pts_ark = pts_cv @ wp.CV_TO_ARKIT.T
    pts_lvl = wp.level_with_gravity(pts_cv, frame.pose)

    up_cam = wp.quat_to_matrix(p).T @ np.array([0.0, 1.0, 0.0])
    n_ark, _ = dominant_normal(pts_ark)
    n_lvl, c = dominant_normal(pts_lvl)

    print(f"\npose quat (x,y,z,w): {p.qx:+.3f} {p.qy:+.3f} {p.qz:+.3f} {p.qw:+.3f}")
    print(f"world-up in ARKit cam frame: {np.round(up_cam, 3)}")
    print(f"dominant plane normal, ARKit raw : {np.round(n_ark, 3)}")
    print(f"dominant plane normal, LEVELED   : {np.round(n_lvl, 3)}   (inliers {c})")
    print("\nInterpretation:")
    print("  pointing at FLOOR -> leveled normal should be ~[0, +/-1, 0] (vertical axis)")
    print("  pointing at WALL  -> leveled normal should be ~horizontal (|y| small)")

    os._exit(0)


if __name__ == "__main__":
    main()
