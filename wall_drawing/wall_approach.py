"""Approach the detected board to a fixed standoff pose (milestone: approach).

Two phases:
  * LOCK  : search only the LOW region (chalkboard is mounted low) for a stable
            dark vertical board; auto-lock once it's consistent for a few frames.
            The robot holds still while searching.
  * TRACK : stop re-detecting -- refit only the locked board (continuity-gated,
            EMA-smoothed, holds through brief LiDAR dropouts) and drive to the
            standoff. This stops the target jumping between surfaces.

Modes: --drive (move the robot), --view (live point cloud; green = locked board).
Pose is camera-relative (iPhone rigidly forward-mounted), so no hand-eye needed.

Usage:
    python wall_drawing/wall_approach.py --view                 # eyeball + viewer
    python wall_drawing/wall_approach.py --drive --robot-port /dev/tty.usbmodemXXXX
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import open3d as o3d

import board_tracker as bt
import wall_perception as wp
from iphone_camera import Record3DCamera

# --- target + controller tunables ---
TARGET_DISTANCE = 0.30
KX, KY, KYAW = 0.25, 0.25, 0.8    # slower/gentler control
V_MAX, W_MAX = 0.07, 12.0         # lower velocity caps so it can't escalate
D_TOL, LAT_TOL, YAW_TOL = 0.03, 0.03, 3.0
MIN_STOP = 0.15
FWD_SIGN, LAT_SIGN, YAW_SIGN = +1.0, -1.0, -1.0
# Camera (iPhone) offset from the base's rotation centre -- MEASURE these.
# The camera sits forward-and-right of centre (over the front-right wheel), so
# camera-relative pose != base-relative. These correct distance / lateral / yaw
# to the base centre. Signs: forward = ahead of centre, right = to the right.
CAMERA_FORWARD_OFFSET = 0.0   # m the camera is AHEAD of the base centre
CAMERA_RIGHT_OFFSET = 0.0     # m the camera is to the RIGHT of the base centre
CAMERA_YAW_OFFSET = 0.0        # deg the phone is rotated from base-forward (0 if straight ahead)
# --- detection / lock tunables ---
MAX_BOARD_HEIGHT = 0.4   # only search for the board below this height above the camera (m)
LOCK_FRAMES = 5          # consecutive consistent detections needed to auto-lock

GREEN = [0.1, 0.9, 0.2]
ZERO_BASE = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


def estimate_wall_pose(plane: np.ndarray, centroid: np.ndarray):
    n = plane[:3].astype(float)
    d = float(plane[3])
    norm = np.linalg.norm(n) or 1.0
    n, d = n / norm, d / norm
    if n[2] < 0:
        n, d = -n, -d
    distance = abs(d)
    yaw_deg = float(np.degrees(np.arctan2(n[0], n[2])))
    lateral = float(centroid[0])
    return distance, lateral, yaw_deg


def _clamp(v: float, lim: float) -> float:
    return float(np.clip(v, -lim, lim))


def compute_base_command(distance: float, lateral: float, yaw_deg: float,
                         target_distance: float = TARGET_DISTANCE):
    e_fwd = distance - target_distance
    arrived = abs(e_fwd) < D_TOL and abs(lateral) < LAT_TOL and abs(yaw_deg) < YAW_TOL
    x = 0.0 if abs(e_fwd) < D_TOL else _clamp(FWD_SIGN * KX * e_fwd, V_MAX)
    y = 0.0 if abs(lateral) < LAT_TOL else _clamp(LAT_SIGN * KY * lateral, V_MAX)
    th = 0.0 if abs(yaw_deg) < YAW_TOL else _clamp(YAW_SIGN * KYAW * yaw_deg, W_MAX)
    if distance < MIN_STOP:
        x = min(x, 0.0) if FWD_SIGN > 0 else max(x, 0.0)
    return {"x.vel": x, "y.vel": y, "theta.vel": th}, arrived


def cloud(frame):
    """iPhone frame -> (leveled+downsampled points, colors)."""
    pts_cv, valid = wp.backproject(frame.depth, frame.intrinsics, frame.rgb.shape[:2])
    pts = wp.level_with_gravity(pts_cv, frame.pose)
    colors = wp.sample_colors(frame.rgb, frame.depth.shape, valid)
    return wp.downsample(pts, colors)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive", action="store_true", help="actually move the robot")
    p.add_argument("--view", action="store_true", help="show the live point-cloud viewer")
    p.add_argument("--robot-port", help="LeKiwi motor-bus serial port (required with --drive)")
    p.add_argument("--robot-id", default="my_lekiwi")
    p.add_argument("--target-distance", type=float, default=TARGET_DISTANCE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cam = Record3DCamera()
    cam.connect()

    robot = None
    arm_hold = {}
    if args.drive:
        if not args.robot_port:
            raise SystemExit("--drive requires --robot-port")
        from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
        robot = LeKiwi(LeKiwiConfig(port=args.robot_port, id=args.robot_id, cameras={}))
        robot.connect()
        arm_hold = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        print("DRIVE mode: holds still while searching, drives once locked. Ctrl-C to stop.")
    else:
        print("EYEBALL mode: no motion.")

    vis, pcd, added = None, None, False
    if args.view:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Approach (green = locked board)")
        pcd = o3d.geometry.PointCloud()

    def hold():
        if robot is not None:
            robot.send_action({**arm_hold, **ZERO_BASE})

    tracker = bt.BoardTracker()

    # Manual lock: YOU pick the target once. On loss it holds -- never auto-grabs
    # a new wall -- so it can't wander around the room.
    from pynput import keyboard
    lock_req = [False]
    release_req = [False]

    def _on_press(key):
        c = getattr(key, "char", None)
        if c == "l":
            lock_req[0] = True
        elif c == "u":
            release_req[0] = True

    keyboard.Listener(on_press=_on_press).start()
    print("Aim at the board, press 'l' to LOCK. Press 'u' to release. (needs terminal focus)")

    try:
        while True:
            frame = cam.wait(timeout_s=2.0)
            if frame is None:
                hold()
                if vis is not None and not vis.poll_events():
                    break
                continue

            pts, colors = cloud(frame)
            highlight = np.zeros(len(pts), bool)

            if not tracker.is_locked:
                # UNLOCKED: show the candidate, hold still, wait for a manual lock.
                low_pts, low_cols = bt.filter_low(pts, colors, MAX_BOARD_HEIGHT)
                plane = None
                if len(low_pts) > 50:
                    thr = wp.adaptive_dark_threshold(low_cols)
                    plane, inliers, _, _ = wp.fit_board_plane(low_pts, low_cols, thr)
                if plane is not None:
                    cand_centroid = low_pts[inliers].mean(axis=0)
                    highlight = (pts[:, 1] <= MAX_BOARD_HEIGHT) & \
                                (np.abs(pts @ plane[:3] + plane[3]) < wp.DIST_THRESH * 2)
                    if lock_req[0]:
                        tracker.lock(plane, cand_centroid)
                        status = "LOCKED"
                    else:
                        status = "candidate ready - press 'l' to LOCK"
                else:
                    status = "searching (no candidate) - aim so green covers the board"
                lock_req[0] = False
                hold()
            elif release_req[0]:
                release_req[0] = False
                tracker.release()
                status = "released - press 'l' to lock"
                hold()
            else:
                # TRACK PHASE: refit the LOCKED board only; drive. On loss -> hold.
                res = tracker.update(pts)
                if res is None:
                    status = "LOST - holding (aim + press 'l' to re-lock)"
                    hold()
                else:
                    plane, centroid = res
                    distance, lateral, yaw = estimate_wall_pose(plane, centroid)
                    # Correct camera-relative pose to the base centre (camera is offset).
                    distance += CAMERA_FORWARD_OFFSET
                    lateral += CAMERA_RIGHT_OFFSET
                    yaw -= CAMERA_YAW_OFFSET
                    cmd, arrived = compute_base_command(distance, lateral, yaw, args.target_distance)
                    status = (f"{'ARRIVED' if arrived else 'approach'}  dist={distance:5.2f} "
                              f"lat={lateral:+5.2f} yaw={yaw:+6.1f} -> "
                              f"x={cmd['x.vel']:+.2f} y={cmd['y.vel']:+.2f} th={cmd['theta.vel']:+5.1f}")
                    if robot is not None:
                        robot.send_action({**arm_hold, **(ZERO_BASE if arrived else cmd)})
                    highlight = (np.abs(pts @ plane[:3] + plane[3]) < tracker.band) & \
                                (np.linalg.norm(pts - centroid, axis=1) < tracker.radius)

            print(f"\r{status:70s}", end="", flush=True)

            if vis is not None:
                vis_colors = colors.copy()
                vis_colors[highlight] = GREEN
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                if not added:
                    vis.add_geometry(pcd)
                    added = True
                else:
                    vis.update_geometry(pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if robot is not None:
            robot.stop_base()
            robot.disconnect()
        if vis is not None:
            vis.destroy_window()
        cam.disconnect()

    os._exit(0)  # dodge the record3d C++ teardown segfault


if __name__ == "__main__":
    main()
