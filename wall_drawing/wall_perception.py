"""Board-plane perception from the iPhone LiDAR (project milestones M2 + M3).

Live feed. Each frame:
  1. back-project the iPhone depth into a point cloud (K scaled to depth res),
  2. LEVEL it with gravity from the ARKit pose -- roll/pitch only, keeping the
     phone's own heading (so the view faces forward, not spun by ARKit's yaw),
  3. voxel-downsample (we don't need full resolution; keeps the feed smooth),
  4. sequential-RANSAC the target board: a plane that is vertical (skip
     floor/ceiling) AND dark (skip the bright wall), then keep its largest
     spatial cluster above a minimum size (drop small scattered dark objects).

Convention: Record3D depth is OpenCV (x-right, y-down, z-forward); ARKit pose is
(x-right, y-up, z-back). CV_TO_ARKIT bridges them.

Usage:
    1. iPhone: Record3D, USB streaming on, pointed at the board.
    2. Run:  python wall_drawing/wall_perception.py
"""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d

from iphone_camera import Record3DCamera

CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0])  # OpenCV cam frame -> ARKit cam frame

# Tunables (adjust to your board / lighting / room).
VOXEL = 0.03            # downsample voxel size (m)
MAX_UP_DOT = 0.5        # |normal . up| below this = vertical plane
DIST_THRESH = 0.02      # RANSAC inlier distance (m)
# Darkness is adaptive (Otsu per frame) rather than a fixed value, so it tracks
# lighting / auto-exposure. The Otsu split is clamped to this range for safety.
DARK_THR_LO = 0.15
DARK_THR_HI = 0.55
MIN_INLIERS = 120       # a board must have at least this many (downsampled) points
CLUSTER_EPS = 0.06      # DBSCAN neighbour distance (m) for isolating the board
CLUSTER_MIN = 10        # DBSCAN min points per cluster


def backproject(depth: np.ndarray, k_rgb: np.ndarray, rgb_hw: tuple[int, int],
                z_min: float = 0.1, z_max: float = 5.0):
    """Depth map -> Nx3 points in the (OpenCV) camera frame, plus validity mask."""
    dh, dw = depth.shape
    rh, rw = rgb_hw
    sx, sy = dw / rw, dh / rh
    fx, fy = k_rgb[0, 0] * sx, k_rgb[1, 1] * sy
    cx, cy = k_rgb[0, 2] * sx, k_rgb[1, 2] * sy

    us, vs = np.meshgrid(np.arange(dw), np.arange(dh))
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts = np.stack([x, y, z], axis=-1)[valid]
    return pts, valid


def sample_colors(rgb: np.ndarray, depth_hw: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    dh, dw = depth_hw
    rh, rw = rgb.shape[:2]
    vs, us = np.nonzero(valid)
    ru = np.clip((us * rw / dw).astype(int), 0, rw - 1)
    rv = np.clip((vs * rh / dh).astype(int), 0, rh - 1)
    return rgb[rv, ru].astype(np.float64) / 255.0


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's method: the brightness split (0-1) between the dark and bright
    classes of the scene. Adapts to lighting instead of a fixed cutoff."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.5
    prob = hist / total
    mids = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)                 # class-0 weight
    mu = np.cumsum(prob * mids)             # class-0 cumulative mean
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, 1), 0.0)
    return float(mids[int(np.argmax(sigma_b))])


def adaptive_dark_threshold(colors: np.ndarray) -> float:
    """Per-frame 'dark' cutoff = Otsu split of the scene, clamped for safety."""
    brightness = colors.mean(axis=1)
    return float(np.clip(otsu_threshold(brightness), DARK_THR_LO, DARK_THR_HI))


def quat_to_matrix(pose) -> np.ndarray:
    """ARKit camera->world rotation from a Record3D pose (qx,qy,qz,qw)."""
    x, y, z, w = pose.qx, pose.qy, pose.qz, pose.qw
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-8:                 # already aligned or opposite
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def level_with_gravity(points_cv: np.ndarray, pose) -> np.ndarray:
    """Express points in the ARKit camera frame and remove roll/pitch using
    gravity from the pose (keeping the camera's heading -> forward-facing view)."""
    pts_ark = points_cv @ CV_TO_ARKIT.T                     # OpenCV -> ARKit cam frame
    up_cam = quat_to_matrix(pose).T @ np.array([0.0, 1.0, 0.0])  # world-up in cam frame
    r_level = _shortest_arc(up_cam, np.array([0.0, 1.0, 0.0]))
    return pts_ark @ r_level.T


def downsample(points: np.ndarray, colors: np.ndarray, voxel: float = VOXEL):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def fit_board_plane(points: np.ndarray, colors: np.ndarray, max_brightness: float,
                    up=(0.0, 1.0, 0.0)):
    """Sequential RANSAC for the drawable board: first plane that is vertical AND
    dark (mean inlier brightness < max_brightness, an adaptive per-frame cutoff);
    then keep its largest spatial cluster above MIN_INLIERS. Returns
    (plane [a,b,c,d] or None, inlier idx or None, label, n_skipped)."""
    up = np.asarray(up, float)
    up /= np.linalg.norm(up)
    active = np.ones(len(points), bool)
    skipped = 0

    for _ in range(8):
        idx = np.where(active)[0]
        if len(idx) < MIN_INLIERS:
            break
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(points[idx])
        model, inl = sub.segment_plane(DIST_THRESH, 3, 300)
        orig_inl = idx[np.asarray(inl)]
        vertical = abs(float(np.asarray(model[:3]) @ up)) < MAX_UP_DOT
        dark = float(colors[orig_inl].mean()) < max_brightness

        if vertical and dark:
            orig_inl = _largest_cluster(points[orig_inl], orig_inl)
            if len(orig_inl) >= MIN_INLIERS:
                return np.asarray(model), orig_inl, "board", skipped
        active[orig_inl] = False
        skipped += 1

    return None, None, "not found", skipped


def _largest_cluster(cluster_pts: np.ndarray, orig_idx: np.ndarray) -> np.ndarray:
    """Keep only the biggest DBSCAN cluster (isolates the board from scattered
    coplanar dark objects)."""
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(cluster_pts)
    labels = np.asarray(sub.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN))
    if labels.max() < 0:
        return orig_idx
    biggest = np.bincount(labels[labels >= 0]).argmax()
    return orig_idx[labels == biggest]


def main() -> None:
    cam = Record3DCamera()
    cam.connect()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Board plane feed (green = dark vertical board)")
    pcd = o3d.geometry.PointCloud()
    added = False

    print("Live feed running. Close the window to stop.")
    try:
        while True:
            frame = cam.wait(timeout_s=2.0)
            if frame is None:
                if not vis.poll_events():
                    break
                continue

            pts_cv, valid = backproject(frame.depth, frame.intrinsics, frame.rgb.shape[:2])
            pts = level_with_gravity(pts_cv, frame.pose)
            colors = sample_colors(frame.rgb, frame.depth.shape, valid)
            pts, colors = downsample(pts, colors)

            dark_thr = adaptive_dark_threshold(colors)
            plane, inliers, label, skipped = fit_board_plane(pts, colors, dark_thr)
            vis_colors = colors.copy()
            n_in = 0
            if inliers is not None and len(inliers):
                vis_colors[inliers] = [0.1, 0.9, 0.2]
                n_in = len(inliers)
            print(f"\r{label:>9}  points={len(pts):>5}  inliers={n_in:>5}  "
                  f"skipped={skipped}  dark<{dark_thr:.2f}   ", end="", flush=True)

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
    finally:
        vis.destroy_window()
        cam.disconnect()

    os._exit(0)  # dodge the record3d C++ teardown segfault


if __name__ == "__main__":
    main()
