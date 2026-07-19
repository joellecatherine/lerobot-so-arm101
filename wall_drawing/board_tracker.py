"""Board tracking: lock a detected board once, then track it (don't re-detect).

Re-running full detection every frame makes the target jump between surfaces when
the board is sparse (a dark surface returns few LiDAR points), so the approach
controller chases a teleporting goal. Instead: detect once, LOCK it, and each
frame refit only the locked region (continuity-gated + EMA-smoothed), holding the
last estimate through brief LiDAR dropouts. Re-detect only if truly lost.

Height prior for the LOCK phase: the chalkboard is mounted low (so the arm can
reach it), so restrict detection to low points. In the gravity-leveled frame y is
height with the camera at y=0, so this works regardless of phone-mount orientation
and rejects high walls / pictures when still far away.
"""

from __future__ import annotations

import numpy as np


def filter_low(pts: np.ndarray, colors: np.ndarray, max_height: float = 0.4):
    """Keep only low points (height y <= max_height above the camera)."""
    keep = pts[:, 1] <= max_height
    return pts[keep], colors[keep]


def fit_plane_pca(pts: np.ndarray):
    """Least-squares plane through points (PCA). Returns ([a,b,c,d] unit n, centroid)."""
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) or 1.0)
    return np.array([n[0], n[1], n[2], -float(n @ c)]), c


def _plane_dist(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return np.abs(pts @ plane[:3] + plane[3])


class BoardTracker:
    """Lock a board plane, then track it frame-to-frame via gated re-fitting."""

    def __init__(self, ema: float = 0.5, band: float = 0.08, radius: float = 0.6,
                 max_normal_deg: float = 25.0, max_dist_jump: float = 0.25,
                 min_pts: int = 40, max_hold: int = 15):
        self.ema = ema
        self.band = band            # inlier band around the locked plane (m)
        self.radius = radius        # keep points within this of the locked centroid (m)
        self.min_cos = np.cos(np.radians(max_normal_deg))
        self.max_dist_jump = max_dist_jump
        self.min_pts = min_pts
        self.max_hold = max_hold
        self.plane: np.ndarray | None = None
        self.centroid: np.ndarray | None = None
        self._hold = 0

    @property
    def is_locked(self) -> bool:
        return self.plane is not None

    def lock(self, plane, centroid) -> None:
        self.plane = np.asarray(plane, float).copy()
        self.centroid = np.asarray(centroid, float).copy()
        self._hold = 0

    def release(self) -> None:
        self.plane = None
        self.centroid = None
        self._hold = 0

    def update(self, pts: np.ndarray):
        """Refit the locked board on nearby points. Returns (plane, centroid) or
        None if the lock is lost. Holds the last estimate through brief dropouts."""
        if self.plane is None:
            return None

        gate = (_plane_dist(pts, self.plane) < self.band) & \
               (np.linalg.norm(pts - self.centroid, axis=1) < self.radius)
        gated = pts[gate]

        if len(gated) >= self.min_pts:
            new_plane, new_centroid = fit_plane_pca(gated)
            if new_plane[:3] @ self.plane[:3] < 0:      # keep normal orientation consistent
                new_plane = -new_plane
            cos = float(new_plane[:3] @ self.plane[:3])
            dist_jump = abs(abs(new_plane[3]) - abs(self.plane[3]))
            if cos >= self.min_cos and dist_jump <= self.max_dist_jump:   # continuity gate
                a = self.ema
                self.plane = a * self.plane + (1 - a) * new_plane
                self.plane[:3] /= (np.linalg.norm(self.plane[:3]) or 1.0)
                self.centroid = a * self.centroid + (1 - a) * new_centroid
                self._hold = 0
                return self.plane, self.centroid

        self._hold += 1                                 # failed this frame -> hold last
        if self._hold <= self.max_hold:
            return self.plane, self.centroid
        self.release()
        return None


def planes_consistent(p1: np.ndarray, p2: np.ndarray,
                      max_normal_deg: float = 15.0, max_dist_jump: float = 0.1) -> bool:
    """Are two planes close enough to be the same surface? (for auto-lock stability)."""
    n1, n2 = p1[:3] / (np.linalg.norm(p1[:3]) or 1), p2[:3] / (np.linalg.norm(p2[:3]) or 1)
    cos = abs(float(n1 @ n2))
    return cos >= np.cos(np.radians(max_normal_deg)) and abs(abs(p1[3]) - abs(p2[3])) <= max_dist_jump
