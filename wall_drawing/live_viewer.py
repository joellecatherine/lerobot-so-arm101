"""Live viewer for the iPhone RGB-D stream (Record3D).

Connects once, holds the session open, and continuously shows the RGB image
next to a colorized depth map. Point the phone at the wall to see what the
LiDAR captures. Press 'q' (or Esc) in the window to quit.

Usage:
    1. iPhone: open Record3D, enable USB (RGBD) streaming, connect via cable.
    2. Run:  python wall_drawing/live_viewer.py
"""

from __future__ import annotations

import cv2
import numpy as np

from iphone_camera import Record3DCamera

# Fixed depth range (metres) for a stable colormap. Adjust to your working
# distance; a fixed range keeps the visualization from flickering frame to frame.
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 3.0


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a metric depth image to an 8-bit colour image for display."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip((d - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M), 0.0, 1.0)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def main() -> None:
    cam = Record3DCamera()
    cam.connect()
    print("Streaming... press 'q' or Esc in the window to quit.")

    try:
        while True:
            frame = cam.wait()
            if frame is None:
                print("Waiting for frames...")
                continue

            # Record3D gives RGB; OpenCV windows expect BGR.
            bgr = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR)
            depth_vis = colorize_depth(frame.depth)

            # Match heights so we can show them side by side.
            h = bgr.shape[0]
            scale = h / depth_vis.shape[0]
            depth_vis = cv2.resize(depth_vis, (int(depth_vis.shape[1] * scale), h))
            canvas = np.hstack([bgr, depth_vis])

            cv2.imshow("iPhone RGB (left)  |  LiDAR depth (right)", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
                break
    finally:
        cv2.destroyAllWindows()
        cam.disconnect()


if __name__ == "__main__":
    main()
