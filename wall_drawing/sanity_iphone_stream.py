"""Sanity check for the iPhone -> Mac RGB-D bridge.

Connects to an iPhone running the Record3D app (USB streaming mode), grabs a
single synchronized frame, and prints the shapes/intrinsics/pose so we can
confirm the whole sensing pipeline works before building perception on top.

Usage:
    1. On the iPhone: open Record3D, choose the LiDAR/USB streaming mode, and
       plug the phone into the Mac via USB.
    2. Run:  python wall_drawing/sanity_iphone_stream.py
"""

from __future__ import annotations

import numpy as np

from iphone_camera import Record3DCamera


def main() -> None:
    cam = Record3DCamera()
    cam.connect()
    frame = cam.wait(timeout_s=10.0)
    if frame is None:
        raise TimeoutError("No frame received. Is Record3D USB streaming running?")

    rgb, depth = frame.rgb, frame.depth
    print("\n--- one frame captured ---")
    print(f"RGB    : shape={rgb.shape} dtype={rgb.dtype}")
    print(f"Depth  : shape={depth.shape} dtype={depth.dtype} "
          f"range=[{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}] m")
    print(f"Intrinsics K (for depth->3D back-projection):\n{frame.intrinsics}")

    pose = frame.pose
    # Record3D returns a pose object with quaternion (qx,qy,qz,qw) + translation (tx,ty,tz).
    if all(hasattr(pose, a) for a in ("qx", "qy", "qz", "qw", "tx", "ty", "tz")):
        print(f"Camera pose (ARKit): quat=({pose.qx:.3f}, {pose.qy:.3f}, "
              f"{pose.qz:.3f}, {pose.qw:.3f})  t=({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})")
    else:
        print(f"Camera pose (ARKit): {pose}")

    print("\nBridge OK  iPhone RGB-D + pose is reaching Python.")
    cam.disconnect()


if __name__ == "__main__":
    main()
