"""Saves iPhone LiDAR frames for sensor noise analysis.

Captures N consecutive depth frames of a static scene and stores them as one
compressed stack per capture, plus the camera intrinsics.

Usage (one run per distance, phone rigidly propped, scene static):
    python wall_drawing/capture_LiDAR_frames.py --out data/lidar_noise/noise_1m
    python wall_drawing/capture_LiDAR_frames.py --out data/lidar_noise/noise_0p5m
    python wall_drawing/capture_LiDAR_frames.py --out data/lidar_noise/noise_2m

Output: <out>.npz with
    depth       (N, H, W) float32, metres
    intrinsics  (3, 3) float64
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from iphone_camera import Record3DCamera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="Output path without extension, e.g. data/lidar_noise/noise_1m")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    out_path = args.out + ".npz"
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} already exists — refusing to overwrite a capture.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not Record3DCamera.is_available():
        raise SystemExit("iPhone (Record3D) not found — check USB + streaming mode.")
    iphone = Record3DCamera()
    iphone.connect()
    print("iPhone (Record3D): CONNECTED\n")

    depths: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    try:
        while len(depths) < args.frames:
            frame = iphone.wait(timeout_s=5.0)
            if frame is None:
                print(f"Timeout waiting for frame {len(depths) + 1} — stopping early.")
                break
            depths.append(frame.depth.copy())
            if intrinsics is None:
                intrinsics = frame.intrinsics
                print("--- first sample ---")
                print(f"  [iphone depth] {frame.depth.shape}  "
                      f"range=[{np.nanmin(frame.depth):.2f}, {np.nanmax(frame.depth):.2f}] m")
            if len(depths) % 50 == 0:
                print(f"  {len(depths)}/{args.frames} frames")
    except KeyboardInterrupt:
        print("\nStopping early.")
    finally:
        iphone.disconnect()

    if depths:
        stack = np.stack(depths).astype(np.float32)
        np.savez_compressed(out_path, depth=stack, intrinsics=intrinsics)
        print("\n--- summary ---")
        print(f"  saved {stack.shape[0]} frames of {stack.shape[1:]} to {out_path}")
        print(f"  depth range over capture: "
              f"[{np.nanmin(stack):.3f}, {np.nanmax(stack):.3f}] m")
    else:
        print("\nNo frames captured — nothing saved.")

    # Exit hard to avoid a segfault in the record3d C++ extension's teardown
    # (all data is already saved above).
    os._exit(0)


if __name__ == "__main__":
    main()
