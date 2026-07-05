"""Combined dataflow check: lekiwi motors + wrist camera + iPhone LiDAR.

Reads full robot observations (joint states + wrist image) and, in the same
loop, the latest iPhone frame (RGB + depth + pose) from Record3D. Reports which
sources are live so you can confirm everything streams together before
recording.

The iPhone is optional here: if no Record3D device is found (e.g. Wi-Fi
streaming isn't exposed to the Python library), the check still validates the
robot + wrist camera and just notes the phone is absent.

Does NOT move the robot. Ctrl-C to stop.

Usage:
    python wall_drawing/check_all_dataflows.py \
        --robot-port /dev/tty.usbmodemXXXX --camera-index 0

    (find your port with `lerobot-find-port`)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from iphone_camera import Record3DCamera
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-port", required=True)
    p.add_argument("--robot-id", default="my_lekiwi")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--seconds", type=float, default=5.0, help="How long to sample")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wrist = OpenCVCameraConfig(index_or_path=args.camera_index, fps=30,
                               width=1920, height=1080, warmup_s=2)
    robot = LeKiwi(LeKiwiConfig(port=args.robot_port, id=args.robot_id,
                                cameras={"wrist": wrist}))

    iphone = Record3DCamera()
    iphone_ok = Record3DCamera.is_available()
    if iphone_ok:
        iphone.connect()
    print(f"iPhone (Record3D): {'CONNECTED' if iphone_ok else 'not found'}\n")

    robot.connect()  # press ENTER if prompted to reuse calibration
    print("Robot + wrist camera connected. Sampling all sources...\n")

    printed = False
    iphone_frames = 0
    robot_frames = 0
    wrist_last = None
    wrist_max = 0.0
    t_end = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < t_end:
            obs = robot.get_observation()
            robot_frames += 1

            phone = iphone.latest() if iphone_ok else None
            if phone is not None:
                iphone_frames += 1

            # Track wrist brightness across ALL frames (first frame is often black).
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    b = float(v.mean())
                    wrist_last, wrist_max = b, max(wrist_max, b)

            if not printed:
                printed = True
                print("--- first full sample ---")
                for k, v in obs.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 2:
                        print(f"  [robot image] {k}: {v.shape}")
                    else:
                        print(f"  [robot state] {k}: {v}")
                if phone is not None:
                    print(f"  [iphone rgb ] {phone.rgb.shape}")
                    print(f"  [iphone depth] {phone.depth.shape}  "
                          f"range=[{np.nanmin(phone.depth):.2f}, {np.nanmax(phone.depth):.2f}] m")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        print("\n--- summary ---")
        print(f"  robot observations : {robot_frames}")
        print(f"  iphone frames      : {iphone_frames if iphone_ok else 'n/a (not connected)'}")
        if wrist_last is not None:
            print(f"  wrist brightness   : last={wrist_last:.1f}  max={wrist_max:.1f} "
                  f"({'looks live' if wrist_max > 10 else 'still dark - check lens/hub'})")
        robot.disconnect()
        if iphone_ok:
            iphone.disconnect()

    # Exit hard to avoid a segfault in the record3d C++ extension's teardown
    # (all data is already collected and printed above).
    os._exit(0)


if __name__ == "__main__":
    main()
