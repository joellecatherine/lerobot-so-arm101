"""Forward kinematics for the SO-101 arm: gripper pose from joint angles.

Wraps LeRobot's placo-based `RobotKinematics` with the SO-101 URDF and maps a
LeKiwi observation (`arm_<joint>.pos`, in degrees) to the gripper's 4x4 pose in
the ARM BASE frame (x-forward, y-left, z-up).

URDF: from the SO-ARM100 repo, `Simulation/SO101/so101_new_calib.urdf`
(https://github.com/TheRobotStudio/SO-ARM100). Point SO101_URDF at your copy, or
pass urdf_path explicitly.

Run directly to verify: it disables arm torque so you can move the follower by
hand and watch the gripper x/y/z update (x forward, y left, z up).

Usage:
    python wall_drawing/arm_kinematics.py --robot-port /dev/tty.usbmodemXXXX
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from lerobot.model import RobotKinematics

DEFAULT_URDF = os.environ.get(
    "SO101_URDF",
    os.path.expanduser(
        "~/Documents/CodingProjects/Claudeplayground/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    ),
)

# URDF joint order (matches RobotKinematics.joint_names for this URDF).
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class GripperFK:
    """SO-101 forward kinematics -> gripper pose in the arm base frame."""

    def __init__(self, urdf_path: str = DEFAULT_URDF, target_frame: str = "gripper"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"SO-101 URDF not found at {urdf_path}. Clone TheRobotStudio/SO-ARM100 "
                f"and set SO101_URDF, or pass urdf_path."
            )
        self.kin = RobotKinematics(urdf_path, target_frame_name=target_frame)

    def from_joints_deg(self, joints_deg) -> np.ndarray:
        """joint angles (deg, ARM_JOINTS order) -> 4x4 gripper pose in base frame."""
        return self.kin.forward_kinematics(np.asarray(joints_deg, dtype=float))

    def from_observation(self, obs: dict) -> np.ndarray:
        """LeKiwi observation (arm_<joint>.pos) -> 4x4 gripper pose."""
        q = np.array([obs[f"arm_{j}.pos"] for j in ARM_JOINTS], dtype=float)
        return self.from_joints_deg(q)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-port", required=True)
    p.add_argument("--robot-id", default="my_lekiwi")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fk = GripperFK(args.urdf)

    from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
    robot = LeKiwi(LeKiwiConfig(port=args.robot_port, id=args.robot_id, cameras={}))
    robot.connect()
    robot.bus.disable_torque(robot.arm_motors)  # so you can move the arm by hand
    print("Arm torque OFF. Move the gripper by hand; watch x(fwd)/y(left)/z(up). Ctrl-C to stop.")

    try:
        while True:
            obs = robot.get_observation()
            T = fk.from_observation(obs)
            x, y, z = T[:3, 3]
            print(f"\rgripper (base frame)  x={x:+.3f}  y={y:+.3f}  z={z:+.3f} m   ",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
