"""Wheel check for the LeKiwi holonomic base.

Commands pure rotate / forward / strafe so you can watch whether ALL THREE wheels
respond. If one wheel barely turns (e.g. the front-right), holonomic navigation
will be wrong for every motion -- fix that before tuning the approach.

Tip: lift the base onto a box so the wheels spin freely in the air while you watch.

Usage:
    python wall_drawing/wheel_test.py --robot-port /dev/tty.usbmodemXXXX
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig


def drive(robot, arm_hold, x, y, th, label, dur=3.0, fps=30):
    print(f"\n>>> {label}")
    t0 = time.time()
    while time.time() - t0 < dur:
        robot.send_action({**arm_hold, "x.vel": x, "y.vel": y, "theta.vel": th})
        time.sleep(1.0 / fps)
    robot.send_action({**arm_hold, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-port", required=True)
    p.add_argument("--robot-id", default="my_lekiwi")
    args = p.parse_args()

    robot = LeKiwi(LeKiwiConfig(port=args.robot_port, id=args.robot_id, cameras={}))
    robot.connect()
    # LeKiwi.send_action needs arm position keys; hold the arm wherever it is.
    arm_hold = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    print("Lift the base so wheels spin freely. Watch each wheel. Ctrl-C to stop.")
    try:
        drive(robot, arm_hold, 0.0, 0.0, 20.0, "ROTATE -- all 3 wheels should spin the same way")
        time.sleep(1.0)
        drive(robot, arm_hold, 0.06, 0.0, 0.0, "FORWARD")
        time.sleep(1.0)
        drive(robot, arm_hold, 0.0, 0.06, 0.0, "STRAFE LEFT")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.stop_base()
        robot.disconnect()


if __name__ == "__main__":
    main()
