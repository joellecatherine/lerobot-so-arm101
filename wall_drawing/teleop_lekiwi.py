"""Tethered teleoperation for LeKiwi (no Raspberry Pi).

  * Arm  : the SO-101 leader arm drives the 6 arm joints.
  * Base : the keyboard drives the holonomic base.

The stock LeRobot example assumes a Pi running `lekiwi_host` and talks to it
over Wi-Fi via `LeKiwiClient` (whose `_from_keyboard_to_base_action` we port
here). This project is USB-tethered to the Mac, so we drive the *direct*
`LeKiwi` class over the serial port.

Keyboard controls (hold keys):
    w / s : forward / backward
    a / d : strafe left / right
    y / x : rotate left / right
    r / f : speed up / down (3 levels)
    esc   : release keyboard

macOS note: pynput needs Accessibility / Input Monitoring permission for your
terminal app (System Settings > Privacy & Security). Without it the base keys
won't register (the arm will still work).

Usage:
    python wall_drawing/teleop_lekiwi.py \
        --robot-port /dev/tty.usbmodemXXXX \
        --leader-port /dev/tty.usbmodemYYYY

    (find each port with `lerobot-find-port`)

Safety: hold the leader roughly in the follower's pose before starting; keep a
hand near the power switch. Ctrl-C to stop (zeros the base and disconnects).
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

FPS = 30


class BaseKeyboardController:
    """Maps currently-pressed keys to holonomic base velocities (x, y, theta)."""

    TELEOP_KEYS = {
        "forward": "w", "backward": "s", "left": "a", "right": "d",
        "rotate_left": "y", "rotate_right": "x",
        "speed_up": "r", "speed_down": "f",
    }
    # xy in m/s, theta in deg/s.
    SPEED_LEVELS = [
        {"xy": 0.1, "theta": 30},
        {"xy": 0.2, "theta": 60},
        {"xy": 0.3, "theta": 90},
    ]

    def __init__(self) -> None:
        self.speed_index = 0

    def to_base_action(self, pressed_keys) -> dict[str, float]:
        k = self.TELEOP_KEYS
        if k["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, len(self.SPEED_LEVELS) - 1)
        if k["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed = self.SPEED_LEVELS[self.speed_index]
        xy, theta = speed["xy"], speed["theta"]

        x_cmd = (xy if k["forward"] in pressed_keys else 0.0) - (xy if k["backward"] in pressed_keys else 0.0)
        y_cmd = (xy if k["left"] in pressed_keys else 0.0) - (xy if k["right"] in pressed_keys else 0.0)
        theta_cmd = (theta if k["rotate_left"] in pressed_keys else 0.0) - (
            theta if k["rotate_right"] in pressed_keys else 0.0
        )
        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-port", required=True, help="LeKiwi motor-bus serial port")
    p.add_argument("--leader-port", required=True, help="SO-101 leader arm serial port")
    p.add_argument("--robot-id", default="my_lekiwi", help="LeKiwi calibration id")
    p.add_argument("--leader-id", default="leader_arm_7v", help="Leader calibration id")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    robot = LeKiwi(LeKiwiConfig(port=args.robot_port, id=args.robot_id, cameras={}))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    base = BaseKeyboardController()

    robot.connect()   # press ENTER if prompted to reuse calibration
    leader.connect()

    base_enabled = True
    try:
        keyboard.connect()
    except Exception as e:  # pragma: no cover - depends on OS permissions
        base_enabled = False
        print(f"Keyboard base control unavailable ({e}); running arm-only.")

    print("Teleop running. Move the leader arm; use WASD/YX for the base. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.perf_counter()

            # Leader gives keys like "shoulder_pan.pos"; lekiwi expects "arm_shoulder_pan.pos".
            arm_action = {f"arm_{k}": v for k, v in leader.get_action().items()}
            base_action = base.to_base_action(keyboard.get_action()) if base_enabled else {
                "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
            }
            robot.send_action({**arm_action, **base_action})

            time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.stop_base()
        robot.disconnect()
        leader.disconnect()
        if base_enabled:
            keyboard.disconnect()


if __name__ == "__main__":
    main()
