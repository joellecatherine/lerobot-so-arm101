#!/usr/bin/env python
"""
Deploy trained High-Five policy to real SO-ARM101 robot.

This script deploys a policy trained in simulation to a real SO-ARM101 robot
for performing high-five/fist-bump interactions with humans.

Usage:
    # Deploy with default settings
    python -m lerobot.scripts.highfive.deploy_real \\
        --policy_path outputs/highfive/checkpoints/best_model \\
        --robot_port /dev/tty.usbmodem12345

    # Deploy with hand detection camera
    python -m lerobot.scripts.highfive.deploy_real \\
        --policy_path outputs/highfive/checkpoints/best_model \\
        --robot_port /dev/tty.usbmodem12345 \\
        --birdseye_camera_index 1

Requirements:
    - Trained SAC policy checkpoint
    - SO-ARM101 robot connected via USB
    - Bird's eye camera mounted above workspace
    - Optional: MediaPipe for hand detection

Safety Notes:
    - Start with slow_mode enabled (default)
    - Test with static hand positions first
    - Gradually increase speed after verifying safe operation
    - Keep emergency stop ready

"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy High-Five policy to real SO-ARM101 robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and robot
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to trained policy checkpoint",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        required=True,
        help="Serial port for SO-ARM101 robot (e.g., /dev/tty.usbmodem12345)",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default=None,
        help="Robot ID for calibration files",
    )

    # Camera
    parser.add_argument(
        "--birdseye_camera_index",
        type=int,
        default=0,
        help="Index of bird's eye camera",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=640,
        help="Camera capture width",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=480,
        help="Camera capture height",
    )
    parser.add_argument(
        "--observation_size",
        type=int,
        default=224,
        help="Observation image size for policy",
    )

    # Control parameters
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for policy inference",
    )
    parser.add_argument(
        "--control_freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--slow_mode",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable slow mode for safety (limits speed)",
    )
    parser.add_argument(
        "--max_relative_target",
        type=float,
        default=5.0,
        help="Maximum relative movement per step (degrees)",
    )

    # Hand detection
    parser.add_argument(
        "--use_hand_detection",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use MediaPipe for hand detection",
    )
    parser.add_argument(
        "--hand_detection_confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for hand detection",
    )

    # Operation modes
    parser.add_argument(
        "--mode",
        type=str,
        default="continuous",
        choices=["continuous", "trigger", "demo"],
        help="Operation mode: continuous tracking, triggered high-five, or demo",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run (None for infinite)",
    )

    # Debugging
    parser.add_argument(
        "--display",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Display camera feed and debug info",
    )
    parser.add_argument(
        "--dry_run",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Dry run without sending commands to robot",
    )

    return parser.parse_args()


class BirdsEyeCamera:
    """Camera handler for bird's eye view."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {self.width}x{self.height}")

    def capture(self) -> np.ndarray | None:
        """Capture a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        """Release the camera."""
        self.cap.release()


class HandDetector:
    """Hand detector using MediaPipe."""

    def __init__(self, min_detection_confidence: float = 0.7):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "MediaPipe is required for hand detection. "
                "Install with: pip install mediapipe"
            )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(
        self, image: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Detect hand in image.

        Returns:
            Tuple of (hand_center, hand_landmarks) or (None, None) if no hand detected.
            hand_center is (x, y) in pixel coordinates.
        """
        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get palm center (wrist and middle finger base)
            wrist = hand_landmarks.landmark[0]
            middle_base = hand_landmarks.landmark[9]

            h, w = image.shape[:2]
            center_x = int((wrist.x + middle_base.x) / 2 * w)
            center_y = int((wrist.y + middle_base.y) / 2 * h)

            return np.array([center_x, center_y]), hand_landmarks

        return None, None

    def draw_landmarks(
        self, image: np.ndarray, landmarks, hand_center: np.ndarray | None = None
    ) -> np.ndarray:
        """Draw hand landmarks on image."""
        import mediapipe as mp

        annotated = image.copy()
        if landmarks:
            self.mp_draw.draw_landmarks(
                annotated,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )
        if hand_center is not None:
            cv2.circle(annotated, tuple(hand_center), 10, (0, 255, 0), -1)
        return annotated


def load_policy(policy_path: str, device: str):
    """Load trained SAC policy."""
    from lerobot.policies.sac.modeling_sac import SACPolicy

    policy = SACPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    print(f"Policy loaded from {policy_path}")
    return policy


def preprocess_image(
    image: np.ndarray,
    target_size: int,
    device: str,
) -> torch.Tensor:
    """Preprocess camera image for policy input."""
    # Resize
    image = cv2.resize(image, (target_size, target_size))

    # Convert to tensor and normalize
    tensor = torch.from_numpy(image).float() / 255.0

    # Add batch dimension and permute to (B, C, H, W)
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)

    return tensor.to(device)


def create_observation(
    image: np.ndarray,
    joint_positions: np.ndarray,
    target_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Create observation dict for policy."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    obs = {
        f"{OBS_IMAGES}.image": preprocess_image(image, target_size, device),
        OBS_STATE: torch.from_numpy(joint_positions).float().unsqueeze(0).to(device),
    }
    return obs


def action_to_joint_positions(
    action: np.ndarray,
    current_positions: np.ndarray,
    max_relative_target: float,
) -> np.ndarray:
    """
    Convert normalized action to joint positions.

    Args:
        action: Normalized action in [-1, 1]
        current_positions: Current joint positions in degrees
        max_relative_target: Maximum relative movement per step

    Returns:
        Target joint positions in degrees
    """
    # Scale action to relative movement
    delta = action * max_relative_target

    # Apply to current positions
    target = current_positions + delta

    return target


def run_deployment(args: argparse.Namespace):
    """Main deployment loop."""
    # Load policy
    print("Loading policy...")
    policy = load_policy(args.policy_path, args.device)

    # Initialize camera
    print("Initializing camera...")
    camera = BirdsEyeCamera(
        camera_index=args.birdseye_camera_index,
        width=args.camera_width,
        height=args.camera_height,
    )

    # Initialize hand detector if requested
    hand_detector = None
    if args.use_hand_detection:
        print("Initializing hand detector...")
        hand_detector = HandDetector(args.hand_detection_confidence)

    # Initialize robot (unless dry run)
    robot = None
    if not args.dry_run:
        print("Connecting to robot...")
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots.factory import make_robot

        robot_config = SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            max_relative_target=args.max_relative_target if args.slow_mode else None,
        )
        robot = make_robot(robot_config)
        robot.connect()
        print("Robot connected!")

    # Signal handler for graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\nShutdown requested...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Control loop
    control_period = 1.0 / args.control_freq
    episode_count = 0

    print("\n" + "=" * 60)
    print("Deployment started! Press Ctrl+C to stop.")
    if args.slow_mode:
        print("SLOW MODE ENABLED - Limited movement speed")
    if args.dry_run:
        print("DRY RUN MODE - No commands sent to robot")
    print("=" * 60 + "\n")

    try:
        while not shutdown_requested:
            if args.num_episodes and episode_count >= args.num_episodes:
                break

            loop_start = time.time()

            # Capture image
            image = camera.capture()
            if image is None:
                print("Failed to capture image!")
                continue

            # Hand detection (optional)
            hand_center = None
            hand_landmarks = None
            if hand_detector:
                hand_center, hand_landmarks = hand_detector.detect(image)

            # Get current joint positions
            if robot:
                obs = robot.get_observation()
                joint_positions = np.array([
                    obs["shoulder_pan.pos"],
                    obs["shoulder_lift.pos"],
                    obs["elbow_flex.pos"],
                    obs["wrist_flex.pos"],
                    obs["wrist_roll.pos"],
                ])
            else:
                # Dummy positions for dry run
                joint_positions = np.zeros(5)

            # Create observation
            policy_obs = create_observation(
                image,
                joint_positions,
                args.observation_size,
                args.device,
            )

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_obs, deterministic=True)
                action = action.squeeze(0).cpu().numpy()

            # Convert to joint positions
            target_positions = action_to_joint_positions(
                action,
                joint_positions,
                args.max_relative_target,
            )

            # Send to robot
            if robot and not args.dry_run:
                robot_action = {
                    "shoulder_pan.pos": target_positions[0],
                    "shoulder_lift.pos": target_positions[1],
                    "elbow_flex.pos": target_positions[2],
                    "wrist_flex.pos": target_positions[3],
                    "wrist_roll.pos": target_positions[4],
                    "gripper.pos": 0.0,  # Keep gripper closed (fist)
                }
                robot.send_action(robot_action)

            # Display
            if args.display:
                display_image = image.copy()

                # Draw hand detection
                if hand_detector and hand_landmarks:
                    display_image = hand_detector.draw_landmarks(
                        display_image, hand_landmarks, hand_center
                    )

                # Add info text
                cv2.putText(
                    display_image,
                    f"Action: {action}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Convert RGB to BGR for OpenCV display
                display_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("High-Five Robot", display_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    shutdown_requested = True
                elif key == ord("r"):
                    # Reset to home position
                    if robot:
                        print("Resetting to home position...")
                        robot.reset()

            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

    finally:
        # Cleanup
        print("\nShutting down...")

        if args.display:
            cv2.destroyAllWindows()

        camera.close()

        if robot:
            print("Disconnecting robot...")
            robot.disconnect()

        print("Deployment complete!")


def main():
    args = parse_args()

    print("=" * 60)
    print("High-Five Robot Deployment")
    print("=" * 60)
    print(f"Policy: {args.policy_path}")
    print(f"Robot port: {args.robot_port}")
    print(f"Device: {args.device}")
    print(f"Control frequency: {args.control_freq} Hz")
    print(f"Slow mode: {args.slow_mode}")
    print(f"Hand detection: {args.use_hand_detection}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    run_deployment(args)


if __name__ == "__main__":
    main()
