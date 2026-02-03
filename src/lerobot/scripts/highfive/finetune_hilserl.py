#!/usr/bin/env python
"""
Fine-tune High-Five policy with Human-in-the-Loop SERL (HIL-SERL).

This script enables fine-tuning of a sim-trained policy using human
interventions via a leader arm. When the robot makes mistakes, a human
can take over control to demonstrate the correct behavior, and these
corrections are used to improve the policy.

Usage:
    python -m lerobot.scripts.highfive.finetune_hilserl \\
        --policy_path outputs/highfive/checkpoints/best_model \\
        --follower_port /dev/tty.usbmodem12345 \\
        --leader_port /dev/tty.usbmodem67890

HIL-SERL Overview:
    1. Robot executes policy autonomously
    2. Human observes and can intervene via leader arm
    3. Interventions are prioritized in training
    4. Policy improves from both autonomous rollouts and human corrections

This approach is especially useful when:
    - Sim-to-real gap is large
    - Policy works sometimes but fails in edge cases
    - Human expertise can guide learning

"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
from pathlib import Path
from queue import Empty, Full
from typing import Any

import cv2
import numpy as np
import torch
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune High-Five policy with HIL-SERL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to pre-trained policy checkpoint",
    )

    # Robot
    parser.add_argument(
        "--follower_port",
        type=str,
        required=True,
        help="Serial port for follower arm",
    )
    parser.add_argument(
        "--follower_id",
        type=str,
        default=None,
        help="Follower robot ID",
    )
    parser.add_argument(
        "--leader_port",
        type=str,
        required=True,
        help="Serial port for leader arm (human control)",
    )
    parser.add_argument(
        "--leader_id",
        type=str,
        default=None,
        help="Leader robot ID",
    )

    # Camera
    parser.add_argument(
        "--birdseye_camera_index",
        type=int,
        default=0,
        help="Index of bird's eye camera",
    )
    parser.add_argument(
        "--observation_size",
        type=int,
        default=224,
        help="Observation image size",
    )

    # Training
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (lower than initial training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--online_buffer_capacity",
        type=int,
        default=10_000,
        help="Online replay buffer capacity",
    )
    parser.add_argument(
        "--intervention_buffer_capacity",
        type=int,
        default=5_000,
        help="Intervention replay buffer capacity",
    )

    # Episode parameters
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--control_freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Save checkpoint every N episodes",
    )

    return parser.parse_args()


def setup_output_dir(args: argparse.Namespace) -> Path:
    """Setup output directory."""
    from datetime import datetime

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"outputs/highfive/hilserl_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    return output_dir


def load_policy(policy_path: str, device: str):
    """Load pre-trained SAC policy."""
    from lerobot.policies.sac.modeling_sac import SACPolicy

    policy = SACPolicy.from_pretrained(policy_path)
    policy.to(device)
    return policy


def preprocess_observation(
    image: np.ndarray,
    joint_positions: np.ndarray,
    target_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Preprocess observation for policy."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    # Resize and normalize image
    image = cv2.resize(image, (target_size, target_size))
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

    # Joint positions
    joint_tensor = torch.from_numpy(joint_positions).float().unsqueeze(0)

    return {
        f"{OBS_IMAGES}.image": image_tensor.to(device),
        OBS_STATE: joint_tensor.to(device),
    }


def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_config_path: str,
    device: str,
    learning_rate: float,
    batch_size: int,
    online_buffer_capacity: int,
    intervention_buffer_capacity: int,
):
    """
    Learner process - trains policy on collected transitions.

    Implements HIL-SERL training:
    - Online buffer stores all transitions
    - Intervention buffer stores only human interventions
    - Each batch is 50% online, 50% intervention data
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy
    from lerobot.rl.buffer import ReplayBuffer
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    print("[LEARNER] Starting learner process...")

    # Load policy
    policy = SACPolicy.from_pretrained(policy_config_path)
    policy.to(device)
    policy.train()

    # Create optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Create replay buffers
    state_keys = [f"{OBS_IMAGES}.image", OBS_STATE]
    online_buffer = ReplayBuffer(
        capacity=online_buffer_capacity,
        device=device,
        state_keys=state_keys,
    )
    intervention_buffer = ReplayBuffer(
        capacity=intervention_buffer_capacity,
        device=device,
        state_keys=state_keys,
    )

    print(f"[LEARNER] Online buffer capacity: {online_buffer_capacity}")
    print(f"[LEARNER] Intervention buffer capacity: {intervention_buffer_capacity}")

    training_step = 0
    min_samples = batch_size

    while not shutdown_event.is_set():
        # Receive transitions from actor
        try:
            transitions = transitions_queue.get(timeout=0.1)
            for transition in transitions:
                # Add to online buffer
                online_buffer.add(**transition)

                # Add interventions to intervention buffer
                if transition.get("is_intervention", False):
                    intervention_buffer.add(**transition)
                    print(
                        f"[LEARNER] Intervention added! "
                        f"Intervention buffer: {len(intervention_buffer)}"
                    )

        except Empty:
            pass

        # Train if enough samples
        have_enough_online = len(online_buffer) >= min_samples
        have_enough_intervention = len(intervention_buffer) >= min_samples // 2

        if have_enough_online and have_enough_intervention:
            # Sample from both buffers (HIL-SERL key mechanism)
            online_batch = online_buffer.sample(batch_size // 2)
            intervention_batch = intervention_buffer.sample(batch_size // 2)

            # Combine batches
            batch = {}
            for key in online_batch:
                if key in intervention_batch:
                    batch[key] = torch.cat(
                        [online_batch[key], intervention_batch[key]], dim=0
                    )
                else:
                    batch[key] = online_batch[key]

            # Training step
            loss, loss_info = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_step += 1

            if training_step % 10 == 0:
                print(
                    f"[LEARNER] Step {training_step} | Loss: {loss.item():.4f} | "
                    f"Online: {len(online_buffer)} | "
                    f"Interventions: {len(intervention_buffer)}"
                )

            # Send updated parameters to actor
            if training_step % 10 == 0:
                try:
                    state_dict = {
                        k: v.cpu() for k, v in policy.state_dict().items()
                    }
                    parameters_queue.put_nowait(state_dict)
                except Full:
                    pass

    print("[LEARNER] Learner process finished")


def run_actor(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_config_path: str,
    follower_port: str,
    follower_id: str | None,
    leader_port: str,
    leader_id: str | None,
    camera_index: int,
    observation_size: int,
    device: str,
    num_episodes: int,
    max_steps: int,
    control_freq: float,
    output_dir: str,
):
    """
    Actor process - interacts with environment and human.

    Handles:
    - Policy rollouts
    - Human intervention detection via leader arm
    - Transition collection
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from lerobot.robots.factory import make_robot
    from lerobot.teleoperators.factory import make_teleoperator

    print("[ACTOR] Starting actor process...")

    # Load policy
    policy = SACPolicy.from_pretrained(policy_config_path)
    policy.to(device)
    policy.eval()

    # Initialize follower robot
    print("[ACTOR] Connecting to follower robot...")
    follower_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        max_relative_target=5.0,  # Safety limit
    )
    follower = make_robot(follower_config)
    follower.connect()

    # Initialize leader arm (for human interventions)
    print("[ACTOR] Connecting to leader arm...")
    leader_config = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
    )
    leader = make_teleoperator(leader_config)
    leader.connect()

    # Initialize camera
    print("[ACTOR] Initializing camera...")
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        raise RuntimeError(f"Failed to open camera at index {camera_index}")

    control_period = 1.0 / control_freq

    try:
        for episode in range(num_episodes):
            if shutdown_event.is_set():
                break

            print(f"\n[ACTOR] Episode {episode + 1}/{num_episodes}")

            episode_transitions = []
            step = 0

            # Reset robot to home position
            follower.reset()
            time.sleep(0.5)

            while step < max_steps and not shutdown_event.is_set():
                loop_start = time.time()

                # Check for updated policy parameters
                try:
                    new_params = parameters_queue.get_nowait()
                    policy.load_state_dict(new_params)
                    print("[ACTOR] Updated policy parameters")
                except Empty:
                    pass

                # Capture observation
                ret, frame = camera.read()
                if not ret:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get follower state
                follower_obs = follower.get_observation()
                follower_joints = np.array([
                    follower_obs["shoulder_pan.pos"],
                    follower_obs["shoulder_lift.pos"],
                    follower_obs["elbow_flex.pos"],
                    follower_obs["wrist_flex.pos"],
                    follower_obs["wrist_roll.pos"],
                ])

                # Get leader state (human input)
                leader_obs = leader.get_observation()
                leader_joints = np.array([
                    leader_obs["shoulder_pan.pos"],
                    leader_obs["shoulder_lift.pos"],
                    leader_obs["elbow_flex.pos"],
                    leader_obs["wrist_flex.pos"],
                    leader_obs["wrist_roll.pos"],
                ])

                # Detect human intervention
                # Simple heuristic: significant difference between leader and follower
                joint_diff = np.abs(leader_joints - follower_joints)
                is_intervention = np.any(joint_diff > 5.0)  # 5 degree threshold

                # Determine action
                if is_intervention:
                    # Human is intervening - use leader arm position
                    action = leader_joints
                    print(f"[ACTOR] Human intervention at step {step}")
                else:
                    # Use policy action
                    policy_obs = preprocess_observation(
                        image, follower_joints, observation_size, device
                    )
                    with torch.no_grad():
                        action_tensor = policy.select_action(
                            policy_obs, deterministic=True
                        )
                        action = action_tensor.squeeze(0).cpu().numpy()

                # Execute action
                robot_action = {
                    "shoulder_pan.pos": float(action[0]),
                    "shoulder_lift.pos": float(action[1]),
                    "elbow_flex.pos": float(action[2]),
                    "wrist_flex.pos": float(action[3]),
                    "wrist_roll.pos": float(action[4]),
                    "gripper.pos": 0.0,  # Keep closed
                }
                follower.send_action(robot_action)

                # Get next observation
                ret, next_frame = camera.read()
                if not ret:
                    continue
                next_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

                next_obs = follower.get_observation()
                next_joints = np.array([
                    next_obs["shoulder_pan.pos"],
                    next_obs["shoulder_lift.pos"],
                    next_obs["elbow_flex.pos"],
                    next_obs["wrist_flex.pos"],
                    next_obs["wrist_roll.pos"],
                ])

                # Create transition
                state = preprocess_observation(
                    image, follower_joints, observation_size, device
                )
                next_state = preprocess_observation(
                    next_image, next_joints, observation_size, device
                )

                transition = {
                    "state": state,
                    "action": action,
                    "reward": 0.0,  # Reward computed elsewhere or sparse
                    "next_state": next_state,
                    "done": False,
                    "truncated": False,
                    "is_intervention": is_intervention,
                }
                episode_transitions.append(transition)

                step += 1

                # Display
                display = frame.copy()
                status = "HUMAN" if is_intervention else "POLICY"
                cv2.putText(
                    display,
                    f"Episode: {episode + 1} | Step: {step} | Mode: {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if not is_intervention else (0, 0, 255),
                    2,
                )
                cv2.imshow("HIL-SERL Fine-tuning", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    shutdown_event.set()
                    break

                # Maintain control frequency
                elapsed = time.time() - loop_start
                if elapsed < control_period:
                    time.sleep(control_period - elapsed)

            # Send episode transitions to learner
            if episode_transitions:
                try:
                    transitions_queue.put(episode_transitions, timeout=1.0)
                except Full:
                    print("[ACTOR] Warning: Queue full, dropping transitions")

            # Count interventions
            n_interventions = sum(
                1 for t in episode_transitions if t.get("is_intervention", False)
            )
            print(
                f"[ACTOR] Episode {episode + 1} complete | "
                f"Steps: {step} | Interventions: {n_interventions}"
            )

    except Exception as e:
        print(f"[ACTOR] Error: {e}")
        raise

    finally:
        # Cleanup
        print("[ACTOR] Cleaning up...")
        cv2.destroyAllWindows()
        camera.release()

        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()

        # Save final policy
        output_path = Path(output_dir) / "checkpoints" / "final_policy"
        policy.save_pretrained(output_path)
        print(f"[ACTOR] Final policy saved to {output_path}")

        print("[ACTOR] Actor process finished")


def main():
    args = parse_args()
    output_dir = setup_output_dir(args)

    print("=" * 60)
    print("HIL-SERL Fine-tuning for High-Five Robot")
    print("=" * 60)
    print(f"Policy: {args.policy_path}")
    print(f"Follower: {args.follower_port}")
    print(f"Leader: {args.leader_port}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print("\nInstructions:")
    print("- Robot will execute policy autonomously")
    print("- Move leader arm to intervene and demonstrate corrections")
    print("- Interventions are prioritized in training")
    print("- Press 'q' to stop")
    print("=" * 60)

    # Create communication channels
    transitions_queue = mp.Queue(maxsize=20)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    # Signal handler
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start processes
    learner_process = mp.Process(
        target=run_learner,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            args.policy_path,
            args.device,
            args.learning_rate,
            args.batch_size,
            args.online_buffer_capacity,
            args.intervention_buffer_capacity,
        ),
    )

    actor_process = mp.Process(
        target=run_actor,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            args.policy_path,
            args.follower_port,
            args.follower_id,
            args.leader_port,
            args.leader_id,
            args.birdseye_camera_index,
            args.observation_size,
            "cpu",  # Actor runs on CPU for inference
            args.num_episodes,
            args.max_steps_per_episode,
            args.control_freq,
            str(output_dir),
        ),
    )

    print("\nStarting learner and actor processes...")
    learner_process.start()
    actor_process.start()

    try:
        actor_process.join()
        shutdown_event.set()
        learner_process.join(timeout=10)

    except KeyboardInterrupt:
        print("\nInterrupted!")
        shutdown_event.set()
        actor_process.join(timeout=5)
        learner_process.join(timeout=10)

    finally:
        if learner_process.is_alive():
            learner_process.terminate()
        if actor_process.is_alive():
            actor_process.terminate()

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
