#!/usr/bin/env python
"""
Collect demonstrations from a trained state-based SAC expert for imitation learning.

The expert uses state observations (joints + EE + hand position) for decision-making,
but we record pixel observations (camera images) alongside state and actions for
training a vision-based student policy (ACT/diffusion).

Usage:
    python -m lerobot.scripts.highfive.collect_demos \
        --checkpoint outputs/highfive/sac_xxx/checkpoints/best_model \
        --num_episodes 100 \
        --output_dir data/highfive_demos

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect demonstrations from a trained SAC expert",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained SAC checkpoint directory (must contain model.safetensors and config.json)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of successful episodes to collect",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/highfive_demos",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="local/highfive_demos",
        help="Dataset repo ID (used for LeRobotDataset metadata)",
    )
    parser.add_argument(
        "--hand_motion_type",
        type=str,
        default="random_walk",
        choices=["static", "random", "sinusoidal", "random_walk", "tracking"],
        help="Type of hand target motion",
    )
    parser.add_argument(
        "--domain_randomization",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable domain randomization",
    )
    parser.add_argument(
        "--observation_size",
        type=int,
        default=84,
        help="Camera image size (square)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Dataset FPS",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=200,
        help="Maximum episode length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for policy inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=0,
        help="Max total rollout attempts (0 = unlimited, stops at num_episodes successes)",
    )
    parser.add_argument(
        "--motion_freq_scale",
        type=float,
        default=0.5,
        help="Scale factor for hand motion frequency",
    )
    parser.add_argument(
        "--start_pose",
        type=str,
        default="folded",
        choices=["folded", "neutral"],
        help="Starting pose for robot arm",
    )
    return parser.parse_args()


def load_expert_policy(checkpoint_path: str, device: str):
    """Load a trained state-based SAC policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory containing model.safetensors
        device: Device to load the policy on

    Returns:
        Loaded policy in eval mode
    """
    from safetensors.torch import load_model as load_model_as_safetensor

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.scripts.highfive.custom_sac import HighFiveSACPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    checkpoint_dir = Path(checkpoint_path)
    model_file = checkpoint_dir / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # State-only expert: 11-dim input (5 joints + 3 EE + 3 hand pos)
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(11,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    }
    encoder_config = {
        "latent_dim": 256,
        "state_dim": 11,
        "encoder_type": "state",
        "single_camera": True,
    }

    # Try to load SAC config from checkpoint, fall back to defaults
    config_file = checkpoint_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
        print(f"Loaded config from {config_file}")
    else:
        saved_config = {}
        print("No config.json found, using defaults")

    policy_config = SACConfig(
        device=device,
        input_features=input_features,
        output_features=output_features,
        discount=saved_config.get("discount", 0.99),
        use_torch_compile=False,
        vision_encoder_name=None,
        shared_encoder=True,
    )

    policy = HighFiveSACPolicy(policy_config, encoder_config=encoder_config)
    load_model_as_safetensor(policy, str(model_file), strict=False)
    policy.to(device)
    policy.eval()

    print(f"Expert policy loaded from {checkpoint_dir}")
    return policy


def create_env(args: argparse.Namespace):
    """Create the environment with pixel observations for recording."""
    from lerobot.envs.highfive.highfive_env import HighFiveEnv

    env = HighFiveEnv(
        episode_length=args.episode_length,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        hand_motion_type=args.hand_motion_type,
        domain_randomization=args.domain_randomization,
        single_camera=False,
        motion_freq_scale=args.motion_freq_scale,
        start_pose=args.start_pose,
        seed=args.seed,
    )
    return env


def get_expert_state(env) -> np.ndarray:
    """Extract the full 11-dim state vector for the expert from the environment.

    The expert was trained with obs_type="state" which provides:
        [joint_pos(5), ee_pos(3), hand_pos(3)] = 11 dims

    Returns:
        11-dim state vector as numpy array
    """
    joint_pos = env._get_joint_positions()  # 5
    ee_pos = env._get_ee_position()  # 3
    hand_pos = env._get_hand_position()  # 3
    return np.concatenate([joint_pos, ee_pos, hand_pos])


def expert_select_action(policy, state: np.ndarray, device: str) -> np.ndarray:
    """Select an action from the expert policy given state.

    Args:
        policy: The loaded SAC expert policy
        state: 11-dim state vector
        device: Torch device

    Returns:
        5-dim action array
    """
    from lerobot.utils.constants import OBS_STATE

    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    obs = {OBS_STATE: state_tensor}

    with torch.no_grad():
        action = policy.select_action(obs)
    return action.cpu().numpy().squeeze(0)


def create_dataset(args: argparse.Namespace):
    """Create a LeRobotDataset for recording demonstrations."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    obs_size = args.observation_size

    features = {
        "observation.images.birdseye": {
            "dtype": "video",
            "shape": (obs_size, obs_size, 3),
            "names": ["height", "width", "channels"],
            "video.fps": args.fps,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (obs_size, obs_size, 3),
            "names": ["height", "width", "channels"],
            "video.fps": args.fps,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll",
                "ee_x", "ee_y", "ee_z",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (5,),
            "names": [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll",
            ],
        },
    }

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            f"Remove it first with: rm -rf {output_dir}"
        )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=output_dir,
        robot_type="so100",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    return dataset


def collect_demos(args: argparse.Namespace):
    """Main collection loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load expert
    print("Loading expert policy...")
    policy = load_expert_policy(args.checkpoint, args.device)

    # Create environment
    print("Creating environment...")
    env = create_env(args)

    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(args)

    successful_episodes = 0
    total_attempts = 0

    print(f"\nCollecting {args.num_episodes} successful episodes...")
    print(f"Hand motion: {args.hand_motion_type}")
    print(f"Domain randomization: {args.domain_randomization}")
    print(f"Image size: {args.observation_size}x{args.observation_size}")
    print()

    while successful_episodes < args.num_episodes:
        if args.max_attempts > 0 and total_attempts >= args.max_attempts:
            print(f"\nReached max attempts ({args.max_attempts})")
            break

        total_attempts += 1
        obs, info = env.reset()
        episode_frames = []
        done = False

        while not done:
            # Get full state for expert decision-making
            expert_state = get_expert_state(env)

            # Expert selects action from state
            action = expert_select_action(policy, expert_state, args.device)

            # Record frame: pixel observations + agent_pos + action
            frame = {
                "observation.images.birdseye": obs["pixels"]["birdseye"],
                "observation.images.wrist": obs["pixels"]["wrist"],
                "observation.state": obs["agent_pos"].astype(np.float32),
                "action": action.astype(np.float32),
                "task": "high-five with moving hand target",
            }
            episode_frames.append(frame)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Check success
        is_success = info.get("is_success", False)
        if hasattr(is_success, "item"):
            is_success = is_success.item()

        if is_success:
            successful_episodes += 1
            # Write frames to dataset
            for frame in episode_frames:
                dataset.add_frame(frame)
            dataset.save_episode()

            distance = info.get("distance", float("nan"))
            print(
                f"  [{successful_episodes:>4d}/{args.num_episodes}] "
                f"Episode saved (attempt {total_attempts}, "
                f"{len(episode_frames)} frames, "
                f"final distance: {distance:.4f})"
            )
        else:
            # Discard failed episode
            if total_attempts % 10 == 0:
                print(
                    f"  ... attempt {total_attempts}, "
                    f"{successful_episodes} successes so far "
                    f"({successful_episodes / total_attempts * 100:.0f}% success rate)"
                )

    # Finalize dataset
    print("\nFinalizing dataset...")
    dataset.finalize()

    env.close()

    success_rate = successful_episodes / total_attempts * 100 if total_attempts > 0 else 0
    print(f"\nCollection complete!")
    print(f"  Successful episodes: {successful_episodes}")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    collect_demos(args)
