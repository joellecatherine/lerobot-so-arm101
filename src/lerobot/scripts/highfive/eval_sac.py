#!/usr/bin/env python
"""
Evaluate/visualize a trained SAC policy for High-Five Robot task.

Usage:
    # Evaluate best model with visualization
    python -m lerobot.scripts.highfive.eval_sac \
        --checkpoint outputs/highfive/sac_2026-02-03_11-15-01/checkpoints/best_model \
        --render

    # Evaluate without rendering (faster)
    python -m lerobot.scripts.highfive.eval_sac \
        --checkpoint outputs/highfive/sac_2026-02-03_11-15-01/checkpoints/step_20000 \
        --num_episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC policy for High-Five Robot task",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--observation_size",
        type=int,
        default=84,
        help="Observation image size (must match training)",
    )
    parser.add_argument(
        "--hand_motion_type",
        type=str,
        default="sinusoidal",
        choices=["static", "random", "sinusoidal", "tracking"],
        help="Type of hand target motion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def create_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    """Create the high-five environment."""
    from lerobot.envs.configs import HighFiveEnvConfig
    from lerobot.envs.factory import make_env

    render_mode = "human" if args.render else "rgb_array"

    env_config = HighFiveEnvConfig(
        episode_length=200,
        obs_type="pixels_agent_pos",
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        hand_motion_type=args.hand_motion_type,
        domain_randomization=False,  # Disable for eval
        render_mode=render_mode,
    )

    env_dict = make_env(env_config, n_envs=1)
    vec_env = env_dict["highfive"][0]
    return vec_env


def create_policy(args: argparse.Namespace, env: gym.vector.VectorEnv):
    """Create and load the SAC policy."""
    from safetensors.torch import load_model as load_model_as_safetensor

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.scripts.highfive.custom_sac import HighFiveSACPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    obs_size = args.observation_size
    input_features = {
        f"{OBS_IMAGES}.birdseye": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, obs_size, obs_size),
        ),
        f"{OBS_IMAGES}.wrist": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, obs_size, obs_size),
        ),
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE,
            shape=(5,),
        ),
    }

    output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(5,),
        ),
    }

    policy_config = SACConfig(
        device=args.device,
        input_features=input_features,
        output_features=output_features,
        vision_encoder_name=None,
        use_torch_compile=False,
    )

    encoder_config = {
        "latent_dim": 256,
        "state_dim": 5,
        "pretrained": True,
        "freeze_backbones": True,
    }

    policy = HighFiveSACPolicy(policy_config, encoder_config=encoder_config)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    model_file = checkpoint_path / "model.safetensors"
    if model_file.exists():
        print(f"Loading checkpoint from {model_file}")
        load_model_as_safetensor(policy, str(model_file), strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    policy.to(args.device)
    policy.eval()
    return policy


def preprocess_observation(obs: dict, device: str) -> dict[str, torch.Tensor]:
    """Preprocess environment observation for policy input."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    processed = {}

    if "pixels" in obs:
        for camera_name in ["birdseye", "wrist"]:
            if camera_name in obs["pixels"]:
                image = obs["pixels"][camera_name]
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float() / 255.0
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    image = image.permute(0, 3, 1, 2)
                processed[f"{OBS_IMAGES}.{camera_name}"] = image.to(device)

    if "agent_pos" in obs:
        agent_pos = obs["agent_pos"]
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
        processed[OBS_STATE] = agent_pos.to(device)

    return processed


def evaluate(args: argparse.Namespace):
    """Run evaluation."""
    print("=" * 60)
    print("High-Five Robot Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Render: {args.render}")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment and policy
    print("Creating environment...")
    env = create_env(args)

    print("Loading policy...")
    policy = create_policy(args, env)

    # Run evaluation
    total_rewards = []
    total_successes = []
    episode_lengths = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                policy_obs = preprocess_observation(obs, args.device)
                action = policy.select_action(policy_obs)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()

            if args.render:
                env.call("render")

            episode_reward += reward.sum()
            episode_length += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check success
        is_success = False
        if "is_success" in info:
            success = info["is_success"]
            if hasattr(success, "__iter__") and not isinstance(success, (str, dict)):
                is_success = success[0] if len(success) > 0 else False
            else:
                is_success = success
        elif "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, (list, tuple)) and len(final_info) > 0:
                final_info = final_info[0]
            if isinstance(final_info, dict):
                is_success = final_info.get("is_success", False)

        total_successes.append(float(is_success))

        print(
            f"Episode {ep + 1:>3d}/{args.num_episodes} | "
            f"Reward: {episode_reward:>8.2f} | "
            f"Length: {episode_length:>3d} | "
            f"Success: {is_success}"
        )

    # Print summary
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Success rate: {np.mean(total_successes) * 100:.1f}%")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print("=" * 60)

    env.close()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
