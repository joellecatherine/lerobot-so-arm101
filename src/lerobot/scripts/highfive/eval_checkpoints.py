#!/usr/bin/env python
"""Evaluate all checkpoints in a directory and print success rates.

Usage:
    python -m lerobot.scripts.highfive.eval_checkpoints \
        --checkpoint_dir outputs/highfive/sac_force_disturbance_80pct/checkpoints \
        --eval_episodes 50 --obs_type state --ee_orientation --facing_reward \
        --randomize_hand_position --hand_motion_type static --palm_target_size 0.06
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint subdirs (best_model, step_10000, etc.)")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--obs_type", type=str, default="state")
    parser.add_argument("--hand_motion_type", type=str, default="static")
    parser.add_argument("--ee_orientation", action="store_true")
    parser.add_argument("--facing_reward", action="store_true")
    parser.add_argument("--randomize_hand_position", action="store_true")
    parser.add_argument("--palm_target_size", type=float, default=0.06)
    parser.add_argument("--force_disturbances", action="store_true")
    parser.add_argument("--force_disturbance_max", type=float, default=10.0)
    parser.add_argument("--force_disturbance_duration", type=int, default=8)
    parser.add_argument("--force_kicks_per_episode", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=100)
    args = parser.parse_args()

    from safetensors.torch import load_model as load_model_safetensor

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.envs.highfive.highfive_env import HighFiveEnv
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.scripts.highfive.custom_sac import HighFiveSACPolicy
    from lerobot.scripts.highfive.train_sac import preprocess_observation
    from lerobot.utils.constants import ACTION, OBS_STATE

    # Create env
    env = HighFiveEnv(
        obs_type=args.obs_type,
        hand_motion_type=args.hand_motion_type,
        render_mode=None,
        ee_orientation=args.ee_orientation,
        facing_reward=args.facing_reward,
        palm_target_size=args.palm_target_size,
        randomize_hand_position=args.randomize_hand_position,
        force_disturbances=args.force_disturbances,
        force_disturbance_max=args.force_disturbance_max,
        force_disturbance_duration=args.force_disturbance_duration,
        force_kicks_per_episode=args.force_kicks_per_episode,
        episode_length=args.episode_length,
    )

    # Build policy structure
    state_dim = 22 if args.ee_orientation else 16
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    encoder_config = {
        "latent_dim": 256, "state_dim": state_dim,
        "encoder_type": "state", "single_camera": True,
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    }
    policy_config = SACConfig(
        device=args.device,
        input_features=input_features,
        output_features=output_features,
        vision_encoder_name=None,
        image_encoder_hidden_dim=32,
        shared_encoder=True,
        use_torch_compile=False,
    )

    # Find all checkpoint subdirs
    checkpoint_dir = Path(args.checkpoint_dir)
    subdirs = sorted(checkpoint_dir.iterdir())
    checkpoints = []
    for d in subdirs:
        if (d / "model.safetensors").exists():
            # Extract step number for sorting
            if "step_" in d.name:
                step = int(d.name.split("step_")[1])
            elif d.name == "best_model":
                step = -1  # show first
            elif d.name == "final_model":
                step = float('inf')  # show last
            else:
                step = 0
            checkpoints.append((step, d))

    checkpoints.sort(key=lambda x: x[0])

    # Evaluate each
    print(f"{'Checkpoint':<20s} | {'Success Rate':>12s} | {'Mean Reward':>11s} | {'Std Reward':>10s} | {'Mean Length':>11s}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*11}-+-{'-'*10}-+-{'-'*11}")

    for _, ckpt_path in checkpoints:
        # Rebuild policy fresh each time to avoid state leakage
        policy = HighFiveSACPolicy(policy_config, encoder_config=encoder_config)
        load_model_safetensor(policy, str(ckpt_path / "model.safetensors"), strict=False)
        policy.to(args.device)
        policy.eval()

        rewards = []
        successes = []
        lengths = []

        for ep in range(args.eval_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                with torch.no_grad():
                    policy_obs = preprocess_observation(obs, args.device)
                    action = policy.select_action(policy_obs, deterministic=True)
                    action = action.cpu().numpy().squeeze()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(float(info.get("is_success", False)))

        sr = np.mean(successes)
        mr = np.mean(rewards)
        sr_std = np.std(rewards)
        ml = np.mean(lengths)

        print(f"{ckpt_path.name:<20s} | {sr:>11.1%} | {mr:>11.2f} | {sr_std:>10.2f} | {ml:>11.1f}")

    env.close()


if __name__ == "__main__":
    main()
