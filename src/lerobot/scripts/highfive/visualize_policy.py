#!/usr/bin/env python
"""Visualize a trained policy: render episodes in MuJoCo viewer or save video.

Usage:
    # Live 3D viewer (default)
    python -m lerobot.scripts.highfive.visualize_policy \
        --checkpoint outputs/highfive/sac_XXXX/checkpoints/final_model \
        --hand_motion_type random_walk --num_episodes 3

    # Save video file
    python -m lerobot.scripts.highfive.visualize_policy \
        --checkpoint outputs/highfive/sac_XXXX/checkpoints/final_model \
        --render_mode rgb_array --save_video out.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hand_motion_type", type=str, default="sinusoidal")
    parser.add_argument("--motion_freq_scale", type=float, default=0.3)
    parser.add_argument("--save_video", type=str, default=None, help="Path to save video (e.g. out.mp4)")
    parser.add_argument("--obs_type", type=str, default="pixels_agent_pos")
    parser.add_argument("--encoder_type", type=str, default="small_cnn")
    parser.add_argument("--observation_size", type=int, default=84)
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"],
                        help="human = live MuJoCo viewer, rgb_array = offscreen (use with --save_video)")
    args = parser.parse_args()

    from lerobot.envs.highfive.highfive_env import HighFiveEnv
    from lerobot.scripts.highfive.train_sac import preprocess_observation

    # Create a single (non-vectorized) env for easy access to internals
    env = HighFiveEnv(
        obs_type=args.obs_type,
        hand_motion_type=args.hand_motion_type,
        motion_freq_scale=args.motion_freq_scale,
        domain_randomization=False,
        single_camera=False,
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        render_mode=args.render_mode,
    )

    # Load policy
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.scripts.highfive.custom_sac import HighFiveSACPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    use_state = args.obs_type == "state"

    if use_state:
        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(11,)),
        }
        encoder_config = {
            "latent_dim": 256, "state_dim": 11,
            "encoder_type": "state", "single_camera": True,
        }
    else:
        obs_size = args.observation_size
        base_channels = 3
        input_features = {
            f"{OBS_IMAGES}.birdseye": PolicyFeature(
                type=FeatureType.VISUAL, shape=(base_channels, obs_size, obs_size)),
            f"{OBS_IMAGES}.wrist": PolicyFeature(
                type=FeatureType.VISUAL, shape=(base_channels, obs_size, obs_size)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        }
        encoder_config = {
            "latent_dim": 256, "state_dim": 5, "pretrained": True,
            "freeze_backbones": True, "fusion": "concat", "use_depth": False,
            "single_camera": False, "bev_depth_wrist_rgb": False,
            "birdseye_channels": base_channels, "wrist_channels": base_channels,
            "encoder_type": args.encoder_type,
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

    policy = HighFiveSACPolicy(policy_config, encoder_config=encoder_config)

    # Load weights
    from safetensors.torch import load_model as load_model_safetensor
    model_file = Path(args.checkpoint) / "model.safetensors"
    load_model_safetensor(policy, str(model_file), strict=False)
    policy.to(args.device)
    policy.eval()
    print(f"Loaded policy from {model_file}")

    # Collect video frames
    video_frames = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        print(f"\n{'='*60}")
        print(f"Episode {ep+1}")
        print(f"{'='*60}")
        print(f"{'Step':>5s} | {'EE pos':^30s} | {'Hand pos':^30s} | {'Dist':>6s} | {'Reward':>7s}")
        print(f"{'-'*5}-+-{'-'*30}-+-{'-'*30}-+-{'-'*6}-+-{'-'*7}")

        while not done:
            # Get positions from env internals
            ee_pos = env._get_ee_position()
            hand_pos = env._get_hand_position()
            dist = np.linalg.norm(ee_pos - hand_pos)

            # Select action
            with torch.no_grad():
                # Wrap obs to look like vectorized env output (add batch dim for images)
                policy_obs = preprocess_observation(obs, args.device)
                action = policy.select_action(policy_obs)
                action = action.cpu().numpy().squeeze()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Print every 10 steps or on final step
            if step % 10 == 1 or done:
                print(f"{step:5d} | [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}] | "
                      f"[{hand_pos[0]:+.3f}, {hand_pos[1]:+.3f}, {hand_pos[2]:+.3f}] | "
                      f"{dist:.4f} | {reward:+.2f}")

            # Render: live viewer or capture frames for video
            if args.render_mode == "human":
                env.render()
                time.sleep(1 / 30)  # throttle to ~30fps
            elif args.save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

        success = "SUCCESS" if info.get("is_success", False) else "FAIL"
        print(f"\nResult: {success} | Total reward: {total_reward:.2f} | Steps: {step}")

        if args.render_mode == "human":
            time.sleep(1.0)  # pause between episodes

    # Save video if requested
    if args.save_video and video_frames:
        try:
            import imageio
            imageio.mimsave(args.save_video, video_frames, fps=30)
            print(f"\nVideo saved to {args.save_video}")
        except ImportError:
            print("\nInstall imageio to save video: pip install imageio imageio-ffmpeg")

    env.close()


if __name__ == "__main__":
    main()
