#!/usr/bin/env python
"""Evaluate a trained ACT policy in the high-five environment.

Usage:
    # Headless eval with stats:
    python -m lerobot.scripts.highfive.eval_act \
        --checkpoint outputs/highfive/act_sinusoidal \
        --dataset_path data/highfive_demos/sac_2026-02-17-sinusoidal/demos \
        --num_episodes 20

    # Live viewer:
    python -m lerobot.scripts.highfive.eval_act \
        --checkpoint outputs/highfive/act_sinusoidal \
        --dataset_path data/highfive_demos/sac_2026-02-17-sinusoidal/demos \
        --render_mode human --num_episodes 5

    # Save video:
    python -m lerobot.scripts.highfive.eval_act \
        --checkpoint outputs/highfive/act_sinusoidal \
        --dataset_path data/highfive_demos/sac_2026-02-17-sinusoidal/demos \
        --save_video out.mp4 --num_episodes 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--hand_motion_type", type=str, default="sinusoidal")
    parser.add_argument("--motion_freq_scale", type=float, default=0.5)
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        choices=["human", "rgb_array"])
    parser.add_argument("--save_video", type=str, default=None)
    parser.add_argument("--force_disturbances", action="store_true")
    parser.add_argument("--force_disturbance_max", type=float, default=2.0)
    parser.add_argument("--show_cameras", action="store_true", help="Show camera feeds in OpenCV window")
    parser.add_argument("--randomize_hand_position", action="store_true",
                        help="Randomize hand base position each episode")
    parser.add_argument("--domain_randomization", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--palm_target_size", type=float, default=0.06)
    parser.add_argument("--device", type=str,
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    from lerobot.envs.highfive.highfive_env import HighFiveEnv

    device = torch.device(args.device)

    # Load policy
    print(f"Loading ACT policy from {args.checkpoint}...")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # Load preprocessor/postprocessor from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  temporal_ensemble_coeff: {policy.config.temporal_ensemble_coeff}")

    # Create environment
    env = HighFiveEnv(
        episode_length=200,
        obs_type="pixels_agent_pos",
        render_mode=args.render_mode,
        observation_width=84,
        observation_height=84,
        hand_motion_type=args.hand_motion_type,
        domain_randomization=args.domain_randomization,
        single_camera=False,
        motion_freq_scale=args.motion_freq_scale,
        start_pose="folded",
        randomize_hand_position=args.randomize_hand_position,
        palm_target_size=args.palm_target_size,
        force_disturbances=args.force_disturbances,
        force_disturbance_max=args.force_disturbance_max,
    )

    successes = 0
    total_rewards = []
    video_frames = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        policy.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            obs_dict = {}

            # Images: (H, W, 3) uint8 -> (1, 3, H, W) float
            for cam_name in ["birdseye", "wrist"]:
                img = obs["pixels"][cam_name]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                obs_dict[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

            # State: agent_pos (13 dims)
            state = obs["agent_pos"].astype("float32")
            obs_dict[OBS_STATE] = torch.from_numpy(state).unsqueeze(0).to(device)

            obs_dict = preprocessor(obs_dict)

            with torch.no_grad():
                action = policy.select_action(obs_dict)

            action = postprocessor(action)
            action_np = action.cpu().numpy().squeeze(0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            if args.render_mode == "human":
                env.render()
                if args.show_cameras:
                    birdseye = obs["pixels"]["birdseye"]
                    wrist = obs["pixels"]["wrist"]
                    frames = [birdseye, wrist]
                    if not hasattr(main, '_cam_fig'):
                        import matplotlib
                        matplotlib.use('TkAgg')
                        import matplotlib.pyplot as plt
                        main._cam_fig, main._cam_axes = plt.subplots(1, 2, figsize=(8, 4))
                        main._cam_ims = []
                        for ax, f, name in zip(main._cam_axes, frames, ["birdseye", "wrist"]):
                            main._cam_ims.append(ax.imshow(f))
                            ax.set_title(name)
                            ax.axis('off')
                        plt.ion()
                        plt.show()
                    else:
                        import matplotlib.pyplot as plt
                        for im, f in zip(main._cam_ims, frames):
                            im.set_data(f)
                        main._cam_fig.canvas.draw_idle()
                        main._cam_fig.canvas.flush_events()
                time.sleep(1 / 30)

            if args.save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

        is_success = info.get("is_success", False)
        if hasattr(is_success, "item"):
            is_success = is_success.item()
        if is_success:
            successes += 1
        total_rewards.append(episode_reward)

        status = "SUCCESS" if is_success else "FAIL"
        dist = info.get("distance", float("nan"))
        print(
            f"  Episode {ep+1:>3d} | {status:>7s} | "
            f"Reward: {episode_reward:>7.1f} | Dist: {dist:.4f} | Steps: {step}"
        )

        if args.render_mode == "human":
            time.sleep(0.5)

    env.close()
    if args.show_cameras and hasattr(main, '_cam_fig'):
        import matplotlib.pyplot as plt
        plt.close(main._cam_fig)

    print(f"\nResults ({args.num_episodes} episodes):")
    print(f"  Success rate: {successes}/{args.num_episodes} "
          f"({100 * successes / args.num_episodes:.0f}%)")
    print(f"  Mean reward: {np.mean(total_rewards):.1f} +/- {np.std(total_rewards):.1f}")

    if args.save_video and video_frames:
        try:
            import imageio
            imageio.mimsave(args.save_video, video_frames, fps=30)
            print(f"  Video saved to {args.save_video}")
        except ImportError:
            print("  Install imageio to save video: pip install imageio imageio-ffmpeg")


if __name__ == "__main__":
    main()
