#!/usr/bin/env python
"""
Train SAC policy for High-Five Robot task in MuJoCo simulation.

This script trains a Soft Actor-Critic (SAC) reinforcement learning policy
to control a SO-ARM101 robot arm to perform high-five/fist-bump interactions
with a moving hand target.

Usage:
    # Basic training with default settings (MPS for Apple Silicon)
    python -m lerobot.scripts.highfive.train_sac

    # Training with custom settings
    python -m lerobot.scripts.highfive.train_sac \\
        --device mps \\
        --num_steps 500000 \\
        --batch_size 256 \\
        --domain_randomization True

    # Training with wandb logging
    python -m lerobot.scripts.highfive.train_sac \\
        --wandb_project highfive_robot \\
        --wandb_run_name sac_v1

Environment Configuration:
    - State space: Bird's eye camera image (224x224 RGB) + joint positions
    - Action space: 5 joint positions (shoulder_pan, shoulder_lift, elbow_flex,
                    wrist_flex, wrist_roll)
    - Reward: -distance(gripper, hand) + contact_bonus
    - Episode: Terminates on contact or timeout (200 steps)

Apple Silicon Notes:
    - Uses MPS (Metal Performance Shaders) for GPU acceleration
    - MuJoCo runs natively on Apple Silicon
    - Reduce batch size if memory constrained

"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC policy for High-Five Robot task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500_000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for all networks",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=50_000,
        help="Replay buffer capacity (reduced for memory efficiency with images)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of random action steps before learning",
    )

    # Environment parameters
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=200,
        help="Maximum episode length",
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
        help="Enable domain randomization for sim-to-real transfer",
    )
    parser.add_argument(
        "--observation_size",
        type=int,
        default=84,
        help="Observation image size (square). 84 is standard for image-based RL.",
    )
    parser.add_argument(
        "--unfreeze_backbones",
        action="store_true",
        help="Unfreeze ResNet backbones for fine-tuning (default: frozen)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "cross_attention"],
        help="Sensor fusion method: concat (default) or cross_attention",
    )
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="Add depth channel to observations (RGBD instead of RGB)",
    )
    parser.add_argument(
        "--single_camera",
        action="store_true",
        help="Use only birdseye camera (simpler architecture)",
    )
    parser.add_argument(
        "--motion_freq_scale",
        type=float,
        default=0.3,
        help="Scale factor for hand motion frequency (0.5 = half speed)",
    )
    parser.add_argument(
        "--start_pose",
        type=str,
        default="folded",
        choices=["folded", "neutral"],
        help="Starting pose for robot arm",
    )
    parser.add_argument(
        "--bev_depth_wrist_rgb",
        action="store_true",
        help="Asymmetric sensors: BEV depth-only (1ch) + Wrist RGB (3ch)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=1,
        help="Number of frames to stack for temporal context (1=no stacking, 4=common choice)",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="small_cnn",
        choices=["resnet", "small_cnn"],
        help="Vision encoder: resnet (ImageNet pretrained) or small_cnn (trained from scratch)",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        default="pixels_agent_pos",
        choices=["pixels_agent_pos", "state"],
        help="Observation type: pixels_agent_pos (vision+joints) or state (joints+hand pos, no cameras)",
    )
    parser.add_argument(
        "--utd_ratio",
        type=int,
        default=1,
        help="Update-to-data ratio: gradient updates per env step (1 for pixels, 10-20 for state)",
    )

    # Logging and saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10_000,
        help="Frequency of checkpoint saving (in steps)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5_000,
        help="Frequency of evaluation (in steps)",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Frequency of logging (in steps)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )

    # Wandb
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name (None to disable)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name",
    )

    # SAC specific
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Target network soft update coefficient",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization during training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> Path:
    """Setup output directory and logging."""
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = f"outputs/highfive/sac_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Save full config as JSON for reproducibility
    config_path = output_dir / "config.json"
    config_dict = vars(args).copy()
    config_dict["timestamp"] = datetime.now().isoformat()
    config_dict["command"] = " ".join(sys.argv)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    return output_dir


def create_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    """Create the high-five environment."""
    from lerobot.envs.configs import HighFiveEnvConfig
    from lerobot.envs.factory import make_env

    render_mode = "human" if args.render else "rgb_array"

    # State mode doesn't need cameras
    single_camera = args.single_camera or args.obs_type == "state"

    env_config = HighFiveEnvConfig(
        episode_length=args.episode_length,
        obs_type=args.obs_type,
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        hand_motion_type=args.hand_motion_type,
        domain_randomization=args.domain_randomization,
        render_mode=render_mode,
        motion_freq_scale=args.motion_freq_scale,
        use_depth=args.use_depth,
        single_camera=single_camera,
        bev_depth_wrist_rgb=args.bev_depth_wrist_rgb,
        start_pose=args.start_pose,
    )

    env_dict = make_env(env_config, n_envs=args.n_envs)
    # Get the vectorized environment (single task, task_id=0)
    vec_env = env_dict["highfive"][0]

    # Apply frame stacking if requested
    if args.frame_stack > 1:
        from lerobot.scripts.highfive.frame_stack import VecFrameStackWrapper
        vec_env = VecFrameStackWrapper(vec_env, n_frames=args.frame_stack)
        print(f"Applied frame stacking with {args.frame_stack} frames")

    return vec_env


def create_policy(args: argparse.Namespace, env: gym.vector.VectorEnv):
    """Create the SAC policy with custom dual-camera encoder."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.scripts.highfive.custom_sac import HighFiveSACPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    use_state_only = args.obs_type == "state"

    if use_state_only:
        # State-only: 11-dim state vector (5 joints + 3 ee pos + 3 hand pos), no images
        input_features = {
            OBS_STATE: PolicyFeature(
                type=FeatureType.STATE,
                shape=(11,),
            ),
        }
        encoder_config = {
            "latent_dim": 256,
            "state_dim": 11,
            "encoder_type": "state",
            "single_camera": True,
        }
    else:
        obs_size = args.observation_size
        n_frames = args.frame_stack

        # Determine number of image channels based on configuration
        if args.bev_depth_wrist_rgb:
            bev_channels = 1 * n_frames
            wrist_channels = 3 * n_frames
        else:
            base_channels = 4 if args.use_depth else 3
            bev_channels = base_channels * n_frames
            wrist_channels = base_channels * n_frames

        input_features = {
            f"{OBS_IMAGES}.birdseye": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(bev_channels, obs_size, obs_size),
            ),
            OBS_STATE: PolicyFeature(
                type=FeatureType.STATE,
                shape=(5,),
            ),
        }

        if not args.single_camera:
            input_features[f"{OBS_IMAGES}.wrist"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(wrist_channels, obs_size, obs_size),
            )

        encoder_config = {
            "latent_dim": 256,
            "state_dim": 5,
            "pretrained": True,
            "freeze_backbones": not args.unfreeze_backbones,
            "fusion": args.fusion,
            "use_depth": args.use_depth,
            "single_camera": args.single_camera,
            "bev_depth_wrist_rgb": args.bev_depth_wrist_rgb,
            "birdseye_channels": bev_channels,
            "wrist_channels": wrist_channels,
            "encoder_type": args.encoder_type,
        }

    output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(5,),  # 5 joint actions
        ),
    }

    # Configure SAC
    policy_config = SACConfig(
        device=args.device,
        input_features=input_features,
        output_features=output_features,
        # SAC hyperparameters
        discount=args.discount,
        critic_target_update_weight=args.tau,
        critic_lr=args.learning_rate,
        actor_lr=args.learning_rate,
        temperature_lr=args.learning_rate,
        online_buffer_capacity=args.buffer_capacity,
        online_step_before_learning=args.warmup_steps,
        # Vision encoder settings (our custom encoder ignores these)
        vision_encoder_name=None,
        image_encoder_hidden_dim=32,
        shared_encoder=True,  # Actor/critic share our custom encoder
        # Disable torch compile for Apple Silicon compatibility
        use_torch_compile=False,
    )

    policy = HighFiveSACPolicy(policy_config, encoder_config=encoder_config)
    policy.to(args.device)

    return policy, policy_config


def preprocess_observation(
    obs: dict[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    """Preprocess environment observation for policy input."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    processed = {}

    # State-only mode: joint positions + hand position
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)
        processed[OBS_STATE] = state.to(device)
        return processed

    # Process dual camera observations (birdseye + wrist)
    if "pixels" in obs:
        for camera_name in ["birdseye", "wrist"]:
            if camera_name in obs["pixels"]:
                image = obs["pixels"][camera_name]
                if isinstance(image, np.ndarray):
                    # Convert to tensor: (H, W, C) -> (B, C, H, W) and normalize to [0, 1]
                    image = torch.from_numpy(image).float() / 255.0
                    if image.dim() == 3:
                        image = image.unsqueeze(0)  # Add batch dimension
                    image = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                processed[f"{OBS_IMAGES}.{camera_name}"] = image.to(device)

    # Process agent position (joint positions)
    if "agent_pos" in obs:
        agent_pos = obs["agent_pos"]
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)  # Add batch dimension
        processed[OBS_STATE] = agent_pos.to(device)

    return processed


def evaluate(
    policy,
    env: gym.vector.VectorEnv,
    num_episodes: int,
    device: str,
) -> dict[str, float]:
    """Evaluate the policy."""
    policy.eval()

    total_rewards = []
    total_successes = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                policy_obs = preprocess_observation(obs, device)
                action = policy.select_action(policy_obs)
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()

            episode_reward += reward.sum()
            episode_length += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check success from info (handle vectorized env format)
        if "is_success" in info:
            success = info["is_success"]
            # Handle array from vectorized env
            if hasattr(success, "__iter__") and not isinstance(success, (str, dict)):
                success = success[0] if len(success) > 0 else False
            total_successes.append(float(success))
        elif "final_info" in info:
            final_info = info["final_info"]
            # Handle list of dicts from vectorized env
            if isinstance(final_info, (list, tuple)) and len(final_info) > 0:
                final_info = final_info[0]
            if isinstance(final_info, dict):
                total_successes.append(float(final_info.get("is_success", False)))
            else:
                total_successes.append(0.0)
        else:
            total_successes.append(0.0)

    policy.train()

    return {
        "eval/mean_reward": np.mean(total_rewards),
        "eval/std_reward": np.std(total_rewards),
        "eval/success_rate": np.mean(total_successes),
        "eval/mean_episode_length": np.mean(episode_lengths),
    }


def train(args: argparse.Namespace):
    """Main training loop."""
    # Setup
    output_dir = setup_logging(args)
    print(f"Output directory: {output_dir}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize wandb if requested
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"highfive_sac_{args.seed}",
                config=vars(args),
            )
        except ImportError:
            print("Wandb not installed, skipping logging")

    # Create environment
    print("Creating environment...")
    env = create_env(args)

    # Create policy
    print(f"Creating SAC policy on device: {args.device}")
    policy, policy_config = create_policy(args, env)

    # Resume from checkpoint if specified
    resume_step = 0
    if args.resume:
        from safetensors.torch import load_model as load_model_as_safetensor

        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            model_file = resume_path / "model.safetensors"
            if model_file.exists():
                load_model_as_safetensor(policy, str(model_file), strict=False)
                print(f"Loaded weights from {model_file}")
            else:
                print(f"Warning: {model_file} not found")

            # Try to extract step number from path (e.g., "step_20000")
            if "step_" in resume_path.name:
                try:
                    resume_step = int(resume_path.name.split("step_")[1])
                    print(f"Resuming from step {resume_step}")
                except ValueError:
                    pass
        else:
            print(f"Warning: Checkpoint path {resume_path} not found, starting from scratch")

    # Create replay buffer
    from lerobot.rl.buffer import ReplayBuffer
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    state_keys = [f"{OBS_IMAGES}.image", OBS_STATE]
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        device=args.device,
        state_keys=state_keys,
    )

    # Create separate optimizers for actor, critic, and temperature (required by SAC)
    # get_optim_params() filters shared encoder params out of the actor optimizer
    optim_params = policy.get_optim_params()
    optimizers = {
        "critic": optim.Adam(optim_params["critic"], lr=args.learning_rate),
        "actor": optim.Adam(optim_params["actor"], lr=args.learning_rate),
        "temperature": optim.Adam([optim_params["temperature"]], lr=args.learning_rate),
    }

    # Training state
    global_step = resume_step
    episode_count = 0
    best_success_rate = 0.0

    # Signal handler for graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\nShutdown requested, finishing current episode...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting training for {args.num_steps} steps...")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Batch size: {args.batch_size}")

    # Initial reset
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0

    while global_step < args.num_steps and not shutdown_requested:
        # Select action
        if global_step < args.warmup_steps:
            # Random action during warmup
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                policy_obs = preprocess_observation(obs, args.device)
                action = policy.select_action(policy_obs)
                # Keep batch dimension for vectorized env (n_envs, action_dim)
                action = action.cpu().numpy()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated.any() or truncated.any()

        # Render if visualization enabled
        if args.render:
            env.call("render")

        episode_reward += reward.sum()
        episode_length += 1
        global_step += 1

        # Store transition in replay buffer
        policy_obs = preprocess_observation(obs, args.device)
        policy_next_obs = preprocess_observation(next_obs, args.device)

        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action)
            # Add batch dim only if action is 1D (single env)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
        else:
            action_tensor = action

        replay_buffer.add(
            state=policy_obs,
            action=action_tensor,
            reward=float(reward.sum()),
            next_state=policy_next_obs,
            done=bool(done),
            truncated=bool(truncated.any()),
        )

        # Train policy (multiple updates per env step when utd_ratio > 1)
        if global_step >= args.warmup_steps and len(replay_buffer) >= args.batch_size:
            for _utd in range(args.utd_ratio):
                batch = replay_buffer.sample(args.batch_size)

                # Train critic
                critic_loss_dict = policy.forward(batch, model="critic")
                critic_loss = critic_loss_dict["loss_critic"]

                optimizers["critic"].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), max_norm=1.0)
                optimizers["critic"].step()

                # Update target networks
                policy.update_target_networks()

                # Train actor and temperature (every other update)
                actor_loss = None
                temperature_loss = None
                if _utd % 2 == 0:
                    # Actor optimization
                    actor_loss_dict = policy.forward(batch, model="actor")
                    actor_loss = actor_loss_dict["loss_actor"]

                    optimizers["actor"].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(optim_params["actor"], max_norm=1.0)
                    optimizers["actor"].step()

                    # Temperature optimization
                    temp_loss_dict = policy.forward(batch, model="temperature")
                    temperature_loss = temp_loss_dict["loss_temperature"]

                    optimizers["temperature"].zero_grad()
                    temperature_loss.backward()
                    torch.nn.utils.clip_grad_norm_([policy.log_alpha], max_norm=1.0)
                    optimizers["temperature"].step()

                    policy.update_temperature()

            # Logging
            if global_step % args.log_freq == 0:
                log_dict = {
                    "train/critic_loss": critic_loss.item(),
                    "train/temperature": policy.temperature,
                    "train/step": global_step,
                    "train/buffer_size": len(replay_buffer),
                }
                if actor_loss is not None:
                    log_dict["train/actor_loss"] = actor_loss.item()
                if temperature_loss is not None:
                    log_dict["train/temperature_loss"] = temperature_loss.item()

                if wandb_run:
                    wandb_run.log(log_dict, step=global_step)

                print(
                    f"Step {global_step:>7d} | Critic: {critic_loss.item():.4f} | "
                    f"Temp: {policy.temperature:.4f} | Buffer: {len(replay_buffer):>6d}"
                )

        # Episode end
        if done:
            episode_count += 1

            is_success = False
            if "is_success" in info:
                is_success = info["is_success"]
            elif "final_info" in info and isinstance(info["final_info"], dict):
                is_success = info["final_info"].get("is_success", False)

            # Convert is_success to scalar if it's an array
            success_value = is_success
            if hasattr(is_success, 'item'):
                success_value = is_success.item()
            elif hasattr(is_success, '__iter__') and not isinstance(is_success, (str, bool)):
                success_value = bool(is_success[0]) if len(is_success) > 0 else False

            log_dict = {
                "episode/reward": episode_reward,
                "episode/length": episode_length,
                "episode/success": float(success_value),
                "episode/count": episode_count,
            }

            if wandb_run:
                wandb_run.log(log_dict, step=global_step)

            print(
                f"Episode {episode_count:>4d} | Reward: {episode_reward:>8.2f} | "
                f"Length: {episode_length:>3d} | Success: {is_success}"
            )

            # Reset for next episode
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # Evaluation
        if global_step % args.eval_freq == 0 and global_step > args.warmup_steps:
            print(f"\nEvaluating at step {global_step}...")
            eval_metrics = evaluate(policy, env, args.eval_episodes, args.device)

            for k, v in eval_metrics.items():
                print(f"  {k}: {v:.4f}")

            if wandb_run:
                wandb_run.log(eval_metrics, step=global_step)

            # Save best model
            if eval_metrics["eval/success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["eval/success_rate"]
                best_path = output_dir / "checkpoints" / "best_model"
                policy.save_pretrained(best_path)
                print(f"  New best model saved! Success rate: {best_success_rate:.2%}")

        # Save checkpoint
        if global_step % args.save_freq == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{global_step}"
            policy.save_pretrained(checkpoint_path)
            print(f"Checkpoint saved at step {global_step}")

    # Final save
    final_path = output_dir / "checkpoints" / "final_model"
    policy.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if wandb_run:
        wandb_run.finish()

    env.close()

    return policy


def main():
    args = parse_args()

    print("=" * 60)
    print("High-Five Robot Training with SAC")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Total steps: {args.num_steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Buffer capacity: {args.buffer_capacity:,}")
    print(f"Observation size: {args.observation_size}x{args.observation_size}")
    print(f"Hand motion type: {args.hand_motion_type}")
    print(f"Motion freq scale: {args.motion_freq_scale}")
    print(f"Domain randomization: {args.domain_randomization}")
    print(f"Fusion method: {args.fusion}")
    print(f"Single camera: {args.single_camera}")
    print(f"Use depth (RGBD): {args.use_depth}")
    print(f"BEV depth + Wrist RGB: {args.bev_depth_wrist_rgb}")
    print(f"Frame stacking: {args.frame_stack}")
    print(f"Observation type: {args.obs_type}")
    print(f"UTD ratio: {args.utd_ratio}")
    print(f"Encoder type: {args.encoder_type}")
    print(f"Start pose: {args.start_pose}")
    print(f"Unfreeze backbones: {args.unfreeze_backbones}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
