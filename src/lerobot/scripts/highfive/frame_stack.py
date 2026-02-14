"""
Frame stacking wrapper for temporal context in RL.

Stacks the last N frames to give the policy information about motion/velocity.
This is a simple but effective approach for tasks requiring temporal reasoning.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper that stacks the last N frames for each image observation.

    For image observations, frames are stacked along the channel dimension:
    - Input: (H, W, C) per frame
    - Output: (H, W, C * n_frames)

    State observations (like agent_pos) are NOT stacked - only the current
    state is returned, as joint positions don't benefit from stacking.

    Args:
        env: The environment to wrap
        n_frames: Number of frames to stack (default: 4)
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames: dict[str, deque] = {}

        # Modify observation space for stacked images
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Update observation space to reflect frame stacking."""
        old_space = self.env.observation_space

        if isinstance(old_space, spaces.Dict):
            new_spaces = {}
            for key, space in old_space.spaces.items():
                if key == "pixels" and isinstance(space, spaces.Dict):
                    # Stack each camera's images
                    new_pixel_spaces = {}
                    for cam_key, cam_space in space.spaces.items():
                        if isinstance(cam_space, spaces.Box):
                            # Stack along channel dimension
                            old_shape = cam_space.shape  # (H, W, C)
                            new_shape = (old_shape[0], old_shape[1], old_shape[2] * self.n_frames)
                            new_pixel_spaces[cam_key] = spaces.Box(
                                low=0,
                                high=255,
                                shape=new_shape,
                                dtype=np.uint8,
                            )
                        else:
                            new_pixel_spaces[cam_key] = cam_space
                    new_spaces[key] = spaces.Dict(new_pixel_spaces)
                elif key == "agent_pos" and isinstance(space, spaces.Box):
                    # Stack joint positions along last dimension
                    old_shape = space.shape
                    new_shape = (old_shape[0] * self.n_frames,)
                    new_spaces[key] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=new_shape, dtype=space.dtype,
                    )
                else:
                    new_spaces[key] = space

            self.observation_space = spaces.Dict(new_spaces)
        else:
            # If not a dict space, assume it's a single image
            if isinstance(old_space, spaces.Box) and len(old_space.shape) == 3:
                old_shape = old_space.shape
                new_shape = (old_shape[0], old_shape[1], old_shape[2] * self.n_frames)
                self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=new_shape,
                    dtype=np.uint8,
                )

    def _get_stacked_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Stack frames for image observations."""
        stacked_obs = {}

        for key, value in obs.items():
            if key == "pixels" and isinstance(value, dict):
                stacked_pixels = {}
                for cam_key, image in value.items():
                    # Initialize frame buffer if needed
                    buffer_key = f"pixels.{cam_key}"
                    if buffer_key not in self._frames:
                        self._frames[buffer_key] = deque(maxlen=self.n_frames)
                        # Fill with copies of first frame
                        for _ in range(self.n_frames):
                            self._frames[buffer_key].append(image.copy())
                    else:
                        self._frames[buffer_key].append(image.copy())

                    # Stack frames along channel dimension
                    stacked = np.concatenate(list(self._frames[buffer_key]), axis=-1)
                    stacked_pixels[cam_key] = stacked

                stacked_obs[key] = stacked_pixels
            elif key == "agent_pos":
                # Stack joint positions for velocity information
                buffer_key = "agent_pos"
                if buffer_key not in self._frames:
                    self._frames[buffer_key] = deque(maxlen=self.n_frames)
                    for _ in range(self.n_frames):
                        self._frames[buffer_key].append(value.copy())
                else:
                    self._frames[buffer_key].append(value.copy())
                stacked_obs[key] = np.concatenate(list(self._frames[buffer_key]), axis=-1)
            else:
                stacked_obs[key] = value

        return stacked_obs

    def reset(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment and frame buffers."""
        obs, info = self.env.reset(**kwargs)

        # Clear frame buffers
        self._frames.clear()

        # Get stacked observation (will initialize buffers with first frame)
        stacked_obs = self._get_stacked_obs(obs)

        return stacked_obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        stacked_obs = self._get_stacked_obs(obs)
        return stacked_obs, reward, terminated, truncated, info


class VecFrameStackWrapper:
    """
    Frame stacking wrapper for vectorized environments.

    Handles the case where observations come from multiple parallel environments.
    """

    def __init__(self, vec_env, n_frames: int = 4):
        self.vec_env = vec_env
        self.n_frames = n_frames
        self._frames: dict[str, deque] = {}

        # Forward attributes from wrapped env
        self.action_space = vec_env.action_space
        self.single_action_space = vec_env.single_action_space
        self.num_envs = vec_env.num_envs

        # Update observation space
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Update observation space for stacked frames."""
        old_space = self.vec_env.single_observation_space

        if isinstance(old_space, spaces.Dict):
            new_spaces = {}
            for key, space in old_space.spaces.items():
                if key == "pixels" and isinstance(space, spaces.Dict):
                    new_pixel_spaces = {}
                    for cam_key, cam_space in space.spaces.items():
                        if isinstance(cam_space, spaces.Box):
                            old_shape = cam_space.shape
                            new_shape = (old_shape[0], old_shape[1], old_shape[2] * self.n_frames)
                            new_pixel_spaces[cam_key] = spaces.Box(
                                low=0, high=255, shape=new_shape, dtype=np.uint8
                            )
                        else:
                            new_pixel_spaces[cam_key] = cam_space
                    new_spaces[key] = spaces.Dict(new_pixel_spaces)
                elif key == "agent_pos" and isinstance(space, spaces.Box):
                    old_shape = space.shape
                    new_shape = (old_shape[0] * self.n_frames,)
                    new_spaces[key] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=new_shape, dtype=space.dtype,
                    )
                else:
                    new_spaces[key] = space
            self.single_observation_space = spaces.Dict(new_spaces)
            self.observation_space = self.single_observation_space
        else:
            self.single_observation_space = old_space
            self.observation_space = old_space

    def _get_stacked_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Stack frames for vectorized observations."""
        stacked_obs = {}

        for key, value in obs.items():
            if key == "pixels" and isinstance(value, dict):
                stacked_pixels = {}
                for cam_key, images in value.items():
                    # images shape: (n_envs, H, W, C)
                    buffer_key = f"pixels.{cam_key}"
                    if buffer_key not in self._frames:
                        self._frames[buffer_key] = deque(maxlen=self.n_frames)
                        for _ in range(self.n_frames):
                            self._frames[buffer_key].append(images.copy())
                    else:
                        self._frames[buffer_key].append(images.copy())

                    # Stack: list of (n_envs, H, W, C) -> (n_envs, H, W, C*n_frames)
                    stacked = np.concatenate(list(self._frames[buffer_key]), axis=-1)
                    stacked_pixels[cam_key] = stacked

                stacked_obs[key] = stacked_pixels
            elif key == "agent_pos":
                buffer_key = "agent_pos"
                if buffer_key not in self._frames:
                    self._frames[buffer_key] = deque(maxlen=self.n_frames)
                    for _ in range(self.n_frames):
                        self._frames[buffer_key].append(value.copy())
                else:
                    self._frames[buffer_key].append(value.copy())
                stacked_obs[key] = np.concatenate(list(self._frames[buffer_key]), axis=-1)
            else:
                stacked_obs[key] = value

        return stacked_obs

    def reset(self, **kwargs):
        """Reset and clear frame buffers."""
        obs, info = self.vec_env.reset(**kwargs)
        self._frames.clear()
        return self._get_stacked_obs(obs), info

    def step(self, action):
        """Step and update frame stack."""
        obs, reward, terminated, truncated, info = self.vec_env.step(action)
        return self._get_stacked_obs(obs), reward, terminated, truncated, info

    def close(self):
        """Close wrapped environment."""
        return self.vec_env.close()

    def call(self, method_name, *args, **kwargs):
        """Forward method calls to wrapped env."""
        return self.vec_env.call(method_name, *args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to wrapped env."""
        return getattr(self.vec_env, name)
