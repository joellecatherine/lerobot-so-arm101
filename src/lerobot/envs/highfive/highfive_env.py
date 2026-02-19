#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-Five Robot Environment using MuJoCo.

This environment simulates a SO-ARM101 robot arm that must track and
intercept a moving hand target for a high-five/fist-bump interaction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# MuJoCo import with fallback error message
try:
    import mujoco
    from mujoco import viewer as mujoco_viewer
except ImportError as e:
    raise ImportError(
        "MuJoCo is required for the HighFive environment. "
        "Install it with: pip install mujoco"
    ) from e


# Environment constants
ACTION_DIM = 5  # 5 arm joints (gripper stays closed)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 200
CONTACT_BONUS = 80.0  # Base bonus reward for contact (decays with timestep)
CONTACT_BONUS_GAMMA = 0.99  # Decay rate for contact bonus per timestep
CONTACT_TERMINATE_STEPS = 0  # Steps to continue after first contact before terminating
REWARD_SCALE = 0.4  # Scale for exponential decay reward
REWARD_SIGMA = 5.0  # Decay rate (per meter) for exponential reward
ACTION_RATE_PENALTY = 0.01  # Penalty for jerky actions (squared diff)
CONTACT_FORCE_PENALTY = 0.001  # Penalty per Newton of contact force (encourages gentle approach)
FACING_REWARD_SCALE = 0.1  # Scale for facing reward (gripper pointing toward palm)

# Starting poses (5 arm joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
START_POSES = {
    "neutral": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "folded": np.array([0.0, -1.5, 1.5, 0.0, 0.0]),
}


def get_asset_path() -> Path:
    """Get the path to the assets directory."""
    return Path(__file__).parent / "assets"


class HighFiveEnv(gym.Env):
    """High-Five Robot Environment.

    A MuJoCo-based environment where a robot arm must track and intercept
    a moving hand target. The robot's gripper stays closed (fist bump style).

    Observation Space:
        - pixels: Bird's eye camera image (224x224 RGB by default)
        - agent_pos: Joint positions (optional, for pixels_agent_pos mode)

    Action Space:
        - 5 continuous joint positions: shoulder_pan, shoulder_lift,
          elbow_flex, wrist_flex, wrist_roll

    Reward:
        - Negative distance between gripper and hand target
        - Bonus reward for contact

    Episode Termination:
        - Contact between gripper and hand
        - Timeout (default 200 steps)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    # Joint names matching SO-ARM101 configuration
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    # Actuator names in the MuJoCo model (from so101_new_calib.xml)
    ACTUATOR_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    def __init__(
        self,
        task_id: int = 0,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 84,
        observation_height: int = 84,
        visualization_width: int = 640,
        visualization_height: int = 480,
        camera_name: str = "birdseye",
        hand_motion_type: str = "sinusoidal",
        domain_randomization: bool = True,
        seed: int | None = None,
        motion_freq_scale: float = 1.0,
        use_depth: bool = False,
        single_camera: bool = False,
        bev_depth_wrist_rgb: bool = False,
        start_pose: str = "folded",
        randomize_hand_position: bool = False,
        force_disturbances: bool = False,
        force_disturbance_max: float = 30.0,
        force_disturbance_interval: tuple[int, int] = (30, 100),
        force_disturbance_duration: int = 15,
        force_kicks_per_episode: int = 1,
        closing_reward: bool = False,
        facing_reward: bool = False,
        palm_target_size: float = 0.05,
    ):
        """Initialize the High-Five environment.

        Args:
            task_id: Task identifier (for multi-task compatibility)
            episode_length: Maximum steps per episode
            obs_type: Observation type ("pixels" or "pixels_agent_pos")
            render_mode: Rendering mode ("rgb_array" or "human")
            observation_width: Width of observation images
            observation_height: Height of observation images
            visualization_width: Width for human visualization
            visualization_height: Height for human visualization
            camera_name: Name of camera for observations
            hand_motion_type: Type of hand motion ("static", "random", "sinusoidal", "tracking")
            domain_randomization: Whether to apply domain randomization
            seed: Random seed
            motion_freq_scale: Scale factor for hand motion frequency (0.5 = half speed)
            use_depth: Add depth channel to observations (RGBD)
            single_camera: Use only birdseye camera
            bev_depth_wrist_rgb: Asymmetric mode - BEV depth-only + Wrist RGB
            randomize_hand_position: Randomize hand base position each episode
            force_disturbances: Apply random Cartesian force kicks to arm bodies
            force_disturbance_max: Max force magnitude (N) applied to arm body
            force_disturbance_interval: (min, max) steps between kicks
            force_disturbance_duration: How many steps each kick lasts
        """
        super().__init__()

        self.task_id = task_id
        self.episode_length = episode_length
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name
        self.hand_motion_type = hand_motion_type
        self.domain_randomization = domain_randomization
        self.motion_freq_scale = motion_freq_scale
        self.use_depth = use_depth
        self.single_camera = single_camera
        self.bev_depth_wrist_rgb = bev_depth_wrist_rgb
        self.start_pose = start_pose
        self.randomize_hand_position = randomize_hand_position
        self.force_disturbances = force_disturbances
        self.force_disturbance_max = force_disturbance_max
        self.force_disturbance_interval = force_disturbance_interval
        self.force_disturbance_duration = force_disturbance_duration
        self.force_kicks_per_episode = force_kicks_per_episode
        self.closing_reward = closing_reward
        self.facing_reward = facing_reward
        self.palm_target_size = palm_target_size

        # Load MuJoCo model
        self._load_model()

        # Initialize state tracking
        self._max_episode_steps = episode_length
        self._step_count = 0
        self._episode_count = 0
        self._rng = np.random.default_rng(seed)

        # Hand motion parameters (scaled by motion_freq_scale)
        # Robot base is at origin. Reachable bounds:
        #   x: [0.35, 0.55], y: [-0.40, +0.40], z: [0.20, 0.30]
        self._hand_base_pos = np.array([0.40, 0.0, 0.25])
        self._hand_motion_amplitude = np.array([0.03, 0.05, 0.03])
        self._hand_motion_freq = np.array([0.5, 0.3, 0.2]) * self.motion_freq_scale
        self._hand_motion_phase = np.zeros(3)

        # Viewer for human rendering
        self._viewer = None

        # Cached renderers for each camera (prevents resource leak)
        self._renderers = {}

        # Define observation space
        self._setup_observation_space()

        # Define action space (5 joint positions, normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _load_model(self):
        """Load the MuJoCo model."""
        model_path = get_asset_path() / "scene.xml"
        if not model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found at {model_path}. "
                "Please ensure the scene.xml file exists in the assets directory."
            )

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        # Cache important indices
        self._joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        ]
        self._actuator_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ACTUATOR_NAMES
        ]

        # Get camera ID
        self._camera_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )

        # Get site IDs for distance computation
        self._ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self._hand_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "hand_contact"
        )

        # Get hand body ID for position control
        self._hand_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "hand_target"
        )

        # Cache joint DOF addresses for velocity readout and gravity compensation
        self._joint_dof_ids = [
            self._model.jnt_dofadr[jid] for jid in self._joint_ids
        ]

        # Get joint address for hand free joint
        self._hand_joint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free"
        )
        self._hand_qpos_addr = self._model.jnt_qposadr[self._hand_joint_id]

        # Gripper actuator (6th DOF) — held closed during high-five
        self._gripper_actuator_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )

        # Body IDs for orientation computation
        self._gripper_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )

        # Geom IDs for physics-based contact detection
        self._fist_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "fist"
        )
        self._palm_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "palm"
        )
        self._palm_target_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "palm_target"
        )
        # Set palm target zone size (half-extents: width, depth, height)
        half = self.palm_target_size / 2.0
        self._model.geom_size[self._palm_target_geom_id] = [half, 0.005, half]
        self._gripper_col_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_collision"
        )
        # All hand geom IDs for contact force computation
        self._hand_geom_ids = {self._palm_geom_id}
        # All gripper geom IDs
        self._gripper_geom_ids = {self._fist_geom_id, self._gripper_col_geom_id}
        # Arm body IDs for xfrc_applied force disturbances
        self._arm_body_names = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
        self._arm_body_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._arm_body_names
        ]
        # Visual kick indicator geom IDs (one per body, group 4 = camera-hidden)
        self._kick_indicator_ids = {
            name: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"kick_ind_{jname}")
            for name, jname in zip(
                self._arm_body_names,
                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            )
        }
        self._kick_active_visual = False

    def _setup_observation_space(self):
        """Set up the observation space based on obs_type."""
        if self.obs_type == "state":
            # State-only: joint_pos(5) + joint_vel(5) + ee_pos(3) + hand_pos(3) + hand_vel(3) = 19 dims
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(19,),
                    dtype=np.float64,
                ),
            })
            return

        # Determine channels for each camera based on configuration
        if self.bev_depth_wrist_rgb:
            # Asymmetric: BEV depth-only (1 channel), Wrist RGB (3 channels)
            birdseye_channels = 1
            wrist_channels = 3
        elif self.use_depth:
            # RGBD mode: 4 channels for both
            birdseye_channels = 4
            wrist_channels = 4
        else:
            # Standard RGB: 3 channels for both
            birdseye_channels = 3
            wrist_channels = 3

        birdseye_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, birdseye_channels),
            dtype=np.uint8,
        )
        wrist_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, wrist_channels),
            dtype=np.uint8,
        )

        # Dual camera observation space (birdseye + wrist)
        pixels_dict = {
            "birdseye": birdseye_space,
            "wrist": wrist_space,
        }

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict(pixels_dict),
            })
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict(pixels_dict),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ACTION_DIM * 2 + 3,),  # 5 joint pos + 5 joint vel + 3 EE position
                    dtype=np.float64,
                ),
            })
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    def _get_camera_image(
        self,
        camera_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
        include_depth: bool = False,
        depth_only: bool = False,
    ) -> np.ndarray:
        """Render an image from the specified camera.

        Args:
            camera_name: Name of camera to render from
            width: Image width
            height: Image height
            include_depth: If True, return RGBD (4 channels) instead of RGB (3 channels)
            depth_only: If True, return depth only (1 channel) - for asymmetric sensor config

        Returns:
            RGB image (H, W, 3), RGBD image (H, W, 4), or depth-only (H, W, 1)
        """
        if camera_name is None:
            camera_name = self.camera_name
        if width is None:
            width = self.observation_width
        if height is None:
            height = self.observation_height

        camera_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )

        # Create or reuse cached renderer for this camera/size combination
        renderer_key = (camera_name, width, height)
        if renderer_key not in self._renderers:
            self._renderers[renderer_key] = mujoco.Renderer(self._model, width=width, height=height)

        renderer = self._renderers[renderer_key]

        # Scene options: hide debug visuals from camera renders (ee_site, hand_contact
        # sites and fist geom are only for distance computation / debugging)
        if not hasattr(self, '_scene_option'):
            self._scene_option = mujoco.MjvOption()
            for i in range(6):
                self._scene_option.sitegroup[i] = 0
            # Hide geom group 4 (debug-only geoms like the fist sphere)
            self._scene_option.geomgroup[4] = 0

        if depth_only:
            # Render depth-only image (1 channel)
            renderer.enable_depth_rendering = True
            renderer.update_scene(self._data, camera=camera_id, scene_option=self._scene_option)
            depth = renderer.render()
            renderer.enable_depth_rendering = False

            # Normalize depth to 0-255 range
            # Clip to reasonable range (0.1m to 2m) and normalize
            depth = np.clip(depth, 0.1, 2.0)
            depth_normalized = ((depth - 0.1) / 1.9 * 255).astype(np.uint8)

            # Return as (H, W, 1) for consistency with other image formats
            return depth_normalized[..., np.newaxis]

        # Standard RGB rendering
        renderer.update_scene(self._data, camera=camera_id, scene_option=self._scene_option)
        image = renderer.render()

        if include_depth:
            # Render depth image
            renderer.enable_depth_rendering = True
            renderer.update_scene(self._data, camera=camera_id, scene_option=self._scene_option)
            depth = renderer.render()
            renderer.enable_depth_rendering = False

            # Normalize depth to 0-255 range for consistency with RGB
            # Clip to reasonable range (0.1m to 2m) and normalize
            depth = np.clip(depth, 0.1, 2.0)
            depth_normalized = ((depth - 0.1) / 1.9 * 255).astype(np.uint8)

            # Stack RGB + Depth as 4-channel image
            image = np.concatenate([image, depth_normalized[..., np.newaxis]], axis=-1)

        return image

    def _get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        positions = np.zeros(ACTION_DIM)
        for i, joint_id in enumerate(self._joint_ids):
            qpos_addr = self._model.jnt_qposadr[joint_id]
            positions[i] = self._data.qpos[qpos_addr]
        return positions

    def _get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = np.zeros(ACTION_DIM)
        for i, dof_id in enumerate(self._joint_dof_ids):
            velocities[i] = self._data.qvel[dof_id]
        return velocities

    def _get_hand_velocity(self) -> np.ndarray:
        """Get hand target velocity (3D).

        Returns non-zero values only for sinusoidal and random_walk motion types,
        where the velocity is meaningful for prediction. Returns zeros otherwise.
        """
        if self.hand_motion_type == "sinusoidal":
            # Analytical derivative of sinusoidal motion in m/step units.
            # Position uses virtual time t = step * dt_step where dt_step = timestep * 10.
            # d(pos)/d(step) = d(pos)/dt * dt_step
            dt_step = self._model.opt.timestep * 10
            t = self._step_count * dt_step
            velocity = (
                self._hand_motion_amplitude
                * 2 * np.pi * self._hand_motion_freq
                * np.cos(2 * np.pi * self._hand_motion_freq * t + self._hand_motion_phase)
                * dt_step
            )
            return velocity
        elif self.hand_motion_type == "random_walk":
            # Velocity is the direction of movement at constant speed
            if hasattr(self, '_hand_walk_pos') and hasattr(self, '_hand_walk_target'):
                diff = self._hand_walk_target - self._hand_walk_pos
                dist = np.linalg.norm(diff)
                if dist > 0.01:
                    move_speed = 0.002 * self.motion_freq_scale
                    return diff / dist * move_speed
            return np.zeros(3)
        elif self.hand_motion_type == "move_and_hold":
            move_steps = 80
            if self._step_count < move_steps and hasattr(self, '_hold_target'):
                # Derivative of smoothstep: 6*a*(1-a) / move_steps * displacement
                alpha = self._step_count / move_steps
                d_alpha = 6 * alpha * (1 - alpha) / move_steps
                return d_alpha * (self._hold_target - self._hand_base_pos)
            return np.zeros(3)
        else:
            return np.zeros(3)

    def _get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        return self._data.site_xpos[self._ee_site_id].copy()

    def _get_hand_position(self) -> np.ndarray:
        """Get hand target position."""
        return self._data.site_xpos[self._hand_site_id].copy()

    def _compute_distance(self) -> float:
        """Compute distance between end effector and hand target."""
        ee_pos = self._get_ee_position()
        hand_pos = self._get_hand_position()
        return np.linalg.norm(ee_pos - hand_pos)

    def _get_gripper_forward(self) -> np.ndarray:
        """Get gripper forward direction (toward fist tip) in world frame.

        The fist is at local (~ -0.008, 0, -0.098), so the gripper points
        along its negative local Z axis.
        """
        # xmat is a 3x3 rotation matrix stored row-major as (9,)
        xmat = self._data.xmat[self._gripper_body_id].reshape(3, 3)
        # Negative Z column = gripper forward direction
        return -xmat[:, 2]

    def _get_palm_normal(self) -> np.ndarray:
        """Get palm face normal in world frame (points toward robot).

        The hand body is rotated 90° around Z, so local +Y maps to
        the palm front face direction.
        """
        xmat = self._data.xmat[self._hand_body_id].reshape(3, 3)
        return xmat[:, 1]

    def _compute_facing_score(self) -> float:
        """Compute how well the gripper faces the palm.

        Returns dot product of gripper forward and direction-to-palm,
        ranging from -1 (facing away) to +1 (facing directly at palm).
        """
        ee_pos = self._get_ee_position()
        hand_pos = self._get_hand_position()
        to_palm = hand_pos - ee_pos
        dist = np.linalg.norm(to_palm)
        if dist < 1e-6:
            return 1.0
        to_palm_unit = to_palm / dist
        gripper_fwd = self._get_gripper_forward()
        return float(np.dot(gripper_fwd, to_palm_unit))

    def _check_contact(self) -> bool:
        """Check if there's contact between fist and palm target zone."""
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self._fist_geom_id and g2 == self._palm_target_geom_id) or \
               (g1 == self._palm_target_geom_id and g2 == self._fist_geom_id):
                return True
        return False

    def _get_contact_force(self) -> float:
        """Get total contact force between gripper and hand geoms (in Newtons)."""
        total_force = 0.0
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            # Check if this contact is between any gripper geom and any hand geom
            is_gripper_hand = (g1 in self._gripper_geom_ids and g2 in self._hand_geom_ids) or \
                              (g1 in self._hand_geom_ids and g2 in self._gripper_geom_ids)
            if is_gripper_hand:
                # Extract 6D contact force (3 normal + 3 friction)
                force = np.zeros(6)
                mujoco.mj_contactForce(self._model, self._data, i, force)
                total_force += np.linalg.norm(force)
        return total_force

    def _update_hand_position(self):
        """Update hand position based on motion type."""
        if self.hand_motion_type == "static":
            # Hand stays at base position
            target_pos = self._hand_base_pos.copy()

        elif self.hand_motion_type == "random":
            # Random position within workspace
            if self._step_count % 50 == 0:  # Change position every 50 steps
                offset = self._rng.uniform(-0.15, 0.15, size=3)
                offset[2] = np.clip(offset[2], -0.1, 0.1)  # Limit vertical motion
                target_pos = self._hand_base_pos + offset
            else:
                return  # Keep current position

        elif self.hand_motion_type == "sinusoidal":
            # Smooth sinusoidal motion
            t = self._step_count * self._model.opt.timestep * 10
            offset = self._hand_motion_amplitude * np.sin(
                2 * np.pi * self._hand_motion_freq * t + self._hand_motion_phase
            )
            target_pos = self._hand_base_pos + offset

        elif self.hand_motion_type == "random_walk":
            # Target-based random walk: pick a random target, move toward it smoothly,
            # then pick a new target on arrival. Large amplitude, controlled speed.
            bounds_lo = np.array([0.20, -0.20, 0.15])
            bounds_hi = np.array([0.40, 0.20, 0.30])
            if not hasattr(self, '_hand_walk_pos'):
                self._hand_walk_pos = self._hand_base_pos.copy()
                self._hand_walk_target = self._rng.uniform(bounds_lo, bounds_hi)
            # Move toward target at constant speed
            move_speed = 0.002 * self.motion_freq_scale
            diff = self._hand_walk_target - self._hand_walk_pos
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                # Reached target, pick a new one
                self._hand_walk_target = self._rng.uniform(bounds_lo, bounds_hi)
            else:
                self._hand_walk_pos += diff / dist * move_speed
            target_pos = self._hand_walk_pos.copy()

        elif self.hand_motion_type == "move_and_hold":
            # Hand starts at base, moves to a random target over ~30 steps, then holds.
            move_steps = 80
            if not hasattr(self, '_hold_target'):
                # Pick random target within reachable bounds
                bounds_lo = np.array([0.25, -0.25, 0.15])
                bounds_hi = np.array([0.45,  0.25, 0.30])
                self._hold_target = self._rng.uniform(bounds_lo, bounds_hi)
            if self._step_count < move_steps:
                # Smooth interpolation from base to target
                alpha = self._step_count / move_steps
                # Smoothstep for natural motion
                alpha = alpha * alpha * (3 - 2 * alpha)
                target_pos = self._hand_base_pos + alpha * (self._hold_target - self._hand_base_pos)
            else:
                target_pos = self._hold_target.copy()

        elif self.hand_motion_type == "tracking":
            # Move toward/away from robot (for difficulty scaling)
            ee_pos = self._get_ee_position()
            direction = ee_pos - self._hand_base_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            # Oscillate toward and away
            t = self._step_count * self._model.opt.timestep * 5
            offset_magnitude = 0.1 * np.sin(t)
            target_pos = self._hand_base_pos + offset_magnitude * direction

        else:
            raise ValueError(f"Unknown hand_motion_type: {self.hand_motion_type}")

        # Update hand position in qpos (free joint: x, y, z, qw, qx, qy, qz)
        self._data.qpos[self._hand_qpos_addr:self._hand_qpos_addr + 3] = target_pos
        # Orientation: +90° around Z so palm faces -X (toward robot)
        # Quaternion for 90° Z rotation: [cos(45°), 0, 0, sin(45°)]
        self._data.qpos[self._hand_qpos_addr + 3:self._hand_qpos_addr + 7] = [0.7071, 0, 0, 0.7071]
        # Zero out hand velocity so gravity doesn't drag it between substeps
        hand_qvel_addr = self._model.jnt_dofadr[self._hand_joint_id]
        self._data.qvel[hand_qvel_addr:hand_qvel_addr + 6] = 0.0

    def _apply_domain_randomization(self):
        """Apply domain randomization for sim-to-real transfer."""
        if not self.domain_randomization:
            return

        # === Hand size randomization (0.8x to 1.2x) ===
        hand_geom_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["palm"]
        ]
        hand_scale = self._rng.uniform(0.8, 1.2)
        for geom_id in hand_geom_ids:
            if geom_id >= 0:
                # Scale geometry size (MuJoCo geom_size stores half-extents)
                # We need to store original sizes on first call
                if not hasattr(self, '_original_hand_sizes'):
                    self._original_hand_sizes = {}
                if geom_id not in self._original_hand_sizes:
                    self._original_hand_sizes[geom_id] = self._model.geom_size[geom_id].copy()
                self._model.geom_size[geom_id] = self._original_hand_sizes[geom_id] * hand_scale

        # === Hand base position randomization ===
        self._hand_base_pos = self._rng.uniform(
            [0.25, -0.25, 0.15],
            [0.45, 0.25, 0.30],
        )

        # === Camera position/angle randomization ===
        # Birdseye camera randomization
        birdseye_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "birdseye")
        if birdseye_cam_id >= 0:
            # Store original pose on first call
            if not hasattr(self, '_original_birdseye_pos'):
                self._original_birdseye_pos = self._model.cam_pos[birdseye_cam_id].copy()
            # Randomize position (±5cm in x,y, ±10cm in z)
            pos_noise = self._rng.uniform([-0.05, -0.05, -0.10], [0.05, 0.05, 0.10])
            self._model.cam_pos[birdseye_cam_id] = self._original_birdseye_pos + pos_noise

        # === Lighting randomization ===
        # Randomize light positions and intensities
        for light_id in range(self._model.nlight):
            # Store original values on first call
            if not hasattr(self, '_original_light_pos'):
                self._original_light_pos = {}
                self._original_light_diffuse = {}
            if light_id not in self._original_light_pos:
                self._original_light_pos[light_id] = self._model.light_pos[light_id].copy()
                self._original_light_diffuse[light_id] = self._model.light_diffuse[light_id].copy()

            # Randomize light position (±20cm)
            pos_noise = self._rng.uniform(-0.2, 0.2, size=3)
            self._model.light_pos[light_id] = self._original_light_pos[light_id] + pos_noise

            # Randomize light intensity (0.5x to 1.5x)
            intensity_scale = self._rng.uniform(0.5, 1.5)
            self._model.light_diffuse[light_id] = np.clip(
                self._original_light_diffuse[light_id] * intensity_scale, 0, 1
            )

        # === Randomize motion parameters ===
        self._hand_motion_phase = self._rng.uniform(0, 2 * np.pi, size=3)
        self._hand_motion_freq = np.array([0.5, 0.3, 0.2]) * self.motion_freq_scale * self._rng.uniform(
            0.8, 1.2, size=3
        )

    def _get_observation(self) -> dict[str, Any]:
        """Get current observation with configurable cameras and depth."""
        if self.obs_type == "state":
            # State-only: joint_pos(5) + joint_vel(5) + ee_pos(3) + hand_pos(3) + hand_vel(3) = 19
            joint_pos = self._get_joint_positions()
            joint_vel = self._get_joint_velocities()
            ee_pos = self._get_ee_position()
            hand_pos = self._get_hand_position()
            hand_vel = self._get_hand_velocity()
            return {"state": np.concatenate([joint_pos, joint_vel, ee_pos, hand_pos, hand_vel])}

        if self.bev_depth_wrist_rgb:
            # Asymmetric mode: BEV depth-only, Wrist RGB
            birdseye_image = self._get_camera_image(
                camera_name="birdseye",
                include_depth=False,  # We'll get depth separately
                depth_only=True,  # Get depth-only (1 channel)
            )
            obs = {
                "pixels": {
                    "birdseye": birdseye_image,
                },
            }
            # Wrist camera RGB (no depth)
            wrist_image = self._get_camera_image(
                camera_name="wrist",
                include_depth=False,
                depth_only=False,
            )
            obs["pixels"]["wrist"] = wrist_image
        else:
            # Standard mode: same channels for both cameras
            birdseye_image = self._get_camera_image(
                camera_name="birdseye",
                include_depth=self.use_depth,
            )
            obs = {
                "pixels": {
                    "birdseye": birdseye_image,
                },
            }
            # Add wrist camera if not single_camera mode
            if not self.single_camera:
                wrist_image = self._get_camera_image(
                    camera_name="wrist",
                    include_depth=self.use_depth,
                )
                obs["pixels"]["wrist"] = wrist_image

        if self.obs_type == "pixels_agent_pos":
            obs["agent_pos"] = np.concatenate([
                self._get_joint_positions(),   # 5 joint angles
                self._get_joint_velocities(),  # 5 joint velocities
                self._get_ee_position(),       # 3 EE position (FK)
            ])

        return obs

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self._model, self._data)

        # Apply domain randomization
        self._apply_domain_randomization()

        # Always randomize motion parameters (even without domain randomization)
        self._hand_motion_phase = self._rng.uniform(0, 2 * np.pi, size=3)
        self._hand_motion_freq = np.array([0.5, 0.3, 0.2]) * self.motion_freq_scale * self._rng.uniform(
            0.8, 1.2, size=3
        )

        # Randomize hand base position (independent of domain randomization)
        if self.randomize_hand_position and not self.domain_randomization:
            self._hand_base_pos = self._rng.uniform(
                [0.25, -0.25, 0.15],
                [0.45, 0.25, 0.30],
            )

        # Set robot starting position
        base_qpos = START_POSES[self.start_pose]
        if self.domain_randomization:
            # Add noise around the start pose (±0.3 rad)
            initial_qpos = base_qpos + self._rng.uniform(-0.3, 0.3, size=5)
        else:
            initial_qpos = base_qpos.copy()

        for i, joint_id in enumerate(self._joint_ids):
            qpos_addr = self._model.jnt_qposadr[joint_id]
            self._data.qpos[qpos_addr] = initial_qpos[i]

        # Initialize hand position
        self._update_hand_position()

        # Step simulation to settle
        for _ in range(10):
            mujoco.mj_step(self._model, self._data)

        # Reset step counter
        self._step_count = 0
        self._episode_count += 1
        self._episode_success = False  # Track if contact happened at any point
        self._first_contact_step = None  # Step when first contact occurred
        self._prev_action = np.zeros(ACTION_DIM)  # For action rate penalty

        # Reset force disturbance state
        if self.force_disturbances:
            self._next_kick_step = self._rng.integers(*self.force_disturbance_interval)
            self._kick_force = np.zeros(3)  # 3D Cartesian force
            self._kick_body_idx = 0  # index into _arm_body_ids
            self._kick_steps_remaining = 0
            self._kicks_fired = 0

        # Reset motion state
        if hasattr(self, '_hand_walk_pos'):
            del self._hand_walk_pos
        if hasattr(self, '_hand_walk_target'):
            del self._hand_walk_target
        if hasattr(self, '_hold_target'):
            del self._hold_target

        self._prev_distance = self._compute_distance()

        observation = self._get_observation()
        info = {
            "task_id": self.task_id,
            "is_success": False,
            "distance": self._prev_distance,
        }

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Joint position targets (5 values, normalized to [-1, 1])

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Action must be 1D with shape ({ACTION_DIM},), got shape {action.shape}"
            )

        # Clip action to valid range
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)

        # Convert normalized action to joint positions
        # Scale from [-1, 1] to joint limits
        for i, actuator_id in enumerate(self._actuator_ids):
            ctrl_range = self._model.actuator_ctrlrange[actuator_id]
            # Map from [-1, 1] to control range
            self._data.ctrl[actuator_id] = (
                ctrl_range[0] + (action[i] + 1) * 0.5 * (ctrl_range[1] - ctrl_range[0])
            )

        # Hold gripper closed (fist for high-five)
        self._data.ctrl[self._gripper_actuator_id] = 0.0

        # Update hand position
        self._update_hand_position()

        # Handle force disturbance kicks (Cartesian forces on arm bodies)
        if self.force_disturbances:
            if self._kick_steps_remaining > 0:
                self._kick_steps_remaining -= 1
                if self._kick_steps_remaining == 0:
                    # Kick ended, hide indicator
                    for gid in self._kick_indicator_ids.values():
                        self._model.geom_rgba[gid] = [0, 0, 0, 0]
                    self._kick_active_visual = False
            elif self._step_count >= self._next_kick_step and self._kicks_fired < self.force_kicks_per_episode:
                # Start a new kick: random Cartesian force on a random arm body
                self._kick_body_idx = self._rng.integers(len(self._arm_body_ids))
                # Random 3D force direction with random magnitude
                direction = self._rng.standard_normal(3)
                direction /= np.linalg.norm(direction) + 1e-8
                magnitude = self._rng.uniform(0.3, 1.0) * self.force_disturbance_max
                self._kick_force = direction * magnitude
                self._kick_steps_remaining = self.force_disturbance_duration
                self._kicks_fired += 1
                # Visual indicator: show red cylinder on kicked body, oriented along force
                # Compute quaternion that rotates Z-axis (cylinder long axis) to force direction
                z = np.array([0.0, 0.0, 1.0])
                d = direction  # unit vector
                cross = np.cross(z, d)
                dot = np.dot(z, d)
                if np.linalg.norm(cross) < 1e-6:
                    # Parallel or anti-parallel
                    quat = np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0 else np.array([0.0, 1.0, 0.0, 0.0])
                else:
                    # quat = [w, x, y, z] where w = 1 + dot, (x,y,z) = cross
                    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
                    quat /= np.linalg.norm(quat)
                body_name = self._arm_body_names[self._kick_body_idx]
                for name, gid in self._kick_indicator_ids.items():
                    if name == body_name:
                        intensity = min(magnitude / self.force_disturbance_max, 1.0)
                        self._model.geom_rgba[gid] = [1.0, 0.0, 0.0, intensity]
                        self._model.geom_quat[gid] = quat
                    else:
                        self._model.geom_rgba[gid] = [0, 0, 0, 0]
                self._kick_active_visual = True
                print(f"  KICK #{self._kicks_fired}: body={body_name} mag={magnitude:.1f}N dir={direction}")
                # Schedule next kick
                self._next_kick_step = self._step_count + self._rng.integers(
                    *self.force_disturbance_interval
                )

        # Step simulation (multiple physics steps per control step)
        n_substeps = 4
        hand_qvel_addr = self._model.jnt_dofadr[self._hand_joint_id]
        for _ in range(n_substeps):
            # Gravity compensation: apply bias forces (gravity + Coriolis) for arm joints
            # so the position controller only needs to handle the task
            for dof_id in self._joint_dof_ids:
                self._data.qfrc_applied[dof_id] = self._data.qfrc_bias[dof_id]
            # Apply Cartesian force disturbance to arm body via xfrc_applied
            if self.force_disturbances and self._kick_steps_remaining > 0:
                body_id = self._arm_body_ids[self._kick_body_idx]
                self._data.xfrc_applied[body_id, :3] = self._kick_force
            # Pin hand in place: zero velocity each substep so contact forces
            # don't push it away between substeps
            self._data.qvel[hand_qvel_addr:hand_qvel_addr + 6] = 0.0
            mujoco.mj_step(self._model, self._data)
        # Clear xfrc_applied after substeps so it doesn't persist
        if self.force_disturbances:
            self._data.xfrc_applied[:] = 0.0

        self._step_count += 1

        # Compute reward: 10 * exp(-5 * d)
        # At 2cm: 9.05, at 10cm: 6.07, at 20cm: 3.68, at 40cm: 1.35
        distance = self._compute_distance()
        reward = REWARD_SCALE * np.exp(-REWARD_SIGMA * distance)

        # Closing velocity reward: reward for reducing distance to target
        if self.closing_reward:
            closing = max(0.0, self._prev_distance - distance)
            reward += 5.0 * closing
        self._prev_distance = distance

        # Facing reward: encourage gripper to point toward palm
        if self.facing_reward:
            facing_score = self._compute_facing_score()
            reward += FACING_REWARD_SCALE * max(0.0, facing_score)

        # Action rate penalty: penalize jerky movements
        action_diff = action - self._prev_action
        reward -= ACTION_RATE_PENALTY * np.sum(action_diff ** 2)
        self._prev_action = action.copy()

        # Check for contact (success) — fist must touch palm target zone (front face)
        is_contact = self._check_contact()
        if is_contact:
            # Decaying contact bonus: reward early contact more than late contact
            reward += CONTACT_BONUS * (CONTACT_BONUS_GAMMA ** self._step_count)
            self._episode_success = True
            if self._first_contact_step is None:
                self._first_contact_step = self._step_count

        # Check termination conditions
        # Terminate early after CONTACT_TERMINATE_STEPS steps post first contact
        if self._first_contact_step is not None:
            steps_since_contact = self._step_count - self._first_contact_step
            terminated = steps_since_contact >= CONTACT_TERMINATE_STEPS
        else:
            terminated = False
        truncated = self._step_count >= self.episode_length

        # Restore floor color before camera capture so kick indicator
        # is only visible in the human viewer, not in training images
        # Hide kick indicators before camera capture (they're group 4 so
        # already hidden from cameras, but be safe)
        kick_rgbas_backup = None
        if self._kick_active_visual:
            kick_rgbas_backup = {gid: self._model.geom_rgba[gid].copy() for gid in self._kick_indicator_ids.values()}
            for gid in self._kick_indicator_ids.values():
                self._model.geom_rgba[gid] = [0, 0, 0, 0]

        # Get observation
        observation = self._get_observation()

        # Restore kick indicators for human viewer
        if kick_rgbas_backup is not None:
            for gid, rgba in kick_rgbas_backup.items():
                self._model.geom_rgba[gid] = rgba

        # Build info dict
        info = {
            "task_id": self.task_id,
            "is_success": self._episode_success,
            "distance": distance,
            "step": self._step_count,
        }

        if terminated or truncated:
            info["final_info"] = {
                "task_id": self.task_id,
                "is_success": self._episode_success,
                "final_distance": distance,
                "episode_length": self._step_count,
            }

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._get_camera_image(
                width=self.visualization_width,
                height=self.visualization_height,
            )
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco_viewer.launch_passive(
                    self._model, self._data
                )
                # Enable geom group 4 in viewer so kick indicators are visible
                self._viewer.opt.geomgroup[4] = 1
            self._viewer.sync()
            return None
        return None

    def close(self):
        """Clean up environment resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        # Close cached renderers
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


def _make_env_fns(
    task_id: int,
    n_envs: int,
    episode_length: int,
    gym_kwargs: dict[str, Any],
) -> list[Callable[[], HighFiveEnv]]:
    """Build n_envs factory callables for a single task."""

    def _make_env(episode_index: int, **kwargs) -> HighFiveEnv:
        return HighFiveEnv(
            task_id=task_id,
            episode_length=episode_length,
            seed=episode_index,
            **kwargs,
        )

    fns: list[Callable[[], HighFiveEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


def create_highfive_envs(
    task: str = "highfive-v0",
    n_envs: int = 1,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
) -> dict[str, dict[int, Any]]:
    """Create vectorized High-Five environments.

    Args:
        task: Task name (currently only "highfive-v0" supported)
        n_envs: Number of parallel environments
        gym_kwargs: Additional kwargs passed to HighFiveEnv
        env_cls: Vectorized environment class (e.g., SyncVectorEnv)
        episode_length: Maximum episode length

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            Mapping from suite name to task_id to vectorized environment
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError(
            "env_cls must be a callable that wraps a list of environment factory callables."
        )
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})

    print(
        f"Creating High-Five envs | task={task} | n_envs={n_envs} | "
        f"episode_length={episode_length}"
    )

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    # Currently only one task variant
    task_id = 0
    fns = _make_env_fns(
        task_id=task_id,
        n_envs=n_envs,
        episode_length=episode_length,
        gym_kwargs=gym_kwargs,
    )
    out["highfive"][task_id] = env_cls(fns)
    print(f"Built vec env | suite=highfive | task_id={task_id} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}
