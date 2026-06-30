#!/usr/bin/env python
"""Quick MuJoCo viewer to test hand and robot positioning.

Usage:
    mjpython -m lerobot.scripts.highfive.test_position
    mjpython -m lerobot.scripts.highfive.test_position --hand_pos 0.05 0.0 0.50
    mjpython -m lerobot.scripts.highfive.test_position --joint_pos 1.0 0 0 0 0 0
    mjpython -m lerobot.scripts.highfive.test_position --random_walk
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


def _run_env(args):
    """Run the full HighFiveEnv with random actions in the MuJoCo viewer."""
    from lerobot.envs.highfive.highfive_env import HighFiveEnv

    env = HighFiveEnv(
        obs_type="state",
        hand_motion_type="random_walk",
        motion_freq_scale=args.motion_freq_scale,
        domain_randomization=True,
        render_mode="human",
        start_pose="folded",
    )

    print("Running env with random actions. Close viewer to stop.")
    print(f"Hand motion: random_walk, freq_scale={args.motion_freq_scale}")
    print(f"Domain randomization: ON, start_pose: folded")

    ep = 0
    while True:
        obs, info = env.reset()
        ep += 1
        print(f"\n--- Episode {ep} | hand_base={env._hand_base_pos.round(3)} ---")
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(1 / 30)
            step += 1
            done = terminated or truncated
        result = "SUCCESS" if info.get("is_success") else "TIMEOUT"
        print(f"  {result} after {step} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_pos", type=float, nargs=3, default=[0.45, 0.0, 0.25],
                        help="Hand body position: x y z (world frame, robot base at origin)")
    parser.add_argument("--joint_pos", type=float, nargs=6, default=[0, -1.5, 1.5, 1.0, 0, 0],
                        help="Joint positions (rad): shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper. "
                             "Ranges: pan ±1.92, lift ±1.75, elbow ±1.69, wflex ±1.66, wroll ±2.74, gripper [-0.17,1.75]")
    parser.add_argument("--random_walk", action="store_true",
                        help="Animate the hand with random walk motion")
    parser.add_argument("--motion_freq_scale", type=float, default=0.5,
                        help="Speed of random walk (default 0.3)")
    parser.add_argument("--run_env", action="store_true",
                        help="Run the full HighFiveEnv with random actions in the viewer")
    args = parser.parse_args()

    if args.run_env:
        _run_env(args)
        return

    model = mujoco.MjModel.from_xml_path(
        str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "envs" / "highfive" / "assets" / "scene.xml")
    )
    data = mujoco.MjData(model)

    hand_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
    addr = model.jnt_qposadr[hand_joint_id]

    # Set robot joint positions
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = args.joint_pos[i]

    # Set hand position
    hand_pos = np.array(args.hand_pos, dtype=float)
    data.qpos[addr:addr + 3] = hand_pos
    data.qpos[addr + 3:addr + 7] = [0.7071, 0, 0, 0.7071]  # 90° Z rotation, palm faces robot

    mujoco.mj_forward(model, data)

    # Print EE position
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_pos = data.site_xpos[ee_site_id]

    print(f"Joint pos (rad): {dict(zip(joint_names, args.joint_pos))}")
    print(f"EE pos:          [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]")
    print(f"Hand body pos:   {args.hand_pos}")
    print(f"Robot base pos:  [0, 0, 0]")

    if args.random_walk:
        print(f"Random walk active (freq_scale={args.motion_freq_scale})")
        print(f"Bounds: x[0.35, 0.55], y[-0.40, 0.40], z[0.20, 0.30]")

    viewer = mujoco.viewer.launch_passive(model, data)
    print("Viewer open — close the window or Ctrl+C to exit.")

    rng = np.random.default_rng()
    bounds_lo = np.array([0.35, -0.40, 0.20])
    bounds_hi = np.array([0.55, 0.40, 0.30])
    # Pick a random target, move toward it at constant speed
    walk_target = rng.uniform(bounds_lo, bounds_hi)
    move_speed = 0.006 * args.motion_freq_scale  # meters per frame

    while viewer.is_running():
        if args.random_walk:
            # Move toward target
            diff = walk_target - hand_pos
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                # Reached target, pick a new one
                walk_target = rng.uniform(bounds_lo, bounds_hi)
            else:
                hand_pos += diff / dist * move_speed

            data.qpos[addr:addr + 3] = hand_pos
            data.qpos[addr + 3:addr + 7] = [0.7071, 0, 0, 0.7071]
            mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
