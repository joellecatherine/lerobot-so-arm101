#!/usr/bin/env python
"""Verify random_walk bounds are reachable by the arm."""

import numpy as np
import mujoco
from lerobot.envs.highfive.highfive_env import HighFiveEnv


def test_hand_position(env, hand_body_pos, label):
    """Set hand to a specific body position and random-search for closest EE."""
    best_dist = float("inf")
    best_ee = None

    for _ in range(2000):
        env.reset()
        # Override hand position
        env._data.qpos[env._hand_qpos_addr:env._hand_qpos_addr + 3] = hand_body_pos
        env._data.qpos[env._hand_qpos_addr + 3:env._hand_qpos_addr + 7] = [0.7071, 0, 0, 0.7071]
        mujoco.mj_forward(env._model, env._data)

        action = np.random.uniform(-1, 1, size=5).astype(np.float32)
        for _ in range(50):
            # Keep hand fixed during search
            env._data.qpos[env._hand_qpos_addr:env._hand_qpos_addr + 3] = hand_body_pos
            env._data.qpos[env._hand_qpos_addr + 3:env._hand_qpos_addr + 7] = [0.7071, 0, 0, 0.7071]
            env.step(action)

        ee = env._get_ee_position()
        hand = env._get_hand_position()
        dist = np.linalg.norm(ee - hand)
        if dist < best_dist:
            best_dist = dist
            best_ee = ee.copy()

    status = "OK" if best_dist < 0.05 else "MARGINAL" if best_dist < 0.10 else "UNREACHABLE"
    hand_contact = hand_body_pos + np.array([-0.03, 0, 0.02])
    print(f"  {label:30s} | body={np.round(hand_body_pos, 3)} | contact≈{np.round(hand_contact, 3)} | "
          f"best_dist={best_dist:.4f} | {status}")
    return best_dist


def main():
    env = HighFiveEnv(
        obs_type="state",
        hand_motion_type="static",
        domain_randomization=False,
        single_camera=True,
    )

    print("=== RANDOM WALK BOUNDS REACHABILITY TEST ===")
    print("  Testing if arm can reach hand at each corner of the walk bounds.")
    print("  Body pos bounds: x[-0.20, 0.00], y[-0.10, +0.10], z[0.35, 0.55]")
    print()

    # Test center
    test_hand_position(env, np.array([-0.129, 0.0, 0.45]), "center (default)")

    # Test x extremes
    test_hand_position(env, np.array([-0.20, 0.0, 0.45]), "x_min (-0.20)")
    test_hand_position(env, np.array([0.00, 0.0, 0.45]), "x_max (0.00)")

    # Test y extremes
    test_hand_position(env, np.array([-0.129, -0.10, 0.45]), "y_min (-0.10)")
    test_hand_position(env, np.array([-0.129, 0.10, 0.45]), "y_max (+0.10)")

    # Test z extremes
    test_hand_position(env, np.array([-0.129, 0.0, 0.35]), "z_min (0.35)")
    test_hand_position(env, np.array([-0.129, 0.0, 0.55]), "z_max (0.55)")

    # Test worst-case corners
    test_hand_position(env, np.array([-0.20, -0.10, 0.35]), "corner: x-/y-/z-")
    test_hand_position(env, np.array([0.00, 0.10, 0.55]), "corner: x+/y+/z+")
    test_hand_position(env, np.array([-0.20, 0.10, 0.55]), "corner: x-/y+/z+")
    test_hand_position(env, np.array([0.00, -0.10, 0.35]), "corner: x+/y-/z-")

    env.close()


if __name__ == "__main__":
    main()
