# Wall-Drawing Mobile Manipulator

A low-cost **mobile manipulator that navigates up to a chalk wall and draws on it** — combining
3D perception, SLAM/navigation, and closed-loop visual control on a sub-$1k robot.

Built on the [LeKiwi](https://github.com/huggingface/lerobot) platform (an [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm on a holonomic mobile base), using an **iPhone 15 Pro as the depth/LiDAR sensor** and a
**MacBook as the compute brain**. LeRobot is used purely as a hardware-abstraction layer for the
motors; all perception, navigation, and control on top are custom.

> **Status:** 🚧 Work in progress — hardware bring-up and calibration. Milestones tracked below.

<!-- TODO: hero demo GIF goes here once the robot is drawing -->
<!-- ![demo](docs/media/demo.gif) -->

---

## Motivation

I'm a 3D computer-vision / reinforcement-learning researcher moving into **robotics**, and this
project is a hands-on way to close the gap between simulation and real hardware. The task —
a mobile base that positions itself and an arm that draws on a vertical surface — is inspired by
large-format robotic wall-drawing systems. It's deliberately chosen because a
*cheap, imprecise* robot makes the interesting problems unavoidable: you can't brute-force
accuracy, so you have to **perceive, plan, and correct**.

## What makes it interesting

- **3D perception:** fit the wall plane from LiDAR, segment the drawable chalk region from the
  surrounding wall, and fuse depth with RGB.
- **Navigation:** start away from the wall, *find* it, and drive to a good drawing pose — then
  **reposition** to tile a drawing larger than the arm's reach.
- **Closed-loop drawing:** the arm is not precise, so a **visual feedback loop** watches the pen
  and corrects its stroke in real time, backed by a **compliant pen mount** that absorbs contact
  force.
- **Learned control:** train a visuomotor policy and deploy it on the real robot — closing the
  deployment / sim-to-real gap that most demos skip.

---

## Hardware

| Component | Role |
|---|---|
| **LeKiwi** — SO-101 arm (6 DoF) on a 3-wheel **holonomic** base (9× Feetech STS3215 servos) | Manipulator + omnidirectional mobile base |
| **MacBook (Apple Silicon)** | Compute brain — perception, planning, control, ML |
| **iPhone 15 Pro**, mounted centre-front below the arm | Sensor pod — solid-state **LiDAR** (RGB-D) + **ARKit** 6-DoF pose |
| **Wrist camera** (wired) | Low-latency close-up of the pen tip for visual correction |
| Compliant / suspended **pen mount** (DIY) | Absorbs contact force so the arm/pen can't be damaged |
| 12 V battery + 3 m USB-C tether | Motor power (battery) + data (USB-C to Mac) |

**On the sensing:** the iPhone 15 Pro's LiDAR is a genuine solid-state direct-time-of-flight
sensor. It's forward-facing and low-resolution (256×192 depth, ~0.25–5 m) rather than a 360°
scanning unit — well suited to near-field wall perception. Frames (RGB + depth + intrinsics +
pose) are streamed to the Mac via [Record3D](https://record3d.app/) and processed with the
`record3d` Python package.

**On the tether:** this is a bench/portfolio robot working in a bounded area near one wall, so a
USB-C tether (data) plus onboard battery (power) is sufficient. The design leaves room to go fully
untethered later by adding a Raspberry Pi or Nvidia Jetson Orin Nano host (LeKiwi's Wi-Fi client/host mode).

---

## System architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 MacBook (brain)               │
                 │                                               │
   iPhone 15 Pro │   Perception ─ wall-plane fit, chalk-region   │
   (LiDAR + pose)│                segmentation, RGB-D fusion     │
        ──USB──▶ │        │                                      │
      (Record3D) │        ▼                                      │
                 │   Navigation ─ localization, find-wall,       │
                 │                drive-to-pose, base tiling      │
                 │        │                                      │
   Wrist camera  │        ▼                                      │
      ──USB────▶ │   Control ─ feedforward stroke plan           │
                 │        + visual correction loop (wrist cam)    │
                 │        │                                      │
                 │        ▼                                      │
                 │   LeRobot HAL ── serial ──▶ [STS3215 × 9]      │
                 └───────────────────────────────USB-C───────────┘
                                                       │
                                              12 V battery → motors
```

### Control: three complementary layers

Drawing accuracy comes from combining, not choosing between, these:

1. **Feedforward geometry (global, iPhone LiDAR).** Fit the wall plane once; plan the full pen
   trajectory in the wall frame.
2. **Visual correction (local, wrist camera, few-Hz loop).** Track the pen tip and the *actual*
   drawn line vs. the target; correct in-plane (x/y) drift as the stroke is drawn. The wired wrist
   camera keeps this loop low-latency.
3. **Passive compliance (mechanical).** The suspended pen mount handles the wall-normal (z) axis —
   contact force — so the control loops only worry about in-plane position.

The **dominant error source is extrinsic calibration** (iPhone ↔ base ↔ pen-tip transforms), so
hand-eye calibration gets first-class attention; the visual loop then forgives residual error.

---

## Roadmap

- [x] Hardware assembly, motor ID setup, LeKiwi calibration
- [x] **M1** — Bring-up: drive the arm from the Mac via LeRobot (tethered teleop)
- [x] **Sensing dataflow** — synchronized robot state + wrist RGB + iPhone LiDAR RGB-D/pose in one loop
- [ ] **M2** — camera/arm **extrinsic calibration** + wall-plane fit (streaming ✅)
- [ ] **M3** — Chalk-region segmentation (distinguish drawable board from wall)
- [ ] **M4** — Static-base drawing: draw a small figure with feedforward + compliance
- [ ] **M5** — Visual correction loop: detect drift mid-stroke and correct
- [ ] **M6** — Navigation: find the wall from across the room and drive to a drawing pose
- [ ] **M7** — Mobile manipulation: tile a larger drawing by repositioning the base
- [ ] **M8** *(stretch)* — VLM + voice: speak a prompt, robot plans and draws it

## Tech stack

Python · PyTorch (Apple MPS) · [LeRobot](https://github.com/huggingface/lerobot) (motor HAL) ·
OpenCV · Open3D / point-cloud processing · ARKit + Record3D (iPhone sensing) · MuJoCo (sim / policy training)

## Acknowledgements

Built on Hugging Face's [LeRobot](https://github.com/huggingface/lerobot) and the open-source
[SO-ARM / LeKiwi](https://github.com/TheRobotStudio/SO-ARM100) hardware designs. iPhone LiDAR
streaming uses [Record3D](https://record3d.app/) by Marek Simoník.
