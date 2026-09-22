# Isaaclab_Parkour

Parkour locomotion for the Unitree Go2 in IsaacLab, based on [Extreme-Parkour](https://extreme-parkour.github.io/).

## Policy versions

The main distinction between `trained_v1.1` and `trained_v1.3` is how the policy is trained and how its inputs are provided.

| Version | Policy architecture and training | Perception and direction inputs |
| --- | --- | --- |
| `trained_v1.1` | Teacher–student distillation with oracle points also supplied to the student. | Depth camera tilted 20° downward; the intended direction of travel is provided through oracle points. |
| `trained_v1.3` | Retains the pretrained teacher policy architecture and adapts its inputs, rather than introducing a separate teacher–student architecture. | L1 LiDAR point clouds are converted into an elevation map; input noise is progressively increased during training. |

### trained_v1.1: Distillation with oracle direction inputs

`trained_v1.1` keeps the teacher–student structure and the same distillation method as the original approach. The key change is that **oracle points are provided to the student as inputs as well**. These points specify the robot's intended direction of travel, so the student does not also have to predict which direction the robot should move next.

The student therefore learns locomotion through distillation while receiving the direction information explicitly.

### trained_v1.3: Input adaptation with the teacher policy architecture

**`trained_v1.3` does not use a separate teacher–student policy architecture.** It starts from a teacher policy already trained with ground-truth (GT) scandots and preserves that policy's network architecture while changing the terrain input:

1. Start with the policy trained on GT scandots.
2. Collect point clouds with the L1 LiDAR and construct an elevation map using `elevation_mapping_cupy`.
3. Sample the elevation map on the same 12 × 11 grid used for the teacher's scandots and feed these samples into the existing scan encoder.
4. Continue training with progressively stronger noise on the elevation-map input so the policy can keep moving under increasingly noisy terrain observations.

The depth camera is removed in this version. The focus is on adapting the existing policy to LiDAR-derived terrain observations, rather than training a separate depth-camera student to reproduce a teacher.

The registered task retains the name `Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-v0`. The current training code also reuses the `EMDistillation` harness and GT teacher action targets for supervision. These implementation details describe the optimization path; the adapted policy retains the teacher's architecture instead of using a distinct student architecture.

### trained_v1.3 observation changes

Two additional observation changes affect compatibility with earlier checkpoints:

- **Fixed terrain-type flags:** the flat/non-flat components at proprioceptive observation indices 11 and 12 are fixed to `non-flat=1` and `flat=0`, regardless of terrain. This removes dependence on an oracle terrain-type signal and targets robust locomotion over rough terrain. The convention applies to training, playback, evaluation, and demos.
- **Estimated explicit privileged observations:** the elevation-map policy receives the frozen estimator's predictions in the nine-dimensional `priv_explicit` slot, including base linear velocity, in place of GT values. The estimator comes from teacher training. This aligns the policy's observations with deployment conditions, where GT base velocity is unavailable.

### trained_v1.4: Slip and disturbance robustness

Retrain the GT-scan teacher with PPO and the new reward, then adapt that teacher to elevation-map inputs using the existing EM training path.

- Ground friction uses `multiply`. Robot friction uses 96 material buckets: 32 each in `[0.25, 0.5)`, `[0.5, 1.0)`, and `[1.0, 2.0)`, giving each interval probability 1/3. Static and dynamic friction match; all robot shapes share the sampled coefficient within an environment.
- During training, world-frame x/y velocities are reset every 8 seconds. Body-frame roll/pitch/yaw angular velocity increments are added every 7 seconds. The bounds below apply independently to each component.
- The teacher's slip reward is `-0.04 * sum(indicator(contact_force_norm >= 5 N) * foot_xy_speed)`, where the comparison is a per-foot indicator and speed is measured in world coordinates. Only current contacts count. EM adaptation still optimizes action imitation, not this reward directly.

| Training progress | x/y bound (m/s) | Angular increment bound (rad/s) |
| --- | --- | --- |
| 0–20% | ±0.2 | ±0.1 |
| 20–40% | ±0.4 | ±0.2 |
| 40–60% | ±0.6 | ±0.3 |
| 60–80% | ±0.8 | ±0.4 |
| 80–100% | ±1.0 | ±0.5 |

The schedule uses the additional iteration budget of the new training run, independent of the pretrained checkpoint's iteration number. Checkpoints save schedule progress: resuming a v1.4 run preserves its original schedule horizon and stays at full strength after that horizon. Starting EM adaptation from a teacher starts a fresh schedule. TensorBoard records `Disturbance/scale`.

EVAL uses full strength and retains its existing 6-second linear interval, with the new 7-second angular interval. PLAY disables both disturbances. The `-0.04` slip weight is an initial value; its measured contribution still needs rollout calibration.

## Installation

Run the following from your IsaacLab directory in an environment configured for IsaacLab:

```bash
git clone https://github.com/JINWOOSEO21/Isaaclab_Parkour.git
cd Isaaclab_Parkour
pip3 install -e .
pip3 install -e ./parkour_tasks
```

Run the commands below from the `Isaaclab_Parkour` repository root.

## Training

### Teacher policy

```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
```

### Depth-camera student policy

This task uses the teacher–student distillation workflow.

```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```

### Elevation-map policy (trained_v1.3)

This task adapts the pretrained teacher architecture to elevation-map inputs. Configure the source teacher checkpoint before training.

```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-v0 --seed 1 --headless
```

## Playback and evaluation

### Teacher policy

[Download the pretrained teacher policy](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing).

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0
```

[Teacher policy demo](https://github.com/user-attachments/assets/ff1f58db-2439-449c-b596-5a047c526f1f)

### Depth-camera student policy

[Download the pretrained student policy](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing).

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0
```

https://github.com/user-attachments/assets/82a5cecb-ffbf-4a46-8504-79188a147c40

### Elevation-map policy (trained_v1.3)

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0 --num_envs 16
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Eval-v0
```

### Playback recording and input panels

In `play.py`, `--video` records one viewport video under `videos/play/`.
The viewer configuration starts at environment 0; there is currently no CLI
argument for choosing a different viewport environment. In the GUI, Numpad 7/9
switch the tracked environment. Headless recording has no keyboard camera control.

`--multicam` records one video per environment under `videos/multicam/` (or
`--out_dir`). Passing both `--video --multicam` saves both sets of videos.

Add `--panels` to `--multicam` to display the policy's terrain input automatically:

| Policy input | Panels beside the RGB video |
| --- | --- |
| Depth camera | Depth input used by the depth encoder |
| Ground-truth scandots | GT scandots |
| LiDAR-derived elevation map | GT scandots and Estimated scandots, side by side |

The elevation-map comparison uses the scan samples actually passed to the policy.
Depth panels retain the last encoder input between encoder updates. `--panels`
does not take a depth/scandots argument and requires `--multicam`; it does not add
panels to the viewport video or attach an extra depth camera to a GT-scan policy.

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0 --headless --multicam --panels --num_envs 2 --video_length 1000
```

## IsaacLab demos

```bash
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0
```

[Go2 demo in IsaacLab](https://github.com/user-attachments/assets/4fb1ba4b-1780-49b0-a739-bff0b95d9b66)

### Viewport controls

The `ParkourViewportCameraController` supports the following controls:

| Key | Action |
| --- | --- |
| `1` / `2` | Select an environment to view. |
| `8` | Move the camera forward. |
| `4` | Move the camera left. |
| `6` | Move the camera right. |
| `5` | Move the camera backward. |
| `0` | Enable the free camera and mouse control. |
| `1` | Return to the default camera mode. |

## Module tests

Module test scripts are available in `parkour_test/`.

## Sim-to-sim and sim-to-real deployment

See [go2_parkour_deploy](https://github.com/CAI23sbP/go2_parkour_deploy) for the deployment project.

- [x] Teacher training code.
- [x] Distillation training code.
- [x] IsaacLab policy demos, based on the [IsaacLab showroom example](https://isaac-sim.github.io/IsaacLab/main/source/overview/showroom.html).
- [x] Sim-to-sim deployment from IsaacLab to MuJoCo.
- [ ] Sim-to-real deployment on the physical robot.

## Project video

https://github.com/user-attachments/assets/aa9f7ece-83c1-404f-be50-6ae6a3ba3530

## Citation

If you use this code for your research, you **must** cite the following papers:

```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/Isaaclab_Parkour
```

## Contact

[sbp0783@hanyang.ac.kr](mailto:sbp0783@hanyang.ac.kr)
