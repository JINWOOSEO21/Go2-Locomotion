# Isaaclab_Parkour

Unitree Go2 parkour locomotion in IsaacLab, with two primary terrain-input paths:
`scandots_input` and `lidar_input`. This fork builds on
[Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour) and
[Extreme-Parkour](https://extreme-parkour.github.io/).

## Input paths

| Input path | Terrain input consumed by the policy | Current training method |
| --- | --- | --- |
| `scandots_input` | 132 ground-truth height samples (12 × 11 scandots) from the simulator | PPO (`PPOWithExtractor`) |
| `lidar_input` | 132 height samples from an elevation map built using L1 LiDAR point clouds and noisy odometry | PPO (`PPOWithExtractor`), optionally initialized from a Scandots PPO checkpoint |

The LiDAR path keeps the scan encoder and actor architecture of the GT-scan
policy. Its data flow is:

```text
L1 LiDAR point clouds + odometry
    → elevation map
    → 12 × 11 height samples
    → existing scan encoder and policy
```

Odometry is used to place the point clouds into the map and query it around the
robot. In simulation, the pose estimate is derived from simulator state with
modeled drift and noise. Raw point clouds and odometry are not concatenated as
new direct inputs to the actor. The map updates every five control steps (10 Hz),
and its latest samples are held between updates.

The environment still provides GT scandots for visualization and diagnostics, but
the LiDAR actor receives the elevation-map samples in that observation slice. The
explicit privileged-state slice, including base linear velocity, is filled by the
same state estimator used by the Scandots PPO path. Proprioception, history and
task-direction inputs remain part of the policy input.

Both input paths now train with the same PPO implementation, network, rewards,
events and terrain distribution. Their runtime differences are the terrain input
and action delay: Scandots disables action delay, while LiDAR keeps the existing
delayed-action configuration. LiDAR training also schedules injected LiDAR and
odometry noise from zero to the maxima stored in the observation configuration.

## Registered tasks

The names below are the exact IDs currently accepted by `--task`.
`scandots_input` and `lidar_input` are descriptive names in this document, not
new CLI options. The public IDs use Scandots, Lidar or Depth plus an explicit
Train, Play or Eval suffix. Previous task IDs are no longer registered;
checkpoint files, configuration class names and storage paths are unchanged.

| Use | `scandots_input` task ID | `lidar_input` task ID |
| --- | --- | --- |
| Train | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Train-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0` |
| Play | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0` |
| Eval | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Eval-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Eval-v0` |

There are nine registered tasks in total. The other three use the older
depth-camera input path and remain available for existing checkpoints:

| Use | Existing depth-camera task ID |
| --- | --- |
| Train | `Isaac-Extreme-Parkour-Depth-Unitree-Go2-Train-v0` |
| Play | `Isaac-Extreme-Parkour-Depth-Unitree-Go2-Play-v0` |
| Eval | `Isaac-Extreme-Parkour-Depth-Unitree-Go2-Eval-v0` |

The registration source is
[`config/go2/__init__.py`](parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py).
Train, Play and Eval select distinct environment configurations; Play disables
scheduled pushes, while Eval retains evaluation disturbances.

## Installation

Use a Python environment configured for IsaacLab. From the desired parent directory:

```bash
git clone --recurse-submodules https://github.com/JINWOOSEO21/Isaaclab_Parkour.git
cd Isaaclab_Parkour
pip install -e .
pip install -e ./parkour_tasks
```

For an existing checkout, initialize the elevation-map submodule with
`git submodule update --init --recursive` if needed. The LiDAR path also requires
the GPU/CuPy dependencies of
[`elevation_mapping_cupy`](elevation_mapping_cupy/README.md); editable installation
of this repository alone does not install that backend's full environment.

Run the commands below from the repository root.

## Training — trained_v1.4

The `trained_v1.4` workflow has two PPO phases. Phase 1 trains with GT scandots.
Phase 2 initializes the policy and estimator weights from that checkpoint, changes
the terrain observation to LiDAR/odometry elevation-map samples, enables action
delay and continues optimizing the same PPO objective with a fresh optimizer.

The v1.4 environment includes slip penalties and scheduled linear/angular
perturbations. Both phases optimize the environment reward directly.

### Phase 1: Scandots input

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Train-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name scandots_v1.4 \
  --seed 42 \
  --headless
```

With the current configuration, this writes to
`logs/rsl_rl/unitree_go2_parkour/trained_v1.4/<timestamp>_scandots_v1.4/`.
After the 15,000-iteration run completes, use its `model_14999.pt` checkpoint.
An already-running v1.4 run can supply this checkpoint; it does not need to be
restarted just because the registered task names changed.

### Phase 2: LiDAR and odometry input

Run after the Phase 1 checkpoint is available:

```bash
SCANDOTS_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/trained_v1.4/<scandots-run>/model_14999.pt"

python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0 \
  --init_checkpoint "$SCANDOTS_CHECKPOINT" \
  --num_envs 192 \
  --max_iterations 30000 \
  --noise_ramp_ratio 0.7 \
  --run_name lidar_v1.4 \
  --seed 42 \
  --headless
```

`--init_checkpoint` is the explicit Phase 1 → Phase 2 boundary. It loads only the
policy, estimator and observation-normalizer weights, resets the iteration counter,
and starts fresh PPO optimizer, disturbance and LiDAR-noise schedules. It cannot be
combined with `--resume`, `--load_run` or `--checkpoint`.

`--noise_ramp_ratio` is the fraction of the Phase 2 iteration budget used by the
linear ramp. The remaining fraction is split equally between a zero-noise prefix
and a maximum-noise suffix. With the command above, 30,000 iterations are divided
into 4,500 iterations at zero noise, 21,000 ramp iterations and 4,500 iterations
at maximum noise. Use `--noise_ramp_ratio 2/3` for an exact 5,000 / 20,000 / 5,000
split. The maximum sensor and odometry noise values remain in
`LidarObservationsCfg`; the schedule multiplies those amplitudes by a value from
0 to 1.

Phase 2 keeps the existing output-directory configuration and adds the
`_lidar_v1.4` run suffix. Use the actual run path printed at launch; changing a
task ID or `--load_run` does not change the output directory. A completed
30,000-iteration Phase 2 run produces `model_29999.pt` in that run.

To resume either phase later, use its matching Train task and
`--resume --load_run <configured-run-folder> --checkpoint <checkpoint-filename> --max_iterations <additional-iterations>`.
`--max_iterations` is the total noise-schedule horizon for a fresh Phase 2 run.
When resuming, it is the additional iteration budget for that invocation while
the original noise horizon, ratio and completed progress are restored from the
LiDAR checkpoint. Resuming an existing phase and initializing Phase 2 from Phase 1
are distinct operations.

## Playback and evaluation — trained_v1.4

Set these variables to the actual checkpoint paths from the two phases.
Replace the placeholders before running the commands:

```bash
SCANDOTS_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/trained_v1.4/<scandots-run>/model_14999.pt"
LIDAR_CHECKPOINT="<absolute-path-to-the-phase-2-run>/model_29999.pt"
```

Using explicit checkpoint paths avoids falling back to older configured model
locations. Both `play.py` and `evaluation.py` accept checkpoint paths here.

### Scandots input

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0 \
  --checkpoint "$SCANDOTS_CHECKPOINT" \
  --num_envs 3 \
  --seed 42

python scripts/rsl_rl/evaluation.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Eval-v0 \
  --checkpoint "$SCANDOTS_CHECKPOINT" \
  --headless
```

### LiDAR and odometry input

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" \
  --num_envs 2 \
  --seed 42

python scripts/rsl_rl/evaluation.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Eval-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" \
  --headless
```

For normal playback/evaluation, match the task to the checkpoint's input path;
use a completed Phase 2 checkpoint to evaluate the LiDAR adaptation. Play disables
scheduled pushes; Eval uses its evaluation configuration rather than restoring
the training terrain curriculum.

### Recording and input panels

| Flags | Output |
| --- | --- |
| `--video` | One viewport video under `videos/play/` |
| `--multicam` | One video per environment under `videos/multicam/`, or `--out_dir` |
| `--video --multicam` | Both sets of videos |

The viewport starts with environment 0. There is no CLI environment selector;
in the GUI, Numpad 7/9 switch environments, Numpad 0 enables free camera mode,
and Numpad 1 returns to robot tracking.

`--panels` is a boolean flag requiring `--multicam`. It automatically shows GT
scandots for the Scandots task, GT and Estimated scandots for the Lidar task,
or the encoder's depth input for a Depth task. It affects only the per-environment
videos when both recording modes are enabled.

To record both v1.4 input phases with their corresponding panels:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0 \
  --checkpoint "$SCANDOTS_CHECKPOINT" \
  --num_envs 3 --seed 42 --headless --multicam --panels --video_length 1000

python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" \
  --num_envs 2 --seed 42 --headless --multicam --panels --video_length 1000
```

Panel values are copied before the environment step. Estimated scandots are the
samples actually passed to the actor; depth panels retain the last encoder input
between updates. `--preset` selects a Play terrain preset.

## Logs and checkpoints — trained_v1.4

Existing files and directory settings are preserved. Training writes under
`logs/rsl_rl/<experiment_name>/<run_subdir>/<timestamp>_<run_name>`. The current
experiment name is `unitree_go2_parkour`. Phase 1 uses `trained_v1.4`; Phase 2
continues to use the existing `student_pretrained` subdirectory so this refactor
does not move or rename prior logs. The v1.4 run names identify new runs without
moving previous outputs.

Renaming task registrations does not rename checkpoint keys, saved `params`, or
existing logs. The active LiDAR configuration classes use neutral `Lidar` names;
the previous internal class names remain as import aliases for serialized configs.
Use the new IDs when launching a new process. No compatibility aliases for the
previous task IDs are registered.

## Development checks

CPU-only task registration and recording/CLI regression tests:

```bash
python -m unittest parkour_test.test_task_registry_cpu parkour_test.test_play_recording_cpu -v
```

Other module tests are in `parkour_test/`; some launch Isaac Sim and should not
be collected indiscriminately while another training job is running.

Ruff lint and formatting settings are in [`pyproject.toml`](pyproject.toml).
Logs, generated outputs and the elevation-map submodule are excluded. No Ruff
command is run automatically by training or playback.

## Upstream attribution

This README describes the current fork's input paths and commands. Earlier
usage documentation remains available in Git history and in the
[upstream repository](https://github.com/CAI23sbP/Isaaclab_Parkour).
Documentation updated on 2026-09-22 to describe the current input paths and tasks.
The upstream research citations and attribution notice are retained below.
The root [`LICENSE`](LICENSE) contains GPL-3.0 text; individual source files also
carry their own notices, including BSD-3-Clause SPDX headers. This README rewrite
does not change those notices or relicense the code.

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

The following attribution excerpt is retained from the earlier README; it is
not a complete license text or a replacement for `LICENSE` and per-file notices.

```
Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/Isaaclab_Parkour
```
