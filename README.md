# Isaaclab_Parkour

Train a Unitree Go2 parkour policy whose terrain perception uses LiDAR point clouds
and odometry. Training starts with ground-truth (GT) scandots in simulation to
initialize locomotion, then adapts the policy to LiDAR-based terrain estimates
using reinforcement learning.

This project builds on [Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)
and [Extreme-Parkour](https://extreme-parkour.github.io/).

## Policy inputs

The deployment target is the LiDAR policy. GT scandots provide terrain information
during pretraining and a reference for visualization during LiDAR training and
playback. The deployed policy does not require GT scandots.

```text
LiDAR point clouds + odometry
    → elevation map
    → 132 terrain-height samples (12 × 11)
    → scan encoder and locomotion policy
```

Odometry places point clouds in the map and locates the sampling grid around the
robot. The map updates every five control steps (10 Hz), holding its latest
samples between updates. In simulation, odometry is modeled from robot state with
injected noise and drift. The policy also uses proprioception, observation history
and task-direction inputs; LiDAR and odometry describe its terrain-perception
pipeline, not the entire control interface.

| Training phase | Terrain observations | Optimization | Action delay |
| --- | --- | --- | --- |
| Phase 1: Scandots pretraining | GT height samples | RL PPO (`PPOWithExtractor`) | Disabled |
| Phase 2: LiDAR training | Height samples from LiDAR and odometry | RL PPO, initialized from Phase 1 | 1 control step (20 ms) |

Phase 2 optimizes environment rewards. It does not match actions from a frozen
reference policy. LiDAR Train, Play and Eval all enable action delay, including
when injected sensor noise is zero. `action_delay_steps=[1, 1]` keeps the delay at
one step; `history_length=8` specifies buffer capacity, not an eight-step delay.

## Installation

Use a Python environment configured for IsaacLab:

```bash
git clone --recurse-submodules https://github.com/JINWOOSEO21/Isaaclab_Parkour.git
cd Isaaclab_Parkour
pip install -e .
pip install -e ./parkour_tasks
```

For an existing checkout, initialize submodules with
`git submodule update --init --recursive`. The LiDAR pipeline requires the GPU/CuPy
dependencies described in [elevation_mapping_cupy](elevation_mapping_cupy/README.md).
Run the following commands from the repository root.

## Tasks

| Use | Scandots pretraining | LiDAR policy |
| --- | --- | --- |
| Train | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Train-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0` |
| Play | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0` |
| Eval | `Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Eval-v0` | `Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Eval-v0` |

Play disables scheduled pushes. Eval uses the evaluation terrain distribution and
disturbances. The task registration source is
[config/go2/__init__.py](parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py).

## Training

Both phases default to **4096 environments × 24 control steps per rollout**
(98,304 transitions per PPO iteration). `--num_envs` overrides the environment
count. LiDAR mapping adds GPU work and memory use; choose a smaller count when
needed for the available hardware.

### Phase 1: Scandots pretraining

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Train-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name scandots \
  --seed 42 \
  --headless
```

The run is saved to
`logs/rsl_rl/unitree_go2_parkour/<timestamp>_scandots/`.
A completed 15,000-iteration run produces `model_14999.pt`.

### Phase 2: LiDAR and odometry

Replace `<timestamp>` with the Phase 1 run's timestamp:

```bash
SCANDOTS_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/<timestamp>_scandots/model_14999.pt"

python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0 \
  --init_checkpoint "$SCANDOTS_CHECKPOINT" \
  --num_envs 4096 \
  --max_iterations 30000 \
  --noise_ramp_ratio 0.7 \
  --run_name lidar \
  --seed 42 \
  --headless
```

The run is saved to `logs/rsl_rl/unitree_go2_parkour/<timestamp>_lidar/`.
A completed 30,000-iteration run produces `model_29999.pt`.

`--init_checkpoint` loads policy, estimator and available observation-normalizer
weights. It starts fresh optimizers, iteration counts and curricula. It cannot be
combined with `--resume`, `--load_run` or `--checkpoint`.

`--noise_ramp_ratio` specifies the fraction of iterations used for the linear
increase in LiDAR and odometry noise. The remaining iterations are split equally
between an initial zero-noise interval and a final maximum-noise interval:

| 30,000-iteration schedule | Zero noise | Linear increase | Maximum noise |
| --- | --- | --- | --- |
| Default `--noise_ramp_ratio 0.7` | 4,500 | 21,000 | 4,500 |
| `--noise_ramp_ratio 2/3` | 5,000 | 20,000 | 5,000 |

Noise amplitudes are scaled from zero to the maxima defined in
`LidarObservationsCfg`. Zero injected noise still includes LiDAR sampling and
mapping errors. The action delay remains enabled throughout the schedule.

### Resuming training

Select the matching Train task and the existing run folder:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0 \
  --resume \
  --load_run '<timestamp>_lidar' \
  --checkpoint model_10000.pt \
  --max_iterations 20000 \
  --headless
```

On resume, `--max_iterations` is the additional iteration budget. The LiDAR noise
schedule restores its original horizon, ratio and completed progress from the
checkpoint. Resume writes a new run directory; it does not move existing files.

## Playback and evaluation

Set the checkpoint paths for the runs to inspect:

```bash
SCANDOTS_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/<timestamp>_scandots/model_14999.pt"
LIDAR_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/<timestamp>_lidar/model_29999.pt"
```

Inspect the pretrained policy:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0 \
  --checkpoint "$SCANDOTS_CHECKPOINT" --num_envs 3 --seed 42
```

Play and evaluate the LiDAR policy:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" --num_envs 3 --seed 42

python scripts/rsl_rl/evaluation.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Eval-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" --headless
```

LiDAR Play and Eval use maximum configured sensor noise and action delay.
Evaluation does not restore the training terrain curriculum.

### Recording

`--multicam` records one video per environment. Videos are written directly to
`<checkpoint-run-directory>/videos/`; `--out_dir` overrides that location in
`play.py`. `--video_length` sets the number of recorded steps.

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
  --checkpoint "$LIDAR_CHECKPOINT" \
  --num_envs 3 --seed 42 --headless \
  --multicam --panels --video_length 1000
```

`--panels` requires `--multicam`. For LiDAR, it shows GT and estimated scandots
side by side; the estimated samples are the terrain inputs passed to the actor.
For Scandots pretraining, it shows GT scandots only. Panel inputs are copied before
the environment step. `--preset` selects a Play terrain preset.

## Logs and checkpoints

Training writes to
`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`.
The default experiment is `unitree_go2_parkour`. `run_name` is a directory-name
suffix: it defaults to `scandots` for Phase 1 and `lidar` for Phase 2. For example,
`--run_name lidar_trial2` creates `<timestamp>_lidar_trial2`. It does not choose
an algorithm or a checkpoint to load.

Existing logs and checkpoints are left in place. Use an explicit checkpoint path
for playback or Phase 2 initialization, and `--load_run` to select a resume source.

## Development checks

Run targeted CPU checks without launching Isaac Sim:

```bash
python -m unittest parkour_test.test_task_registry_cpu parkour_test.test_lidar_ppo_cpu parkour_test.test_lidar_noise_cpu parkour_test.test_play_recording_cpu -v
```

Some other tests launch Isaac Sim; avoid collecting them indiscriminately during
an active training run. Ruff lint and formatting settings are in
[pyproject.toml](pyproject.toml):

```bash
ruff check .
ruff format .
```
