# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.", allow_abbrev=False)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--noise_ramp_ratio", type=cli_args.parse_noise_ramp_ratio, default=0.7,
    help="LiDAR noise ramp fraction (default: 0.7; accepts 2/3). Remaining iterations split equally before/after.",
)
parser.add_argument(
    "--init_checkpoint", type=str, default=None,
    help="Initialize PPO model/estimator weights from this checkpoint path, with fresh optimizer and schedules.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
unknown_options = [arg for arg in hydra_args if arg.startswith("-")]
if unknown_options:
    parser.error(f"unrecognized arguments: {' '.join(unknown_options)}")
if args_cli.max_iterations is not None and args_cli.max_iterations <= 0:
    parser.error("--max_iterations must be positive.")
if args_cli.init_checkpoint:
    if args_cli.resume or args_cli.load_run is not None or args_cli.checkpoint is not None:
        parser.error("--init_checkpoint cannot be combined with --resume, --load_run or --checkpoint.")
    args_cli.init_checkpoint = os.path.abspath(os.path.expanduser(args_cli.init_checkpoint))
    if not os.path.isfile(args_cli.init_checkpoint):
        parser.error(f"Initialization checkpoint does not exist: {args_cli.init_checkpoint}")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import pickle
from datetime import datetime

import gymnasium as gym
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.hydra import hydra_task_config

# import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.checkpoint_utils import get_checkpoint_path_with_fallback
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ParkourManagerBasedRLEnv | ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent."""

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # The tiled camera is used by play.py --multicam only. Training never reads it,
    # so remove it to avoid allocating a renderer for every environment.
    if getattr(env_cfg.scene, "record_camera", None) is not None:
        env_cfg.scene.record_camera = None
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if agent_cfg.max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if getattr(agent_cfg, "input_mode", "scandots_input") == "lidar_input":
        agent_cfg.noise_ramp_ratio = args_cli.noise_ramp_ratio
        # Initial scene/reset observations must also be noise-free.
        env_cfg.lidar_noise_scale = 0.0
    if args_cli.init_checkpoint and agent_cfg.algorithm.class_name != "PPOWithExtractor":
        raise ValueError("--init_checkpoint is supported for PPO tasks only.")
    if (
        agent_cfg.algorithm.class_name == "PPOWithExtractor"
        and not agent_cfg.resume
        and (args_cli.load_run is not None or args_cli.checkpoint is not None)
    ):
        raise ValueError("Use --init_checkpoint PATH for a new PPO phase, or --resume to continue an existing run.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The task's configured input mode determines the run directory suffix.
    input_suffix = {"scandots_input": "scandots", "lidar_input": "lidar"}[agent_cfg.input_mode]
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir += f"_{input_suffix}"

    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Resolve a checkpoint only when continuing an existing PPO run.
    if agent_cfg.resume:
        resume_path = get_checkpoint_path_with_fallback(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # # create runner from rsl-rl
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.init_checkpoint:
        print(f"[INFO]: Initializing PPO weights from: {args_cli.init_checkpoint}")
        runner.load(args_cli.init_checkpoint, warm_start=True)
    elif agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        # 학습 재개에서만 지형 커리큘럼까지 되살린다. 체크포인트에 terrain_levels 가
        # 없거나(구 체크포인트) env 수가 달라지면 조용히 건너뛰고 경고만 남긴다.
        runner.load(resume_path, restore_terrain_curriculum=True)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # isaaclab.utils.io.dump_pickle 는 IsaacLab 2.3 에서 제거됨 (upstream 도 pickle 덤프를 뺐다).
    # 이 pkl 은 repo 안에서 다시 읽지 않으므로 표준 pickle 로 그대로 남겨둔다.
    for _name, _cfg in (("env", env_cfg), ("agent", agent_cfg)):
        _path = os.path.join(log_dir, "params", f"{_name}.pkl")
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "wb") as _f:
            pickle.dump(_cfg, _f)

    # # # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
