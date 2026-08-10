# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import os 
import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

import gymnasium as gym
import os
import pickle
import torch
from datetime import datetime

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
# import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_isaaclab.envs import (
ParkourManagerBasedRLEnv
)
# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ParkourManagerBasedRLEnv |ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: ParkourRslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # student 씬의 record_camera(TiledCamera)는 play.py --multicam / demo.py 녹화 전용이다.
    # 학습에서는 이 카메라의 프레임을 읽는 곳이 없는데도, 씬에 있으면 --enable_cameras
    # 없이는 스폰 단계에서 RuntimeError 가 나고, 켜면 env 수만큼 960x540 렌더가
    # 매 스텝 돌아가 VRAM 과 속도를 잡아먹는다. 학습 경로에서는 항상 떼어낸다.
    # (--video 녹화는 뷰포트(render_mode="rgb_array")를 쓰므로 이 카메라와 무관하다.)
    if getattr(env_cfg.scene, "record_camera", None) is not None:
        env_cfg.scene.record_camera = None
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

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
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"

    # run_subdir 는 쓰기 경로에만 끼운다. 아래 get_checkpoint_path 의 뿌리는 여전히
    # log_root_path 라서, student 산출물을 student_pretrained/ 아래로 모으면서도
    # load_run="teacher_pretrained" 로 teacher 체크포인트를 그대로 찾아온다.
    run_subdir = getattr(agent_cfg, "run_subdir", None)
    if run_subdir:
        log_dir = os.path.join(run_subdir, log_dir)

    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        load_run, load_checkpoint = agent_cfg.load_run, agent_cfg.load_checkpoint
        # distillation 의 출발점은 teacher 다. agent_cfg.load_run 은 play/evaluation 이
        # 집어올 student 를 가리키므로 그대로 쓰면 student 를 이어 학습하게 된다.
        # --resume 으로 student 학습을 이어가는 경우와 --load_run 을 손으로 준 경우는
        # 사용자의 지정이 이기게 둔다.
        if (
            agent_cfg.algorithm.class_name == "DistillationWithExtractor"
            and not agent_cfg.resume
            and args_cli.load_run is None
            and getattr(agent_cfg, "distill_load_run", None)
        ):
            load_run = agent_cfg.distill_load_run
            if args_cli.checkpoint is None and agent_cfg.distill_load_checkpoint:
                load_checkpoint = agent_cfg.distill_load_checkpoint
            print(f"[INFO] Distillation source (teacher): run='{load_run}', checkpoint='{load_checkpoint}'")
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    # # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # # create runner from rsl-rl
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "DistillationWithExtractor":
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
