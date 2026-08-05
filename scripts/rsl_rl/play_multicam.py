# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""한 번의 실행으로 지형 종류별 영상을 동시에 뽑는 스크립트.

play.py 는 뷰포트 카메라 1대를 gym.wrappers.RecordVideo 로 녹화하므로 영상이 1개만 나온다.
여기서는 씬에 붙인 TiledCamera(record_camera, env 마다 1대)를 직접 읽어
env 별로 별도의 mp4 를 기록한다.

env 와 지형의 대응은 terrain_types = floor(arange(num_envs) / (num_envs/num_cols)) 로 정해진다.
PLAY 설정이 num_cols=4 이므로 --num_envs 4 로 실행하면
  env0 parkour_gap / env1 parkour_hurdle / env2 parkour_step / env3 parkour(램프)
가 1마리씩 배정된다.

사용 예:
  PARKOUR_DIFFICULTY=0.8 python scripts/rsl_rl/play_multicam.py \
      --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 \
      --num_envs 4 --headless --video_length 1000 --out_dir videos/d0.8
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record per-terrain videos from one run.")
parser.add_argument("--video_length", type=int, default=1000, help="Number of steps to record.")
parser.add_argument("--out_dir", type=str, default=None, help="Directory to write the mp4 files into.")
parser.add_argument("--fps", type=int, default=50, help="Output video fps (sim is 1/step_dt = 50).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# TiledCamera 는 RTX 렌더가 필요하므로 항상 켠다.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import parkour_tasks  # noqa: F401


def main():
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # render_mode 는 주지 않는다. 프레임은 뷰포트가 아니라 record_camera 에서 읽는다.
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    is_distill = agent_cfg.algorithm.class_name == "DistillationWithExtractor"
    if is_distill:
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "videos", "multicam")

    obs, extras = env.get_observations()
    # 지형 배정은 리셋 후에야 확정되므로 첫 관측 뒤에 레코더를 만든다.
    recorder = PerEnvVideoRecorder(env, out_dir, fps=args_cli.fps)

    depth_latent = None
    yaw = None
    # mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로 재생 불가가
    # 되므로 어떤 경로로 빠져나가든 반드시 close 되게 감싼다.
    try:
      while simulation_app.is_running() and recorder.written < args_cli.video_length:
        with torch.inference_mode():
            if not is_distill:
                obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                    obs[:, :num_prop]
                )
                actions = policy(obs, hist_encoding=True)
            else:
                depth_camera = extras["observations"]["depth_camera"].to(env.device)
                if env.unwrapped.common_step_counter % 5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5 * yaw
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        # 스텝 중 렌더가 일어나므로 그 전에 카메라를 현재 로봇 위치로 옮겨둔다.
        recorder.track_camera()
        obs, _, _, extras = env.step(actions)

        if recorder.capture() and recorder.written % 100 == 0:
            print(f"[INFO] {recorder.written}/{args_cli.video_length} frames", flush=True)
    finally:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
