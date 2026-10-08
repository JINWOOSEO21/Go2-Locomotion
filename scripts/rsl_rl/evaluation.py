# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a PPO checkpoint with optional per-environment recording."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import statistics
from collections import deque

import numpy as np
from isaaclab.app import AppLauncher
from tqdm import tqdm

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.", allow_abbrev=False)
parser.add_argument("--multicam", action="store_true", help="Record each environment under the checkpoint run/videos directory.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive.")
# always enable cameras to record video
if args_cli.multicam:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import parse_env_cfg

from locomotion_tasks.default_cfg import RECORD_CAMERA_CFG
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder

from locomotion_tasks.locomotion_task.config.go2.agents.locomotion_rl_cfg import LocomotionRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.checkpoint_utils import get_checkpoint_path_with_fallback
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import LocomotionRslRlVecEnvWrapper


def main():
    """Evaluate with RSL-RL agent."""
    # parse configuration
    if args_cli.task.find("Eval") == -1:
        print("[INFO] task argument must have 'Eval'")
        return

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: LocomotionRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.multicam:
        env_cfg.scene.record_camera = copy.deepcopy(RECORD_CAMERA_CFG)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path_with_fallback(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = LocomotionRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    reward_feet_edge = env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
    base_goal = env.unwrapped.goal_manager.get_term("base_goal")
    recorder = None
    try:
        if args_cli.multicam:
            recorder = PerEnvVideoRecorder(env, os.path.join(log_dir, "videos"), fps=round(1.0 / dt))
        for i in tqdm(range(1500)):
            start_time = time.time()
            with torch.inference_mode():
                obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                    obs[:, :num_prop]
                )
                actions = policy(obs, hist_encoding=True)
            if recorder is not None and timestep < args_cli.video_length:
                recorder.track_camera()
            cur_goal_idx = base_goal.cur_goal_idx.clone()
            obs, rews, dones, extras = env.step(actions)
            if recorder is not None and timestep < args_cli.video_length:
                if recorder.capture():
                    timestep += 1

            edge_violation_buffer.extend(reward_feet_edge.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())
            cur_reward_sum += rews
            cur_episode_length += 1
            cur_time_from_start += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float) / 7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float) / 7.0)

    edge_violation_mean = np.mean(edge_violation_buffer)
    edge_violation_std = np.std(edge_violation_buffer)

    print("Mean reward: {:.2f} ± {:.2f}".format(rew_mean, rew_std))
    print("Mean episode length: {:.2f} ± {:.2f}".format(len_mean, len_std))
    print("Mean number of waypoints: {:.2f} ± {:.2f}".format(num_waypoints_mean, num_waypoints_std))
    print("Mean edge violation: {:.2f} ± {:.2f}".format(edge_violation_mean, edge_violation_std))


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
