# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--spawn_roll",
    type=float,
    default=0.0,
    help=(
        "Roll the robot by this many degrees at spawn (one-off; 0 keeps the usual upright pose). "
        "Positive tilts the body toward its own right side, negative toward its left."
    ),
)
parser.add_argument(
    "--spawn_yaw",
    type=float,
    default=0.0,
    help=(
        "Yaw the robot by this many degrees at spawn (one-off; 0 keeps the usual heading). "
        "Positive turns it left (counter-clockwise seen from above), negative turns it right."
    ),
)
parser.add_argument(
    "--spawn_offset_y",
    type=float,
    default=0.0,
    help=(
        "Shift the spawn point along the world y axis by this many metres. Use it to give a yawed "
        "robot room before it walks off the 4 m wide tile. Positive is world +y (left of the course)."
    ),
)
parser.add_argument(
    "--fixed_heading",
    action="store_true",
    default=False,
    help=(
        "Student policy only. Ignore the heading estimated by the depth encoder and instead set the "
        "observation's delta-yaw terms (index 6 and 7) to (world +X) - (current robot heading) every step, "
        "so the robot always steers toward the world forward direction."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, wrap_to_pi
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def apply_spawn_offset(env_cfg):
    """--spawn_offset_y 만큼 스폰 지점을 월드 y 로 밀어 준다.

    events.reset_root_state 의 위치 계산은
        positions = default_root_state[:, 0:3] + env_origin - (back_from_center, 0, 0)
    이고 앞 3개는 ArticulationCfg.init_state.pos 에서 온다. y 만 바꾸면 타일 안에서
    좌우로 밀린다. 타일 y 폭은 4m(중심에서 +-2m)다.
    """
    if not args_cli.spawn_offset_y:
        return
    pos = list(env_cfg.scene.robot.init_state.pos)
    pos[1] += args_cli.spawn_offset_y
    env_cfg.scene.robot.init_state.pos = tuple(float(v) for v in pos)
    print(f"[INFO] 스폰 위치 이동: y {args_cli.spawn_offset_y:+.3f}m -> pos={env_cfg.scene.robot.init_state.pos}")


def apply_spawn_rotation(env_cfg):
    """--spawn_roll / --spawn_yaw 만큼 로봇의 초기 자세를 돌린다.

    스폰 자세는 events.reset_root_state 가 default_root_state[:, 3:7] 를 그대로
    write_root_pose_to_sim 에 넘겨서 정해지고, 그 값은 ArticulationCfg.init_state.rot
    에서 온다. GO2 기본 cfg 에는 rot 이 없어 항등 쿼터니언 (1,0,0,0) 이다.
    따라서 여기서 rot 만 바꿔 두면 리셋될 때마다 그 자세로 놓인다.

    부호는 오른손 법칙(+Z 위, +X 앞, +Y 왼쪽) 기준이다.
      roll > 0 : +X 축 회전이라 왼쪽면이 올라간다 = 몸이 오른쪽으로 기운다.
      yaw  > 0 : 위에서 봤을 때 반시계 = 왼쪽으로 돈다. 오른쪽은 음수다.
                 keyboard_teleop 의 좌회전이 +dyaw 인 것과 같은 규약이다.
    """
    if not (args_cli.spawn_roll or args_cli.spawn_yaw):
        return
    roll = math.radians(args_cli.spawn_roll)
    yaw = math.radians(args_cli.spawn_yaw)
    delta = quat_from_euler_xyz(
        torch.tensor([roll]), torch.tensor([0.0]), torch.tensor([yaw])
    )
    base = torch.tensor([env_cfg.scene.robot.init_state.rot])
    # 월드 기준 회전을 기존 자세에 얹는 순서. 기본값이 항등이라 지금은 delta 와 같지만,
    # 나중에 cfg 에 rot 이 생겨도 의도한 대로 동작하도록 곱해 둔다.
    rot = quat_mul(delta, base)[0]
    env_cfg.scene.robot.init_state.rot = tuple(float(v) for v in rot)
    print(
        f"[INFO] 스폰 자세 변경: roll={args_cli.spawn_roll:+.1f}deg "
        f"yaw={args_cli.spawn_yaw:+.1f}deg -> quat(w,x,y,z)="
        f"({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f})"
    )


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    apply_spawn_rotation(env_cfg)
    apply_spawn_offset(env_cfg)
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device) 
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    is_distill = agent_cfg.algorithm.class_name == "DistillationWithExtractor"
    if args_cli.fixed_heading:
        if is_distill:
            print("[INFO] --fixed_heading: depth encoder 의 heading 예측 대신 월드 +X 방향을 목표로 삼는다.")
        else:
            print("[WARN] --fixed_heading 은 student(Distillation) 정책 전용이다. 이 태스크에서는 무시된다.")
    robot = env.unwrapped.scene["robot"]

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding = True)
            # env stepping
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                if args_cli.fixed_heading:
                    # obs index 6,7 은 원래 delta_yaw / delta_next_yaw, 즉
                    #   (goal point 방향의 월드 각도) - (로봇의 현재 진행 각도)
                    # 다. 여기서는 depth encoder 가 추정한 heading(yaw)을 쓰지 않고,
                    # goal 방향을 월드 앞 방향(+X, 각도 0)으로 고정해 같은 식으로 계산한다.
                    #
                    # heading_w = atan2(forward_w.y, forward_w.x) 로, 관측 코드가 쓰는
                    # wrap_to_pi(euler_xyz_from_quat(root_quat_w)[2]) 와 같은 값이다.
                    # 로봇 자세가 매 스텝 바뀌므로 encoder 갱신 주기(5스텝)와 무관하게
                    # 여기서 매 스텝 다시 계산한다.
                    # delta 는 순수한 각도(rad)로 두고, obs 에 넣을 때만 아래 encoder
                    # 경로의 `1.5*yaw` 와 동일한 배율 1.5 를 곱한다.
                    delta = wrap_to_pi(0.0 - robot.data.heading_w)
                    obs[:, 6] = 1.5 * delta
                    obs[:, 7] = 1.5 * delta
                else:
                    obs[:, 6:8] = 1.5*yaw
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        obs, _, _, extras = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
