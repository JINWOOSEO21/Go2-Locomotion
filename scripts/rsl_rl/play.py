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
    "--multicam",
    action="store_true",
    default=False,
    help=(
        "뷰포트 1대를 녹화하는 --video 대신, 씬의 record_camera(TiledCamera, env 당 1대)를 읽어 "
        "env 마다 mp4 를 하나씩 만든다. PLAY 설정은 컬럼 수 = 지형 종류 수 이므로 "
        "--num_envs 를 지형 종류 수(현재 5)와 같게 주면 지형별로 1마리씩 배정된다."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="--multicam 으로 만든 mp4 를 저장할 디렉터리. 기본값은 체크포인트 옆 videos/multicam.",
)
parser.add_argument("--fps", type=int, default=50, help="--multicam 출력 fps (sim 은 1/step_dt = 50).")
parser.add_argument(
    "--cam_offset",
    type=str,
    default="0,2.6,1.6",
    help=(
        "--multicam 추격 카메라의 로봇 기준 오프셋 'X,Y,Z' (m). 기본값은 원본 VIEWER 와 같은 "
        "좌측 측면 시점이다. 카메라는 로봇의 z 도 따라가므로, 구덩이형 지형"
        "(parkour_pyramid_stairs 등)에서는 로봇과 같이 내려가 구덩이 턱에 시야가 막힌다. "
        "그럴 때는 높이를 올리거나(예: '0,3.0,2.4') 코스를 정면으로 보는 후방 시점"
        "(예: '-3.5,0,2.0')을 쓰면 계단이 잘 보인다. "
        "X 가 음수면 argparse 가 옵션 이름으로 오해하므로 반드시 "
        "--cam_offset=-3.5,0,2.0 처럼 '=' 로 붙여 써야 한다."
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
# (--multicam 이 읽는 TiledCamera 도 RTX 렌더가 필요하므로 같이 켠다.)
if args_cli.video or args_cli.multicam:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import wrap_to_pi
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
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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
    # --multicam 은 뷰포트가 아니라 씬의 record_camera 에서 프레임을 읽으므로 render_mode 가 필요 없다.
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
    # env 별 지형 배정은 리셋 후에야 확정되므로 첫 관측 뒤에 레코더를 만든다.
    recorder = None
    if args_cli.multicam:
        out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "videos", "multicam")
        cam_offset = tuple(float(v) for v in args_cli.cam_offset.split(","))
        if len(cam_offset) != 3:
            raise ValueError(f"--cam_offset 은 'X,Y,Z' 세 값이어야 한다: {args_cli.cam_offset!r}")
        recorder = PerEnvVideoRecorder(env, out_dir, fps=args_cli.fps, cam_offset=cam_offset)
    # simulate environment
    try:
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
        # 스텝 중에 렌더가 일어나므로 그 전에 카메라를 현재 로봇 위치로 옮겨둔다.
        if recorder is not None:
            recorder.track_camera()
        obs, _, _, extras = env.step(actions)
        if recorder is not None and recorder.capture():
            if recorder.written % 100 == 0:
                print(f"[INFO] {recorder.written}/{args_cli.video_length} frames", flush=True)
            if recorder.written >= args_cli.video_length:
                break
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    finally:
        # mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로
        # 재생 불가가 되므로 어떤 경로로 빠져나가든 반드시 close 되게 감싼다.
        if recorder is not None:
            recorder.close()

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
