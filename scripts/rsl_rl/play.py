# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL checkpoint and optionally record videos.

--video records a single viewport video in <load_run>/videos/play/.
--multicam records one video per environment in <load_run>/videos/multicam/
(or --out_dir). Both flags may be used together; output paths are unchanged.

--panels requires --multicam and automatically shows the policy's terrain input:
  depth policy: depth camera input used by the encoder;
  GT-scan policy: GT scandots;
  elevation-map policy: GT and Estimated scandots side by side.

The viewport starts with viewer.env_index=0; play has no CLI environment selector.
In the GUI, Numpad 7/9 change the tracked environment.

Example:
  python scripts/rsl_rl/play.py --headless --multicam --panels \
      --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
      --num_envs 2 --video_length 1000
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record a single viewport video during playback."
)
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--multicam",
    action="store_true",
    default=False,
    help=(
        "Record one mp4 per environment from the scene's record_camera instead of a single viewport "
        "video. With --num_envs equal to the PLAY config's terrain-column count this yields one video "
        "per terrain in a single run. Works with all policy input modes."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="--multicam only. Directory to write the per-env mp4 files into (default: <load_run dir>/videos/multicam).",
)
parser.add_argument("--fps", type=int, default=50, help="--multicam only. Output video fps (sim is 1/step_dt = 50).")
parser.add_argument(
    "--panels",
    action="store_true",
    help=(
        "--multicam only. Show the policy's terrain input: depth for depth policies, "
        "GT scandots for GT-scan policies, or GT and Estimated scandots for elevation-map policies."
    ),
)
parser.add_argument(
    "--preset",
    type=str,
    default=None,
    help=(
        "Override the PLAY config's terrain distribution with a TERRAIN_PRESETS key "
        "(e.g. 'trapezoid_only' for ramp + stairs only). Applied with one_col_per_terrain=True, "
        "so --num_envs equal to the preset's terrain count yields one robot per terrain."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help=(
        "Seed for the environment (terrain noise, reset noise, DR). Same behaviour as train.py: "
        "omit for non-deterministic play, -1 samples a random seed."
    ),
)
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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.panels and not args_cli.multicam:
    parser.error("--panels requires --multicam.")
# always enable cameras to record video
# --multicam 은 TiledCamera 를 쓰므로 RTX 렌더가 필요하다. --video 의 뷰포트 녹화도 마찬가지다.
if args_cli.video or args_cli.multicam:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import math
import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import parse_env_cfg

from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
from parkour_tasks.default_cfg import RECORD_CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from scripts.rsl_rl.checkpoint_utils import get_checkpoint_path_with_fallback
from scripts.rsl_rl.exporter import (
    export_deploy_policy_as_jit,
    export_deploy_policy_as_onnx,
    export_teacher_policy_as_jit,
    export_teacher_policy_as_onnx,
)
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


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
    delta = quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([0.0]), torch.tensor([yaw]))
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


def apply_record_camera(env_cfg):
    """--multicam 일 때 씬에 env 별 녹화 카메라를 꽂아 넣는다.

    student 씬은 record_camera 를 상시로 들고 있지만(학습 때만 train.py 가 떼어낸다),
    teacher 씬은 학습이 6144 env 라 상시로 두면 --enable_cameras 가 강제되고 env 수만큼
    960x540 렌더가 돌아 VRAM 이 감당이 안 된다. 그래서 teacher 는 None 으로 비워 두고
    녹화하는 이 경로에서만 채운다.
    """
    if not args_cli.multicam:
        return
    if getattr(env_cfg.scene, "record_camera", None) is None:
        env_cfg.scene.record_camera = copy.deepcopy(RECORD_CAMERA_CFG)
        print("[INFO] --multicam: record_camera 가 없어 default_cfg.RECORD_CAMERA_CFG 를 씬에 추가한다.")


def apply_terrain_override(env_cfg):
    """--preset 이 주어지면 PLAY cfg 의 지형 분포를 그 프리셋으로 갈아끼운다.

    one_col_per_terrain=True 로 붙여서 커리큘럼 컬럼→지형 매핑이 1:1 로 떨어지게 한다
    (PLAY cfg 들이 쓰는 것과 같은 방식). noise_range 도 PLAY 와 같은 값으로 맞춘다.
    """
    if not args_cli.preset:
        return
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        print("[WARN] --preset: terrain_generator 가 없어 무시한다.")
        return
    apply_terrain_preset(
        generator,
        args_cli.preset,
        one_col_per_terrain=True,
        active_overrides={"noise_range": (0.02, 0.02)},
    )
    active = [k for k, v in generator.sub_terrains.items() if v.proportion > 0]
    print(f"[INFO] --preset {args_cli.preset}: 지형 {active} (num_cols={generator.num_cols})")
    if args_cli.num_envs != len(active):
        print(
            f"[WARN] --num_envs {args_cli.num_envs} != 지형 수 {len(active)}. "
            "지형당 1마리를 원하면 --num_envs 를 지형 수와 같게 줄 것."
        )


def snapshot_policy_panels(obs, num_prop, num_scan, *, depth=None, estimated_obs=None):
    """Copy the terrain inputs before env.step can mutate their backing buffers."""
    if depth is not None:
        return depth.detach().cpu().numpy().copy(), None

    scan = slice(num_prop, num_prop + num_scan)
    panels = [("GT", obs[:, scan].detach().cpu().numpy().copy())]
    if estimated_obs is not None:
        panels.append(("Estimated", estimated_obs[:, scan].detach().cpu().numpy().copy()))
    return None, panels


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    apply_spawn_rotation(env_cfg)
    apply_spawn_offset(env_cfg)
    apply_record_camera(env_cfg)
    apply_terrain_override(env_cfg)
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # --seed 가 주어졌을 때만 env 에 심는다 (train.py:127 과 같은 경로). env 초기화가
    # torch/numpy 전역 시드를 잡아 지형 노이즈·리셋 노이즈·DR 이 재현된다.
    # 플래그가 없으면 기존처럼 seed=None(비결정적)을 유지한다. -1 은 cli_args 가
    # 이미 랜덤 시드로 바꿔 agent_cfg.seed 에 넣어 둔다.
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed
        print(f"[INFO] --seed {args_cli.seed}: environment seed 를 {agent_cfg.seed} 로 설정한다.")

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
        # student_pretrained/ 바로 아래의 승격된 checkpoint 를 우선하고, 없으면
        # (승격 전이라면) 최근 하위 run 폴더에서 찾는다.
        resume_path = get_checkpoint_path_with_fallback(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 녹화물(videos/)은 checkpoint 가 하위 run 폴더에서 나왔더라도 load_run 폴더
    # (student_pretrained/ 또는 teacher_pretrained/) 아래에 모은다. --checkpoint 로
    # 임의 경로를 줬거나 load_run 폴더가 없으면 기존대로 checkpoint 옆에 둔다.
    log_dir = os.path.join(log_root_path, str(agent_cfg.load_run))
    if not os.path.isdir(log_dir):
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
        print("[INFO] Recording viewport video during playback.")
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
        export_deploy_policy_as_jit(
            policy_nn, estimator, depth_encoder, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_deploy_policy_as_onnx(
            policy_nn,
            estimator,
            depth_encoder,
            agent_cfg,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    elif agent_cfg.algorithm.class_name == "EMDistillation":
        # elevation-map student: depth encoder 가 없고 depth_actor 만 있다.
        # 배포용 export 는 EM 파이프라인(센서→em_cupy→샘플)이 정책 밖에 살아서
        # 아직 정의되지 않았다 — 재생/평가만 지원한다.
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        print("[INFO] EMDistillation: JIT/ONNX export 는 지원하지 않는다 (EM 파이프라인이 정책 밖).")
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    is_em = agent_cfg.algorithm.class_name == "EMDistillation"
    is_distill = agent_cfg.algorithm.class_name in ("DistillationWithExtractor", "EMDistillation")
    record_depth = args_cli.panels and is_distill and not is_em
    record_scandots = args_cli.panels and not record_depth
    if args_cli.panels:
        content = "Depth" if record_depth else "GT | Estimated" if is_em else "GT"
        print(f"[INFO] --panels: {content}")

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()

    # --multicam 녹화기. 지형 이름은 리셋 후에야 확정되므로 첫 관측 뒤에 만든다.
    recorder = None
    if args_cli.multicam:
        # --video 와 같은 규칙: load_run 폴더(student_pretrained/) 아래 videos/ 에 모은다.
        out_dir = args_cli.out_dir or os.path.join(log_dir, "videos", "multicam")
        recorder = PerEnvVideoRecorder(env, out_dir, fps=args_cli.fps)
        if record_scandots:
            grid = recorder.scandots_grid
            # 센서 격자와 obs 의 scan 폭이 어긋나면 reshape 이 녹화 도중에 터진다.
            # 그러면 mp4 가 close 되기 전에 죽으므로 여기서 미리 끈다.
            if grid is None or grid[0] * grid[1] != num_scan:
                print(
                    f"[WARN] --panels: height_scanner 격자 {grid} 가 num_scan={num_scan} 과 "
                    "맞지 않는다. scandots 패널을 끈다."
                )
                record_scandots = False

    timestep = 0
    panel_depth = None
    # mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로 재생 불가가
    # 되므로 어떤 경로로 빠져나가든 반드시 close 되게 감싼다.
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            if not is_distill:
                with torch.inference_mode():
                    # agent stepping
                    obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                        obs[:, :num_prop]
                    )
                    actions = policy(obs, hist_encoding=True)
                # env stepping
            elif is_em:
                # EM student: obs 의 scan 구간만 em_scan(10Hz 갱신, 사이 step 은 최신값)
                # 으로 갈아끼우고 depth_actor 의 자체 scan_encoder 가 인코딩한다.
                em_scan = extras["observations"]["em_scan"].to(env.device)
                with torch.inference_mode():
                    obs_em = obs.clone()
                    obs_em[:, num_prop : num_prop + num_scan] = em_scan
                    # 학습(learn_em)·배포와 동일: priv_explicit 은 estimator 추정값
                    obs_em[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                        obs_em[:, :num_prop]
                    )
                    actions = policy(obs_em, hist_encoding=True)
            else:
                depth_camera = extras["observations"]["depth_camera"].to(env.device)
                with torch.inference_mode():
                    if env.unwrapped.common_step_counter % 5 == 0:
                        obs_student = obs[:, :num_prop].clone()
                        obs_student[:, 6:8] = 0
                        depth_encoder_out = depth_encoder(depth_camera, obs_student)
                        if record_depth:
                            # Keep the input that produced the held latent, even if the
                            # sensor buffer changes before the next encoder update.
                            panel_depth = depth_camera.detach().clone()
                        # 지금 encoder 는 depth embedding 32 차원만 낸다. heading 을
                        # 예측하던 시절의 체크포인트는 34 차원이라 뒤 2 개가 더 붙는데,
                        # 그건 재생 호환을 위해 폭으로 갈라서 예전처럼 obs 에 덮어쓴다.
                        depth_latent = depth_encoder_out[:, :32]
                        yaw = depth_encoder_out[:, 32:] if depth_encoder_out.shape[1] > 32 else None
                    if yaw is not None:
                        # 34 차원 구 체크포인트 전용 경로. 새 정책은 teacher 와 똑같이
                        # obs 의 oracle heading 을 그대로 쓰므로 여기서 덮어쓰지 않는다.
                        obs[:, 6:8] = 1.5 * yaw
                    # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
            depth_np, scandots_np = None, None
            if record_depth or record_scandots:
                depth_np, scandots_np = snapshot_policy_panels(
                    obs,
                    num_prop,
                    num_scan,
                    depth=panel_depth if record_depth else None,
                    estimated_obs=obs_em if is_em else None,
                )
            # 스텝 중에 렌더가 일어나므로 그 전에 카메라를 현재 로봇 위치로 옮겨둔다.
            if recorder is not None:
                recorder.track_camera()
            obs, _, _, extras = env.step(actions)

            if recorder is not None:
                if not recorder.capture(depth=depth_np, scandots=scandots_np):
                    # 아직 카메라 출력이 안 나왔다. 프레임 수로 세지 않는다.
                    continue

            if args_cli.video or recorder is not None:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep >= args_cli.video_length:
                    break
                if recorder is not None and timestep % 100 == 0:
                    print(f"[INFO] {timestep}/{args_cli.video_length} frames", flush=True)

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if recorder is not None:
            recorder.close()

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
