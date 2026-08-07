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
parser.add_argument(
    "--with_depth",
    action="store_true",
    default=False,
    help=(
        "Paste the depth map the policy actually consumes onto the right of each frame, so one video "
        "shows the robot and its depth input side by side. Student (distillation) tasks only."
    ),
)
parser.add_argument(
    "--spawn_roll",
    type=float,
    default=0.0,
    help="Roll the robot by this many degrees at spawn. Positive tilts the body toward its own right side.",
)
parser.add_argument(
    "--spawn_yaw",
    type=float,
    default=0.0,
    help="Yaw the robot by this many degrees at spawn. Positive turns it left (CCW), negative right.",
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# TiledCamera 는 RTX 렌더가 필요하므로 항상 켠다.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import math
import numpy as np
import os
import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from scripts.rsl_rl.video_overlay import depth_to_panel

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import parkour_tasks  # noqa: F401


def resolve_terrain_names(env) -> list[str]:
    """env 별로 배정된 서브지형 이름을 돌려준다.

    TerrainImporter 가 env 를 (level=row, type=col) 로 배정하고
    ParkourTerrainGenerator 가 terrain_names[row, col] 에 이름을 채워둔다.
    그 배열을 우선 쓰고, 접근이 안 되면 generator 와 동일한 규칙으로
    컬럼→서브지형 매핑을 직접 계산한다.
    """
    terrain = env.unwrapped.scene.terrain
    types = terrain.terrain_types.cpu().numpy()
    levels = terrain.terrain_levels.cpu().numpy()

    # 1순위: 생성기가 기록해둔 이름 배열
    try:
        names_arr = terrain.terrain_generator_class.terrain_names
        return [str(np.asarray(names_arr[levels[i], types[i]]).reshape(-1)[0]) for i in range(len(types))]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] terrain_names 조회 실패, 설정에서 재계산한다: {exc!r}")

    # 2순위: proportion 으로 컬럼 매핑을 재현 (ParkourTerrainGenerator 와 같은 식)
    gen_cfg = env.unwrapped.scene.cfg.terrain.terrain_generator
    keys = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    props = props / props.sum()
    num_cols = gen_cfg.num_cols
    col_names = [
        keys[int(np.min(np.where(i / num_cols + 0.001 < np.cumsum(props))[0]))] for i in range(num_cols)
    ]
    return [col_names[t] for t in types]


def apply_spawn_offset(env_cfg):
    """--spawn_offset_y 만큼 스폰 지점을 월드 y 로 밀어 준다.

    events.reset_root_state 의 위치 계산은
        positions = default_root_state[:, 0:3] + env_origin - (back_from_center, 0, 0)
    이고 default_root_state 의 앞 3개는 ArticulationCfg.init_state.pos 에서 온다.
    그래서 init_state.pos 의 y 만 바꾸면 지형 타일 안에서 좌우로 밀린다.

    타일 y 폭이 4m(중심에서 +-2m)이므로 25cm 는 걷기 시작하는 유효 통로
    (half_valid_width 0.6~1.2m) 안에 들어간다.
    """
    if not args_cli.spawn_offset_y:
        return
    pos = list(env_cfg.scene.robot.init_state.pos)
    pos[1] += args_cli.spawn_offset_y
    env_cfg.scene.robot.init_state.pos = tuple(float(v) for v in pos)
    print(f"[INFO] 스폰 위치 이동: y {args_cli.spawn_offset_y:+.3f}m -> pos={env_cfg.scene.robot.init_state.pos}")


def apply_spawn_rotation(env_cfg):
    """--spawn_roll / --spawn_yaw 만큼 로봇의 초기 자세를 돌린다. (play.py 와 같은 규약)

    스폰 자세는 events.reset_root_state 가 default_root_state[:, 3:7] 을 그대로
    write_root_pose_to_sim 에 넘겨서 정해지고, 그 값은 ArticulationCfg.init_state.rot
    에서 온다. GO2 기본 cfg 에는 rot 이 없어 항등 쿼터니언이다.

    부호는 오른손 법칙(+Z 위, +X 앞, +Y 왼쪽) 기준이다.
      roll > 0 : 왼쪽면이 올라간다 = 몸이 오른쪽으로 기운다.
      yaw  > 0 : 위에서 봤을 때 반시계 = 왼쪽. 오른쪽은 음수다.
    """
    if not (args_cli.spawn_roll or args_cli.spawn_yaw):
        return
    roll = math.radians(args_cli.spawn_roll)
    yaw = math.radians(args_cli.spawn_yaw)
    delta = quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([0.0]), torch.tensor([yaw]))
    base = torch.tensor([env_cfg.scene.robot.init_state.rot])
    rot = quat_mul(delta, base)[0]
    env_cfg.scene.robot.init_state.rot = tuple(float(v) for v in rot)
    print(
        f"[INFO] 스폰 자세 변경: roll={args_cli.spawn_roll:+.1f}deg yaw={args_cli.spawn_yaw:+.1f}deg "
        f"-> quat(w,x,y,z)=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f})"
    )


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

    apply_spawn_rotation(env_cfg)
    apply_spawn_offset(env_cfg)

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

    camera = env.unwrapped.scene["record_camera"]
    robot = env.unwrapped.scene["robot"]
    # 카메라 자세는 고정하고 위치만 로봇을 따라간다.
    # 오프셋이 월드 기준 상수이므로 eye->target 방향도 항상 같고,
    # 결과적으로 자세는 고정된 채 평행이동만 하게 된다.
    # (로봇 base 에 부착하면 몸체의 pitch/roll 을 물려받아 화면이 같이 기운다.)
    #
    # 오프셋은 원본 VIEWER(eye=(-0., 2.6, 1.6), origin_type='asset_root') 와 동일한
    # 좌측 측면 시점이다. +Y 가 좌측, +Z 가 위.
    cam_offset = torch.tensor([0.0, 2.6, 1.6], device=env.unwrapped.device)

    def track_camera():
        """로봇 위치 + 고정 오프셋에서 로봇을 바라보게 카메라를 옮긴다."""
        target = robot.data.root_pos_w
        camera.set_world_poses_from_view(eyes=target + cam_offset, targets=target)

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "videos", "multicam")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    obs, extras = env.get_observations()
    # 지형 이름은 리셋 후에야 확정되므로 첫 관측 뒤에 읽는다.
    terrain_names = resolve_terrain_names(env)
    writers = []
    for i, name in enumerate(terrain_names):
        path = os.path.join(out_dir, f"env{i}_{name}.mp4")
        # quality 8 은 imageio-ffmpeg 기준 고화질(기본값 5보다 높음).
        writers.append(imageio.get_writer(path, fps=args_cli.fps, quality=8, macro_block_size=1))
        print(f"[INFO] env {i} -> {name} -> {path}", flush=True)

    # depth 패널은 student 정책일 때만 의미가 있다. teacher 는 depth 관측 자체가 없다.
    record_depth = bool(args_cli.with_depth and is_distill)
    if args_cli.with_depth and not is_distill:
        print("[WARN] --with_depth 는 student(Distillation) 태스크 전용이다. depth 없이 녹화한다.")
    if record_depth:
        print("[INFO] --with_depth: 각 프레임 오른쪽에 정책 입력 depth map 을 붙인다.")

    depth_camera = None
    depth_latent = None
    yaw = None
    written = 0
    # mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로 재생 불가가
    # 되므로 어떤 경로로 빠져나가든 반드시 close 되게 감싼다.
    try:
      while simulation_app.is_running() and written < args_cli.video_length:
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
        track_camera()
        obs, _, _, extras = env.step(actions)

        rgb = camera.data.output["rgb"]
        if rgb is None:
            continue
        frames = rgb[..., :3].detach().cpu().numpy()
        if frames.dtype != np.uint8:
            # float 로 나오는 경우 0..1 로 보고 변환한다.
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        # 정책 호출에 쓴 것과 같은 텐서를 그대로 그린다. 버퍼가 5 스텝마다 갱신되므로
        # 사이 스텝에서는 직전 프레임이 유지되는데, 그게 정책이 보고 있는 실제 입력이다.
        depth_np = None
        if record_depth:
            depth_np = depth_camera.detach().cpu().numpy()
        for i, w in enumerate(writers):
            frame = frames[i]
            if depth_np is not None:
                frame = np.hstack([frame, depth_to_panel(depth_np[i], frame.shape[0])])
            w.append_data(frame)
        written += 1
        if written % 100 == 0:
            print(f"[INFO] {written}/{args_cli.video_length} frames", flush=True)
    finally:
        for w in writers:
            try:
                w.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] writer close 실패: {exc!r}", flush=True)
        print(f"[INFO] wrote {written} frames to {out_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
