# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL.

녹화 방식이 두 가지다. teacher / student 어느 태스크든 똑같이 쓸 수 있다.

  --video     뷰포트 카메라 1대를 gym.wrappers.RecordVideo 로 녹화한다.
              결과는 <체크포인트 폴더>/videos/play/ 아래 mp4 1개다.

  --multicam  씬의 record_camera(TiledCamera, env 당 1대)를 직접 읽어
              env 마다 mp4 를 하나씩 쓴다. env 와 지형의 대응이
                  terrain_types = floor(arange(num_envs) / (num_envs/num_cols))
              이므로 --num_envs 를 PLAY 설정의 num_cols(= 활성 지형 수)와 같게 주면
              지형당 1마리가 되어 "지형별 영상"이 한 번에 나온다.
                teacher PLAY : trapezoid_train_all + one_col_per_terrain -> 3종
                               (flat / 사다리꼴 램프 / 사다리꼴 계단) 이므로 --num_envs 3
                student PLAY : trapezoid_only + one_col_per_terrain -> 2종 이므로 --num_envs 2
              파일 이름에 실제 지형 이름이 들어간다 (envN_<terrain>.mp4).

              teacher 씬은 학습이 6144 env 라 record_camera 를 상시로 둘 수 없어
              None 으로 비워 두었고, 이 스크립트가 --multicam 일 때만 꽂아 넣는다.

  --with_depth  --multicam 과 같이 쓴다. depth map 을 각 프레임 오른쪽에 붙인다.
                student 는 정책이 실제로 먹는 관측이 그대로 그려진다.
                teacher 는 depth 관측이 없으므로, student 와 같은 카메라/전처리를
                녹화용으로만 씬에 꽂아 넣어 "같은 자리에서 보면 이렇게 보인다"를
                보여 준다 (정책 입력에는 전혀 쓰이지 않는다).

  --with_scandots  --multicam 과 같이 쓴다. teacher 가 실제로 먹는 height scan
                (obs 의 num_scan=132 구간) 을 로봇 기준 탑뷰 격자로 그려 붙인다.
                teacher 에게는 이게 "정책이 보는 지형" 그 자체다. --with_depth 의
                teacher 패널이 정책과 무관한 참고 화면인 것과 대비된다.
                student 는 이 구간을 depth latent 로 대체해 쓰지 않으므로,
                student 에서 켜면 "teacher 라면 봤을 값" 을 보여 주는 셈이다.
                --with_depth 와 같이 주면 RGB | depth | scandots 로 붙는다.

  --preset      PLAY cfg 의 지형 분포를 TERRAIN_PRESETS 의 다른 키로 바꾼다.
                one_col_per_terrain 로 붙이므로 --num_envs 를 프리셋의 지형 수와
                같게 주면 지형당 1마리가 된다.
                  trapezoid_only      : 사다리꼴 램프 + 계단 2종
                  trapezoid_train_all : 위 2종 + flat 3종 (teacher PLAY 기본값)

사용 예:
  # teacher, 지형 3종 영상 3개
  python scripts/rsl_rl/play.py --headless --multicam \
      --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
      --num_envs 3 --video_length 1000 --out_dir videos/teacher

  # teacher, 램프/계단 2종만 + 정책 입력 scandots 패널, 10초(50fps x 500 step)
  python scripts/rsl_rl/play.py --headless --multicam --with_scandots \
      --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
      --preset trapezoid_only --num_envs 2 --video_length 500

  # student, 지형 2종 + depth 패널
  python scripts/rsl_rl/play.py --headless --multicam --with_depth \
      --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 \
      --num_envs 2 --video_length 1000

  # 기존 방식(뷰포트 1대)
  python scripts/rsl_rl/play.py --video --video_length 1000 \
      --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
"""

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
    "--multicam",
    action="store_true",
    default=False,
    help=(
        "Record one mp4 per environment from the scene's record_camera instead of a single viewport "
        "video. With --num_envs equal to the PLAY config's terrain-column count this yields one video "
        "per terrain in a single run. Works for both teacher and student tasks."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="--multicam only. Directory to write the per-env mp4 files into (default: <checkpoint>/videos/multicam).",
)
parser.add_argument("--fps", type=int, default=50, help="--multicam only. Output video fps (sim is 1/step_dt = 50).")
parser.add_argument(
    "--with_depth",
    action="store_true",
    default=False,
    help=(
        "--multicam only. Paste the depth map onto the right of each frame, so one video shows the robot "
        "and the depth view side by side. For student tasks this is the policy's actual input; for teacher "
        "tasks a record-only depth camera (same cfg as the student's) is attached to the scene."
    ),
)
parser.add_argument(
    "--with_scandots",
    action="store_true",
    default=False,
    help=(
        "--multicam only. Paste the height scan the teacher policy actually consumes (the num_scan "
        "slice of the observation) onto the right of each frame, drawn as a robot-centred top-down "
        "grid. Combine with --with_depth to get RGB | depth | scandots side by side."
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
# --multicam 은 TiledCamera 를 쓰므로 RTX 렌더가 필요하다. --video 의 뷰포트 녹화도 마찬가지다.
if args_cli.video or args_cli.multicam:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import math
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, wrap_to_pi
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
from parkour_tasks.default_cfg import CAMERA_CFG, RECORD_CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import StudentObservationsCfg
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
        generator, args_cli.preset,
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


def apply_depth_camera(env_cfg):
    """--with_depth 인데 씬에 depth 관측이 없으면(=teacher) 녹화용으로 꽂아 넣는다.

    teacher 는 depth 를 아예 안 보는 정책이라 씬에도 관측에도 카메라가 없다. 그래서
    student 와 똑같은 CAMERA_CFG(RayCasterCamera) + image_features 관측군을 여기서
    붙여 "로봇 머리 위치에서 보면 이렇게 보인다"를 영상에 같이 남긴다. 관측군 이름이
    policy 가 아니므로 정책 입력에는 절대 섞이지 않는다 (wrapper 는 policy 군만 읽는다).

    RayCasterCamera 는 mesh raycast 라 RTX 렌더를 추가로 돌리지 않는다. env 2~3 대
    규모에서는 비용이 사실상 없다.
    """
    if not (args_cli.with_depth and args_cli.multicam):
        return
    if getattr(env_cfg.scene, "depth_camera", None) is not None:
        return
    env_cfg.scene.depth_camera = copy.deepcopy(CAMERA_CFG)
    env_cfg.scene.depth_camera.update_period = env_cfg.sim.dt * env_cfg.decimation
    obs_cfg = copy.deepcopy(StudentObservationsCfg.DepthCameraPolicyCfg())
    # 헤드리스에서 cv2 창을 띄우려다 경고만 남긴다. 녹화에는 필요 없다.
    obs_cfg.depth_cam.params["debug_vis"] = False
    env_cfg.observations.depth_camera = obs_cfg
    print("[INFO] --with_depth: 씬에 depth 관측이 없어 녹화 전용 depth 카메라를 추가한다 (정책 입력 아님).")


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
    apply_depth_camera(env_cfg)
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
            print("[INFO] --fixed_heading: 관측의 oracle heading 대신 월드 +X 방향을 목표로 삼는다.")
        else:
            print("[WARN] --fixed_heading 은 student(Distillation) 정책 전용이다. 이 태스크에서는 무시된다.")

    record_depth = bool(args_cli.with_depth and args_cli.multicam)
    if args_cli.with_depth and not args_cli.multicam:
        print("[WARN] --with_depth 는 --multicam 과 같이 써야 한다. 무시한다.")
        record_depth = False
    if record_depth:
        what = "정책 입력" if is_distill else "녹화 전용(정책 입력 아님)"
        print(f"[INFO] --with_depth: 각 프레임 오른쪽에 {what} depth map 을 붙인다.")

    record_scandots = bool(args_cli.with_scandots and args_cli.multicam)
    if args_cli.with_scandots and not args_cli.multicam:
        print("[WARN] --with_scandots 는 --multicam 과 같이 써야 한다. 무시한다.")
        record_scandots = False
    if record_scandots:
        # student 는 actor 에 scandots_latent 를 직접 넘기므로 obs 의 scan 구간을
        # 쓰지 않는다 (actor_critic_with_encoder 가 latent 가 주어지면 scan encoder
        # 를 건너뛴다). 그래도 관측 자체는 계산돼 들어 있어서 그릴 수는 있다.
        what = "정책 입력" if not is_distill else "참고용(student 는 depth latent 로 대체)"
        print(f"[INFO] --with_scandots: 각 프레임 오른쪽에 {what} scandots 격자를 붙인다.")

    robot = env.unwrapped.scene["robot"]

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
        out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "videos", "multicam")
        recorder = PerEnvVideoRecorder(env, out_dir, fps=args_cli.fps)
        if record_scandots:
            grid = recorder.scandots_grid
            # 센서 격자와 obs 의 scan 폭이 어긋나면 reshape 이 녹화 도중에 터진다.
            # 그러면 mp4 가 close 되기 전에 죽으므로 여기서 미리 끈다.
            if grid is None or grid[0] * grid[1] != num_scan:
                print(
                    f"[WARN] --with_scandots: height_scanner 격자 {grid} 가 num_scan={num_scan} 과 "
                    "맞지 않는다. scandots 패널을 끈다."
                )
                record_scandots = False

    timestep = 0
    depth_camera = None
    # mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로 재생 불가가
    # 되므로 어떤 경로로 빠져나가든 반드시 close 되게 감싼다.
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
                # teacher 는 depth 를 안 먹지만, --with_depth 면 녹화용 관측군이 붙어 있다.
                if record_depth:
                    depth_camera = extras["observations"]["depth_camera"].to(env.device)
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
                        depth_encoder_out = depth_encoder(depth_camera, obs_student)
                        # 지금 encoder 는 depth embedding 32 차원만 낸다. heading 을
                        # 예측하던 시절의 체크포인트는 34 차원이라 뒤 2 개가 더 붙는데,
                        # 그건 재생 호환을 위해 폭으로 갈라서 예전처럼 obs 에 덮어쓴다.
                        depth_latent = depth_encoder_out[:, :32]
                        yaw = depth_encoder_out[:, 32:] if depth_encoder_out.shape[1] > 32 else None
                    if args_cli.fixed_heading:
                        # obs index 6,7 은 원래 delta_yaw / delta_next_yaw, 즉
                        #   (goal point 방향의 월드 각도) - (로봇의 현재 진행 각도)
                        # 다. 여기서는 관측이 주는 oracle heading 을 쓰지 않고,
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
                    elif yaw is not None:
                        # 34 차원 구 체크포인트 전용 경로. 새 정책은 teacher 와 똑같이
                        # obs 의 oracle heading 을 그대로 쓰므로 여기서 덮어쓰지 않는다.
                        obs[:, 6:8] = 1.5*yaw
                    # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
            # env.step 이 obs 를 갈아끼우므로, 방금 정책에 넣은 scan 구간을 먼저 떠 둔다.
            # depth 쪽과 마찬가지로 "이 프레임의 행동을 만든 입력"을 그려야 한다.
            scandots_np = (
                obs[:, num_prop:num_prop + num_scan].detach().cpu().numpy() if record_scandots else None
            )
            # 스텝 중에 렌더가 일어나므로 그 전에 카메라를 현재 로봇 위치로 옮겨둔다.
            if recorder is not None:
                recorder.track_camera()
            obs, _, _, extras = env.step(actions)

            if recorder is not None:
                # 정책 호출에 쓴 것과 같은 텐서를 그대로 그린다. 버퍼가 5 스텝마다 갱신되므로
                # 사이 스텝에서는 직전 프레임이 유지되는데, 그게 정책이 보고 있는 실제 입력이다.
                depth_np = depth_camera.detach().cpu().numpy() if record_depth else None
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
