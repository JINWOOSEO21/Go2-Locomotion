"""
Code reference:
1. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/enum_namespacecarb_1_1input_1a41f626f5bfc1020c9bd87f5726afdec1.html#namespacecarb_1_1input_1a41f626f5bfc1020c9bd87f5726afdec1
2. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/enum_namespacecarb_1_1input_1af1c4ed7e318b3719809f13e2a48e2f2d.html#namespacecarb_1_1input_1af1c4ed7e318b3719809f13e2a48e2f2d
3. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/docs/python/bindings.html#carb.input.GamepadInput
"""

import argparse
import copy
import os
import sys
import time
import weakref

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import cli_args  # isort: skip
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run an interactive locomotion policy demo.", allow_abbrev=False)
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
    "--input",
    type=str,
    default="keyboard",
    choices=["keyboard", "gamepad", "none"],
    help="Teleop input device. 'keyboard' reads the SSH terminal (stdin) and the Kit window if one exists.",
)
parser.add_argument(
    "--mjpeg_port",
    type=int,
    default=0,
    help=(
        "Serve the follow camera as MJPEG on this port (0 = off). Bound to localhost; "
        "reach it with an SSH tunnel: ssh -L 8080:localhost:8080 <host>."
    ),
)
parser.add_argument("--mjpeg_quality", type=int, default=80, help="JPEG quality for --mjpeg_port (1-100).")
parser.add_argument(
    "--mjpeg_every", type=int, default=2, help="Encode one frame every N sim steps (sim is 50Hz, so 2 -> 25fps)."
)
parser.add_argument(
    "--record_res",
    type=str,
    default="640x360",
    help=(
        "Follow-camera render resolution WxH for --mjpeg_port. The scene default is 960x540, which is the "
        "dominant per-step cost when streaming; lowering it raises the sim rate."
    ),
)
parser.add_argument(
    "--follow_env",
    type=int,
    default=0,
    help=(
        "Env index the keyboard drives when no prim is selected in the viewport. "
        "Set to -1 to require an explicit viewport selection (original behaviour)."
    ),
)
parser.add_argument(
    "--multicam",
    action="store_true",
    help="Record one mp4 per environment until the demo exits.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="--multicam only. Output directory (default: <checkpoint run>/videos).",
)
parser.add_argument(
    "--record_fps",
    type=int,
    default=0,
    help="Playback fps of --multicam. 0 derives it from the simulation rate.",
)
parser.add_argument(
    "--record_every",
    type=int,
    default=1,
    help="Write one frame every N simulation steps for --multicam.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    choices=[
        "gap",
        "hurdle",
        "step",
        "obstacle_course",
        "flat",
        "demo",
        "pyramid",
        "pyramid_up",
        "discrete",
        "grid",
    ],
    help=(
        "Spawn every robot on this one sub-terrain instead of the default mix. "
        "'flat' is the obstacle-free course (flat), which also flips the terrain-type "
        "flags in the observation (index 11/12) by itself."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# MJPEG streaming and multicamera recording require the TiledCamera.
if args_cli.mjpeg_port or args_cli.multicam:
    args_cli.enable_cameras = True

# LiDAR preprocessing requires cameras even without recording.
if args_cli.task is not None and "Lidar" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import carb
import isaaclab.sim as sim_utils
import numpy as np
import omni
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, wrap_to_pi
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from locomotion_isaaclab.envs import LocomotionManagerBasedRLEnv
from locomotion_tasks.locomotion_task.config.go2.agents.locomotion_rl_cfg import LocomotionRslRlOnPolicyRunnerCfg
from locomotion_tasks.default_cfg import RECORD_CAMERA_CFG
from locomotion_tasks.locomotion_task.config.go2.locomotion_em_student_cfg import UnitreeGo2LidarLocomotionEnvCfg_PLAY
from locomotion_tasks.locomotion_task.config.go2.locomotion_teacher_cfg import UnitreeGo2TeacherLocomotionEnvCfg_PLAY
from scripts.rsl_rl.checkpoint_utils import get_checkpoint_path_with_fallback
from scripts.rsl_rl.keyboard_teleop import KeyboardTeleop, KeyboardTeleopState
from scripts.rsl_rl.mjpeg_server import MjpegStreamer
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.multicam_recorder import PerEnvVideoRecorder
from scripts.rsl_rl.vecenv_wrapper import LocomotionRslRlVecEnvWrapper

# --terrain 값 -> terrain_generator.sub_terrains 의 키.
_TERRAIN_KEYS = {
    "gap": "gap",
    "hurdle": "hurdle",
    "step": "step",
    "obstacle_course": "obstacle_course",
    "flat": "flat",
    "demo": "demo",
    "pyramid": "pyramid_stairs",
    "pyramid_up": "pyramid_stairs_up",
    "discrete": "discrete_obstacles",
    "grid": "random_grid",
}


def force_single_terrain(env_cfg, choice: str):
    """모든 컬럼이 한 종류의 sub-terrain 만 쓰도록 비율을 바꾼다.

    지형 종류는 컬럼별로 정해진다(locomotion_terrain_generator.py):
        sub_index = min(where(col/num_cols + 0.001 < cumsum(정규화된 proportion)))
    고른 것만 proportion 을 남기면 그 앞의 cumsum 이 전부 0 이라 어느 컬럼이든 같은
    종류가 걸린다. 즉 --num_envs 를 몇으로 주든 전부 이 지형에 스폰된다.

    'flat'(flat)은 hurdle 지형에 apply_flat=True 를 준 것이라 장애물만 안 파이고
    goal point 8 개와 코스 길이는 그대로다. 그래서 방향 조종은 똑같이 동작한다.

    관측의 지형 타입 플래그(index 11 = non-flat, 12 = flat)는 observations.py 가
    goal_event.env_per_terrain_name 을 'flat' 과 문자열 비교해 매 스텝 만든다.
    실제로 그 지형에 스폰시키면 플래그도 저절로 뒤집히므로 obs 를 손댈 필요가 없다.
    """
    gen = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if gen is None:
        print(f"[demo] 이 씬에는 terrain_generator 가 없어 --terrain {choice} 를 무시한다.")
        return
    key = _TERRAIN_KEYS[choice]
    if key not in gen.sub_terrains:
        print(f"[demo] sub_terrains 에 '{key}' 가 없다. --terrain 을 무시한다.")
        return
    for name, sub_terrain in gen.sub_terrains.items():
        sub_terrain.proportion = 1.0 if name == key else 0.0
    # PLAY 설정의 루프는 flat 을 건너뛰어서 노이즈 폭이 기본값(0.02~0.06)으로
    # 남는다. 다른 지형과 같은 조건으로 맞춰 준다.
    gen.sub_terrains[key].noise_range = (0.02, 0.02)
    print(f"[demo] 모든 env 를 '{key}' 지형에 스폰한다.")


class LocomotionDemoGO2:
    def __init__(self):
        agent_cfg: LocomotionRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        if args_cli.use_pretrained_checkpoint:
            checkpoint = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
            if not checkpoint:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            checkpoint = retrieve_file_path(args_cli.checkpoint)
        else:
            checkpoint = get_checkpoint_path_with_fallback(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        self.agent_cfg = agent_cfg
        self.log_dir = os.path.dirname(checkpoint)
        # create envionrment
        env_cfg = (
            UnitreeGo2LidarLocomotionEnvCfg_PLAY()
            if agent_cfg.input_mode == "lidar_input"
            else UnitreeGo2TeacherLocomotionEnvCfg_PLAY()
        )
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        # 지형 생성은 env 를 만들 때 한 번만 돌아가므로 그 전에 비율을 바꿔야 한다.
        if args_cli.terrain is not None:
            force_single_terrain(env_cfg, args_cli.terrain)
        # 추격 카메라 렌더는 스트리밍 시 스텝당 비용의 대부분을 차지한다.
        # 씬 기본값 960x540 은 녹화용이고, 실시간 조종에는 과하다.
        if getattr(env_cfg.scene, "record_camera", None) is not None:
            if args_cli.mjpeg_port or args_cli.multicam:
                w, _, h = args_cli.record_res.lower().partition("x")
                env_cfg.scene.record_camera.width = int(w)
                env_cfg.scene.record_camera.height = int(h)
            else:
                # 스트리밍을 안 하면 이 카메라의 프레임을 읽는 곳이 없다. 그래도 씬에 있으면
                # 매 스텝 렌더는 그대로 돌아가므로 꺼 둔다(원래 play.py --multicam 의 녹화용이다).
                env_cfg.scene.record_camera = None
        elif args_cli.mjpeg_port or args_cli.multicam:
            env_cfg.scene.record_camera = copy.deepcopy(RECORD_CAMERA_CFG)
            w, _, h = args_cli.record_res.lower().partition("x")
            env_cfg.scene.record_camera.width = int(w)
            env_cfg.scene.record_camera.height = int(h)
        self.env_cfg = env_cfg
        # wrap around environment for rsl-rl
        self.env = LocomotionRslRlVecEnvWrapper(LocomotionManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunnerWithExtractor(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.estimator = ppo_runner.get_estimator_inference_policy(device=self.device)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self._fps_t0 = time.perf_counter()
        self._fps_steps = 0
        self._last_delta_yaw = None
        self.set_up_mjpeg()
        self.teleop = None
        if args_cli.input == "keyboard":
            self.set_up_keyboard()
        elif args_cli.input == "gamepad":
            self.set_up_gamepad()
        self._prim_selection = omni.usd.get_context().get_selection()
        # 뷰포트에서 프림을 고를 수 없는 상황(헤드리스/스트리밍)에서도 키보드가 로봇을
        # 조종할 수 있도록 기본 대상 env 를 정해 둔다. -1 이면 기존처럼 선택을 강제한다.
        self._default_id = args_cli.follow_env if args_cli.follow_env >= 0 else None
        self._selected_id = self._default_id
        self._previous_selected_id = None
        self._follow_camera_applied = False
        # self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self._camera_local_transform = torch.tensor([-0.0, 2.6, 1.6], device=self.device)
        # _selected_id 를 쓰므로 그 뒤에 만든다.
        self.set_up_teleop_marker()

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        # --headless(스트리밍도 없음)로 띄우면 뷰포트 윈도우 자체가 없어 None 이 온다.
        # 이때는 카메라 조작을 전부 건너뛴다.
        if self.viewport is None:
            print("[demo] 뷰포트가 없다(완전 헤드리스). 3인칭 카메라 전환을 건너뛴다.")
            return
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_mjpeg(self):
        """record_camera 를 붙이고, 요청된 출력(브라우저 스트리밍 / mp4)을 준비한다."""
        self.mjpeg = None
        self.record_camera = None
        self.recorder = None
        self._record_steps = 0
        if not (args_cli.mjpeg_port or args_cli.multicam):
            return
        # 스트리밍 또는 녹화를 요청하면 RGB 카메라가 씬에 추가된다.
        if "record_camera" not in self.env.unwrapped.scene.sensors:
            print("[demo] 이 태스크의 씬에는 record_camera 가 없어 스트리밍/녹화를 건너뛴다.")
            return
        self.record_camera = self.env.unwrapped.scene["record_camera"]
        if args_cli.mjpeg_port:
            self.mjpeg = MjpegStreamer(
                port=args_cli.mjpeg_port, quality=args_cli.mjpeg_quality, every=args_cli.mjpeg_every
            )
        # play.py --multicam 과 같은 좌측 측면 시점. 월드 기준 상수 오프셋이라
        # 자세는 고정된 채 평행이동만 하고, 로봇이 점프해도 화면이 기울지 않는다.
        self._record_cam_offset = torch.tensor([0.0, 2.6, 1.6], device=self.device)

    def start_multicam_recording(self):
        """Open per-environment writers after the initial reset.

        fps 는 기본적으로 시뮬 속도(1/step_dt)에서 뽑는다. 실시간 배속으로 재생되게
        하려는 것이고, --record_every 로 프레임을 솎으면 그만큼 나눠 준다.
        데모는 실측 40~70Hz 로 도는데 그건 렌더가 느려서지 시뮬 dt 가 바뀐 게 아니므로
        벽시계 속도가 아니라 step_dt 를 기준으로 삼는 것이 맞다.
        """
        if not args_cli.multicam or self.record_camera is None:
            return
        every = max(1, int(args_cli.record_every))
        fps = args_cli.record_fps or max(1, round(1.0 / self.env.unwrapped.step_dt / every))
        out_dir = args_cli.out_dir or os.path.join(self.log_dir, "videos")
        self.recorder = PerEnvVideoRecorder(self.env, out_dir, fps=fps)

    def track_record_camera(self):
        """스텝 중에 렌더가 일어나므로, step() 직전에 카메라를 현재 로봇 위치로 옮긴다."""
        if self.recorder is not None:
            self.recorder.track_camera()
            return
        if self.record_camera is None:
            return
        target = self.env.unwrapped.scene["robot"].data.root_pos_w
        self.record_camera.set_world_poses_from_view(eyes=target + self._record_cam_offset, targets=target)

    def publish_frame(self):
        """선택한 환경은 스트리밍하고, 녹화는 모든 환경의 RGB를 저장한다."""
        if self.mjpeg is None and self.recorder is None:
            return
        rgb = self.record_camera.data.output.get("rgb")
        if rgb is None:
            return
        if self.mjpeg is not None:
            idx = self._selected_id if self._selected_id is not None else 0
            frame = rgb[idx, ..., :3].detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            self.mjpeg.push(frame)
        if self.recorder is not None:
            self._record_steps += 1
            if self._record_steps % max(1, int(args_cli.record_every)) == 0:
                self.recorder.capture()

    def set_up_keyboard(self):
        """키보드 텔레오퍼레이션을 켠다.

        입력은 두 경로로 들어온다(keyboard_teleop.py 주석 참고).
          - SSH 터미널 stdin : 디스플레이가 전혀 없어도 동작
          - Kit 앱 윈도우    : GUI 또는 WebRTC 스트리밍 클라이언트로 붙었을 때 동작
        """
        self.teleop_state = KeyboardTeleopState()
        self.teleop = KeyboardTeleop(self.teleop_state)
        if not self.teleop.any_enabled:
            print("[demo] 사용할 수 있는 키보드 입력 경로가 없다. 방향은 정면(0)으로 고정된다.")

    def set_up_teleop_marker(self):
        """키보드로 정한 목표 방향을 **주황색 화살표**로 그린다.

        씬에 원래 떠 있는 화살표들은 키보드와 아무 상관이 없다.
          - 초록 화살표(goal_vel_visualizer) : command manager 가 만든 속도 명령
            self.command[:, :2] 를 **로봇 기준 프레임**으로 그린 것이다. lin_vel_y 가
            항상 0 이라 언제나 로봇의 코 방향을 가리키며, 로봇이 돌 때만 같이 돈다.
          - 파랑 화살표(current_vel_visualizer) : 실제 몸통 속도.
          - 초록 구슬 8개(current_arrow_visualizer) : 파쿠르 goal point 로 가는 점선.
        키보드는 정책 호출 직전에 obs[6],[7] 만 덮어쓰고 command manager 나 goal point 는
        건드리지 않으므로, 위 어느 것도 키 입력에 반응하지 않는다. 그래서 별도로 그린다.

        cfg 를 GREEN_ARROW_X_MARKER_CFG.replace() 로 만들면 안 된다. configclass 의
        replace 는 dataclasses.replace 라 얕은 복사이고, markers 딕셔너리가 원본과
        **공유**된다. 거기서 색을 바꾸면 씬의 진짜 초록 화살표까지 주황색이 된다.
        그래서 cfg 를 처음부터 새로 만든다(화살표 USD 자체는 같은 것을 쓴다).
        """
        self.teleop_marker = None
        if self.teleop is None:
            return
        try:
            self.teleop_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/keyboard_goal_dir",
                    markers={
                        "arrow": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
                        )
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - 마커가 없어도 조종 자체는 되어야 한다
            print(f"[demo] 키보드 방향 화살표를 만들지 못했다({exc}). 화살표 없이 계속한다.")
            self.teleop_marker = None
            return
        # 크기는 visualize(scales=...) 로 매번 넘긴다. UsdFileCfg 의 scale 은 프로토타입
        # 프림에만 붙고 PointInstancer 인스턴스에는 반영되지 않아서, 안 넘기면 배율 1 인
        # 원본 크기(로봇보다 큰 화살표)가 그대로 나온다. 속도 화살표들도 같은 이유로
        # _resolve_xy_velocity_to_arrow() 에서 scales 를 계산해 넘긴다.
        # 초록 화살표의 실효 크기가 (0.5*0.5*3*|v|, 0.25, 0.25) 이므로 그와 비슷하되
        # 조금 길게 잡아 눈에 띄게 한다. 방향만 나타내므로 길이는 고정이다.
        self._teleop_marker_scale = torch.tensor([[0.5, 0.22, 0.22]], device=self.device)

    def update_teleop_marker(self):
        """주황 화살표를 로봇 위로 옮기고 목표 방향으로 돌린다.

        target_yaw 는 월드 기준 절대 각도이므로 초록 화살표와 달리 base_quat_w 를
        곱하지 않는다. 로봇이 어느 쪽을 보든 이 화살표는 같은 세계 방향을 가리키고,
        로봇의 코와 이 화살표가 이루는 각이 곧 obs[6],[7] 에 들어가는 delta 다.

        z 오프셋은 0.8 이다. 초록/파랑 화살표가 +0.5 에 있어 겹치지 않게 띄운다.
        """
        if self.teleop_marker is None:
            return
        root_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w
        idx = slice(None) if self._selected_id is None else slice(self._selected_id, self._selected_id + 1)
        pos = root_pos_w[idx].clone()
        pos[:, 2] += 0.8
        yaw = torch.full((pos.shape[0],), self.teleop_state.target_yaw, device=self.device)
        zeros = torch.zeros_like(yaw)
        self.teleop_marker.visualize(
            translations=pos,
            orientations=quat_from_euler_xyz(zeros, zeros, yaw),
            scales=self._teleop_marker_scale.repeat(pos.shape[0], 1),
        )

    def tick_fps(self):
        """실측 시뮬 속도를 2초마다 갱신해 상태줄에 띄운다."""
        now = time.perf_counter()
        self._fps_steps += 1
        elapsed = now - self._fps_t0
        if elapsed >= 2.0:
            if self.teleop is not None:
                self.teleop_state.sim_fps = self._fps_steps / elapsed
                # obs[6],[7] 계산에 쓰인 각도 차이(rad). obs 에는 여기에 1.5 를 곱한
                # 값이 들어간다. 같은 단위인 goal= 과 바로 비교하려고 배율 없는 쪽을
                # 보여 준다. 매 스텝 갱신하면 상태줄이 쉴 새 없이 다시 그려지므로
                # 여기서 2초에 한 번만 반영한다.
                self.teleop_state.delta_yaw = self._last_delta_yaw
                self.teleop_state.dirty = True
            self._fps_t0 = now
            self._fps_steps = 0

    def apply_teleop(self):
        """키 입력을 읽고, 일회성 요청(리셋/카메라)을 처리한다.

        방향값은 정책 호출 직전에 apply_teleop_yaw()에서 관측에 반영한다.
        """
        if self.teleop is None:
            return
        self.teleop.poll()
        state = self.teleop_state

        if state.align_requested:
            state.align_requested = False
            # space: 목표 방향을 로봇이 지금 향한 쪽으로 맞춘다 -> delta 가 0 이 되어 직진.
            state.set_target_yaw(float(self.robot_heading()[self._selected_id or 0]))
        if state.reset_requested:
            state.reset_requested = False
            self.env.reset()
        if state.toggle_camera_requested:
            state.toggle_camera_requested = False
            if self.viewport is not None:
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)

    def robot_heading(self):
        """로봇의 현재 진행 방향(월드 기준 yaw, rad). Shape (num_envs,).

        관측 코드가 쓰는 wrap_to_pi(euler_xyz_from_quat(root_quat_w)[2]) 와 같은 값이다
        (heading_w = atan2(forward_w.y, forward_w.x)). 파쿠르 command term 도 이걸 쓴다.
        """
        return self.env.unwrapped.scene["robot"].data.heading_w

    def apply_teleop_yaw(self, obs):
        """obs 의 index 6,7 을 사용자가 지정한 목표 방향 기준으로 다시 계산해 **덮어쓴다**.

        원본 관측에서 6,7 은 각각 delta_yaw / delta_next_yaw 이고,
            delta = (goal point 방향의 월드 각도) - (로봇의 현재 진행 각도)
        로 만들어진다. 여기서는 사용자가 방향키로 정한 목표 방향을
        goal point 방향으로 간주해 같은 식으로 다시 계산한다.

        index 7(원래는 '다음 goal' 방향)은 요청대로 index 6 과 같은 값을 넣는다.

        차이는 wrap_to_pi 로 감아 항상 최단 회전 방향이 되게 한다. 원본 관측은
        차이 자체를 감지 않아 +-2pi 까지 나올 수 있지만, 조종 입력으로는 최단 회전이
        맞고 값의 범위도 학습 시 범위 안에 들어온다.

        delta 는 순수한 각도(rad)로 두고, obs 에 넣을 때만 원래 코드의
        `obs[:, 6:8] = 1.5*yaw` 와 같은 배율 1.5 를 곱한다.

        공통 관측 생성부와 동일한 배율이며, 정책 호출 직전에 조종 방향으로 덮어쓴다.
        """
        if self.teleop is None:
            return
        delta = wrap_to_pi(self.teleop_state.target_yaw - self.robot_heading())
        # 정수 인덱스 대신 길이 1 슬라이스를 써서 delta 와 차원을 맞춘다.
        idx = slice(None) if self._selected_id is None else slice(self._selected_id, self._selected_id + 1)
        obs[idx, 6] = 1.5 * delta[idx]
        obs[idx, 7] = 1.5 * delta[idx]
        self._last_delta_yaw = float(delta[self._selected_id or 0])

    def close(self):
        if self.teleop is not None:
            self.teleop.close()
            self.teleop = None
        if getattr(self, "mjpeg", None) is not None:
            self.mjpeg.close()
            self.mjpeg = None
        if getattr(self, "recorder", None) is not None:
            self.recorder.close()
            self.recorder = None

    def set_up_gamepad(self):
        self._input = carb.input.acquire_input_interface()
        self._gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self.dead_zone = 0.01
        self.v_x_sensitivity = 0.8
        self.v_y_sensitivity = 0.8
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: self.env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: self.env_cfg.commands.base_velocity.ranges.lin_vel_x[0],
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: self.env_cfg.commands.base_velocity.ranges.heading[0],
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: self.env_cfg.commands.base_velocity.ranges.heading[1],
        }

    def _on_gamepad_event(self, event):
        if event.type == carb.input.GamepadConnectionEventType.CONNECTED:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            cur_val = event.value
            if abs(cur_val) < self.dead_zone:
                cur_val = 0
            if event.input in self._INPUT_STICK_VALUE_MAPPING:
                if self._selected_id:
                    value = self._INPUT_STICK_VALUE_MAPPING[event.input.name]
                    self.commands[self._selected_id] = value * cur_val
            # Escape key exits out of the current selected robot view
            elif event.input == "LEFT_SHOULDER":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input == "RIGHT_SHOULDER":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.GamepadConnectionEventType.DISCONNECTED:
            if self._selected_id:
                self.commands[self._selected_id] = torch.zeros(1, 3).to(self.device)

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            # 아무것도 선택되지 않은 상태. --follow_env 가 주어졌으면 그 env 를 계속 따라간다.
            self._selected_id = self._default_id
            if self._selected_id is None:
                if self.viewport is not None:
                    self.viewport.set_active_camera(self.perspective_path)
            else:
                # --follow_env 로 대상이 정해져 있으면 추격 카메라를 실제로 켜 준다.
                # 예전에는 카메라 위치만 갱신하고 뷰포트는 Perspective 에 머물러서,
                # GUI 로 띄우면 로봇이 화면 밖에 있어 아무것도 안 보였다.
                # 딱 한 번만 전환한다. 그래야 사용자가 c 로 되돌리거나 뷰포트를
                # 직접 돌려 놓은 것을 매 프레임 덮어쓰지 않는다.
                if not self._follow_camera_applied and self.viewport is not None:
                    self._follow_camera_applied = True
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id and self.viewport is not None:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a GO2 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""
        if self.viewport is None:
            return

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_go2 = LocomotionDemoGO2()
    actor_param = demo_go2.agent_cfg.policy.actor
    num_priv_explicit = actor_param.num_priv_explicit
    num_scan = actor_param.num_scan
    num_prop = actor_param.num_prop
    obs, extras = demo_go2.env.reset()
    demo_go2.start_multicam_recording()
    # 관측의 지형 타입 플래그. observations.py 가 이제 지형과 무관하게
    #   index 11 = 1 (non-flat 고정), index 12 = 0 (flat 고정)
    # 으로 만든다 — 실기에는 flat 감지 오라클이 없어 정책이 이 신호에 의존하지
    # 않게 상수화했다. --terrain flat 은 지형 기하만 바꾸고 이 플래그는 안 바뀐다.
    print(f"[demo] obs 지형 타입 플래그(상수)  non-flat(11)={obs[:, 11].tolist()}  flat(12)={obs[:, 12].tolist()}")
    try:
        run_loop(demo_go2, obs, extras, num_prop, num_scan, num_priv_explicit)
    finally:
        demo_go2.close()
        # env.close() 없이 곧장 simulation_app.close() 로 가면 Kit 이 렌더 프로덕트를
        # 붙잡은 채 종료 루프를 무한히 돈다(메인 스레드가 CPU 100% 로 R 상태). 원래
        # 데모는 루프를 빠져나올 방법 자체가 없어 드러나지 않던 문제고, play.py 는
        # 처음부터 env.close() 를 먼저 부른다. 같은 순서를 맞춘다.
        demo_go2.env.close()


def run_loop(demo_go2, obs, extras, num_prop, num_scan, num_priv_explicit):
    while simulation_app.is_running():
        # 키 입력을 읽는다(리셋/카메라/종료 요청도 여기서 처리).
        demo_go2.apply_teleop()
        if demo_go2.teleop is not None and demo_go2.teleop_state.quit_requested:
            break
        # check for selected robots
        demo_go2.update_selected_object()
        # 마커 갱신은 USD 쓰기라 inference_mode 블록 밖에서 한다.
        demo_go2.update_teleop_marker()
        with torch.inference_mode():
            # obs index 9(command velocity)는 건드리지 않는다. 환경의 command manager 가
            # 정한 값이 관측에 이미 들어 있고, 사용자는 방향만 조작한다.
            # 게임패드 모드는 기존 동작을 그대로 유지한다.
            if args_cli.input == "gamepad":
                obs[:, 9] = demo_go2.commands[:, 0]
            priv_states_estimated = demo_go2.estimator.inference(obs[:, :num_prop])
            obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = priv_states_estimated
            demo_go2.apply_teleop_yaw(obs)
            action = demo_go2.policy(obs, hist_encoding=True)
            demo_go2.track_record_camera()
            obs, _, _, extras = demo_go2.env.step(action)
            demo_go2.publish_frame()
            demo_go2.tick_fps()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
