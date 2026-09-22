from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_isaaclab.sensors import L1ScanRayCasterCfg
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter


@configclass
class ParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type=ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.actuators["base_legs"] = ParkourDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 35.0,
                ".*_thigh_joint": 40.0,
                ".*_calf_joint": 40.0,
            },
            saturation_effort={
                ".*_hip_joint": 35.0,
                ".*_thigh_joint": 45.0,
                ".*_calf_joint": 45.0,
            },
            velocity_limit={
                ".*_hip_joint": 52.4,
                ".*_thigh_joint": 30.1,
                ".*_calf_joint": 30.1,
            },
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        )


# LiDAR 정책 씬의 L1 LiDAR (elevation_map_scan 관측이 elevation map 을 만든다).
# lidar 브랜치의 검증된 설정을 그대로 가져왔다:
# - 반구(수평 360° x 수직 90°) 패턴, 마운트는 완전 하향: 돔 축(+Z)을 아래로
#   (X축 180° 회전). 실기 extrinsic 확정 시 이 rot 만 교체.
# - 레이 방향은 unilidar_sdk 의 이중 모터 운동학(L1ScanRayCaster)으로 매 갱신 계산.
# - 로봇 자신은 collisions 프리미티브로 캐스트 타깃에 포함한다 (self-occlusion).
#   .*/visuals 는 Head_upper/lower 에 visual 메시가 없어(17 vs rigid body 19)
#   추적 뷰의 1:1 대조에서 RuntimeError 가 난다.
# - update_mesh_ids 는 켜지 않는다: range 만 쓰므로 불필요하고, 켜면 (N,B,1) 버퍼
#   vs (N,B) 반환의 업스트림 shape 버그에 걸린다.
GO2_LIDAR_CFG = L1ScanRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=L1ScanRayCasterCfg.OffsetCfg(
        pos=(0.28, 0.0, 0.10),
        rot=(0.0, 1.0, 0.0, 0.0),
    ),
    ray_alignment="base",
    # 프레임당 유효 포인트 수 (기본 2,160 = 21,600 pts/s x 0.1 s).
    rays_per_frame=2160,
    mesh_prim_paths=[
        "/World/ground",
        L1ScanRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Robot/.*/collisions",
            track_mesh_transforms=True,
        ),
    ],
    max_distance=10.0,
    debug_vis=False,
)

# 녹화 전용 추격 카메라. env 마다 1대씩 생기고 로봇을 따라다닌다.
# TiledCamera 는 모든 env 를 한 번의 렌더 패스로 처리하므로, env 별 카메라를
# 따로 두는 것보다 싸다. data.output["rgb"] 가 (num_envs, H, W, 3) 로 나온다.
#
# 로봇의 자식이 아니라 env 바로 아래에 둔다. 로봇 base 에 부착하면 몸체의
# pitch/roll/yaw 를 그대로 물려받아 점프할 때마다 화면이 같이 기운다.
# 여기서는 자세를 고정하고 위치만 따라가야 하므로, play.py --multicam 이 매 스텝
# camera.set_world_poses_from_view(로봇위치 + 고정오프셋, 로봇위치) 로 갱신한다.
# 아래 offset 은 스폰 시 초기값일 뿐이고 이후에는 스크립트가 덮어쓴다.
#
# convention="world": +X 가 카메라 정면, +Z 가 위. rot 은 Y축 회전(하향 pitch).
# 로봇 뒤(-X) 3.0m, 위(+Z) 1.3m 이면 base 가 atan(1.3/3.0)=23.4도 아래에 보이고,
# 20도 숙이면 로봇이 화면 중앙 살짝 아래 + 전방 지형도 들어온다.
# (처음엔 10도로 뒀다가 로봇이 화면 하단에 잘려 20도로 키웠다.)
#
# 이 카메라는 씬에 있기만 해도 --enable_cameras 가 필요하고, env 수만큼 렌더가
# 돌아 VRAM 과 속도를 먹는다. 녹화할 때만 씬에 추가한다.
RECORD_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/record_cam",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-3.0, 0.0, 1.3),
        rot=(0.9848078, 0.0, 0.1736482, 0.0),
        convention="world",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 60.0),
    ),
    # 환경별 RGB 녹화 해상도.
    width=960,
    height=540,
)

VIEWER = ViewerCfg(
    eye=(-0.0, 2.6, 1.6),
    asset_name="robot",
    origin_type="asset_root",
)
