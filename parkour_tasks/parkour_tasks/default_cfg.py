from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_tasks.extreme_parkour_task.config.go2 import agents 
from isaaclab.sensors import RayCasterCameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch 
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg

def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())

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
        class_type= ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
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
        self.robot.actuators['base_legs'] = ParkourDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                        '.*_hip_joint':35.0,
                        '.*_thigh_joint':40.0,
                        '.*_calf_joint':40.0,
                        },
            saturation_effort={
                        '.*_hip_joint':35.0,
                        '.*_thigh_joint':45.0,
                        '.*_calf_joint':45.0,
                        },
            velocity_limit={
                        '.*_hip_joint':52.4,
                        '.*_thigh_joint':30.1,
                        '.*_calf_joint':30.1,
                        },
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        )

## we are now using a raycaster based camera, not a pinhole camera. see tail issue https://github.com/isaac-sim/IsaacLab/issues/719
CAMERA_CFG = RayCasterCameraCfg( 
    prim_path= '{ENV_REGEX_NS}/Robot/base',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08), 
        rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,70,-90])))), 
        convention="ros"
        ),
    depth_clipping_behavior = 'max',
    pattern_cfg = PinholeCameraPatternCfg(
        focal_length=11.041, 
        horizontal_aperture=20.955,
        vertical_aperture = 12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance = 2.,
)

CAMERA_USD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0],'d435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.33, 0.0, 0.08), 
            rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,90,-90]))))
    )
)
# 녹화 전용 추격 카메라. env 마다 1대씩 생기고 로봇을 따라다닌다.
# TiledCamera 는 모든 env 를 한 번의 렌더 패스로 처리하므로, env 별 카메라를
# 따로 두는 것보다 싸다. data.output["rgb"] 가 (num_envs, H, W, 3) 로 나온다.
#
# 로봇의 자식이 아니라 env 바로 아래에 둔다. 로봇 base 에 부착하면 몸체의
# pitch/roll/yaw 를 그대로 물려받아 점프할 때마다 화면이 같이 기운다.
# 여기서는 자세를 고정하고 위치만 따라가야 하므로, play_multicam.py 가 매 스텝
# camera.set_world_poses_from_view(로봇위치 + 고정오프셋, 로봇위치) 로 갱신한다.
# 아래 offset 은 스폰 시 초기값일 뿐이고 이후에는 스크립트가 덮어쓴다.
#
# convention="world": +X 가 카메라 정면, +Z 가 위. rot 은 Y축 회전(하향 pitch).
# 로봇 뒤(-X) 3.0m, 위(+Z) 1.3m 이면 base 가 atan(1.3/3.0)=23.4도 아래에 보이고,
# 20도 숙이면 로봇이 화면 중앙 살짝 아래 + 전방 지형도 들어온다.
# (처음엔 10도로 뒀다가 로봇이 화면 하단에 잘려 20도로 키웠다.)
#
# 이 카메라는 씬에 있기만 해도 --enable_cameras 가 필요하고, env 수만큼 렌더가
# 돌아 VRAM 과 속도를 먹는다. 그래서 teacher 씬은 record_camera=None 으로 두고
# play_multicam.py 가 녹화할 때만 이 cfg 를 꽂아 넣는다. student 씬은 예전부터
# 상시로 들고 있고, train.py 가 학습 경로에서 떼어낸다.
RECORD_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/record_cam",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-3.0, 0.0, 1.3),
        rot=(0.9848078, 0.0, 0.1736482, 0.0),
        convention="world",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 60.0),
    ),
    # play.py 의 뷰포트 녹화는 1280x720 이지만 그건 카메라 1대 기준이다.
    # 여기는 env 마다 1대라 같은 해상도면 픽셀 수가 env 수만큼 늘고,
    # 실측상 1280x720x4 는 CUDA OOM 이 났다(가용 6.8GiB, 필요 7GiB+).
    # 960x540 은 1280x720 의 56%, 기존 640x360 의 2.25배.
    # GPU 가 비면 1280x720 으로 올릴 수 있다.
    width=960,
    height=540,
)

VIEWER = ViewerCfg(
    eye=(-0., 2.6, 1.6),
    asset_name = "robot",
    origin_type = 'asset_root',
)
