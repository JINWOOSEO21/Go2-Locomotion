import os

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
# isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import  CAMERA_USD_CFG, CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourTeacherSceneCfg
@configclass
class ParkourStudentSceneCfg(ParkourTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None

    # 녹화 전용 추격 카메라. env 마다 1대씩 생기고 로봇 base 에 붙어 따라다닌다.
    # TiledCamera 는 모든 env 를 한 번의 렌더 패스로 처리하므로, env 별 카메라를
    # 따로 두는 것보다 싸다. data.output["rgb"] 가 (num_envs, H, W, 3) 로 나온다.
    #
    # 로봇의 자식이 아니라 env 바로 아래에 둔다. 로봇 base 에 부착하면 몸체의
    # pitch/roll/yaw 를 그대로 물려받아 점프할 때마다 화면이 같이 기운다.
    # 여기서는 자세를 고정하고 위치만 따라가야 하므로, play_multicam.py 가 매 스텝
    # camera.set_world_poses(로봇위치 + 고정오프셋, 고정쿼터니언) 로 갱신한다.
    # 아래 offset 은 스폰 시 초기값일 뿐이고 이후에는 스크립트가 덮어쓴다.
    #
    # convention="world": +X 가 카메라 정면, +Z 가 위. rot 은 Y축 회전(하향 pitch).
    # 로봇 뒤(-X) 3.0m, 위(+Z) 1.3m 이면 base 가 atan(1.3/3.0)=23.4도 아래에 보이고,
    # 20도 숙이면 로봇이 화면 중앙 살짝 아래 + 전방 지형도 들어온다.
    # (처음엔 10도로 뒀다가 로봇이 화면 하단에 잘려 20도로 키웠다.)
    record_camera = TiledCameraCfg(
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
        # 여기는 env 마다 1대라 같은 해상도면 픽셀 수가 4배가 되고,
        # 실측상 1280x720x4 는 CUDA OOM 이 났다(가용 6.8GiB, 필요 7GiB+).
        # 960x540 은 1280x720 의 56%, 기존 640x360 의 2.25배.
        # GPU 가 비면 1280x720 으로 올릴 수 있다.
        width=960,
        height=540,
    )


    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        # 기하 관련 설정은 분포와 무관하게 전 지형에 적용한다.
        for sub_terrain in self.terrain.terrain_generator.sub_terrains.values():
            sub_terrain: ExtremeParkourRoughTerrainCfg
            sub_terrain.use_simplified = True
            sub_terrain.horizontal_scale = 0.1
        # 지형 분포는 parkour.py 의 TERRAIN_PRESETS 에서 관리한다.
        # 이식 지형 4종을 포함해 재학습하려면 student_train 프리셋을 고칠 것.
        apply_terrain_preset(self.terrain.terrain_generator, "student_train")
        # gap/hurdle/step 은 코스 중심선을 살짝 흔든다 ('parkour' 는 자체 y_range 를 쓴다).
        for key in ('parkour_gap', 'parkour_hurdle', 'parkour_step'):
            self.terrain.terrain_generator.sub_terrains[key].y_range = (-0.1, 0.1)



@configclass
class UnitreeGo2StudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourStudentSceneCfg = ParkourStudentSceneCfg(num_envs=192, env_spacing=1.)
    # Basic settings
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # PhysX 는 이 값에 비례해 contact 버퍼를 고정 할당한다 (contact 당 80 byte).
        # IsaacLab 기본값 2**23 은 640MB 를 한 번에 잡아서 VRAM 이 빠듯하면 OOM 이 난다
        # (Unable to allocate memory of size 671088640 for mGpuContactPairsDev).
        # 2**20 이면 80MB 이고, 소수 환경의 사족보행 + parkour 지형에는 충분하다.
        # 환경 수를 크게 늘릴 때 contact buffer overflow 경고가 뜨면 이 값을 다시 올릴 것.
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        # update sensor update periods
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8



@configclass
class UnitreeGo2StudentParkourEnvCfg_EVAL(UnitreeGo2StudentParkourEnvCfg):
    viewer = VIEWER 
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.commands.base_velocity.debug_vis = True

        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None

        self.observations.depth_camera.depth_cam.params['debug_vis'] = True

        self.commands.base_velocity.resampling_time_range = (60.,60.)
        self.commands.base_velocity.debug_vis = True

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        self.events.random_camera_position.params['rot_noise_range'] = {'pitch':(0, 1)}
        
        # 실전 장애물 8종 균등 (flat/demo 제외) — 기존 else 블록과 같은 분포다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator, "all_obstacles",
            active_overrides={"noise_range": (0.02, 0.02)},
        )

@configclass
class UnitreeGo2StudentParkourEnvCfg_PLAY(UnitreeGo2StudentParkourEnvCfg_EVAL):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 60.

        if self.scene.terrain.terrain_generator is not None:
            # 난이도 스윕용. PARKOUR_DIFFICULTY 를 주면 그 값으로 고정된다.
            # 지형 생성은 difficulty = lower + (upper-lower)*t 이므로
            # 상하한을 같게 두면 t 와 무관하게 정확히 그 값이 나온다.
            _d = os.environ.get("PARKOUR_DIFFICULTY")
            if _d is not None:
                self.scene.terrain.terrain_generator.difficulty_range = (float(_d), float(_d))
            else:
                self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
            # 기본 16m 지형은 원본 설정에서도 넘칠 수 있다.
            #   필요 길이 = platform_len(2.5) + (num_goals-1=7) * 최대간격
            #   gap    difficulty 1.0 -> 최대간격 1.5+0.8 = 2.3 -> 18.6m
            #   hurdle x_range (1.2, 2.2)         -> 2.2      -> 17.9m
            # 넘치면 장애물은 numpy 슬라이싱이 빈 슬라이스가 되어 그려지지 않는데
            # 중간 goal 은 클램프되지 않아 지형 밖에 찍힌다(= 태스크 파손).
            # 24m 면 5종 지형 모두 잘림 없이 들어간다. generator.size 는
            # super().__init__ 에서 sub_cfg.size 로 전파된다.
            self.scene.terrain.terrain_generator.size = (24.0, 4.0)
            # num_rows 는 부모(EVAL)의 5 를 그대로 쓴다 = 커리큘럼 5단계.
            # VRAM 이 부족할 때는 여기서 줄일 수 있다. difficulty_range 를 (d, d) 로
            # 고정한 경우에는 모든 행이 같은 난이도라 로봇이 겪는 코스가 달라지지 않는다.
            # 다만 난이도를 범위로 쓸 때는 커리큘럼 단계 수가 줄어드니 주의.
        self.events.push_by_setting_velocity = None
        # 실전 장애물 8종 균등 + 로봇 1마리당 1종 배정.
        # one_col_per_terrain=True 가 num_cols 를 활성 지형 수에 자동으로 맞춰 주므로
        # (커리큘럼 컬럼→지형 매핑이 1:1 로 떨어지는 조건), 프리셋에 지형을 추가하면
        # 컬럼 수도 같이 늘어난다. 예전처럼 num_cols 를 따로 맞출 필요가 없다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator, "all_obstacles",
            one_col_per_terrain=True,
            active_overrides={"noise_range": (0.02, 0.02)},
        )

