import os

from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
# isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import  CAMERA_USD_CFG, CAMERA_CFG, RECORD_CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourTeacherSceneCfg
@configclass
class ParkourStudentSceneCfg(ParkourTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None

    # 녹화 전용 추격 카메라. 정의는 default_cfg.RECORD_CAMERA_CFG 한 곳에 있고
    # teacher(play.py --multicam 이 주입)와 student 가 같은 화각을 쓴다.
    # student 씬은 예전부터 이걸 상시로 들고 있고, 학습 경로에서는 train.py 가
    # env_cfg.scene.record_camera = None 으로 떼어낸다.
    record_camera = RECORD_CAMERA_CFG


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
        # 원래의 원조 5종 분포로 되돌리려면 "student_train" 으로 바꾸면 된다.
        apply_terrain_preset(self.terrain.terrain_generator, "trapezoid_train")
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

        # oracle goal 마커(현재 goal 초록/이후 goal 빨강 구)와 그 방향을 가리키는
        # 점선(작은 구 8 개)을 그린다. student 도 teacher 와 똑같이 oracle heading 을
        # 관측으로 받으므로 teacher PLAY/EVAL 과 같은 것을 보여 준다.
        self.parkours.base_parkour.debug_vis = True

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
                self.scene.terrain.terrain_generator.difficulty_range = (0.7, 0.7)
            # 기본 16m 지형은 원본 설정에서도 넘칠 수 있다.
            #   필요 길이 = platform_len(2.5) + (num_goals-1=7) * 최대간격
            #   gap    difficulty 1.0 -> 최대간격 1.5+0.8 = 2.3 -> 18.6m
            #   hurdle x_range (1.2, 2.2)         -> 2.2      -> 17.9m
            # 넘치면 장애물은 numpy 슬라이싱이 빈 슬라이스가 되어 그려지지 않는데
            # 중간 goal 은 클램프되지 않아 지형 밖에 찍힌다(= 태스크 파손).
            # 24m 면 5종 지형 모두 잘림 없이 들어간다. generator.size 는
            # super().__init__ 에서 sub_cfg.size 로 전파된다.
            self.scene.terrain.terrain_generator.size = (24.0, 4.0)
            # 난이도가 (d, d) 로 고정이므로 행을 여러 개 둘 이유가 없다.
            # row 1개 = 모든 env 가 같은 난이도의 같은 행에 선다.
            self.scene.terrain.terrain_generator.num_rows = 1
        self.events.push_by_setting_velocity = None
        # 학습 지형(사다리꼴 2종)만 균등. flat 은 학습 보조라 뺀다.
        # one_col_per_terrain=True 가 num_cols 를 활성 지형 수(=2)에 자동으로 맞춘다
        # (커리큘럼 컬럼→지형 매핑이 1:1 로 떨어지는 조건). 프리셋에 지형을 더하면
        # 컬럼 수도 같이 늘어나므로 num_cols 를 따로 맞출 필요가 없다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator, "trapezoid_only",
            one_col_per_terrain=True,
            active_overrides={"noise_range": (0.02, 0.02)},
        )

