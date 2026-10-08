from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from locomotion_isaaclab.terrains.locomotion_terrains.config.locomotion import (  # isort: skip
    LOCOMOTION_TERRAINS_CFG,
    apply_terrain_preset,
)
from locomotion_isaaclab.envs import LocomotionManagerBasedRLEnvCfg
from locomotion_tasks.default_cfg import VIEWER, LocomotionDefaultSceneCfg

from .locomotion_mdp_cfg import *


@configclass
class LocomotionTeacherSceneCfg(LocomotionDefaultSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        # attach_yaw_only=True 와 동일한 동작이지만 이쪽이 후속 API 다.
        # 구 파라미터를 쓰면 IsaacLab 이 _update_ray_infos() 안에서, 즉 센서가 갱신되는
        # 매 스텝마다 deprecation 경고를 찍는다(초당 50줄). 동작에는 영향이 없고 로그만 더럽힌다.
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2, track_air_time=True, debug_vis=False, force_threshold=1.0
    )
    # 지형별 영상 녹화용 추격 카메라 자리. 기본은 None 이다.
    # InteractiveScene._add_entities_from_cfg 가 None 인 필드는 건너뛰므로
    # 학습(4096 env)/EVAL/play.py 경로에는 카메라가 아예 생기지 않는다.
    # play.py --multicam 이 녹화할 때만 default_cfg.RECORD_CAMERA_CFG 를 꽂아 넣는다.
    # (필드를 미리 선언해 두는 이유는 configclass 인스턴스에 없는 속성을 나중에
    #  붙이는 것보다 이쪽이 명시적이고 to_dict 등에서도 안전하기 때문이다.)
    record_camera = None

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = LOCOMOTION_TERRAINS_CFG


@configclass
class UnitreeGo2TeacherLocomotionEnvCfg(LocomotionManagerBasedRLEnvCfg):
    scene: LocomotionTeacherSceneCfg = LocomotionTeacherSceneCfg(num_envs=4096, env_spacing=1.0)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    goals: GoalEventsCfg = GoalEventsCfg()
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
        # update sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        # 지형 분포는 locomotion.py 의 TERRAIN_PRESETS 한 곳에서만 관리한다.
        # LOCOMOTION_TERRAINS_CFG 는 모듈 레벨 공유 객체라 다른 env cfg 가
        # 먼저 proportion 을 바꿨을 수 있으므로, 기본값에 기대지 말고 항상 명시한다.
        apply_terrain_preset(self.scene.terrain.terrain_generator, "trapezoid_train")
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1


@configclass
class UnitreeGo2TeacherLocomotionEnvCfg_EVAL(UnitreeGo2TeacherLocomotionEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.0
        self.goals.base_goal.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        # 학습(trapezoid_train)과 같은 3종을 균등하게 평가한다.
        # 여기가 "eval_core"(원조 4종)였는데, 지금 teacher 는 그 4종을 한 번도
        # 본 적이 없으므로 학습 안 한 지형 위에서 평가하는 꼴이었다.
        # 원조 4종 체크포인트를 평가할 때는 "eval_core" 로 되돌릴 것.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator,
            "trapezoid_train_all",
            active_overrides={"noise_range": (0.02, 0.02)},
        )


@configclass
class UnitreeGo2TeacherLocomotionEnvCfg_PLAY(UnitreeGo2TeacherLocomotionEnvCfg_EVAL):
    viewer = VIEWER

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 16
        self.goals.base_goal.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            # 상하한을 같게 두면 random_difficulty 의 샘플링과 무관하게 정확히 이 값이 된다
            # (difficulty = lower + (upper-lower)*t). row 도 1개로 줄여 모든 env 가
            # 같은 난이도의 같은 행에 선다.
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 0.7)
            self.scene.terrain.terrain_generator.num_rows = 1
        self.events.push_by_setting_velocity = None
        self.events.push_angular_velocity = None
        # 학습에 실제로 쓰는 3종(사다리꼴 램프/계단 + flat)만 균등하게 본다.
        # one_col_per_terrain=True 가 num_cols 를 활성 지형 수(=3)에 맞춰
        # 커리큘럼 컬럼→지형 매핑을 1:1 로 떨어뜨린다. 부모(EVAL)가 잡아 둔
        # num_cols=5 는 여기서 덮어써진다.
        # 사다리꼴 2종만 보고 싶으면 "trapezoid_only" 로 바꾸면 된다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator,
            "trapezoid_train_all",
            one_col_per_terrain=True,
            active_overrides={"noise_range": (0.02, 0.02)},
        )
