
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import (  # isort: skip
    EXTREME_PARKOUR_TERRAINS_CFG,
    apply_terrain_preset,
)
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER

@configclass
class ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        
@configclass
class UnitreeGo2TeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourTeacherSceneCfg = ParkourTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
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
        # update sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        # 지형 분포는 parkour.py 의 TERRAIN_PRESETS 한 곳에서만 관리한다.
        # EXTREME_PARKOUR_TERRAINS_CFG 는 모듈 레벨 공유 객체라 다른 env cfg 가
        # 먼저 proportion 을 바꿨을 수 있으므로, 기본값에 기대지 말고 항상 명시한다.
        apply_terrain_preset(self.scene.terrain.terrain_generator, "trapezoid_train")
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

@configclass
class UnitreeGo2TeacherParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        self.commands.base_velocity.resampling_time_range = (60.,60.)
        # 원조 4종 균등. 예전 코드는 `if key == [리스트]` 라 조건이 항상 False 여서
        # 이 블록이 한 번도 실행되지 않았다 (= 학습 분포 그대로 평가되고 있었다).
        # 프리셋으로 바꾸면서 원래 의도(4종 0.25 + noise 고정)대로 살렸다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator, "eval_core",
            active_overrides={"noise_range": (0.02, 0.02)},
        )

@configclass
class UnitreeGo2TeacherParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.
        self.scene.num_envs = 16
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
        self.events.push_by_setting_velocity = None
        # flat 만 빼고 전부 균등 (demo + 이식 지형 4종 포함) — 기존 else 블록과 같은 분포다.
        apply_terrain_preset(
            self.scene.terrain.terrain_generator, "all_with_demo",
            active_overrides={"noise_range": (0.02, 0.02)},
        )


