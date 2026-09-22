"""elevation-map student 환경 cfg (계획서 docs/emcupy_student_plan.md).

depth student(parkour_student_cfg.py)에서 파생하되:
- depth camera → L1 LiDAR(GO2_LIDAR_CFG) 교체
- 관측 그룹 depth_camera → em_scan(132) 교체
- 카메라 자세 랜덤화 이벤트 제거 (카메라가 없다)
지형/보상/커리큘럼 등 나머지는 depth student 와 동일하게 유지해
두 student 를 같은 조건에서 비교할 수 있게 한다.
"""

import os

from isaaclab.utils import configclass

from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
from parkour_tasks.default_cfg import GO2_LIDAR_CFG, VIEWER

from .parkour_mdp_cfg import *
from .parkour_student_cfg import ParkourStudentSceneCfg, UnitreeGo2StudentParkourEnvCfg


@configclass
class ParkourEMStudentSceneCfg(ParkourStudentSceneCfg):
    # depth 파이프라인 제거, L1 LiDAR 장착
    depth_camera = None
    depth_camera_usd = None
    lidar = GO2_LIDAR_CFG


@configclass
class UnitreeGo2EMStudentParkourEnvCfg(UnitreeGo2StudentParkourEnvCfg):
    scene: ParkourEMStudentSceneCfg = ParkourEMStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: EMStudentObservationsCfg = EMStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # SensorBase 는 lazy 갱신이라 update_period 는 "이보다 오래된 데이터면 재계산"
        # 문턱이다. 소비는 em_scan 이 10Hz(%5)로만 하므로 프레임(0.1s)이 정확히
        # 타일링된다 (계획서 §2.2).
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        # depth camera 가 없으니 카메라 자세 랜덤화도 없다.
        self.events.random_camera_position = None


@configclass
class UnitreeGo2EMStudentParkourEnvCfg_EVAL(UnitreeGo2EMStudentParkourEnvCfg):
    """depth student 의 EVAL 설정에서 카메라 관련 항목만 뺀 것."""

    viewer = VIEWER
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.0
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)

        apply_terrain_preset(
            self.scene.terrain.terrain_generator,
            "all_obstacles",
            active_overrides={"noise_range": (0.02, 0.02)},
        )


@configclass
class UnitreeGo2EMStudentParkourEnvCfg_PLAY(UnitreeGo2EMStudentParkourEnvCfg_EVAL):
    """depth student 의 PLAY 기반 지형 설정 (parkour_student_cfg 참조).

    난이도는 여기서만 다르다 — 기본을 (0.7, 0.7) 고정 + num_rows=1 로 두어
    모든 env 가 같은 난이도(단차 17.6cm)를 밟는다. PARKOUR_DIFFICULTY 환경변수로
    다른 값을 고정할 수 있다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 60.0

        if self.scene.terrain.terrain_generator is not None:
            _d = os.environ.get("PARKOUR_DIFFICULTY")
            if _d is not None:
                self.scene.terrain.terrain_generator.difficulty_range = (float(_d), float(_d))
            else:
                # 고정 난이도 0.7 — row 가 1개뿐이라 모든 env 가 같은 난이도다.
                self.scene.terrain.terrain_generator.difficulty_range = (0.7, 0.7)
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.size = (24.0, 4.0)
        self.events.push_by_setting_velocity = None
        self.events.push_angular_velocity = None
        apply_terrain_preset(
            self.scene.terrain.terrain_generator,
            "trapezoid_only",
            one_col_per_terrain=True,
            active_overrides={"noise_range": (0.02, 0.02)},
        )
