"""LiDAR-input environments using the corresponding Scandots task settings."""

from isaaclab.utils import configclass

from parkour_tasks.default_cfg import GO2_LIDAR_CFG

from .parkour_mdp_cfg import LidarObservationsCfg
from .parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
)


@configclass
class ParkourLidarSceneCfg(ParkourTeacherSceneCfg):
    lidar = GO2_LIDAR_CFG


@configclass
class UnitreeGo2LidarParkourEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    scene: ParkourLidarSceneCfg = ParkourLidarSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2LidarParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    scene: ParkourLidarSceneCfg = ParkourLidarSceneCfg(num_envs=256, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2LidarParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    scene: ParkourLidarSceneCfg = ParkourLidarSceneCfg(num_envs=16, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


# Keep old config symbols importable for serialized configs without exposing them
# through the new task registrations.
ParkourEMStudentSceneCfg = ParkourLidarSceneCfg
UnitreeGo2EMStudentParkourEnvCfg = UnitreeGo2LidarParkourEnvCfg
UnitreeGo2EMStudentParkourEnvCfg_EVAL = UnitreeGo2LidarParkourEnvCfg_EVAL
UnitreeGo2EMStudentParkourEnvCfg_PLAY = UnitreeGo2LidarParkourEnvCfg_PLAY
