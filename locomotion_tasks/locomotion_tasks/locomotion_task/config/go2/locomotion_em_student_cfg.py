"""LiDAR-input environments using the corresponding Scandots task settings."""

from isaaclab.utils import configclass

from locomotion_tasks.default_cfg import GO2_LIDAR_CFG

from .locomotion_mdp_cfg import LidarObservationsCfg
from .locomotion_teacher_cfg import (
    LocomotionTeacherSceneCfg,
    UnitreeGo2TeacherLocomotionEnvCfg,
    UnitreeGo2TeacherLocomotionEnvCfg_EVAL,
    UnitreeGo2TeacherLocomotionEnvCfg_PLAY,
)


@configclass
class LocomotionLidarSceneCfg(LocomotionTeacherSceneCfg):
    lidar = GO2_LIDAR_CFG


@configclass
class UnitreeGo2LidarLocomotionEnvCfg(UnitreeGo2TeacherLocomotionEnvCfg):
    scene: LocomotionLidarSceneCfg = LocomotionLidarSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2LidarLocomotionEnvCfg_EVAL(UnitreeGo2TeacherLocomotionEnvCfg_EVAL):
    scene: LocomotionLidarSceneCfg = LocomotionLidarSceneCfg(num_envs=256, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2LidarLocomotionEnvCfg_PLAY(UnitreeGo2TeacherLocomotionEnvCfg_PLAY):
    scene: LocomotionLidarSceneCfg = LocomotionLidarSceneCfg(num_envs=16, env_spacing=1.0)
    observations: LidarObservationsCfg = LidarObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


# Keep old config symbols importable for serialized configs without exposing them
# through the new task registrations.
LocomotionEMStudentSceneCfg = LocomotionLidarSceneCfg
UnitreeGo2EMStudentLocomotionEnvCfg = UnitreeGo2LidarLocomotionEnvCfg
UnitreeGo2EMStudentLocomotionEnvCfg_EVAL = UnitreeGo2LidarLocomotionEnvCfg_EVAL
UnitreeGo2EMStudentLocomotionEnvCfg_PLAY = UnitreeGo2LidarLocomotionEnvCfg_PLAY
