"""LiDAR-input environments using the corresponding Scandots task settings.

Class/module names remain stable for existing serialized configurations. Sensor
sampling and odometry noise are configured in EMStudentObservationsCfg.
"""

from isaaclab.utils import configclass

from parkour_tasks.default_cfg import GO2_LIDAR_CFG

from .parkour_mdp_cfg import EMStudentObservationsCfg
from .parkour_teacher_cfg import (
    ParkourTeacherSceneCfg,
    UnitreeGo2TeacherParkourEnvCfg,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
)


@configclass
class ParkourEMStudentSceneCfg(ParkourTeacherSceneCfg):
    lidar = GO2_LIDAR_CFG


@configclass
class UnitreeGo2EMStudentParkourEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    # Fewer parallel environments keep the mapping backend's resource use bounded.
    # --num_envs overrides this without changing the PPO or task settings.
    scene: ParkourEMStudentSceneCfg = ParkourEMStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: EMStudentObservationsCfg = EMStudentObservationsCfg()
    lidar_noise_scale: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2EMStudentParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    scene: ParkourEMStudentSceneCfg = ParkourEMStudentSceneCfg(num_envs=256, env_spacing=1.0)
    observations: EMStudentObservationsCfg = EMStudentObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2EMStudentParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    scene: ParkourEMStudentSceneCfg = ParkourEMStudentSceneCfg(num_envs=16, env_spacing=1.0)
    observations: EMStudentObservationsCfg = EMStudentObservationsCfg()
    lidar_noise_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
