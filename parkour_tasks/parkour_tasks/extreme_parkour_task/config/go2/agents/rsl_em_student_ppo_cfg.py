"""LiDAR and odometry PPO configuration."""

from isaaclab.utils import configclass

from .rsl_teacher_ppo_cfg import UnitreeGo2ParkourTeacherPPORunnerCfg


@configclass
class UnitreeGo2ParkourLidarPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    input_mode = "lidar_input"
    max_iterations = 30000
    noise_ramp_ratio = 0.7
    run_name = "lidar"
    load_run = ".*_lidar"


# Compatibility alias for serialized configs created before the task rename.
UnitreeGo2ParkourEMStudentPPORunnerCfg = UnitreeGo2ParkourLidarPPORunnerCfg
