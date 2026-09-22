"""LiDAR-input PPO configuration; retain the module/class path for saved configs."""

from isaaclab.utils import configclass

from .rsl_teacher_ppo_cfg import UnitreeGo2ParkourTeacherPPORunnerCfg


@configclass
class UnitreeGo2ParkourLidarPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    """Same PPO and network as Scandots, with elevation-map terrain inputs."""

    input_mode = "lidar_input"
    max_iterations = 30000
    noise_ramp_ratio = 0.7
    depth_encoder = None
    em_distillation = None
    # Keep the existing output/lookup directories; the new checkpoints identify
    # PPO and their input mode in metadata. Initialization is explicit via CLI.
    run_subdir = "student_pretrained"
    load_run = "student_pretrained"
    distill_load_run = None
    distill_load_checkpoint = None


# Compatibility alias for saved configs; new task registrations use the neutral name.
UnitreeGo2ParkourEMStudentPPORunnerCfg = UnitreeGo2ParkourLidarPPORunnerCfg
