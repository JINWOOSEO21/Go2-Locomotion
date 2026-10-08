# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for velocity-based locomotion environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Locomotion-Scandots-Unitree-Go2-Train-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_teacher_cfg:UnitreeGo2TeacherLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2LocomotionTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Locomotion-Scandots-Unitree-Go2-Play-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_teacher_cfg:UnitreeGo2TeacherLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2LocomotionTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Locomotion-Scandots-Unitree-Go2-Eval-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_teacher_cfg:UnitreeGo2TeacherLocomotionEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2LocomotionTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Locomotion-Lidar-Unitree-Go2-Train-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_em_student_cfg:UnitreeGo2LidarLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_em_student_ppo_cfg:UnitreeGo2LocomotionLidarPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Locomotion-Lidar-Unitree-Go2-Play-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_em_student_cfg:UnitreeGo2LidarLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_em_student_ppo_cfg:UnitreeGo2LocomotionLidarPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Locomotion-Lidar-Unitree-Go2-Eval-v0",
    entry_point="locomotion_isaaclab.envs:LocomotionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_em_student_cfg:UnitreeGo2LidarLocomotionEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_em_student_ppo_cfg:UnitreeGo2LocomotionLidarPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_locomotion_ppo_cfg.yaml",
    },
)
