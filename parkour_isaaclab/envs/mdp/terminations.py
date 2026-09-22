# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from parkour_isaaclab.envs.mdp import ParkourEvent

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


def time_out(env: ParkourManagerBasedRLEnv) -> torch.Tensor:
    """An artificial episode limit; the continuing task's value is bootstrapped."""
    return env.episode_length_buf >= env.max_episode_length


def goal_reached(env: ParkourManagerBasedRLEnv) -> torch.Tensor:
    """Successful task completion is terminal, not a time-limit truncation."""
    parkour_event: ParkourEvent = env.parkour_manager.get_term("base_parkour")
    return parkour_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals


def fallen(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Physical failure ends the return without a future-value bootstrap."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:, 3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > 1.5
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > 1.5
    height_cutoff = asset.data.root_state_w[:, 2] < -0.25
    return roll_cutoff | pitch_cutoff | height_cutoff
