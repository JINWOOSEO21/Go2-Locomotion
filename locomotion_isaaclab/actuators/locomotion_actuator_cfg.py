# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable

from isaaclab.actuators.actuator_cfg import DCMotorCfg
from isaaclab.utils import configclass

from . import locomotion_actuator_pd


@configclass
class LocomotionDCMotorCfg(DCMotorCfg):
    class_type: type = locomotion_actuator_pd.LocomotionDCMotor

    saturation_effort: dict[str, float] | None = None
    """Peak motor force/torque of the electric DC motor (in N-m)."""
