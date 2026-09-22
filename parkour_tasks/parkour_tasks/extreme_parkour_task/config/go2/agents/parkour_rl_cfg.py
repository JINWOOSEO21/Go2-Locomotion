# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

#########################
# Policy configurations #
#########################


@configclass
class ParkourRslRlBaseCfg:
    num_priv_explicit: int = 3 + 3 + 3  # 9
    num_priv_latent: int = 4 + 1 + 12 + 12  # 29
    num_prop: int = 3 + 2 + 3 + 4 + 36 + 5  # 53
    num_scan: int = 132
    num_hist: int = 10


@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder"
    channel_size: int = 10


@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator"
    train_with_estimated_states: bool = True
    learning_rate: float = 1.0e-4
    hidden_dims: list[int] = MISSING


@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticRMA"
    tanh_encoder_output: bool = False
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING


@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOWithExtractor"
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float] = [0, 0.1, 2000, 3000]


@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    input_mode: str = "scandots_input"
    noise_ramp_ratio: float = 0.7
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    algorithm: ParkourRslRlPpoAlgorithmCfg = MISSING
