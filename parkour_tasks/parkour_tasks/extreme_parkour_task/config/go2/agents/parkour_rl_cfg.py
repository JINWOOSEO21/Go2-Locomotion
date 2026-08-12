# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

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
    num_priv_explicit: int = 3 + 3 + 3 # 9
    num_priv_latent: int = 4 + 1 + 12 +12 # 29
    num_prop: int = 3 + 2 + 3 + 4 + 36 + 5 # 53
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class ParkourRslRlDepthEncoderCfg(ParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87" 
    encoder_class_name: str = "RecurrentDepthBackbone" 
    depth_shape: tuple[int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class ParkourRslRlEMDistillationCfg(ParkourRslRlBaseCfg):
    """elevation-map student distillation 설정 (depth encoder 없음).

    depth 파이프라인의 ParkourRslRlDepthEncoderCfg 자리를 대신한다 —
    encoder 관련 필드는 없고, DAgger 수집 길이와 학습률만 남는다.
    """
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING

@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float]= [0, 0.1, 2000, 3000]

@configclass
class ParkourRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    depth_encoder: ParkourRslRlDepthEncoderCfg | None = None
    em_distillation: ParkourRslRlEMDistillationCfg | None = None
    """elevation-map student distillation. depth_encoder 와 상호 배타 —
    이 값이 있으면 runner 가 learn_em(algorithm class EMDistillation)으로 돈다."""
    algorithm: ParkourRslRlPpoAlgorithmCfg | ParkourRslRlDistillationAlgorithmCfg = MISSING

    distill_load_run: str | None = None
    """distillation 출발점(teacher)이 있는 run 폴더. None 이면 load_run 을 쓴다.

    load_run 은 play/evaluation 이 집어올 student 체크포인트를 가리켜야 하므로,
    학습 출발점을 그쪽에 겹쳐 쓰면 student 를 이어 학습하게 된다. 그래서 분리했다.
    algorithm.class_name == "DistillationWithExtractor" 이고 --resume 이 아닐 때만 쓰인다.
    """

    distill_load_checkpoint: str | None = None
    """distillation 출발점 체크포인트 파일명. None 이면 load_checkpoint 를 쓴다."""

    run_subdir: str | None = None
    """산출물을 담을 중간 폴더. None 이면 기존대로
    logs/rsl_rl/<experiment_name>/<timestamp> 에 쓴다.

    체크포인트를 찾는 뿌리(get_checkpoint_path 의 log_root_path)는 그대로
    logs/rsl_rl/<experiment_name> 이고 이 값은 쓰기 경로에만 끼워 넣는다.
    student distillation 은 "student_pretrained" 를 넣어
    logs/rsl_rl/<experiment_name>/student_pretrained/<timestamp> 로 떨어뜨리면서도
    teacher 체크포인트(load_run="teacher_pretrained")를 그대로 찾아올 수 있다.
    """

