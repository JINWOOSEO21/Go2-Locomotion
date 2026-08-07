from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlDistillationAlgorithmCfg,
ParkourRslRlDepthEncoderCfg
)
from isaaclab.utils import configclass

@configclass
class UnitreeGo2ParkourStudentPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 50000 
    save_interval = 100
    experiment_name = "unitree_go2_parkour"
    empirical_normalization = False
    # run 폴더를 logs/rsl_rl/unitree_go2_parkour/student_pretrained 로 고정한다.
    # 기본값 load_run=".*" 은 정규식이라 run 폴더 중 가장 최근 것을 집어 오므로, 학습
    # 로그가 쌓이면 pretrained 대신 엉뚱한 모델이 로드된다.
    #
    # 파일명은 고정하지 않는다(기본값 "model_.*.pt"). 규약은 이렇다.
    #   student_pretrained/model_*.pt            <- 지금 쓰는 체크포인트, 항상 1개만 둔다
    #   student_pretrained/weight_candidates/    <- 대기 중인 후보들 (스캔 대상 아님)
    # 후보를 시험하려면 scripts/rsl_rl/swap_checkpoint.py 로 둘을 맞바꾼다.
    # --load_run / --checkpoint 로 여전히 덮어쓸 수 있다.
    load_run = "student_pretrained"
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
        priv_encoder_dims = [64, 20],
        activation="elu",
        actor = ParkourRslRlActorCfg(
            class_name = "Actor",
            state_history_encoder = ParkourRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder" 
            )
        )
    )
    estimator = ParkourRslRlEstimatorCfg(
            hidden_dims = [128, 64]
    )
    depth_encoder = ParkourRslRlDepthEncoderCfg(
        hidden_dims = 512,
        learning_rate= 1e-3,
        num_steps_per_env = 24*5
    )

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

