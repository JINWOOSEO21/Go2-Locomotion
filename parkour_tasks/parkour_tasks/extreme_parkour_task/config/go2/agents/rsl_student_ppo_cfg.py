from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlActorCfg,
    ParkourRslRlDepthEncoderCfg,
    ParkourRslRlDistillationAlgorithmCfg,
    ParkourRslRlEstimatorCfg,
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoActorCriticCfg,
    ParkourRslRlStateHistEncoderCfg,
)


@configclass
class UnitreeGo2ParkourStudentPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    input_mode = "depth_input"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 1000
    experiment_name = "unitree_go2_parkour"
    empirical_normalization = False
    # 규약은 이렇다.
    #   teacher_pretrained/model_14999.pt        <- distillation 출발점
    #   student_pretrained/model_*.pt            <- 재생/평가용 student, 항상 1개만 둔다
    #   student_pretrained/weight_candidates/    <- 대기 중인 후보들 (스캔 대상 아님)
    #   student_pretrained/<timestamp>/          <- run_subdir 로 떨어지는 학습 산출물
    # 학습 산출물을 승격할 때는 최상위 model_*.pt 를 weight_candidates/ 로 내리고
    # 새 체크포인트를 그 자리에 올린다.
    #
    # load_run 은 play/evaluation/export 가 집어올 student 체크포인트를 가리킨다.
    # 기본값 ".*" 은 정규식이라 run 폴더 중 가장 최근 것을 집어 오므로, 학습 로그가
    # 쌓이면 pretrained 대신 엉뚱한 모델이 로드된다. 그래서 못박아 둔다.
    # 파일명은 고정하지 않는다(기본값 "model_.*.pt").
    load_run = "student_pretrained"
    # 학습(distillation)의 출발점만은 student 가 아니라 teacher 다. 같은 load_run 을
    # 쓰면 student 를 이어 학습하게 되므로 출발점 전용으로 따로 둔다 (train.py 참고).
    distill_load_run = "teacher_pretrained"
    distill_load_checkpoint = "model_14999.pt"
    # 산출물은 logs/rsl_rl/unitree_go2_parkour/student_pretrained/<timestamp> 로 간다.
    # 체크포인트 탐색 뿌리는 여전히 logs/rsl_rl/unitree_go2_parkour 다 (train.py 참고).
    run_subdir = "student_pretrained"
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=ParkourRslRlActorCfg(
            class_name="Actor", state_history_encoder=ParkourRslRlStateHistEncoderCfg(class_name="StateHistoryEncoder")
        ),
    )
    estimator = ParkourRslRlEstimatorCfg(hidden_dims=[128, 64])
    depth_encoder = ParkourRslRlDepthEncoderCfg(hidden_dims=512, learning_rate=1e-3, num_steps_per_env=24 * 5)

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
