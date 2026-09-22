from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlActorCfg,
    ParkourRslRlEstimatorCfg,
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoActorCriticCfg,
    ParkourRslRlPpoAlgorithmCfg,
    ParkourRslRlStateHistEncoderCfg,
)


@configclass
class UnitreeGo2ParkourTeacherPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_parkour"
    run_subdir = "trained_v1.4"
    empirical_normalization = False
    # student 와 같은 규약. run 폴더는 teacher_pretrained 로 고정하고, 그 안에 지금 쓰는
    # 체크포인트 model_*.pt 를 하나만 둔다. 후보는 weight_candidates/ 에 넣어 두고
    # scripts/rsl_rl/swap_checkpoint.py 로 맞바꿔 가며 시험한다.
    #
    # 주의: train.py 에서 --resume 으로 '진행 중이던 학습'을 이어받으려면 이 값 때문에
    # teacher_pretrained 를 보게 되므로, --load_run <타임스탬프 폴더> 를 함께 줘야 한다.
    load_run = "teacher_pretrained"
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
    depth_encoder = None
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0],
    )
