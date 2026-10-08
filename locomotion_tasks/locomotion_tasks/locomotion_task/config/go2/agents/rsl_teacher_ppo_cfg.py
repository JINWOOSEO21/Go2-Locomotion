from isaaclab.utils import configclass

from locomotion_tasks.locomotion_task.config.go2.agents.locomotion_rl_cfg import (
    LocomotionRslRlActorCfg,
    LocomotionRslRlEstimatorCfg,
    LocomotionRslRlOnPolicyRunnerCfg,
    LocomotionRslRlPpoActorCriticCfg,
    LocomotionRslRlPpoAlgorithmCfg,
    LocomotionRslRlStateHistEncoderCfg,
)


@configclass
class UnitreeGo2LocomotionTeacherPPORunnerCfg(LocomotionRslRlOnPolicyRunnerCfg):
    input_mode = "scandots_input"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_locomotion"
    load_run = ".*_scandots"
    empirical_normalization = False
    policy = LocomotionRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=LocomotionRslRlActorCfg(
            class_name="Actor", state_history_encoder=LocomotionRslRlStateHistEncoderCfg(class_name="StateHistoryEncoder")
        ),
    )
    estimator = LocomotionRslRlEstimatorCfg(hidden_dims=[128, 64])
    algorithm = LocomotionRslRlPpoAlgorithmCfg(
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
