from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlEMDistillationCfg,
    ParkourRslRlDistillationAlgorithmCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import (
    UnitreeGo2ParkourStudentPPORunnerCfg,
)
from isaaclab.utils import configclass


@configclass
class UnitreeGo2ParkourEMStudentPPORunnerCfg(UnitreeGo2ParkourStudentPPORunnerCfg):
    """elevation-map student (계획서 docs/emcupy_student_plan.md).

    depth student 설정을 그대로 물려받되:
    - depth_encoder(ConvNet+GRU) 제거 → em_distillation 로 교체
    - algorithm class 는 EMDistillation (depth_actor 만 학습; scan_encoder 는
      teacher deepcopy 로 초기화된 채 함께 fine-tune — Q6)
    - 산출물/로드 경로는 depth student 와 분리 (student_em_pretrained)
    """
    depth_encoder = None
    em_distillation = ParkourRslRlEMDistillationCfg(
        learning_rate=1.e-3,
        num_steps_per_env=24 * 5,
    )
    load_run = "student_em_pretrained"
    run_subdir = "student_em_pretrained"
    # distillation 출발점(teacher). depth student cfg 는 model_14999.pt 를 가리키지만
    # 현재 teacher_pretrained/ 에는 resume 재학습 산출물인 model_16999.pt 만 있다
    # — 존재하는 파일로 못박는다.
    distill_load_checkpoint = "model_16999.pt"

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        class_name="EMDistillation",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
