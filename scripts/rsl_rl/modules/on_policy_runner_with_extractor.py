from __future__ import annotations

import math
import os
import statistics
import time
import warnings
from collections import deque
from copy import copy

import rsl_rl
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    EmpiricalNormalization,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from .actor_critic_with_encoder import ActorCriticRMA
from .feature_extractors import DefaultEstimator
from .ppo_with_extractor import PPOWithExtractor

_DISTURBANCE_SCALES = (0.2, 0.4, 0.6, 0.8, 1.0)


def _disturbance_scale_for_progress(completed_iterations: int, total_iterations: int) -> float:
    """Return the five-stage disturbance scale for the active learn call."""
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    stage = min(
        len(_DISTURBANCE_SCALES) - 1,
        completed_iterations * len(_DISTURBANCE_SCALES) // total_iterations,
    )
    return _DISTURBANCE_SCALES[stage]


def lidar_noise_scale_for_progress(
    completed_iterations: int, total_iterations: int, ramp_ratio: float = 0.7
) -> float:
    """Return the lidar noise scale for a warmup, linear ramp, and plateau schedule."""
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if not math.isfinite(ramp_ratio) or not 0.0 < ramp_ratio <= 1.0:
        raise ValueError("ramp_ratio must be finite and in the interval (0, 1]")
    warmup_iterations = (1.0 - ramp_ratio) * total_iterations / 2.0
    ramp_iterations = ramp_ratio * total_iterations
    progress = (completed_iterations - warmup_iterations) / ramp_iterations
    return min(1.0, max(0.0, progress))


class OnPolicyRunnerWithExtractor(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.estimator_cfg = train_cfg["estimator"]
        self.policy_cfg = train_cfg["policy"]
        self.input_mode = train_cfg.get("input_mode", "scandots_input")
        self.lidar_noise_ramp_ratio = float(train_cfg.get("noise_ramp_ratio", 0.7))
        if self.input_mode not in ("scandots_input", "lidar_input"):
            raise ValueError(
                f"Unsupported input_mode {self.input_mode!r}; expected 'scandots_input' or 'lidar_input'."
            )
        if not math.isfinite(self.lidar_noise_ramp_ratio) or not 0.0 < self.lidar_noise_ramp_ratio <= 1.0:
            raise ValueError("noise_ramp_ratio must be finite and in the interval (0, 1]")
        self.device = device
        self.env = env
        self.mean_hist_latent_loss = 0.0
        self._configure_multi_gpu()

        self.algorithm_class_name = self.alg_cfg["class_name"]
        if self.algorithm_class_name != "PPOWithExtractor":
            raise ValueError(f"Unsupported algorithm {self.algorithm_class_name!r}; expected 'PPOWithExtractor'.")
        self.training_type = "rl"
        if self.input_mode == "lidar_input":
            configure_policy_input = getattr(self.env, "configure_policy_input", None)
            if configure_policy_input is None:
                raise TypeError("lidar_input requires an environment wrapper with configure_policy_input()")
            configure_policy_input(
                self.input_mode,
                self.estimator_cfg["num_prop"],
                self.estimator_cfg["num_scan"],
            )

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        estimator_class = eval(self.estimator_cfg.pop("class_name"))
        estimator: DefaultEstimator = estimator_class(**self.estimator_cfg).to(self.device)
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticRMA = policy_class(num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(
            self.device
        )

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        self.learn = self.learn_rl
        self.dagger_update_freq = self.alg_cfg.pop("dagger_update_freq")
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPOWithExtractor = alg_class(
            policy,
            estimator,
            self.estimator_cfg,
            **self.alg_cfg,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._disturbance_schedule_total_iterations = None
        self._disturbance_schedule_completed_iterations = 0
        self._disturbance_schedule_resume_pending = False
        self._lidar_noise_schedule_total_iterations = None
        self._lidar_noise_schedule_completed_iterations = 0
        self._lidar_noise_schedule_ramp_ratio = self.lidar_noise_ramp_ratio
        self._lidar_noise_schedule_resume_pending = False
        self.git_status_repos = [rsl_rl.__file__]

    def learn_rl(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self.alg: PPOWithExtractor
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # The curriculum is local to this learn() call. Resuming restores its saved position.
        self._begin_disturbance_schedule(num_learning_iterations)
        self._begin_lidar_noise_schedule(num_learning_iterations)

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            self._apply_disturbance_schedule()
            self._apply_lidar_noise_schedule()
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs, hist_encoding)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                self.mean_hist_latent_loss = self.alg.update_dagger()
            loss_dict["hist_latent"] = self.mean_hist_latent_loss

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self._advance_disturbance_schedule()
            self._advance_lidar_noise_schedule()
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                self.writer.add_scalar("Disturbance/scale", self.env.unwrapped.disturbance_scale, it)
                if self.input_mode == "lidar_input":
                    self.writer.add_scalar("Lidar/noise_scale", self.env.unwrapped.lidar_noise_scale, it)
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _begin_disturbance_schedule(self, num_learning_iterations: int):
        """Start or resume the five-stage curriculum used by one learn() call.

        ``num_learning_iterations`` is the number of additional iterations passed
        to this invocation. It deliberately does not use the absolute checkpoint
        iteration, so a fresh phase begins at scale 0.2. Checkpoints
        produced during this curriculum persist the original total and completed
        counts, allowing an interrupted run to continue at the same stage.
        """
        if self._disturbance_schedule_resume_pending:
            self._disturbance_schedule_resume_pending = False
        else:
            self._disturbance_schedule_total_iterations = max(1, int(num_learning_iterations))
            self._disturbance_schedule_completed_iterations = 0
        self._apply_disturbance_schedule()

    def _apply_disturbance_schedule(self):
        total = self._disturbance_schedule_total_iterations
        if total is None:
            return
        scale = _disturbance_scale_for_progress(
            self._disturbance_schedule_completed_iterations,
            total,
        )
        self.env.unwrapped.disturbance_scale = scale

    def _advance_disturbance_schedule(self):
        total = self._disturbance_schedule_total_iterations
        if total is not None:
            self._disturbance_schedule_completed_iterations = min(
                self._disturbance_schedule_completed_iterations + 1,
                total,
            )

    def _begin_lidar_noise_schedule(self, num_learning_iterations: int):
        """Start a fresh lidar schedule or continue the schedule restored by ``load``."""
        if self.input_mode != "lidar_input":
            return
        if num_learning_iterations <= 0:
            raise ValueError("num_learning_iterations must be positive for lidar noise scheduling")
        if self._lidar_noise_schedule_resume_pending:
            self._lidar_noise_schedule_resume_pending = False
        else:
            self._lidar_noise_schedule_total_iterations = int(num_learning_iterations)
            self._lidar_noise_schedule_completed_iterations = 0
            self._lidar_noise_schedule_ramp_ratio = self.lidar_noise_ramp_ratio
        self._apply_lidar_noise_schedule()

    def _apply_lidar_noise_schedule(self):
        if self.input_mode != "lidar_input":
            return
        total = self._lidar_noise_schedule_total_iterations
        if total is None:
            return
        self.env.unwrapped.lidar_noise_scale = lidar_noise_scale_for_progress(
            self._lidar_noise_schedule_completed_iterations,
            total,
            self._lidar_noise_schedule_ramp_ratio,
        )

    def _advance_lidar_noise_schedule(self):
        if self.input_mode != "lidar_input":
            return
        total = self._lidar_noise_schedule_total_iterations
        if total is not None:
            self._lidar_noise_schedule_completed_iterations = min(
                self._lidar_noise_schedule_completed_iterations + 1,
                total,
            )

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "input_mode": self.input_mode,
            "algorithm": self.algorithm_class_name,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Preserve auxiliary PPO optimizer momentum when resuming training.
        for name in ("estimator_optimizer", "hist_encoder_optimizer"):
            optimizer = getattr(self.alg, name, None)
            if optimizer is not None:
                saved_dict[f"{name}_state_dict"] = optimizer.state_dict()
        # -- 지형 커리큘럼 단계. 모델 밖에 있는 유일한 학습 상태라 같이 저장한다.
        #    없으면 재개할 때 randint(0, max_init_terrain_level+1) 로 다시 뽑혀
        #    커리큘럼이 바닥부터 다시 올라간다.
        terrain_levels = self._get_terrain_levels()
        if terrain_levels is not None:
            saved_dict["terrain_levels"] = terrain_levels.detach().cpu()
        # -- PPO 의 업데이트 카운터. priv_reg_coef_schedual 이 이 값만 보고 계수를
        #    올리는데, terrain_levels 와 마찬가지로 모델 밖에 살아서 재개하면 0 으로
        #    돌아간다. 기본 스케줄 [0, 0.1, 2000, 3000] 기준으로, 복원하지 않으면
        #    재개 후 2000 iteration 동안 priv_reg_coef 가 0.1 -> 0.0 으로 꺼진다.
        #    iteration 과 1:1 이 아니다: update() 마다 +1, 20 iteration 마다 도는
        #    update_dagger() 에서도 +1 이라 iteration 보다 5% 쯤 크다.
        alg_counter = getattr(self.alg, "counter", None)
        if alg_counter is not None:
            saved_dict["alg_counter"] = int(alg_counter)
        if self._disturbance_schedule_total_iterations is not None:
            saved_dict["disturbance_schedule"] = {
                "total_iterations": self._disturbance_schedule_total_iterations,
                "completed_iterations": self._disturbance_schedule_completed_iterations,
            }
        if self.input_mode == "lidar_input" and self._lidar_noise_schedule_total_iterations is not None:
            saved_dict["lidar_noise_schedule"] = {
                "total_iterations": self._lidar_noise_schedule_total_iterations,
                "completed_iterations": self._lidar_noise_schedule_completed_iterations,
                "ramp_ratio": self._lidar_noise_schedule_ramp_ratio,
            }
        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        restore_terrain_curriculum: bool = False,
        warm_start: bool = False,
    ):
        """restore_terrain_curriculum 은 학습 재개(train.py)에서만 켠다.

        play/evaluation 은 cfg 가 정한 난이도 분포(EVAL 은 max_init_terrain_level=None
        + random_difficulty)로 돌아야 하는데, 학습 중의 커리큘럼 단계를 되살리면
        그 분포를 덮어써 버린다.
        """
        loaded_dict = torch.load(path, weights_only=False)
        if "depth_actor_state_dict" in loaded_dict or "depth_encoder_state_dict" in loaded_dict:
            raise ValueError(
                "PPO cannot load a legacy imitation checkpoint. Use a Scandots or Lidar PPO checkpoint."
            )
        checkpoint_algorithm = loaded_dict.get("algorithm")
        if checkpoint_algorithm is not None and checkpoint_algorithm != "PPOWithExtractor":
            raise ValueError(
                f"PPO cannot load a checkpoint created by algorithm {checkpoint_algorithm!r}."
            )
        if self.input_mode == "lidar_input" and not warm_start:
            checkpoint_mode = loaded_dict.get("input_mode")
            if checkpoint_mode != "lidar_input" or checkpoint_algorithm != "PPOWithExtractor":
                raise ValueError(
                    "A normal Lidar PPO load requires a Lidar PPO checkpoint with input_mode='lidar_input' "
                    "and algorithm='PPOWithExtractor'. Use --init_checkpoint for a Scandots PPO warm start."
                )
            if "lidar_noise_schedule" not in loaded_dict:
                raise ValueError(
                    "Lidar PPO resume checkpoint has no 'lidar_noise_schedule'; use --init_checkpoint "
                    "to start a fresh Lidar training schedule."
                )

        if warm_start:
            self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
            if self.empirical_normalization:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            self.current_learning_iteration = 0
            if getattr(self.alg, "counter", None) is not None:
                self.alg.counter = 0
            self._lidar_noise_schedule_total_iterations = None
            self._lidar_noise_schedule_completed_iterations = 0
            self._lidar_noise_schedule_ramp_ratio = self.lidar_noise_ramp_ratio
            self._lidar_noise_schedule_resume_pending = False
            self._disturbance_schedule_total_iterations = None
            self._disturbance_schedule_completed_iterations = 0
            self._disturbance_schedule_resume_pending = False
            return loaded_dict.get("infos")

        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # Restore auxiliary PPO optimizers when the checkpoint contains them.
            for name in ("estimator_optimizer", "hist_encoder_optimizer"):
                optimizer = getattr(self.alg, name, None)
                if optimizer is None:
                    continue
                key = f"{name}_state_dict"
                if key in loaded_dict:
                    optimizer.load_state_dict(loaded_dict[key])
                    print(f"Restored {name} state.")
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            self._restore_alg_counter(loaded_dict)
        self._restore_disturbance_schedule(loaded_dict)
        self._restore_lidar_noise_schedule(loaded_dict)
        # -- 지형 커리큘럼 복원. 모델 로드가 다 끝난 뒤에 한다.
        if restore_terrain_curriculum:
            self._restore_terrain_levels(loaded_dict.get("terrain_levels"))
        return loaded_dict["infos"]

    def _restore_disturbance_schedule(self, loaded_dict):
        """Restore an interrupted disturbance schedule when available."""
        self._disturbance_schedule_total_iterations = None
        self._disturbance_schedule_completed_iterations = 0
        self._disturbance_schedule_resume_pending = False
        state = loaded_dict.get("disturbance_schedule")
        if state is None:
            return
        try:
            total = int(state["total_iterations"])
            completed = int(state["completed_iterations"])
        except (KeyError, TypeError, ValueError):
            warnings.warn("invalid disturbance schedule in checkpoint; starting a fresh schedule")
            return
        if total <= 0 or completed < 0:
            warnings.warn("invalid disturbance schedule in checkpoint; starting a fresh schedule")
            return
        self._disturbance_schedule_total_iterations = total
        self._disturbance_schedule_completed_iterations = min(completed, total)
        self._disturbance_schedule_resume_pending = True

    def _restore_lidar_noise_schedule(self, loaded_dict):
        """Restore Lidar PPO noise progress without changing evaluation environment noise."""
        self._lidar_noise_schedule_total_iterations = None
        self._lidar_noise_schedule_completed_iterations = 0
        self._lidar_noise_schedule_ramp_ratio = self.lidar_noise_ramp_ratio
        self._lidar_noise_schedule_resume_pending = False
        if self.input_mode != "lidar_input":
            return

        state = loaded_dict["lidar_noise_schedule"]
        try:
            total = int(state["total_iterations"])
            completed = int(state["completed_iterations"])
            ramp_ratio = float(state["ramp_ratio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid lidar_noise_schedule in checkpoint") from exc
        if total <= 0 or completed < 0:
            raise ValueError("invalid lidar_noise_schedule counts in checkpoint")
        if not math.isfinite(ramp_ratio) or not 0.0 < ramp_ratio <= 1.0:
            raise ValueError("invalid lidar_noise_schedule ramp_ratio in checkpoint")
        if not math.isclose(ramp_ratio, self.lidar_noise_ramp_ratio):
            warnings.warn(
                "checkpoint lidar noise ramp_ratio "
                f"({ramp_ratio}) differs from the current config ({self.lidar_noise_ramp_ratio}); "
                "the saved ratio remains authoritative for resume"
            )
        self._lidar_noise_schedule_total_iterations = total
        self._lidar_noise_schedule_completed_iterations = min(completed, total)
        self._lidar_noise_schedule_ramp_ratio = ramp_ratio
        self._lidar_noise_schedule_resume_pending = True

    def _restore_alg_counter(self, loaded_dict):
        """PPO 업데이트 카운터를 되살린다. priv_reg_coef_schedual 이 이 값만 본다."""
        if getattr(self.alg, "counter", None) is None:
            return
        if loaded_dict["iter"] == 0:
            self.alg.counter = 0
            return
        counter = loaded_dict.get("alg_counter")
        if counter is None:
            # 이 기능 이전에 저장된 체크포인트. iteration 수로 근사한다. 실제 카운터는
            # dagger 갱신 때문에 이보다 5% 쯤 크지만, 스케줄이 (counter-2000)/3000 을
            # 1 로 자르는 구조라 counter 가 5000 만 넘으면 어느 쪽이든 계수는 같다.
            counter = self.current_learning_iteration
            warnings.warn(
                "checkpoint has no 'alg_counter' (saved before counter persistence);"
                f" approximating it from the iteration number ({counter})"
            )
        self.alg.counter = int(counter)
        print(f"Restored PPO update counter = {self.alg.counter}.")

    def _parkour_terms(self):
        """씬에 붙어 있는 파쿠르 이벤트 term 들. 없으면 빈 리스트."""
        manager = getattr(self.env.unwrapped, "parkour_manager", None)
        if manager is None:
            return []
        try:
            return [manager.get_term(name) for name in manager.active_terms]
        except (AttributeError, KeyError):
            return list(getattr(manager, "_terms", {}).values())

    def _get_terrain_levels(self):
        """지형 커리큘럼 단계 텐서. 커리큘럼이 없는 씬이면 None."""
        terrain = getattr(self.env.unwrapped.scene, "terrain", None)
        return getattr(terrain, "terrain_levels", None) if terrain is not None else None

    def _restore_terrain_levels(self, levels):
        """체크포인트의 커리큘럼 단계를 되살린다.

        복원 후 env 를 한 번 리셋한다. 로봇이 옛 타일 위에 서 있는 채로 origin 만
        바뀌면 goal 까지의 상대 위치가 한 에피소드 내내 어긋나기 때문이다.

        리셋은 각 env 의 _resample_command 를 부르고, 그게 move_up/move_down 으로
        단계를 한 칸씩 흔든다(리셋 직후라 이동거리가 0 이라 대부분 강등된다).
        그래서 리셋 뒤에 한 번 더 덮어쓴다. 이 시점의 로봇은 이미 복원된 origin 에
        놓여 있고 dis_to_start_pos / cur_goal_idx 도 리셋이 0 으로 만들어 둔 뒤다.
        """
        if levels is None:
            # 이 기능이 들어오기 전에 저장된 체크포인트다. 조용히 넘어가면 커리큘럼이
            # 바닥부터 다시 올라가는데도 로그에 아무 흔적이 없어, 재개 직후 리워드가
            # 떨어지는 이유를 나중에 찾기 어렵다.
            warnings.warn(
                "checkpoint has no 'terrain_levels' (saved before terrain-curriculum persistence);"
                " the terrain curriculum restarts from randint(0, max_init_terrain_level + 1)"
            )
            return
        current = self._get_terrain_levels()
        if current is None:
            warnings.warn("checkpoint has terrain_levels but this scene has no terrain curriculum; skipping")
            return
        if levels.shape != current.shape:
            warnings.warn(
                f"terrain_levels shape mismatch (checkpoint {tuple(levels.shape)} vs env {tuple(current.shape)}),"
                " probably a different --num_envs; the terrain curriculum restarts from scratch"
            )
            return
        terms = self._parkour_terms()
        if not terms:
            warnings.warn("no parkour term found; not restoring terrain levels")
            return

        for term in terms:
            term.restore_terrain_levels(levels)
        self.env.reset()
        for term in terms:
            term.restore_terrain_levels(levels)
        print(f"Restored terrain curriculum from checkpoint (mean level {levels.float().mean():.2f}).")

    def get_estimator_inference_policy(self, device=None):
        self.alg: PPOWithExtractor
        self.alg.estimator.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy
