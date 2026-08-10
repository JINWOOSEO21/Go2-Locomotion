
from __future__ import annotations

import os
import statistics
import time
import torch
import torch.nn as nn
from collections import deque

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    EmpiricalNormalization,
)
from .actor_critic_with_encoder import ActorCriticRMA
from rsl_rl.utils import store_code_state
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from .feature_extractors import DefaultEstimator
from .ppo_with_extractor import PPOWithExtractor 
from .distillation_with_extractor import DistillationWithExtractor 
from copy import copy 
import warnings 

class OnPolicyRunnerWithExtractor(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.mean_hist_latent_loss = 0.
        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] == "PPOWithExtractor":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "DistillationWithExtractor":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        estimator_class = eval(self.estimator_cfg.pop("class_name"))
        estimator: DefaultEstimator = estimator_class(**self.estimator_cfg).to(self.device)
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticRMA = policy_class(
                                             num_privileged_obs, self.env.num_actions, **self.policy_cfg
                                            ).to(self.device)

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

        # initialize algorithm

        self.learn = self.learn_rl if self.depth_encoder_cfg is None else self.learn_vision
        if self.depth_encoder_cfg is not None:
            alg_class = eval(self.alg_cfg.pop("class_name"))
            self.alg: DistillationWithExtractor = alg_class(
                                                    policy = policy, 
                                                    estimator= estimator, 
                                                    estimator_paras= self.estimator_cfg,
                                                    depth_encoder_cfg = self.depth_encoder_cfg,
                                                    learning_rate = self.alg_cfg['learning_rate'],
                                                    policy_cfg = self.policy_cfg, 
                                                    max_grad_norm = self.alg_cfg['max_grad_norm'],
                                                    device=self.device, 
                                                    multi_gpu_cfg=self.multi_gpu_cfg
                                                    )
        else:
            self.dagger_update_freq = self.alg_cfg.pop("dagger_update_freq")
            alg_class = eval(self.alg_cfg.pop("class_name"))
            self.alg: PPOWithExtractor = alg_class(
                                                    policy, 
                                                    estimator, 
                                                    self.estimator_cfg,
                                                    **self.alg_cfg, 
                                                    device=self.device, 
                                                    multi_gpu_cfg=self.multi_gpu_cfg
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
        if self.depth_encoder_cfg is None:

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
            loss_dict['hist_latent'] = self.mean_hist_latent_loss

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
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

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        if not isinstance(self.alg, DistillationWithExtractor):
            raise TypeError('A algorithm must be DistillationWithExtractor, not a ', self.alg)
        else:
            self.alg: DistillationWithExtractor
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

        obs, extras = self.env.get_observations()
        additional_obs = {}
        additional_obs["depth_camera"] = extras["observations"]['depth_camera'].to(self.device)
        obs = obs.to(self.device)

        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        lenbuffer = deque(maxlen=100)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        num_pretrain_iter = 0
        for it in range(start_iter, tot_iter):
            start = time.time()
            actions_buffer = []
            for _ in range(self.depth_encoder_cfg['num_steps_per_env']):
                if self.env.unwrapped.common_step_counter %5 == 0:
                    obs_prop_depth = obs[:, :self.depth_encoder_cfg['num_prop']].clone()
                    # encoder 입력에서만 heading 을 가린다. depth 로부터 지형을 읽는 것이
                    # 목적이라 목표 방향을 흘려 주지 않는다. obs 본체의 6:8 은 건드리지
                    # 않으므로 policy 는 teacher 와 똑같은 oracle heading 을 받는다.
                    obs_prop_depth[:, 6:8] = 0
                    depth_latent = self.alg.depth_encoder(additional_obs["depth_camera"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                with torch.no_grad():
                    actions_teacher = self.alg.policy.act_inference(obs, hist_encoding=True, scandots_latent=None)
                actions_student = self.alg.depth_actor(obs, hist_encoding=True, scandots_latent=depth_latent)
                actions_buffer.append(actions_teacher.detach() - actions_student)
                
                if it < num_pretrain_iter:
                    # Step the environment
                    obs, _, dones, infos = self.env.step(actions_teacher.detach().to(self.env.device))
                    # Move to device
                    obs, dones = (obs.to(self.device), dones.to(self.device))
                else:
                    # Step the environment
                    obs, _, dones, infos = self.env.step(actions_student.detach().to(self.env.device))
                    # Move to device
                    obs, dones = (obs.to(self.device), dones.to(self.device))
                additional_obs['depth_camera'] = infos["observations"]['depth_camera']
                # perform normalization
                obs = self.obs_normalizer(obs)
                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    # Update episode length
                    cur_episode_length += 1
                    # Clear data for completed episodes
                    # -- common
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop
            actions_buffer = torch.cat(actions_buffer, dim=0)
            loss_dict = self.alg.update_depth_actor(actions_buffer)

            stop = time.time()
            learn_time = stop - start
            self.alg.depth_encoder.detach_hidden_states()

            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log_vision(locals())
                # 예전에는 초반을 촘촘히 찍는 3단 계단이었지만, 이제는 save_interval
                # 하나로 균일하게 찍는다 (cfg 기본값 1000 -> model_0, 1000, ... 4000).
                # 마지막 iteration 은 루프를 빠져나온 뒤 model_4999.pt 로 따로 저장된다.
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

    def log_vision(self, locs, width=80, pad=35):
        
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss_depth/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["lenbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["lenbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        if self.depth_encoder_cfg is not None :
            saved_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            saved_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        # -- 지형 커리큘럼 단계. 모델 밖에 있는 유일한 학습 상태라 같이 저장한다.
        #    없으면 재개할 때 randint(0, max_init_terrain_level+1) 로 다시 뽑혀
        #    커리큘럼이 바닥부터 다시 올라간다 (student 기준 10단계 중 0~2 로 리셋).
        terrain_levels = self._get_terrain_levels()
        if terrain_levels is not None:
            saved_dict["terrain_levels"] = terrain_levels.detach().cpu()
        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, restore_terrain_curriculum: bool = False):
        """restore_terrain_curriculum 은 학습 재개(train.py)에서만 켠다.

        play/evaluation 은 cfg 가 정한 난이도 분포(EVAL 은 max_init_terrain_level=None
        + random_difficulty)로 돌아야 하는데, 학습 중의 커리큘럼 단계를 되살리면
        그 분포를 덮어써 버린다.
        """
        loaded_dict = torch.load(path, weights_only=False)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.depth_encoder_cfg is not None:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                depth_encoder_sd = loaded_dict['depth_encoder_state_dict']
                # heading 예측을 떼기 전의 student 체크포인트는 output_mlp 가 34 차원
                # (= depth latent 32 + heading 2) 이다. 지금 모델은 32 차원이라
                # strict load 가 size mismatch 로 죽는다. 옛 가중치를 재생만이라도
                # 할 수 있게, 체크포인트의 헤드 폭을 읽어 그 폭으로 되돌린 뒤 싣는다.
                # 추론부는 앞 32 차원만 latent 로 쓰고 남는 2 차원은 폭 기준으로 분기한다.
                # 학습은 teacher 체크포인트(= 이 키 자체가 없음)에서 시작하므로 무관하다.
                head = self.alg.depth_encoder.output_mlp[0]
                ckpt_out_dim = depth_encoder_sd['output_mlp.0.weight'].shape[0]
                if ckpt_out_dim != head.out_features:
                    print(
                        f"Legacy depth encoder head detected ({ckpt_out_dim} != {head.out_features}),"
                        " restoring the checkpoint's output width for replay..."
                    )
                    self.alg.depth_encoder.output_mlp[0] = nn.Linear(head.in_features, ckpt_out_dim).to(
                        head.weight.device
                    )
                self.alg.depth_encoder.load_state_dict(depth_encoder_sd)
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.policy.actor.state_dict())
                # teacher 체크포인트에서 distillation 을 시작하는 경로다. teacher 의
                # 'iter' (예: 14999) 를 그대로 물려받으면 student 첫 저장이
                # model_15000.pt 가 되어 버린다. student 학습은 0 부터 새로 센다.
                # (ActorCriticRMA.load_state_dict 가 항상 True 를 돌려주는 탓에
                #  아래 resumed_training 분기만으로는 이걸 막을 수 없다.)
                self.current_learning_iteration = 0
                loaded_dict["iter"] = 0

        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        # -- 지형 커리큘럼 복원. 모델 로드가 다 끝난 뒤에 한다.
        if restore_terrain_curriculum:
            self._restore_terrain_levels(loaded_dict.get("terrain_levels"))
        return loaded_dict["infos"]

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
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator
    
    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

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

    def get_inference_depth_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        policy = self.alg.depth_actor
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.depth_actor(self.obs_normalizer(x))  # noqa: E731
        return policy
