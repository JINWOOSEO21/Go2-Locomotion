import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


class ParkourRslRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ParkourManagerBasedRLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ParkourManagerBasedRLEnv):
            raise ValueError(
                f"The environment must be inherited from ParkourManagerBasedRLEnv. Environment type: {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self._policy_input_mode = "scandots_input"
        self._policy_num_prop = None
        self._policy_num_scan = None

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ParkourManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        obs_dict = self._adapt_policy_observations(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def configure_policy_input(self, input_mode: str, num_prop: int, num_scan: int) -> None:
        """Select the source used for the policy's scan slice.

        The default ``scandots_input`` path is intentionally a no-op. In
        ``lidar_input`` mode, the wrapper replaces only the policy observation's
        scan slice with ``em_scan`` while preserving the original GT scandots in
        the returned extras.
        """
        if input_mode not in ("scandots_input", "lidar_input"):
            raise ValueError(
                f"Unsupported policy input mode {input_mode!r}; expected 'scandots_input' or 'lidar_input'."
            )
        if num_prop < 0 or num_scan <= 0:
            raise ValueError(f"num_prop must be non-negative and num_scan must be positive, got {num_prop}, {num_scan}")
        self._policy_input_mode = input_mode
        self._policy_num_prop = int(num_prop)
        self._policy_num_scan = int(num_scan)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        obs_dict = self._adapt_policy_observations(obs_dict)
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._adapt_policy_observations(obs_dict)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            # A simultaneous fall/success is terminal even at the time limit.
            extras["time_outs"] = truncated & ~terminated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

    def _adapt_policy_observations(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return lidar-adapted observations without mutating environment-owned tensors."""
        if self._policy_input_mode != "lidar_input":
            return obs_dict

        policy = obs_dict.get("policy")
        em_scan = obs_dict.get("em_scan")
        if policy is None:
            raise KeyError("lidar_input requires obs_dict['policy'], but it is missing")
        if em_scan is None:
            raise KeyError("lidar_input requires obs_dict['em_scan'], but it is missing")

        scan_start = self._policy_num_prop
        scan_end = scan_start + self._policy_num_scan
        if policy.ndim != 2 or policy.shape[1] < scan_end:
            raise ValueError(
                "lidar_input policy observation has an invalid shape: "
                f"expected [num_envs, >= {scan_end}], got {tuple(policy.shape)}"
            )
        expected_shape = (policy.shape[0], self._policy_num_scan)
        if tuple(em_scan.shape) != expected_shape:
            raise ValueError(
                "lidar_input em_scan shape mismatch: "
                f"expected {expected_shape}, got {tuple(em_scan.shape)}"
            )

        adapted_policy = policy.clone()
        gt_scandots = policy[:, scan_start:scan_end].clone()
        adapted_policy[:, scan_start:scan_end] = em_scan
        adapted_obs_dict = dict(obs_dict)
        adapted_obs_dict["policy"] = adapted_policy
        adapted_obs_dict["gt_scandots"] = gt_scandots
        return adapted_obs_dict
