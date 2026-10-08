from dataclasses import MISSING

from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.utils import configclass


@configclass
class LocomotionManagerBasedEnvCfg(ManagerBasedEnvCfg):
    goals: object = MISSING
