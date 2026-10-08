from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.utils import configclass

# from isaaclab.utils.modifiers import ModifierCfg
# from isaaclab.utils.noise import NoiseCfg

# from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .goal_manager import GoalTerm


@configclass
class GoalTermCfg:
    class_type: type[GoalTerm] = MISSING

    debug_vis: bool = False
