"""CPU-only tests for the contact-gated foot-slip reward."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
REWARDS = REPO / "parkour_isaaclab/envs/mdp/rewards.py"


def _load_reward_feet_slip():
    tree = ast.parse(REWARDS.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reward_feet_slip")
    namespace = {
        "torch": torch,
        "ParkourManagerBasedRLEnv": object,
        "SceneEntityCfg": object,
        "Articulation": object,
        "ContactSensor": object,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), REWARDS, "exec"), namespace)
    return namespace["reward_feet_slip"]


class _Scene(dict):
    def __init__(self, robot, contact_sensor):
        super().__init__(robot=robot)
        self.sensors = {"contact_forces": contact_sensor}


def _make_env(body_vel, current_forces, previous_forces=None):
    if previous_forces is None:
        previous_forces = torch.zeros_like(current_forces)
    force_history = torch.stack((current_forces, previous_forces), dim=1)
    robot = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=body_vel))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force_history))
    return SimpleNamespace(scene=_Scene(robot, sensor))


def _cfg(name, body_ids):
    return SimpleNamespace(name=name, body_ids=body_ids)


def test_reward_sums_world_xy_speed_only_for_current_contacts():
    reward_feet_slip = _load_reward_feet_slip()
    body_vel = torch.tensor([[[3.0, 4.0, 12.0], [0.0, 2.0, 99.0], [6.0, 8.0, 0.0], [1.0, 0.0, 0.0]]])
    current_forces = torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 4.999], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
    previous_forces = torch.full_like(current_forces, 100.0)
    env = _make_env(body_vel, current_forces, previous_forces)

    reward = reward_feet_slip(
        env,
        asset_cfg=_cfg("robot", [0, 1, 2, 3]),
        sensor_cfg=_cfg("contact_forces", [0, 1, 2, 3]),
    )

    # Feet 0 and 2 qualify at the inclusive 5 N threshold: 5 + 10.
    torch.testing.assert_close(reward, torch.tensor([15.0]))


def test_reward_uses_configured_body_order_and_custom_threshold():
    reward_feet_slip = _load_reward_feet_slip()
    body_vel = torch.tensor([[[1.0, 0.0, 0.0], [4.0, 3.0, 0.0], [8.0, 6.0, 0.0]]])
    current_forces = torch.tensor([[[0.0, 0.0, 8.0], [0.0, 0.0, 9.0], [0.0, 0.0, 0.0]]])
    env = _make_env(body_vel, current_forces)

    reward = reward_feet_slip(
        env,
        asset_cfg=_cfg("robot", [2, 0]),
        sensor_cfg=_cfg("contact_forces", [1, 0]),
        threshold=9.0,
    )

    torch.testing.assert_close(reward, torch.tensor([10.0]))
