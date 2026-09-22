"""CPU regression tests using the installed IsaacLab angle wrapping helper."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _reward(target_degrees, yaw_degrees, quaternion_sign=1):
    namespace = {
        "torch": torch,
        "ParkourManagerBasedRLEnv": object,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
    }
    isaaclab_path = Path(next(iter(importlib.util.find_spec("isaaclab").submodule_search_locations)))
    _load_function(isaaclab_path / "utils/math.py", "wrap_to_pi", namespace)
    reward = _load_function(ROOT / "parkour_isaaclab/envs/mdp/rewards.py", "reward_tracking_yaw", namespace)
    yaw = torch.deg2rad(torch.tensor(yaw_degrees, dtype=torch.float64))
    quat = torch.zeros(len(yaw), 4, dtype=torch.float64)
    quat[:, 0] = torch.cos(yaw / 2)
    quat[:, 3] = torch.sin(yaw / 2)
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=quat * quaternion_sign))
    event = SimpleNamespace(target_yaw=torch.deg2rad(torch.tensor(target_degrees, dtype=torch.float64)))
    env = SimpleNamespace(scene={"robot": robot}, parkour_manager=SimpleNamespace(get_term=lambda _: event))
    return reward(env, "base_parkour")


def test_boundary_crossing_uses_shortest_angle_in_both_directions():
    actual = _reward([179, -179, 180, -180], [-179, 179, -180, 180])
    expected = torch.exp(-torch.deg2rad(torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float64)))
    torch.testing.assert_close(actual, expected)


def test_periodicity_and_quaternion_sign_do_not_change_reward():
    expected = _reward([10, 10, 10], [20, 20, 20])
    torch.testing.assert_close(_reward([10, 370, -350], [20, 20, 20]), expected)
    torch.testing.assert_close(_reward([10, 370, -350], [20, 20, 20], -1), expected)


def test_aligned_opposite_and_ordinary_angles_have_expected_rewards():
    actual = _reward([0, 180, -180, 30, -30], [0, 0, 0, 0, 0])
    expected = torch.exp(-torch.deg2rad(torch.tensor([0.0, 180.0, 180.0, 30.0, 30.0], dtype=torch.float64)))
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert ((actual > 0) & (actual <= 1)).all()
