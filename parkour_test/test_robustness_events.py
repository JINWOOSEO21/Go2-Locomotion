"""CPU regression checks for event math without launching Isaac Sim."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_events():
    tree = ast.parse((ROOT / "parkour_isaaclab/envs/mdp/events.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in ("push_by_setting_velocity", "randomize_rigid_body_material")
    ]

    def uniform(low, high, shape, device):
        return low + torch.rand(shape, device=device) * (high - low)

    def rotate(q, v):
        t = 2 * torch.cross(q[:, 1:], v, dim=-1)
        return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)

    class Rigid:
        pass

    class Manager:
        def __init__(self, *args):
            pass

    ns = dict(
        torch=torch,
        SceneEntityCfg=lambda name: SimpleNamespace(name=name),
        ManagerTermBase=Manager,
        RigidObject=Rigid,
        Articulation=type("Articulation", (Rigid,), {}),
        math_utils=SimpleNamespace(sample_uniform=uniform, quat_apply=rotate),
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "events.py", "exec"), ns)
    return ns


class EventTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_events()
        self.asset = self.ns["RigidObject"]()
        self.asset.device = "cpu"
        self.asset.data = SimpleNamespace(
            root_vel_w=torch.ones(4, 6), root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        )

        def write(value, env_ids):
            self.assertEqual(value.ndim, 2)
            self.asset.data.root_vel_w[env_ids] = value

        self.asset.write_root_velocity_to_sim = write
        self.env = SimpleNamespace(scene={"robot": self.asset}, num_envs=4, disturbance_scale=0.2)
        self.push = self.ns["push_by_setting_velocity"]

    def test_global_linear_preserves_other_axes(self):
        self.push(self.env, None, {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
        self.assertTrue((self.asset.data.root_vel_w[:, :2].abs() <= 0.2).all())
        torch.testing.assert_close(self.asset.data.root_vel_w[:, 2:], torch.ones(4, 4))

    def test_subset_angular_and_same_step_order(self):
        linear = {"x": (1.0, 1.0), "y": (-1.0, -1.0)}
        angular = {"roll": (0.5, 0.5), "pitch": (0.5, 0.5), "yaw": (0.5, 0.5)}
        self.push(self.env, torch.tensor([1, 3]), angular)
        torch.testing.assert_close(self.asset.data.root_vel_w[[0, 2]], torch.ones(2, 6))
        torch.testing.assert_close(self.asset.data.root_vel_w[[1, 3], :3], torch.ones(2, 3))
        self.asset.data.root_vel_w.fill_(1)
        self.push(self.env, None, linear)
        self.push(self.env, None, angular)
        expected = self.asset.data.root_vel_w.clone()
        self.asset.data.root_vel_w.fill_(1)
        self.push(self.env, None, angular)
        self.push(self.env, None, linear)
        torch.testing.assert_close(self.asset.data.root_vel_w, expected)
        torch.testing.assert_close(expected[0], torch.tensor([0.2, -0.2, 1.0, 1.1, 1.1, 1.1]))

    def test_body_angular_rotation(self):
        self.asset.data.root_quat_w[:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
        self.push(self.env, None, {"roll": (0.5, 0.5)})
        torch.testing.assert_close(self.asset.data.root_vel_w[:, 3:], torch.tensor([[1.0, 1.1, 1.0]]).repeat(4, 1))

    def test_material_bins_and_equal_category_sampling(self):
        torch.manual_seed(42)
        n = 30000
        materials = torch.zeros(n, 2, 3)
        self.asset.root_physx_view = SimpleNamespace(
            max_shapes=2, get_material_properties=lambda: materials, set_material_properties=lambda values, ids: None
        )
        self.env.scene = type("Scene", (dict,), {"num_envs": n})(robot=self.asset)
        params = dict(
            asset_cfg=SimpleNamespace(name="robot", body_ids=slice(None)),
            friction_range=(0.25, 2.0),
            friction_intervals=((0.25, 0.5), (0.5, 1.0), (1.0, 2.0)),
            num_buckets=96,
        )
        event = self.ns["randomize_rigid_body_material"](SimpleNamespace(params=params), self.env)
        for chunk, (low, high) in zip(event.material_buckets[:, 0].chunk(3), params["friction_intervals"]):
            self.assertTrue(((chunk >= low) & (chunk < high)).all())
        event(self.env, None, **params)
        values = materials[:, 0, 0]
        counts = torch.stack(
            (
                (values < 0.5).float().mean(),
                ((values >= 0.5) & (values < 1.0)).float().mean(),
                (values >= 1.0).float().mean(),
            )
        )
        self.assertTrue(((counts - 1 / 3).abs() < 0.015).all())
        torch.testing.assert_close(materials[:, 0], materials[:, 1])
        torch.testing.assert_close(materials[:, :, 0], materials[:, :, 1])


if __name__ == "__main__":
    unittest.main()
