"""CPU-only regression tests for the LiDAR/odometry noise curriculum.

The production module imports Isaac Sim, so the tests AST-extract the real
``elevation_map_scan._update`` method and execute it with small tensor stubs.
No simulator, GPU, CuPy, or elevation-map allocation is required.
"""

from __future__ import annotations

import ast
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_FILE = REPO_ROOT / "parkour_isaaclab/envs/mdp/observations.py"


def _identity_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.eye(3).expand(quaternions.shape[0], 3, 3).clone()


def _extract_method(name: str):
    tree = ast.parse(OBSERVATIONS_FILE.read_text(encoding="utf-8"))
    namespace = {
        "np": np,
        "torch": torch,
        "matrix_from_quat": _identity_matrices,
        "euler_xyz_from_quat": lambda quat: tuple(torch.zeros(quat.shape[0]) for _ in range(3)),
        "wrap_to_pi": lambda angle: angle,
        "quat_apply": lambda quat, vector: vector,
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "elevation_map_scan":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    item.decorator_list = []
                    ast.fix_missing_locations(item)
                    exec(compile(ast.Module(body=[item], type_ignores=[]), OBSERVATIONS_FILE, "exec"), namespace)
                    return namespace[name]
    raise AssertionError(f"method not found: {name}")


UPDATE = _extract_method("_update")
RESAMPLE_YAW_BIAS = _extract_method("_resample_yaw_bias")
RESAMPLE_POS_BIAS = _extract_method("_resample_pos_scale_bias")


class _Backend:
    def update(self, points, sensor_rotation, sensor_position, base_position, base_rotation):
        self.points = points
        self.sensor_rotation = sensor_rotation.clone()
        self.sensor_position = sensor_position.clone()
        self.base_position = base_position.clone()
        self.base_rotation = base_rotation.clone()

    def sample(self, points_xy, base_z):
        shape = points_xy.shape[:2]
        zeros = torch.zeros(shape)
        return zeros, zeros.clone(), zeros.clone()


def _make_term(scale: float | None, *, cfg_scale: float = 1.0, **noise):
    env = SimpleNamespace(cfg=SimpleNamespace(lidar_noise_scale=cfg_scale))
    if scale is not None:
        env.lidar_noise_scale = scale

    lidar = SimpleNamespace(
        data=SimpleNamespace(ray_hits_w=torch.tensor([[[1.0, 0.0, 0.0]]])),
        _ray_starts_w=torch.zeros(1, 1, 3),
        _ray_directions_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
    )
    asset_data = SimpleNamespace(
        body_pos_w=torch.zeros(1, 0, 3),
        body_quat_w=torch.zeros(1, 0, 4),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_pos_w=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    defaults = {
        "range_std": 0.0,
        "ray_dir_std_rad": 0.0,
        "odom_scale_std": 0.0,
        "odom_pos_walk_std": 0.0,
        "odom_scale_bias_max": 0.0,
        "odom_yaw_walk_std_rad": 0.0,
        "odom_rp_std_rad": 0.0,
    }
    defaults.update(noise)
    return SimpleNamespace(
        env=env,
        lidar=lidar,
        asset=SimpleNamespace(data=asset_data),
        _capsules=[],
        _R_offset=torch.eye(3),
        num_envs=1,
        device="cpu",
        backend=_Backend(),
        _odom_needs_init=torch.tensor([False]),
        _prev_base_pos=torch.zeros(1, 3),
        _odom_pos_err=torch.zeros(1, 3),
        _odom_yaw_err=torch.zeros(1),
        _odom_yaw_bias=torch.tensor([0.0]),
        _odom_pos_scale_bias=torch.zeros(1, 3),
        _scan_offsets_xy=torch.zeros(1, 2),
        **defaults,
    )


def _run(scale: float, *, seed: int = 17, **noise):
    term = _make_term(scale, **noise)
    torch.manual_seed(seed)
    UPDATE(term)
    return term


class TestLidarNoiseScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sensor_stub = types.ModuleType("parkour_isaaclab.sensors")
        sensor_stub.ray_capsule_penetrates = lambda *args, **kwargs: torch.zeros_like(args[3], dtype=torch.bool)
        cls._previous_sensor_module = sys.modules.get("parkour_isaaclab.sensors")
        sys.modules["parkour_isaaclab.sensors"] = sensor_stub

    @classmethod
    def tearDownClass(cls):
        if cls._previous_sensor_module is None:
            sys.modules.pop("parkour_isaaclab.sensors", None)
        else:
            sys.modules["parkour_isaaclab.sensors"] = cls._previous_sensor_module

    def assertHalfAmplitude(self, zero: torch.Tensor, half: torch.Tensor, full: torch.Tensor, atol=1e-6):
        self.assertTrue(torch.allclose(half - zero, (full - zero) * 0.5, atol=atol))

    def test_range_noise_zero_midpoint_and_full(self):
        zero = _run(0.0, range_std=0.1).backend.points[0].norm(dim=-1)
        half = _run(0.5, range_std=0.1).backend.points[0].norm(dim=-1)
        full = _run(1.0, range_std=0.1).backend.points[0].norm(dim=-1)
        self.assertHalfAmplitude(zero, half, full)

    def test_ray_direction_noise_scales_azimuth_and_elevation(self):
        half = _run(0.5, ray_dir_std_rad=0.1).backend.points[0][0]
        full = _run(1.0, ray_dir_std_rad=0.1).backend.points[0][0]
        half_az, full_az = torch.atan2(half[1], half[0]), torch.atan2(full[1], full[0])
        half_el = torch.asin(half[2] / half.norm())
        full_el = torch.asin(full[2] / full.norm())
        self.assertTrue(torch.allclose(half_az, full_az * 0.5, atol=1e-6))
        self.assertTrue(torch.allclose(half_el, full_el * 0.5, atol=1e-6))

    def test_each_odometry_increment_uses_linear_amplitude(self):
        for field, value in (("odom_scale_std", 0.1), ("odom_pos_walk_std", 0.1)):
            with self.subTest(field=field):
                zero = _run(0.0, **{field: value})._odom_pos_err
                half = _run(0.5, **{field: value})._odom_pos_err
                full = _run(1.0, **{field: value})._odom_pos_err
                self.assertHalfAmplitude(zero, half, full)

        terms = [_make_term(scale) for scale in (0.0, 0.5, 1.0)]
        for term in terms:
            term._odom_pos_scale_bias[:] = torch.tensor([[0.2, -0.1, 0.05]])
            term.odom_scale_bias_max = 0.2
            UPDATE(term)
        self.assertHalfAmplitude(terms[0]._odom_pos_err, terms[1]._odom_pos_err, terms[2]._odom_pos_err)

    def test_yaw_bias_walk_and_roll_pitch_use_linear_amplitude(self):
        bias_terms = [_make_term(scale) for scale in (0.0, 0.5, 1.0)]
        for term in bias_terms:
            term._odom_yaw_bias[:] = 0.2
            torch.manual_seed(3)
            UPDATE(term)
        self.assertHalfAmplitude(
            bias_terms[0]._odom_yaw_err, bias_terms[1]._odom_yaw_err, bias_terms[2]._odom_yaw_err
        )

        walk = [_run(scale, odom_yaw_walk_std_rad=0.1)._odom_yaw_err for scale in (0.0, 0.5, 1.0)]
        self.assertHalfAmplitude(walk[0], walk[1], walk[2])

        rp = [_run(scale, odom_rp_std_rad=0.1).backend.base_rotation for scale in (0.0, 0.5, 1.0)]
        self.assertHalfAmplitude(rp[0], rp[1], rp[2])

    def test_cfg_zero_then_runtime_scale_activates_presampled_biases(self):
        term = _make_term(None, cfg_scale=0.0)
        term._odom_pos_scale_bias[:] = torch.tensor([[0.2, 0.0, 0.0]])
        term._odom_yaw_bias[:] = 0.1
        term.odom_scale_bias_max = 0.2
        UPDATE(term)
        self.assertTrue(torch.equal(term._odom_pos_err, torch.zeros_like(term._odom_pos_err)))
        self.assertTrue(torch.equal(term._odom_yaw_err, torch.zeros_like(term._odom_yaw_err)))

        term.env.lidar_noise_scale = 1.0
        term.asset.data.root_pos_w[:, 0] = 2.0
        UPDATE(term)
        self.assertAlmostEqual(float(term._odom_pos_err[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(term._odom_yaw_err[0]), 0.1, places=6)
        self.assertAlmostEqual(float(term._odom_pos_scale_bias.abs().max()), 0.2, places=6)
        self.assertAlmostEqual(float(term._odom_yaw_bias.abs().max()), 0.1, places=6)

    def test_episode_bias_sampling_keeps_configured_maxima(self):
        term = SimpleNamespace(
            device="cpu",
            odom_scale_bias_max=0.03,
            _odom_pos_scale_bias=torch.zeros(4096, 3),
            odom_yaw_bias_range_dps=(0.01, 0.05),
            _tick_dt=0.1,
            _odom_yaw_bias=torch.zeros(4096),
        )
        env_ids = torch.arange(4096)
        torch.manual_seed(11)
        RESAMPLE_POS_BIAS(term, env_ids)
        RESAMPLE_YAW_BIAS(term, env_ids)
        self.assertLessEqual(float(term._odom_pos_scale_bias.abs().max()), term.odom_scale_bias_max)
        self.assertGreater(float(term._odom_pos_scale_bias.abs().max()), term.odom_scale_bias_max * 0.95)
        yaw_abs = term._odom_yaw_bias.abs()
        lo = math.radians(0.01) * term._tick_dt
        hi = math.radians(0.05) * term._tick_dt
        self.assertGreaterEqual(float(yaw_abs.min()), lo)
        self.assertLessEqual(float(yaw_abs.max()), hi)
        self.assertGreater(float(yaw_abs.max()), hi * 0.95)


if __name__ == "__main__":
    unittest.main()
