"""CPU-only regression tests for ``play.py`` recording helpers.

The production modules start Isaac Sim while importing, so these tests execute
only the relevant AST nodes with small stubs.  No simulator, GPU, or video
encoder is required.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAY_PATH = os.path.join(REPO_ROOT, "scripts", "rsl_rl", "play.py")
RECORDER_PATH = os.path.join(REPO_ROOT, "scripts", "rsl_rl", "multicam_recorder.py")


class _Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


def _load_recorder_namespace():
    tree = ast.parse(Path(RECORDER_PATH).read_text(encoding="utf-8"), RECORDER_PATH)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"pad_to_even", "PerEnvVideoRecorder"}
    ]
    calls = {"depth": [], "scan": [], "labels": []}

    def depth_to_panel(value, height):
        scalar = int(np.asarray(value).reshape(-1)[0])
        calls["depth"].append(scalar)
        return np.full((height, 2, 3), scalar, dtype=np.uint8)

    def scandots_to_panel(value, height, grid, robot_cell=None):
        scalar = int(np.asarray(value).reshape(-1)[0])
        calls["scan"].append((scalar, grid, robot_cell))
        return np.full((height, 3, 3), scalar, dtype=np.uint8)

    def label_panel(panel, label):
        calls["labels"].append(label)
        panel[0, 0, 0] = 250

    ns = {
        "np": np,
        "os": os,
        "torch": torch,
        "imageio": types.SimpleNamespace(get_writer=lambda *args, **kwargs: _Writer()),
        "resolve_terrain_names": lambda env: ["flat", "stairs"],
        "resolve_scandots_grid": lambda env: ((1, 2), (0.0, 0.0)),
        "depth_to_panel": depth_to_panel,
        "scandots_to_panel": scandots_to_panel,
        "label_panel": label_panel,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), RECORDER_PATH, "exec"), ns)
    ns["calls"] = calls
    return ns


def _parse_play_args(argv):
    """Execute only the top-level parser setup, before AppLauncher starts."""
    tree = ast.parse(Path(PLAY_PATH).read_text(encoding="utf-8"), PLAY_PATH)
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "parser" for target in node.targets)
    )
    stop = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "app_launcher" for target in node.targets)
    )
    body = tree.body[start:stop]
    cli_args = types.SimpleNamespace(add_rsl_rl_args=lambda parser: None)
    app_launcher = types.SimpleNamespace(add_app_launcher_args=lambda parser: None)
    ns = {"argparse": argparse, "cli_args": cli_args, "AppLauncher": app_launcher}
    with mock.patch.object(sys, "argv", [PLAY_PATH, *argv]):
        exec(compile(ast.Module(body=body, type_ignores=[]), PLAY_PATH, "exec"), ns)
    return ns["args_cli"]


def _load_snapshot_policy_panels():
    tree = ast.parse(Path(PLAY_PATH).read_text(encoding="utf-8"), PLAY_PATH)
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "snapshot_policy_panels"
    )
    ns = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), PLAY_PATH, "exec"), ns)
    return ns["snapshot_policy_panels"]


def _load_play_loop_try():
    tree = ast.parse(Path(PLAY_PATH).read_text(encoding="utf-8"), PLAY_PATH)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return next(
        node
        for node in main.body
        if isinstance(node, ast.Try) and any(isinstance(child, ast.While) for child in node.body)
    )


def _run_depth_play_loop(record_depth):
    """Run the production while/try/finally block with CPU fakes."""

    class FakeSimulationApp:
        def is_running(self):
            return True

    class FakeEnv:
        def __init__(self, obs, extras):
            self.unwrapped = types.SimpleNamespace(common_step_counter=0)
            self.device = "cpu"
            self.obs = obs
            self.extras = extras
            self.actions = []

        def step(self, actions):
            self.actions.append(actions.detach().clone())
            self.unwrapped.common_step_counter += 1
            self.extras["observations"]["depth_camera"].fill_(self.unwrapped.common_step_counter)
            return self.obs, None, None, self.extras

    class FakeRecorder:
        def __init__(self):
            self.depth_frames = []
            self.closed = False

        def track_camera(self):
            pass

        def capture(self, *, depth, scandots):
            self.depth_frames.append(None if depth is None else depth.copy())
            return True

        def close(self):
            self.closed = True

    obs = torch.arange(11, dtype=torch.float32).reshape(1, 11)
    extras = {"observations": {"depth_camera": torch.zeros(1, 1, 1)}}
    env = FakeEnv(obs, extras)
    recorder = FakeRecorder()

    def depth_encoder(depth, obs_student):
        return torch.full((1, 32), float(depth.item()))

    def policy(policy_obs, *, hist_encoding, scandots_latent):
        return policy_obs[:, :1] + scandots_latent[:, :1]

    ns = {
        "torch": torch,
        "time": time,
        "simulation_app": FakeSimulationApp(),
        "is_distill": True,
        "is_em": False,
        "is_lidar": False,
        "record_depth": record_depth,
        "record_scandots": False,
        "obs": obs,
        "extras": extras,
        "env": env,
        "num_prop": 8,
        "num_scan": 2,
        "num_priv_explicit": 1,
        "depth_encoder": depth_encoder,
        "policy": policy,
        "snapshot_policy_panels": _load_snapshot_policy_panels(),
        "recorder": recorder,
        "args_cli": types.SimpleNamespace(video=False, video_length=6, real_time=False),
        "timestep": 0,
        "panel_depth": None,
        "dt": 0.0,
    }
    exec(compile(ast.Module(body=[_load_play_loop_try()], type_ignores=[]), PLAY_PATH, "exec"), ns)
    return env.actions, recorder


class TestPerEnvVideoRecorder(unittest.TestCase):
    def setUp(self):
        self.ns = _load_recorder_namespace()
        camera = types.SimpleNamespace(
            data=types.SimpleNamespace(
                output={
                    "rgb": torch.from_numpy(
                        np.stack(
                            [
                                np.full((3, 5, 3), 10, dtype=np.uint8),
                                np.full((3, 5, 3), 20, dtype=np.uint8),
                            ]
                        )
                    )
                }
            )
        )
        self.writers = [_Writer(), _Writer()]
        self.recorder = self.ns["PerEnvVideoRecorder"].__new__(self.ns["PerEnvVideoRecorder"])
        self.recorder.camera = camera
        self.recorder.writers = self.writers
        self.recorder.scandots_grid = (1, 2)
        self.recorder.scandots_robot_cell = (0.0, 0.0)
        self.recorder._warned_no_grid = False
        self.recorder.written = 0
        self.recorder.out_dir = "/unused"
        self.recorder.paths = ["env0.mp4", "env1.mp4"]

    def test_capture_rgb_only_is_env_specific_and_even_padded(self):
        self.assertTrue(self.recorder.capture())
        self.assertEqual(self.recorder.written, 1)
        self.assertEqual([writer.frames[0].shape for writer in self.writers], [(4, 6, 3), (4, 6, 3)])
        self.assertTrue(np.all(self.writers[0].frames[0] == 10))
        self.assertTrue(np.all(self.writers[1].frames[0] == 20))

    def test_capture_depth_only_uses_each_environment(self):
        depth = np.array([[31.0], [47.0]])
        self.assertTrue(self.recorder.capture(depth=depth))
        self.assertEqual(self.ns["calls"]["depth"], [31, 47])
        self.assertEqual([writer.frames[0].shape for writer in self.writers], [(4, 8, 3), (4, 8, 3)])

    def test_capture_single_gt_scandots(self):
        gt = np.array([[3.0, 4.0], [8.0, 9.0]])
        self.assertTrue(self.recorder.capture(scandots=gt))
        self.assertEqual([call[0] for call in self.ns["calls"]["scan"]], [3, 8])
        self.assertEqual(self.ns["calls"]["labels"], [])
        self.assertEqual([writer.frames[0].shape for writer in self.writers], [(4, 8, 3), (4, 8, 3)])

    def test_capture_gt_and_estimated_scandots_with_labels(self):
        gt = np.array([[3.0, 4.0], [8.0, 9.0]])
        estimated = np.array([[5.0, 6.0], [12.0, 13.0]])
        entries = [("GT", gt), ("Estimated", estimated)]
        self.assertTrue(self.recorder.capture(scandots=entries))
        self.assertEqual([call[0] for call in self.ns["calls"]["scan"]], [3, 5, 8, 12])
        self.assertEqual(self.ns["calls"]["labels"], ["GT", "Estimated", "GT", "Estimated"])
        self.assertEqual([writer.frames[0].shape for writer in self.writers], [(4, 12, 3), (4, 12, 3)])

    def test_capture_returns_false_when_rgb_is_unavailable(self):
        self.recorder.camera.data.output["rgb"] = None
        self.assertFalse(self.recorder.capture())
        self.assertEqual(self.recorder.written, 0)
        self.assertTrue(all(not writer.frames for writer in self.writers))

    def test_close_closes_every_writer(self):
        self.recorder.close()
        self.assertTrue(all(writer.closed for writer in self.writers))


class TestPlayArgumentParser(unittest.TestCase):
    def test_recording_defaults_are_stable(self):
        args = _parse_play_args([])
        self.assertFalse(args.video)
        self.assertFalse(args.multicam)
        self.assertEqual(args.num_envs, 1)
        self.assertEqual(args.video_length, 500)

    def test_viewport_and_multicam_can_be_enabled_together(self):
        args = _parse_play_args(["--video", "--multicam"])
        self.assertTrue(args.video)
        self.assertTrue(args.multicam)
        self.assertTrue(args.enable_cameras)

    def test_panels_can_be_enabled_with_multicam(self):
        args = _parse_play_args(["--multicam", "--panels"])
        self.assertTrue(args.multicam)
        self.assertTrue(args.panels)
        self.assertTrue(args.enable_cameras)

    def test_viewport_multicam_and_panels_can_be_enabled_together(self):
        args = _parse_play_args(["--video", "--multicam", "--panels"])
        self.assertTrue(args.video)
        self.assertTrue(args.multicam)
        self.assertTrue(args.panels)
        self.assertTrue(args.enable_cameras)

    def test_panels_requires_multicam(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse_play_args(["--panels"])

    def test_removed_debug_and_panel_flags_are_rejected(self):
        for flag in ("--fixed_heading", "--em_error_plot", "--with_depth", "--with_scandots"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                _parse_play_args([flag])


class TestSnapshotPolicyPanels(unittest.TestCase):
    def setUp(self):
        self.snapshot = _load_snapshot_policy_panels()
        self.obs = torch.tensor(
            [
                [10.0, 11.0, 1.0, 2.0, 3.0, 90.0],
                [20.0, 21.0, 4.0, 5.0, 6.0, 91.0],
            ]
        )

    def test_depth_mode_returns_only_a_copied_depth_panel(self):
        depth = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        depth_panel, scandots = self.snapshot(self.obs, 2, 3, depth=depth)
        depth[0, 0, 0] = 99.0
        self.assertIsNone(scandots)
        np.testing.assert_array_equal(depth_panel, np.array([[[1.0, 2.0]], [[3.0, 4.0]]]))

    def test_gt_scandots_mode_returns_one_labeled_copy(self):
        depth, scandots = self.snapshot(self.obs, 2, 3)
        self.obs[0, 2] = 99.0
        self.assertIsNone(depth)
        self.assertEqual([label for label, _ in scandots], ["GT"])
        np.testing.assert_array_equal(scandots[0][1], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_estimated_scandots_mode_returns_gt_and_estimated_copies(self):
        estimated = self.obs.clone()
        estimated[:, 2:5] += 10.0
        depth, scandots = self.snapshot(self.obs, 2, 3, estimated_obs=estimated)
        self.obs[:, 2:5] = -1.0
        estimated[:, 2:5] = -2.0
        self.assertIsNone(depth)
        self.assertEqual([label for label, _ in scandots], ["GT", "Estimated"])
        np.testing.assert_array_equal(scandots[0][1], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(scandots[1][1], np.array([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]))


class TestPlayLoopRecordingIntegration(unittest.TestCase):
    def test_depth_panel_holds_encoder_input_without_changing_actions(self):
        actions_with_panels, recorder = _run_depth_play_loop(record_depth=True)
        actions_without_panels, recorder_without_panels = _run_depth_play_loop(record_depth=False)

        self.assertEqual([float(frame.item()) for frame in recorder.depth_frames], [0, 0, 0, 0, 0, 5])
        self.assertTrue(all(frame is None for frame in recorder_without_panels.depth_frames))
        self.assertEqual(len(actions_with_panels), len(actions_without_panels))
        for with_panels, without_panels in zip(actions_with_panels, actions_without_panels, strict=True):
            torch.testing.assert_close(with_panels, without_panels)
        self.assertTrue(recorder.closed)
        self.assertTrue(recorder_without_panels.closed)


if __name__ == "__main__":
    unittest.main()
