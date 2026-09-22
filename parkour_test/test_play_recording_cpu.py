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
    calls = {"scan": [], "labels": []}


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


class TestPlayLoopRecordingIntegration(unittest.TestCase):
    def test_lidar_panels_copy_inputs_without_changing_actions(self):
        def run(panels):
            obs = torch.tensor([[1.0, 2.0, 30.0, 40.0, 0.0]])
            gt = torch.tensor([[3.0, 4.0]])
            extras = {"observations": {"gt_scandots": gt}}
            actions_seen, frames = [], []
            recorder = types.SimpleNamespace(closed=False, track_camera=lambda: None)

            def capture(*, scandots):
                frames.append(scandots)
                return True

            def close():
                recorder.closed = True

            recorder.capture, recorder.close = capture, close

            def step(actions):
                actions_seen.append(actions.clone())
                obs[:, 2:4].add_(1.0)
                gt.add_(1.0)
                return obs, None, None, extras

            namespace = dict(
                torch=torch, time=time,
                simulation_app=types.SimpleNamespace(is_running=lambda: True),
                env=types.SimpleNamespace(step=step),
                obs=obs, extras=extras, is_lidar=True, record_scandots=panels,
                num_prop=2, num_scan=2, num_priv_explicit=1,
                estimator=types.SimpleNamespace(inference=lambda prop: torch.zeros(1, 1)),
                policy=lambda value, **kwargs: value[:, 2:3].clone(),
                snapshot_policy_panels=_load_snapshot_policy_panels(), recorder=recorder,
                args_cli=types.SimpleNamespace(video_length=2, real_time=False),
                timestep=0, dt=0.02,
            )
            exec(compile(ast.Module(body=[_load_play_loop_try()], type_ignores=[]), PLAY_PATH, "exec"), namespace)
            self.assertTrue(recorder.closed)
            return actions_seen, frames

        with_panels, frames = run(True)
        without_panels, empty_frames = run(False)
        for recorded, plain in zip(with_panels, without_panels, strict=True):
            torch.testing.assert_close(recorded, plain)
        self.assertEqual(empty_frames, [None, None])
        np.testing.assert_array_equal(frames[0][0][1], [[3.0, 4.0]])
        np.testing.assert_array_equal(frames[0][1][1], [[30.0, 40.0]])


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
        self.assertFalse(hasattr(args, "video"))
        self.assertFalse(args.multicam)
        self.assertEqual(args.num_envs, 1)
        self.assertEqual(args.video_length, 500)


    def test_panels_can_be_enabled_with_multicam(self):
        args = _parse_play_args(["--multicam", "--panels"])
        self.assertTrue(args.multicam)
        self.assertTrue(args.panels)
        self.assertTrue(args.enable_cameras)


    def test_panels_requires_multicam(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse_play_args(["--panels"])

    def test_removed_debug_and_panel_flags_are_rejected(self):
        for flag in ("--video", "--fixed_heading", "--em_error_plot", "--with_depth", "--with_scandots"):
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


    def test_gt_scandots_mode_returns_one_labeled_copy(self):
        scandots = self.snapshot(self.obs, 2, 3)
        self.obs[0, 2] = 99.0
        self.assertEqual([label for label, _ in scandots], ["GT"])
        np.testing.assert_array_equal(scandots[0][1], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_estimated_scandots_mode_returns_gt_and_estimated_copies(self):
        estimated = self.obs.clone()
        estimated[:, 2:5] += 10.0
        gt = self.obs[:, 2:5].clone()
        scandots = self.snapshot(estimated, 2, 3, gt_scandots=gt)
        gt.fill_(-3.0)
        self.obs[:, 2:5] = -1.0
        estimated[:, 2:5] = -2.0
        self.assertEqual([label for label, _ in scandots], ["GT", "Estimated"])
        np.testing.assert_array_equal(scandots[0][1], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_array_equal(scandots[1][1], np.array([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]))


if __name__ == "__main__":
    unittest.main()
