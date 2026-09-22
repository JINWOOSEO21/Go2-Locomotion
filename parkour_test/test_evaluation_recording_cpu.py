"""Exercise evaluation recording without starting Isaac Sim or writing videos."""

import ast
import os
import time
import types
import unittest
from collections import deque
from pathlib import Path

import torch


class EvaluationRecordingTest(unittest.TestCase):
    def test_recording_limit_does_not_truncate_evaluation_and_resources_close(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/evaluation.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        block = next(n for n in main.body if isinstance(n, ast.Try))

        class Recorder:
            def __init__(self, env, output, fps):
                self.output, self.fps = output, fps
                self.frames = 0
                self.closed = False

            def track_camera(self):
                pass

            def capture(self):
                self.frames += 1
                return True

            def close(self):
                self.closed = True

        class Env:
            def __init__(self):
                self.steps, self.closed = 0, False

            def step(self, actions):
                self.steps += 1
                return torch.zeros(2, 5), torch.ones(2), torch.ones(2), {}

            def close(self):
                self.closed = True

        env = Env()
        namespace = dict(
            torch=torch, os=os, time=time, env=env,
            args_cli=types.SimpleNamespace(multicam=True, video_length=2, real_time=False),
            recorder=None, PerEnvVideoRecorder=Recorder, log_dir="/unused/run", dt=0.02,
            tqdm=lambda steps: range(4), timestep=0, obs=torch.zeros(2, 5),
            num_prop=2, num_scan=2, num_priv_explicit=1,
            estimator=types.SimpleNamespace(inference=lambda obs: torch.zeros(2, 1)),
            policy=lambda obs, **kwargs: torch.zeros(2, 1),
            base_parkour=types.SimpleNamespace(cur_goal_idx=torch.zeros(2)),
            reward_feet_edge=types.SimpleNamespace(feet_at_edge=torch.zeros(2, 4)),
            cur_reward_sum=torch.zeros(2), cur_episode_length=torch.zeros(2),
            cur_time_from_start=torch.zeros(2), rewbuffer=deque(), lenbuffer=deque(),
            num_waypoints_buffer=deque(), edge_violation_buffer=deque(),
        )
        code = compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")
        exec(code, namespace)
        recorder = namespace["recorder"]
        self.assertEqual(env.steps, 4)
        self.assertEqual(recorder.frames, 2)
        self.assertEqual(recorder.output, "/unused/run/videos")
        self.assertEqual(recorder.fps, 50)
        self.assertTrue(env.closed and recorder.closed)

        def fail(actions):
            raise RuntimeError("step failed")

        env.closed = False
        env.step = fail
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            exec(code, namespace)
        self.assertTrue(env.closed and namespace["recorder"].closed)


if __name__ == "__main__":
    unittest.main()
