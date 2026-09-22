"""CPU-only contract tests for the Lidar PPO runner and observation adapter.

The repository's default Python does not include Isaac Lab or PyTorch, so these
tests execute the small pure-Python scheduling helpers and wrapper methods from
their AST. Full PPO integration remains covered by the simulator environment.
"""

from __future__ import annotations

import ast
import math
import types
import unittest
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "modules" / "on_policy_runner_with_extractor.py"
WRAPPER_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "vecenv_wrapper.py"


class _Tensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def clone(self):
        return _Tensor(self.rows)

    def __getitem__(self, key):
        row_key, col_key = key
        rows = self.rows[row_key]
        if isinstance(row_key, int):
            rows = [rows]
        return _Tensor([row[col_key] for row in rows])

    def __setitem__(self, key, value):
        row_key, col_key = key
        indices = range(len(self.rows))[row_key]
        if isinstance(indices, int):
            indices = [indices]
        for source_index, target_index in enumerate(indices):
            self.rows[target_index][col_key] = value.rows[source_index]

    def to(self, dtype=None):
        return self


class _DoneVector:
    def __init__(self, values):
        self.values = list(values)

    def __or__(self, other):
        return _DoneVector([left or right for left, right in zip(self.values, other.values)])

    def to(self, dtype=None):
        return self


def _class_methods(path: Path, class_name: str, method_names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), path)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    exec(compile(ast.Module(body=methods, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _runner_namespace():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), RUNNER_PATH)
    names = {
        "lidar_noise_scale_for_progress",
        "_begin_lidar_noise_schedule",
        "_apply_lidar_noise_schedule",
        "_advance_lidar_noise_schedule",
        "_restore_lidar_noise_schedule",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunnerWithExtractor":
            nodes.extend(
                method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in names
            )
    namespace = {"math": math, "warnings": warnings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), RUNNER_PATH, "exec"), namespace)
    return namespace


class _FakeEnv:
    def __init__(self, observations):
        self.observations = observations
        self.unwrapped = types.SimpleNamespace(
            _get_observations=lambda: self.observations,
            cfg=types.SimpleNamespace(is_finite_horizon=True),
        )
        self.step_extras = {"marker": 1}

    def reset(self):
        return self.observations, {}

    def step(self, actions):
        return self.observations, "reward", _DoneVector([False, True]), _DoneVector([False, False]), self.step_extras


class _WrapperHarness:
    @property
    def unwrapped(self):
        return self.env.unwrapped


class LidarObservationAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = types.SimpleNamespace(Tensor=_Tensor, long="long")
        namespace = _class_methods(
            WRAPPER_PATH,
            "ParkourRslRlVecEnvWrapper",
            {"configure_policy_input", "_adapt_policy_observations", "get_observations", "reset", "step"},
            {"torch": torch},
        )
        for name, function in namespace.items():
            if callable(function):
                setattr(_WrapperHarness, name, function)

    def _wrapper(self):
        policy = _Tensor([[1, 2, 10, 11, 90], [3, 4, 12, 13, 91]])
        critic = _Tensor([[100, 101], [102, 103]])
        em_scan = _Tensor([[20, 21], [22, 23]])
        observations = {"policy": policy, "critic": critic, "em_scan": em_scan}
        wrapper = _WrapperHarness()
        wrapper.env = _FakeEnv(observations)
        wrapper.clip_actions = None
        wrapper._policy_input_mode = "scandots_input"
        wrapper._policy_num_prop = None
        wrapper._policy_num_scan = None
        wrapper.configure_policy_input("lidar_input", 2, 2)
        return wrapper, observations

    def _assert_adapted(self, obs, extras, raw):
        self.assertEqual(obs.rows, [[1, 2, 20, 21, 90], [3, 4, 22, 23, 91]])
        self.assertEqual(extras["observations"]["policy"].rows, obs.rows)
        self.assertEqual(extras["observations"]["gt_scandots"].rows, [[10, 11], [12, 13]])
        self.assertEqual(raw["policy"].rows, [[1, 2, 10, 11, 90], [3, 4, 12, 13, 91]])
        self.assertEqual(raw["critic"].rows, [[100, 101], [102, 103]])
        self.assertIsNot(obs, raw["policy"])
        self.assertIsNot(extras["observations"]["gt_scandots"], raw["policy"])

    def test_get_reset_and_step_replace_only_policy_scan(self):
        wrapper, raw = self._wrapper()
        obs, extras = wrapper.get_observations()
        self._assert_adapted(obs, extras, raw)

        obs, extras = wrapper.reset()
        self._assert_adapted(obs, extras, raw)

        obs, reward, dones, extras = wrapper.step("actions")
        self.assertEqual(reward, "reward")
        self.assertEqual(dones.values, [False, True])
        self._assert_adapted(obs, extras, raw)

    def test_missing_or_wrong_shape_em_scan_fails_clearly(self):
        wrapper, raw = self._wrapper()
        without_em = dict(raw)
        without_em.pop("em_scan")
        with self.assertRaisesRegex(KeyError, "em_scan"):
            wrapper._adapt_policy_observations(without_em)
        wrong_shape = dict(raw)
        wrong_shape["em_scan"] = _Tensor([[1], [2]])
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            wrapper._adapt_policy_observations(wrong_shape)

    def test_scandots_mode_is_an_identity_path(self):
        wrapper, raw = self._wrapper()
        wrapper.configure_policy_input("scandots_input", 2, 2)
        self.assertIs(wrapper._adapt_policy_observations(raw), raw)


class LidarNoiseScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _runner_namespace()

    def _runner(self, ratio=0.7):
        runner = types.SimpleNamespace(
            input_mode="lidar_input",
            lidar_noise_ramp_ratio=ratio,
            env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(lidar_noise_scale=99.0)),
            _lidar_noise_schedule_total_iterations=None,
            _lidar_noise_schedule_completed_iterations=0,
            _lidar_noise_schedule_ramp_ratio=ratio,
            _lidar_noise_schedule_resume_pending=False,
        )
        for name in (
            "_begin_lidar_noise_schedule",
            "_apply_lidar_noise_schedule",
            "_advance_lidar_noise_schedule",
            "_restore_lidar_noise_schedule",
        ):
            setattr(runner, name, types.MethodType(self.namespace[name], runner))
        return runner

    def test_30k_schedule_with_two_thirds_ramp(self):
        scale = self.namespace["lidar_noise_scale_for_progress"]
        ratio = 2.0 / 3.0
        self.assertEqual(scale(0, 30_000, ratio), 0.0)
        self.assertEqual(scale(5_000, 30_000, ratio), 0.0)
        self.assertAlmostEqual(scale(15_000, 30_000, ratio), 0.5)
        self.assertEqual(scale(25_000, 30_000, ratio), 1.0)
        self.assertEqual(scale(30_000, 30_000, ratio), 1.0)

    def test_default_ratio_and_validation(self):
        scale = self.namespace["lidar_noise_scale_for_progress"]
        self.assertEqual(scale(4_500, 30_000), 0.0)
        self.assertAlmostEqual(scale(15_000, 30_000), 0.5)
        self.assertEqual(scale(25_500, 30_000), 1.0)
        for ratio in (0.0, -0.1, 1.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                scale(0, 30_000, ratio)
        with self.assertRaises(ValueError):
            scale(0, 0, 0.7)

    def test_resume_preserves_saved_horizon_ratio_and_progress(self):
        runner = self._runner(0.7)
        checkpoint = {
            "lidar_noise_schedule": {
                "total_iterations": 30_000,
                "completed_iterations": 15_000,
                "ramp_ratio": 2.0 / 3.0,
            }
        }
        with self.assertWarnsRegex(UserWarning, "saved ratio remains authoritative"):
            runner._restore_lidar_noise_schedule(checkpoint)
        runner._begin_lidar_noise_schedule(99_999)
        self.assertEqual(runner._lidar_noise_schedule_total_iterations, 30_000)
        self.assertAlmostEqual(runner.env.unwrapped.lidar_noise_scale, 0.5)
        runner._advance_lidar_noise_schedule()
        self.assertEqual(runner._lidar_noise_schedule_completed_iterations, 15_001)


class LidarCheckpointContractTest(unittest.TestCase):
    def test_checkpoint_metadata_and_warm_start_boundaries_are_present(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), RUNNER_PATH)
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)}
        save_source = ast.unparse(methods["save"])
        load_source = ast.unparse(methods["load"])
        self.assertIn("lidar_noise_schedule", save_source)
        self.assertIn("input_mode", save_source)
        self.assertIn("algorithm", save_source)
        self.assertIn("warm_start", load_source)
        self.assertIn("depth_actor_state_dict", load_source)
        self.assertIn("Use --init_checkpoint", load_source)
        warm_start_branch = next(
            node
            for node in ast.walk(methods["load"])
            if isinstance(node, ast.If) and ast.unparse(node.test) == "warm_start"
        )
        warm_start_source = ast.unparse(warm_start_branch)
        self.assertIn("model_state_dict", warm_start_source)
        self.assertIn("estimator_state_dict", warm_start_source)
        self.assertNotIn("optimizer_state_dict", warm_start_source)
        self.assertNotIn("_restore_terrain_levels", warm_start_source)

    def test_rl_loop_applies_and_advances_noise_after_update(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), RUNNER_PATH)
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        learn = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "learn_rl")
        calls = [ast.unparse(node.func) for node in ast.walk(learn) if isinstance(node, ast.Call)]
        self.assertIn("self._begin_lidar_noise_schedule", calls)
        self.assertIn("self._apply_lidar_noise_schedule", calls)
        self.assertIn("self._advance_lidar_noise_schedule", calls)
        source = ast.unparse(learn)
        self.assertLess(
            source.index("loss_dict = self.alg.update()"),
            source.index("self._advance_lidar_noise_schedule()"),
        )
        self.assertIn("Lidar/noise_scale", source)


if __name__ == "__main__":
    unittest.main()
