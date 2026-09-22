import ast
import types
import unittest
import warnings
from pathlib import Path

RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "rsl_rl" / "modules" / "on_policy_runner_with_extractor.py"


def _load_schedule_namespace():
    tree = ast.parse(RUNNER_PATH.read_text())
    wanted_module_nodes = {
        "_DISTURBANCE_SCALES",
        "_disturbance_scale_for_progress",
    }
    wanted_methods = {
        "_begin_disturbance_schedule",
        "_apply_disturbance_schedule",
        "_advance_disturbance_schedule",
        "_restore_disturbance_schedule",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_module_nodes for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_module_nodes:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunnerWithExtractor":
            nodes.extend(
                method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in wanted_methods
            )
    namespace = {"warnings": warnings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), RUNNER_PATH, "exec"), namespace)
    return namespace


class _ScheduleRunner:
    def __init__(self, namespace):
        self.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace())
        self.current_learning_iteration = 16_999
        self._disturbance_schedule_total_iterations = None
        self._disturbance_schedule_completed_iterations = 0
        self._disturbance_schedule_resume_pending = False
        for name in (
            "_begin_disturbance_schedule",
            "_apply_disturbance_schedule",
            "_advance_disturbance_schedule",
            "_restore_disturbance_schedule",
        ):
            setattr(self, name, types.MethodType(namespace[name], self))


class DisturbanceScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _load_schedule_namespace()

    def test_five_equal_stages(self):
        scale_for = self.namespace["_disturbance_scale_for_progress"]
        self.assertEqual(
            [scale_for(completed, 10) for completed in range(10)],
            [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0],
        )
        self.assertEqual(scale_for(10, 10), 1.0)
        with self.assertRaises(ValueError):
            scale_for(0, 0)

    def test_fresh_schedule_uses_additional_iterations_not_checkpoint_iteration(self):
        runner = _ScheduleRunner(self.namespace)
        runner._begin_disturbance_schedule(5)
        scales = []
        for _ in range(5):
            runner._apply_disturbance_schedule()
            scales.append(runner.env.unwrapped.disturbance_scale)
            runner._advance_disturbance_schedule()
        self.assertEqual(scales, [0.2, 0.4, 0.6, 0.8, 1.0])

    def test_checkpoint_schedule_resumes_then_a_new_learn_call_restarts(self):
        runner = _ScheduleRunner(self.namespace)
        runner._restore_disturbance_schedule(
            {"disturbance_schedule": {"total_iterations": 10, "completed_iterations": 4}}
        )
        runner._begin_disturbance_schedule(100)
        self.assertEqual(runner.env.unwrapped.disturbance_scale, 0.6)
        self.assertEqual(runner._disturbance_schedule_total_iterations, 10)

        runner._begin_disturbance_schedule(5)
        self.assertEqual(runner.env.unwrapped.disturbance_scale, 0.2)
        self.assertEqual(runner._disturbance_schedule_total_iterations, 5)

    def test_checkpoint_without_schedule_starts_a_fresh_schedule(self):
        runner = _ScheduleRunner(self.namespace)
        runner._restore_disturbance_schedule({})
        runner._begin_disturbance_schedule(5)
        self.assertEqual(runner.env.unwrapped.disturbance_scale, 0.2)
        self.assertEqual(runner._disturbance_schedule_completed_iterations, 0)

    def test_ppo_learning_loop_applies_and_advances_the_schedule(self):
        tree = ast.parse(RUNNER_PATH.read_text())
        runner_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunnerWithExtractor"
        )
        methods = {method.name: method for method in runner_class.body if isinstance(method, ast.FunctionDef)}
        calls = [
            node.func.attr
            for node in ast.walk(methods["learn_rl"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("_begin_disturbance_schedule", calls)
        self.assertIn("_apply_disturbance_schedule", calls)
        self.assertIn("_advance_disturbance_schedule", calls)
        self.assertNotIn("learn_vision", methods)
        self.assertNotIn("learn_em", methods)


if __name__ == "__main__":
    unittest.main()
