"""CPU-only checks for the Go2 task registry.

The registry calls are executed from the AST with a fake Gym collector so the
test does not import Isaac Sim, Gymnasium, or the task configuration modules.
"""

from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "parkour_tasks" / "parkour_tasks" / "extreme_parkour_task" / "config" / "go2" / "__init__.py"
DEMO_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "demo.py"
PACKAGE_NAME = "parkour_tasks.extreme_parkour_task.config.go2"
AGENTS_NAME = f"{PACKAGE_NAME}.agents"


def _collect_registrations():
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"), REGISTRY_PATH)
    register_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "register"
    ]
    registrations = []
    gym = types.SimpleNamespace(register=lambda **kwargs: registrations.append(kwargs))
    namespace = {
        "__name__": PACKAGE_NAME,
        "agents": types.SimpleNamespace(__name__=AGENTS_NAME),
        "gym": gym,
    }
    exec(compile(ast.Module(body=register_calls, type_ignores=[]), REGISTRY_PATH, "exec"), namespace)
    return registrations


def _demo_enables_cameras(task_id):
    tree = ast.parse(DEMO_PATH.read_text(encoding="utf-8"), DEMO_PATH)
    task_camera_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and "args_cli.task" in ast.unparse(node.test)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "enable_cameras" for target in child.targets)
            for child in node.body
        )
    )
    args_cli = types.SimpleNamespace(task=task_id, enable_cameras=False)
    exec(compile(ast.Module(body=[task_camera_guard], type_ignores=[]), DEMO_PATH, "exec"), {"args_cli": args_cli})
    return args_cli.enable_cameras


class TestTaskRegistry(unittest.TestCase):
    def test_registry_ids_and_config_mapping(self):
        registrations = _collect_registrations()
        expected_configs = {
            "Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Train-v0": (
                "parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg",
                "rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0": (
                "parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_PLAY",
                "rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Eval-v0": (
                "parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_EVAL",
                "rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Depth-Unitree-Go2-Train-v0": (
                "parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg",
                "rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Depth-Unitree-Go2-Play-v0": (
                "parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_PLAY",
                "rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Depth-Unitree-Go2-Eval-v0": (
                "parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_EVAL",
                "rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Train-v0": (
                "parkour_em_student_cfg:UnitreeGo2EMStudentParkourEnvCfg",
                "rsl_em_student_ppo_cfg:UnitreeGo2ParkourEMStudentPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0": (
                "parkour_em_student_cfg:UnitreeGo2EMStudentParkourEnvCfg_PLAY",
                "rsl_em_student_ppo_cfg:UnitreeGo2ParkourEMStudentPPORunnerCfg",
            ),
            "Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Eval-v0": (
                "parkour_em_student_cfg:UnitreeGo2EMStudentParkourEnvCfg_EVAL",
                "rsl_em_student_ppo_cfg:UnitreeGo2ParkourEMStudentPPORunnerCfg",
            ),
        }

        self.assertEqual(len(registrations), 9)
        self.assertEqual(len({registration["id"] for registration in registrations}), 9)
        self.assertEqual({registration["id"] for registration in registrations}, set(expected_configs))
        self.assertTrue(all(registration["disable_env_checker"] for registration in registrations))
        self.assertTrue(
            all(
                registration["entry_point"] == "parkour_isaaclab.envs:ParkourManagerBasedRLEnv"
                for registration in registrations
            )
        )
        for registration in registrations:
            env_cfg, runner_cfg = expected_configs[registration["id"]]
            self.assertEqual(registration["kwargs"]["env_cfg_entry_point"], f"{PACKAGE_NAME}.{env_cfg}")
            self.assertEqual(registration["kwargs"]["rsl_rl_cfg_entry_point"], f"{AGENTS_NAME}.{runner_cfg}")
            self.assertEqual(
                registration["kwargs"]["skrl_cfg_entry_point"], f"{AGENTS_NAME}:skrl_parkour_ppo_cfg.yaml"
            )

    def test_legacy_ids_are_absent(self):
        task_ids = {registration["id"] for registration in _collect_registrations()}
        self.assertFalse(any("Teacher" in task_id or "Student" in task_id for task_id in task_ids))

    def test_demo_enables_cameras_for_sensor_input_tasks(self):
        self.assertTrue(_demo_enables_cameras("Isaac-Extreme-Parkour-Depth-Unitree-Go2-Play-v0"))
        self.assertTrue(_demo_enables_cameras("Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0"))
        self.assertFalse(_demo_enables_cameras("Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Play-v0"))
        self.assertFalse(_demo_enables_cameras(None))


if __name__ == "__main__":
    unittest.main()
