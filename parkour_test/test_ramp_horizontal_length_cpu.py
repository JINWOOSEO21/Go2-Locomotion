"""CPU regression tests for trapezoid-ramp horizontal-run sampling.

The terrain module normally imports Isaac Lab through its mesh decorator.  To
keep this test usable while the simulator (and GPU) are occupied, the pure
height-field functions are extracted from the source AST and executed with
small configuration stubs.
"""

from __future__ import annotations

import ast
import math
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAIN_PATH = (
    REPO_ROOT
    / "parkour_isaaclab"
    / "terrains"
    / "extreme_parkour"
    / "extreme_parkour_terrians.py"
)
CFG_PATH = (
    REPO_ROOT
    / "parkour_isaaclab"
    / "terrains"
    / "extreme_parkour"
    / "extreme_parkour_terrains_cfg.py"
)
PARKOUR_CFG_PATH = (
    REPO_ROOT
    / "parkour_isaaclab"
    / "terrains"
    / "extreme_parkour"
    / "config"
    / "parkour.py"
)


def _without_annotations(node: ast.FunctionDef) -> ast.FunctionDef:
    """Make one extracted function independent of unavailable config types."""

    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    if node.args.vararg is not None:
        node.args.vararg.annotation = None
    if node.args.kwarg is not None:
        node.args.kwarg.annotation = None
    return node


def _load_ramp_height_field_function(name="parkour_trapezoid_ramp_terrain"):
    tree = ast.parse(TERRAIN_PATH.read_text(encoding="utf-8"), TERRAIN_PATH)
    names = {
        "padding_height_field_raw",
        "_lay_goals_over_trapezoid",
        "_course_span",
        "parkour_trapezoid_ramp_terrain",
        "parkour_trapezoid_stairs_terrain",
    }
    nodes = [
        _without_annotations(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        # The tested configurations disable roughness, but the name must still
        # exist in the extracted function's globals.
        "random_uniform_terrain": lambda difficulty, cfg, height_field: height_field,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), TERRAIN_PATH, "exec"), namespace)
    return namespace[name]


def _class_field_values(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), path)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    values = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            values[node.target.id] = ast.literal_eval(node.value)
    return values


def _parkour_ramp_kwargs() -> dict[str, object]:
    tree = ast.parse(PARKOUR_CFG_PATH.read_text(encoding="utf-8"), PARKOUR_CFG_PATH)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ExtremeParkourTrapezoidRampTerrainCfg"
    )
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}


def _terrain_cfg(run_range: tuple[float, float], plateau_range: tuple[float, float]):
    return types.SimpleNamespace(
        size=(16.0, 4.0),
        horizontal_scale=0.08,
        vertical_scale=0.005,
        platform_len=2.5,
        platform_height=0.0,
        course_width_range=(4.0, 4.0),
        y_range=(-0.4, 0.4),
        slope_angle="10 + 25*difficulty",
        ramp_length_range=run_range,
        plateau_len_range=plateau_range,
        end_margin=1.5,
        pad_width=0.1,
        pad_height=0.0,
        apply_roughness=False,
    )


class TestRampHorizontalLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generate = staticmethod(_load_ramp_height_field_function())

    def test_cfg_and_training_config_use_new_ranges(self):
        defaults = _class_field_values(CFG_PATH, "ExtremeParkourTrapezoidRampTerrainCfg")
        self.assertEqual(defaults["slope_angle"], "10 + 25*difficulty")
        self.assertEqual(defaults["ramp_length_range"], (2.5, 4.5))
        self.assertEqual(defaults["plateau_len_range"], (1.0, 2.0))
        self.assertNotIn("plateau_height_range", defaults)

        configured = _parkour_ramp_kwargs()
        self.assertEqual(configured["slope_angle"], "10 + 25*difficulty")
        self.assertEqual(configured["ramp_length_range"], (2.5, 4.5))
        self.assertEqual(configured["plateau_len_range"], (1.0, 2.0))
        self.assertNotIn("plateau_height_range", configured)

    def test_training_uses_eleven_levels(self):
        tree = ast.parse(PARKOUR_CFG_PATH.read_text())
        generator = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "ParkourTerrainGeneratorCfg"
        )
        rows = next(keyword.value for keyword in generator.keywords if keyword.arg == "num_rows")
        self.assertEqual(ast.literal_eval(rows), 11)
        student = REPO_ROOT / "parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_student_cfg.py"
        assignments = [
            node for node in ast.walk(ast.parse(student.read_text()))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "num_rows" for target in node.targets)
        ]
        # Training override plus the existing EVAL and PLAY overrides.
        self.assertEqual([ast.literal_eval(node.value) for node in assignments], [11, 5, 1])

    def test_stair_risers_at_all_eleven_levels(self):
        defaults = _class_field_values(CFG_PATH, "ExtremeParkourTrapezoidStairsTerrainCfg")
        generate = _load_ramp_height_field_function("parkour_trapezoid_stairs_terrain")
        cfg = _terrain_cfg((2.5, 2.5), (1.5, 1.5))
        cfg.step_height = defaults["step_height"]
        cfg.num_steps_range = (3, 3)
        cfg.x_range = (0.4, 0.48)
        expected_risers_cm = [5, 7, 8.5, 10.5, 12, 14, 16, 17.5, 19.5, 21, 23]
        for level, expected_cm in enumerate(expected_risers_cm):
            with self.subTest(level=level):
                field, _, _ = generate(level / 10, cfg, num_goals=8)
                profile = field[:, field.shape[1] // 2]
                rises = np.diff(profile)
                positive = rises[rises > 0] * cfg.vertical_scale
                negative = -rises[rises < 0] * cfg.vertical_scale
                np.testing.assert_allclose(positive, [expected_cm / 100] * 3)
                np.testing.assert_allclose(negative, [expected_cm / 100] * 3)

    def test_all_curriculum_angles_preserve_sampled_runs_and_fit_tile(self):
        horizontal_scale = 0.08
        vertical_scale = 0.005
        width_pixels = round(16.0 / horizontal_scale)
        platform_pixels = round(2.5 / horizontal_scale)
        end_margin_pixels = round(1.5 / horizontal_scale)

        for level in range(11):
            difficulty = level / 10
            slope_degrees = 10 + 25 * difficulty
            for run_metres in (2.5, 4.5):
                for plateau_metres in (1.0, 2.0):
                    with self.subTest(
                        slope_degrees=slope_degrees,
                        run_metres=run_metres,
                        plateau_metres=plateau_metres,
                    ):
                        cfg = _terrain_cfg((run_metres, run_metres), (plateau_metres, plateau_metres))
                        # All sampling ranges are fixed at their endpoint in
                        # this subtest, so random sampling remains deterministic.
                        height_field, _, _ = self.generate(difficulty, cfg, num_goals=8)

                        run_pixels = round(run_metres / horizontal_scale)
                        plateau_pixels = round(plateau_metres / horizontal_scale)
                        up_start = platform_pixels
                        up_end = up_start + run_pixels
                        down_start = up_end + plateau_pixels
                        down_end = down_start + run_pixels
                        centreline = height_field[:, height_field.shape[1] // 2]

                        expected_height = round(
                            run_pixels
                            * horizontal_scale
                            * math.tan(math.radians(slope_degrees))
                            / vertical_scale
                        )
                        uphill = centreline[up_start:up_end]
                        plateau = centreline[up_end:down_start]
                        downhill = centreline[down_start:down_end]

                        self.assertEqual(len(uphill), run_pixels)
                        self.assertEqual(len(downhill), run_pixels)
                        self.assertEqual(len(plateau), plateau_pixels)
                        self.assertEqual(int(uphill[-1]), expected_height)
                        np.testing.assert_array_equal(
                            plateau,
                            np.full(plateau_pixels, expected_height, dtype=height_field.dtype),
                        )
                        np.testing.assert_array_equal(
                            uphill + downhill,
                            np.full(run_pixels, expected_height, dtype=height_field.dtype),
                        )
                        self.assertEqual(int(downhill[-1]), 0)
                        self.assertLessEqual(down_end + end_margin_pixels, width_pixels)


if __name__ == "__main__":
    unittest.main()
