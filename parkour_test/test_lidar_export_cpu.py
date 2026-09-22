"""CPU tests for the split-input LiDAR PPO deployment exporter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = REPO_ROOT / "deploy" / "tools" / "export_em_student_onnx.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("lidar_policy_export", EXPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LidarPolicyExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = _load_export_module()

    def _checkpoint(self):
        export = self.export
        actor = export.Actor(
            3,
            [5, 2],
            [8, 4],
            [4],
            export.resolve_nn_activation("elu"),
            num_prop=6,
            num_scan=4,
            num_hist=10,
            num_priv_latent=3,
            num_priv_explicit=2,
            state_history_encoder={"class_name": "StateHistoryEncoder", "channel_size": 2},
        )
        estimator = export.DefaultEstimator(
            num_prop=6,
            num_priv_explicit=2,
            hidden_dims=[7, 5],
            activation="elu",
        )
        model_state = {f"actor.{key}": value.clone() for key, value in actor.state_dict().items()}
        model_state["critic.0.weight"] = torch.zeros(1, 1)
        return {
            "algorithm": "PPOWithExtractor",
            "input_mode": "lidar_input",
            "model_state_dict": model_state,
            "estimator_state_dict": {key: value.clone() for key, value in estimator.state_dict().items()},
        }

    def test_builds_from_actor_prefixed_ppo_checkpoint_and_matches_production_path(self):
        checkpoint = self._checkpoint()
        self.export.validate_checkpoint(checkpoint)
        actor_state = self.export.extract_actor_state_dict(checkpoint["model_state_dict"])
        self.assertTrue(actor_state)
        self.assertTrue(all(not key.startswith("actor.") for key in actor_state))
        self.assertNotIn("critic.0.weight", actor_state)

        arch = self.export.infer_arch(actor_state, checkpoint["estimator_state_dict"])
        policy = self.export.build_policy(checkpoint, arch, "elu", False)
        prop, scan, hist, expected = self.export.check_against_production_path(policy, arch, n=4)
        actual = policy(prop, scan, hist)
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)
        self.assertEqual(tuple(actual.shape), (4, 3))

    def test_rejects_wrong_mode_algorithm_and_normalized_checkpoints(self):
        base = self._checkpoint()
        invalid = (
            {**base, "input_mode": "scandots_input"},
            {**base, "algorithm": "Other"},
            {**base, "obs_norm_state_dict": {}},
        )
        for checkpoint in invalid:
            with self.subTest(keys=checkpoint.keys()), self.assertRaises(ValueError):
                self.export.validate_checkpoint(checkpoint)

    def test_rejects_missing_or_unprefixed_actor_weights(self):
        checkpoint = self._checkpoint()
        with self.assertRaisesRegex(ValueError, "required keys"):
            self.export.validate_checkpoint({})
        with self.assertRaisesRegex(ValueError, "actor\\.\\*"):
            self.export.extract_actor_state_dict({"critic.0.weight": torch.zeros(1, 1)})


if __name__ == "__main__":
    unittest.main()
