"""Test the collision oracle using real CSV contents, without model imports."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smoke_collision", REPO / "scripts/reproduce_smoke_output_collision.py"
)
collision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collision)


class SmokeCollisionTests(unittest.TestCase):
    def test_current_advertised_smoke_preserves_full_run_result(self):
        result = collision.diagnose(REPO)
        self.assertEqual(result["status"], "isolated")
        self.assertEqual(result["paper_before"], result["paper_after"])
        self.assertNotEqual(result["paper_after"], result["smoke_result"])
        self.assertEqual((result["paper_epochs"], result["smoke_epochs"]), (20, 1))

    def test_restoring_historical_smoke_arguments_actually_overwrites_csv(self):
        arguments = collision.smoke_arguments(REPO)
        root_index = arguments.index("--result-root")
        del arguments[root_index:root_index + 2]
        with patch.object(collision, "smoke_arguments", return_value=arguments):
            result = collision.diagnose(REPO)
        self.assertEqual(result["status"], "collision")
        self.assertIn("0.8,0.9", result["paper_before"])
        self.assertIn("0.1,0.2", result["paper_after"])
        self.assertEqual(result["paper_after"], result["smoke_result"])

    def test_oracle_observes_writer_semantics_not_just_path_equality(self):
        # An append-only writer is a controlled negative example: the same path
        # collides, but historical content remains. Read actual bytes to prove it.
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "model").mkdir()
            source = (REPO / "model/our_model.py").read_text()
            original = 'f"{self.seed}_dci.csv"), "w", newline=""'
            self.assertEqual(source.count(original), 1)
            (repo / "model/our_model.py").write_text(
                source.replace(original, 'f"{self.seed}_dci.csv"), "a", newline=""')
            )
            target = repo / "results/a/0_dci.csv"
            result = collision.simulate_writes(repo, target, target, 0)
        self.assertIn("0.8,0.9", result["paper_after"])
        self.assertIn("0.1,0.2", result["paper_after"])
        self.assertFalse(result["paper_result_overwritten"])
        self.assertTrue(result["paper_result_modified"])
        self.assertNotEqual(result["paper_before"], result["paper_after"])

    def test_shell_expressions_are_rejected_as_data(self):
        for command in (
            "python run_all.py --ours; touch /tmp/should-not-exist",
            "python run_all.py --result-root $(pwd)",
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                (repo / "scripts").mkdir()
                (repo / "scripts/smoke_test.sh").write_text(command)
                with self.assertRaises(ValueError):
                    collision.smoke_arguments(repo)


if __name__ == "__main__":
    unittest.main()
