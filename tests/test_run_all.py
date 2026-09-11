"""Check launcher output isolation without importing models or running training."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import shlex
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def launch_commands(arguments):
    with (
        patch("sys.argv", ["run_all.py", *arguments]),
        patch("subprocess.run") as run,
        patch("os.makedirs"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        runpy.run_path(str(REPO_ROOT / "run_all.py"), run_name="__main__")
    for call in run.call_args_list:
        assert call.kwargs == {"check": True}
    return [call.args[0] for call in run.call_args_list]


def result_directory(command):
    return next(arg.split("=", 1)[1] for arg in command if arg.startswith("--result-dir="))


class ResultRootTests(unittest.TestCase):
    def test_default_output_layout_is_preserved_for_all_models_and_modes(self):
        commands = launch_commands(["--all", "--seeds", "0", "1"])
        self.assertEqual(len(commands), 15)
        expected = {
            f"./result_{model}_{mode}"
            for model in ("ours", "GIN", "iVAE")
            for mode in ("synthetic", "flow", "pendulum")
        }
        self.assertEqual({result_directory(command) for command in commands}, expected)

    def test_custom_root_changes_only_result_paths(self):
        arguments = ["--all", "--seeds", "0", "1"]
        original = launch_commands(arguments)
        custom_root = "/tmp/crl output isolation"
        relocated = launch_commands([*arguments, "--result-root", custom_root])
        self.assertEqual(len(original), len(relocated))
        for old, new in zip(original, relocated):
            expected = [
                f"--result-dir={os.path.join(custom_root, os.path.basename(result_directory(old)))}"
                if arg.startswith("--result-dir=") else arg
                for arg in old
            ]
            self.assertEqual(new, expected)

    def test_advertised_smoke_command_cannot_overwrite_default_paper_results(self):
        script = (REPO_ROOT / "scripts" / "smoke_test.sh").read_text()
        arguments = shlex.split(script.split("python run_all.py", 1)[1].replace("\\\n", " "))
        smoke = launch_commands(arguments)
        self.assertEqual(len(smoke), 1)
        self.assertIn("--max-epochs=1", smoke[0])
        self.assertIn("--seed=0", smoke[0])
        paper = launch_commands(["--ours", "--synthetic", "--dgp-list", "a", "--seeds", "0", "1"])
        self.assertNotEqual(
            os.path.normpath(result_directory(smoke[0])),
            os.path.normpath(result_directory(paper[0])),
        )
        self.assertEqual(result_directory(smoke[0]), "./results_smoke/result_ours_synthetic")


if __name__ == "__main__":
    unittest.main()
