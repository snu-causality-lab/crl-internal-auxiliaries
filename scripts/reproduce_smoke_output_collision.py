#!/usr/bin/env python3
"""Reproduce smoke/full-run output collisions without training or dependencies.

Run against this checkout or a historical checkout:
    python scripts/reproduce_smoke_output_collision.py --repo /path/to/checkout

Exit 0: isolated outputs; exit 1: smoke and full run share the same result path;
exit 2: unsupported source/command layout or another diagnostic error.
Only temporary files are written. The shell script is parsed, never executed;
runner subprocesses and directory creation are intercepted. The original DCI
CSV write block is extracted with AST, without importing the model module.
The metrics are synthetic sentinels, not new experimental measurements.
"""

import argparse
import ast
import builtins
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def smoke_arguments(repo):
    """Read the single advertised launcher command as data, not shell code."""
    script = (repo / "scripts/smoke_test.sh").read_text()
    commands = []
    for line in script.replace("\\\n", " ").splitlines():
        if not line.strip().startswith(("python run_all.py", "python3 run_all.py")):
            continue
        tokens = shlex.split(line, comments=True)
        if any(any(char in token for char in "$`;|&<>") for token in tokens):
            raise ValueError("Smoke command contains unsupported shell syntax")
        commands.append(tokens[2:])
    if len(commands) != 1:
        raise ValueError("Expected one literal python run_all.py smoke command")
    return commands[0]


def capture_commands(repo, arguments):
    """Execute only the lightweight launcher; never launch child processes."""
    with (
        patch.object(sys, "argv", ["run_all.py", *arguments]),
        patch("subprocess.run") as run,
        patch("subprocess.Popen", side_effect=AssertionError("Unexpected subprocess")),
        patch("os.makedirs"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        runpy.run_path(str(repo / "run_all.py"), run_name="__main__")
    commands = []
    for call in run.call_args_list:
        if call.kwargs != {"check": True} or not isinstance(call.args[0], list):
            raise ValueError("Expected argv-list subprocess dispatch with check=True")
        commands.append(call.args[0])
    return commands


def option(command, name, default=None):
    for index, token in enumerate(command):
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
        if token == name:
            return command[index + 1]
    if default is not None:
        return default
    raise ValueError(f"Missing command option: {name}")


def result_path(repo, command):
    return (
        repo / option(command, "--result-dir") / option(command, "--dgp")
        / f"{int(option(command, '--seed'))}_dci.csv"
    ).resolve()


def dci_write_block(repo):
    """Extract the actual CSV writer, not a rewritten simulation of its mode."""
    source_path = repo / "model/our_model.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "open"
            and any(
                isinstance(value, ast.Constant) and value.value == "_dci.csv"
                for value in ast.walk(item.context_expr)
            )
            for item in node.items
        )
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one DCI CSV write block in model/our_model.py")
    node = candidates[0]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    return compile(module, str(source_path), "exec"), node.lineno


def simulate_writes(repo, paper_path, smoke_path, seed):
    """Use the original writer, with all opens constrained to a temporary tree."""
    block, line = dci_write_block(repo)
    with tempfile.TemporaryDirectory(prefix="crl-smoke-collision-") as temporary:
        temporary_root = Path(temporary).resolve()

        def mapped(path):
            # Absolute normalized paths retain their equality under this mapping.
            return temporary_root.joinpath(*path.parts[1:])

        def guarded_open(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if temporary_root not in resolved.parents:
                raise ValueError("Writer attempted to access outside the temporary tree")
            return builtins.open(resolved, *args, **kwargs)

        def write(path, values):
            path.parent.mkdir(parents=True, exist_ok=True)
            exec(block, {
                "csv": csv, "os": os, "open": guarded_open,
                "self": SimpleNamespace(seed=seed),
                "result_dgp_dir": str(path.parent), "dci_metric": values,
            })

        paper = mapped(paper_path)
        smoke = mapped(smoke_path)
        write(paper, {"disentanglement": 0.8, "completeness": 0.9})
        before = paper.read_text()
        write(smoke, {"disentanglement": 0.1, "completeness": 0.2})
        after = paper.read_text()
        smoke_contents = smoke.read_text()
    return {
        "paper_result_overwritten": before not in after,
        "paper_result_modified": before != after,
        "paper_before": before,
        "paper_after": after,
        "smoke_result": smoke_contents,
        "writer": f"model/our_model.py:{line}",
    }


def diagnose(repo):
    repo = Path(repo).resolve()
    smoke_commands = capture_commands(repo, smoke_arguments(repo))
    if len(smoke_commands) != 1:
        raise ValueError("Expected one smoke training dispatch")
    smoke = smoke_commands[0]
    if (option(smoke, "-m") != "experiments.Ours.main"
            or option(smoke, "--mode", "synthetic") != "synthetic"):
        raise ValueError("This reproducer covers the advertised Ours/synthetic smoke run")
    seed = int(option(smoke, "--seed"))
    dgp = option(smoke, "--dgp")
    paper_commands = capture_commands(repo, [
        "--ours", "--synthetic", "--dgp-list", dgp,
        "--seeds", str(seed), str(seed + 1),
    ])
    if len(paper_commands) != 1:
        raise ValueError("Expected one matching full-run training dispatch")
    paper = paper_commands[0]
    paper_path, smoke_path = result_path(repo, paper), result_path(repo, smoke)
    result = simulate_writes(repo, paper_path, smoke_path, seed)
    return {
        "status": "collision" if paper_path == smoke_path else "isolated",
        "repo": str(repo), "model": "ours", "mode": "synthetic", "dgp": dgp,
        "seed": seed,
        "paper_epochs": int(option(paper, "--max-epochs")),
        "smoke_epochs": int(option(smoke, "--max-epochs")),
        "paper_path": str(paper_path), "smoke_path": str(smoke_path),
        "paths_collide": paper_path == smoke_path,
        "training_executed": False,
        "sentinel_metrics_only": True,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = diagnose(args.repo)
    except (OSError, ValueError, AssertionError, SyntaxError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return int(result["paths_collide"])


if __name__ == "__main__":
    sys.exit(main())
