#!/usr/bin/env python3
"""Score an outside attempt at any task, the way Harbor would.

    python factory/grade.py --task task6 --patch C:\\path\\to\\solver_output

`--patch` is a directory holding the package (or containing it). Runs the task's
real suite against it and reports the reward, plus which fail_to_pass tests are
still failing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate", HERE / "gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def grade(task: pathlib.Path, patch: pathlib.Path) -> tuple[int, dict]:
    config = json.loads((task / "tests" / "config.json").read_text(encoding="utf-8"))
    f2p, p2p = config["fail_to_pass"], config["pass_to_pass"]

    shipped = next(p for p in (task / "environment").iterdir()
                   if p.is_dir() and (p / "__init__.py").exists())
    package = patch if (patch / "__init__.py").exists() else patch / shipped.name
    if not (package / "__init__.py").exists():
        raise SystemExit(f"no {shipped.name} package found under {patch}")

    tag = f"gate-{task.name}:latest"
    if not gate.docker_build(task, tag):
        raise SystemExit("image build failed")

    import tomllib
    gpus = bool(tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
                .get("environment", {}).get("gpus"))

    tmp = pathlib.Path(tempfile.mkdtemp())
    tests = gate.stage(task / "tests", tmp / "tests")
    staged = tmp / "graded" / shipped.name
    staged.parent.mkdir(parents=True)
    gate.stage(package, staged)

    reward, report = gate.run_suite(tag, staged, tests, tmp / "logs", gpus)
    still_failing = [n for n in f2p if report.get(n) != "PASSED"]
    regressed = [n for n in p2p if report.get(n) != "PASSED"]

    print()
    print("=" * 74)
    print(f"reward Harbor would give: {reward}")
    print("=" * 74)
    print(f"  fail_to_pass fixed     {len(f2p) - len(still_failing)}/{len(f2p)}")
    print(f"  pass_to_pass intact    {len(p2p) - len(regressed)}/{len(p2p)}")
    for name in still_failing:
        print(f"  !! still failing   {name} -> {report.get(name)}")
    for name in regressed:
        print(f"  !! regressed       {name} -> {report.get(name)}")
    print()
    if reward == "1":
        print("VERDICT: this attempt would score 1. The task did not defeat it.")
        return 1, {"reward": reward, "still_failing": still_failing,
                   "regressed": regressed}
    print("VERDICT: this attempt would score 0. The task defeated it.")
    return 0, {"reward": reward, "still_failing": still_failing,
               "regressed": regressed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--json", help="also write the result here")
    args = parser.parse_args()

    task = pathlib.Path(args.task).resolve()
    patch = pathlib.Path(args.patch).resolve()
    code, result = grade(task, patch)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
