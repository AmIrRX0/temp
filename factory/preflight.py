#!/usr/bin/env python3
"""Is the factory ready to build a task? Run before starting an agent.

    python factory/preflight.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBLEMS: list[str] = []


def need(label: str, condition: bool, detail: str, fix: str) -> None:
    if condition:
        print(f"  ok    {label}: {detail}")
    else:
        print(f"  FAIL  {label}: {detail}")
        print(f"        fix: {fix}")
        PROBLEMS.append(label)


def main() -> int:
    print("=" * 74)
    print("factory preflight")
    print("=" * 74)

    standards = REPO / "factory" / "STANDARDS.md"
    need("standards", standards.exists(),
         str(standards) if standards.exists() else "not found",
         "save your acceptance-standards document as factory/STANDARDS.md. "
         "Without it the agent invents its own bar.")

    ideas = REPO / "task_ideas.md"
    need("idea pool", ideas.exists(),
         str(ideas) if ideas.exists() else "not found",
         "put task_ideas.md at the repo root.")

    reference = REPO / "task5" / "tests" / "test.sh"
    need("reference task", reference.exists(),
         "task5/ present" if reference.exists() else "task5/ missing",
         "the agent copies task5's harness shape; keep it in the repo.")

    docker = shutil.which("docker")
    need("docker", bool(docker), docker or "not on PATH",
         "install Docker Desktop and start it. The gate shells out to docker.")

    if docker:
        info = subprocess.run(["docker", "info"], capture_output=True, text=True)
        need("docker daemon", info.returncode == 0,
             "running" if info.returncode == 0 else "not reachable",
             "start Docker Desktop.")

        gpu = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
             "python", "-c", "import torch; print(torch.cuda.get_device_name(0))"],
            capture_output=True, text=True)
        if gpu.returncode == 0:
            print(f"  ok    gpu: {gpu.stdout.strip()}")
        else:
            print("  note  gpu: not reachable from a container")
            print("        Only matters if you build a GPU task. CPU-only ideas")
            print("        are unaffected.")

    need("python", sys.version_info >= (3, 11),
         f"{sys.version_info.major}.{sys.version_info.minor}",
         "gate.py uses tomllib, which needs 3.11+.")

    print()
    if PROBLEMS:
        print(f"NOT READY: {', '.join(PROBLEMS)}")
        return 1
    print("READY. Start a Claude Code session and paste factory/BUILD_PROMPT.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
