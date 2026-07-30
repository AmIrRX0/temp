#!/usr/bin/env python3
"""Scaffold a new task directory. Refuses to touch an existing one.

    python factory/new_task.py --task task6 --idea 1 --name mylib [--gpu]

Copies the task-agnostic harness verbatim from the reference task and leaves
stubs for everything that has to be written fresh.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
REFERENCE = REPO / "task5"
HARNESS = ["tests/test.sh", "tests/run_script.sh", "tests/parser.py"]

CPU_DOCKERFILE = """FROM python:3.11-slim

WORKDIR /workspace

COPY {name}/ /workspace/{name}/

RUN pip install --no-cache-dir pytest==8.3.4 {deps}

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONHASHSEED=0
"""

GPU_DOCKERFILE = """FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Triton builds a CPython extension for its CUDA driver shim on the first kernel
# launch, so it needs a C compiler and the Python headers at run time. The
# -runtime base image ships neither.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends gcc \\
 && rm -rf /var/lib/apt/lists/*

COPY {name}/ /workspace/{name}/

RUN pip install --no-cache-dir pytest==8.3.4

RUN gcc --version | head -1 \\
 && python -c "import os, sys, sysconfig, torch; scheme = sysconfig.get_default_scheme(); scheme = 'posix_prefix' if scheme == 'posix_local' else scheme; inc = sysconfig.get_paths(scheme=scheme)['include']; open('/tmp/pyinc', 'w').write(inc); sys.exit('MISSING ' + os.path.join(inc, 'Python.h')) if not os.path.exists(os.path.join(inc, 'Python.h')) else print('headers ok')" \\
 && printf '#include <Python.h>\\nint main(void){{return 0;}}\\n' > /tmp/cc_probe.c \\
 && gcc /tmp/cc_probe.c -I"$(cat /tmp/pyinc)" -o /tmp/cc_probe \\
 && rm -f /tmp/cc_probe.c /tmp/cc_probe /tmp/pyinc

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONHASHSEED=0 \\
    TRITON_CACHE_DIR=/tmp/triton-cache
"""

SOLVE = """#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python3 - <<'PY'
import pathlib
import sys

# One edit per defect. Assert the match count so a half-applied patch is loud.
path = pathlib.Path("{name}/REPLACE_ME.py")
source = path.read_text()
old = "REPLACE_ME\\n"
new = "REPLACE_ME\\n"
if source.count(old) != 1:
    sys.exit("REPLACE_ME.py: expected exactly one match")
path.write_text(source.replace(old, new))
PY

echo "solution applied"
"""

TOML = '''version = "1.0"

[metadata]
category = "REPLACE_ME"
secondary_categories = []
tags = []
difficulty = "hard"
difficulty_explanation = """
REPLACE_ME: what makes this hard, in terms of what a reader can and cannot see
from any single file.
"""
solution_explanation = """
REPLACE_ME: what each fix is.
"""
verification_explanation = """
REPLACE_ME: what fail_to_pass covers, what pass_to_pass protects, and why no
comparison goes through the library's own output.
"""
expert_time_estimate_hours = 6.0

[verifier]
timeout_sec = 1800

[agent]
timeout_sec = 1800

[environment]
build_timeout_sec = 1800
cpus = 4
{gpus}memory_mb = 16384
storage_mb = 16384
'''

RECORD = """# Adversarial run record for {task}

Fill this in after handing `factory/make_solver_zip.py --task {task}` output to a
solver agent in a SEPARATE session, and grading its patch yourself. The gate
refuses to pass without it, because a task nobody tried to break is not known to
be hard.

reward: REPLACE_ME

solver model:
date:
what it found:
what it missed:
shortcut used (if it succeeded):
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="new directory, e.g. task6, or 'auto' for the next free number")
    parser.add_argument("--idea", required=True, help="idea number from task_ideas.md")
    parser.add_argument("--name", required=True, help="python package name")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--deps", default="", help="extra pip deps for a CPU task")
    args = parser.parse_args()

    name = args.task
    if name == "auto":
        used = [int(m.group(1)) for d in REPO.iterdir()
                if d.is_dir() and (m := re.fullmatch(r"task(\d+)", d.name))]
        name = f"task{max(used) + 1 if used else 1}"
        print(f"auto-selected {name}")
    task = REPO / name
    if task.exists():
        raise SystemExit(
            f"{task} already exists. Every idea gets its own directory and an "
            f"existing one is never edited in place."
        )

    (task / "environment" / args.name).mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()

    for rel in HARNESS:
        shutil.copy(REFERENCE / rel, task / rel)

    template = GPU_DOCKERFILE if args.gpu else CPU_DOCKERFILE
    (task / "environment" / "Dockerfile").write_text(
        template.format(name=args.name, deps=args.deps), encoding="utf-8")

    (task / "environment" / args.name / "__init__.py").write_text(
        f'"""REPLACE_ME: what {args.name} is for."""\n', encoding="utf-8")

    (task / "solution" / "solve.sh").write_text(
        SOLVE.format(name=args.name), encoding="utf-8")
    (task / "tests" / "config.json").write_text(
        '{\n  "fail_to_pass": [],\n  "pass_to_pass": []\n}\n', encoding="utf-8")
    (task / "tests" / "test_behavior.py").write_text(
        'import sys\n\nsys.path.insert(0, "/workspace")\n\n'
        '# REPLACE_ME: every assertion compares against a reference computed\n'
        '# outside the library, never against the library\'s own output.\n',
        encoding="utf-8")
    (task / "task.toml").write_text(
        TOML.format(gpus="gpus = 1\n" if args.gpu else ""), encoding="utf-8")
    (task / "instruction.md").write_text(
        "REPLACE_ME: the contract and the symptom class. Never when it fires.\n",
        encoding="utf-8")
    (task / "task_reasoning.md").write_text(
        "REPLACE_ME: for the reviewer. Name the mechanisms here, unlike\n"
        "instruction.md.\n", encoding="utf-8")

    records = REPO / "factory" / "records"
    records.mkdir(exist_ok=True)
    record = records / f"{task.name}-adversarial.md"
    if not record.exists():
        record.write_text(RECORD.format(task=args.task), encoding="utf-8")

    print(f"scaffolded {task} for idea {args.idea}, package {args.name}")
    print(f"  harness copied verbatim from {REFERENCE.name}")
    print(f"  adversarial record stub at {record}")
    print()
    print(f"Next: python factory/gate.py --task {task.name} --static-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
