#!/usr/bin/env python3
"""Where every task stands. Run this to orient at the start of a session.

    python factory/status.py
    python factory/status.py --full     # also runs each task's gate
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
FACTORY = REPO / "factory"


def tasks() -> list[pathlib.Path]:
    return sorted((d for d in REPO.iterdir()
                   if d.is_dir() and re.fullmatch(r"task\d+", d.name)),
                  key=lambda d: int(d.name[4:]))


def describe(task: pathlib.Path, full: bool) -> None:
    print(f"\n{task.name}")
    print("-" * len(task.name))

    config_path = task / "tests" / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(f"  tests            {len(config.get('fail_to_pass', []))} fail_to_pass, "
              f"{len(config.get('pass_to_pass', []))} pass_to_pass")
    else:
        print("  tests            none yet")

    package = [p.name for p in (task / "environment").iterdir()
               if p.is_dir() and (p / "__init__.py").exists()] \
        if (task / "environment").exists() else []
    print(f"  package          {package[0] if package else 'none yet'}")

    stubs = [str(p.relative_to(task)) for p in task.rglob("*")
             if p.is_file() and p.suffix in {".md", ".py", ".toml", ".sh"}
             and "REPLACE_ME" in p.read_text(encoding="utf-8", errors="ignore")]
    if stubs:
        print(f"  unfinished       {len(stubs)} file(s) still hold REPLACE_ME: "
              f"{', '.join(stubs[:4])}")

    record = FACTORY / "records" / f"{task.name}-adversarial.md"
    if record.exists():
        text = record.read_text(encoding="utf-8")
        reward = re.search(r"^reward:\s*(\S+)", text, re.M)
        attempts = re.search(r"^attempts:\s*(\d+)", text, re.M)
        reward = reward.group(1) if reward else "?"
        count = attempts.group(1) if attempts else "1"
        verdict = ("solvers all failed" if reward == "0"
                   else "A SOLVER REPAIRED IT" if reward == "1"
                   else "unclear")
        print(f"  adversarial      reward {reward}, {count} attempt(s) -- {verdict}")
    else:
        print("  adversarial      not run")

    if (task / "SUBMITTED").exists():
        print(f"  submitted        {(task / 'SUBMITTED').read_text().strip()}")

    if full:
        out = subprocess.run([sys.executable, str(FACTORY / "gate.py"),
                              "--task", task.name],
                             capture_output=True, text=True, cwd=REPO)
        verdict = "READY" if out.returncode == 0 else "NOT READY"
        fails = [l.strip() for l in out.stdout.splitlines()
                 if l.strip().startswith("FAIL")]
        print(f"  gate             {verdict}")
        for line in fails[:6]:
            print(f"                   {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    found = tasks()
    print("=" * 74)
    print(f"{len(found)} task(s) in {REPO}")
    print("=" * 74)
    for task in found:
        describe(task, args.full)
    used = [int(t.name[4:]) for t in found]
    print(f"\nnext free directory: task{max(used) + 1 if used else 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
