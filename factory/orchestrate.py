#!/usr/bin/env python3
"""Drive the whole build-and-break loop with two Claude Code sessions.

    python factory/orchestrate.py --check-cli
    python factory/orchestrate.py --task task6 --idea 1 --name mylib --rounds 3

Each round:
  1. a builder session works in the repo until the gate's static checks pass
  2. the gate runs for real, docker and all
  3. an answer-free package is handed to a solver session in a clean directory,
     with no access to the repo and therefore none to solve.sh
  4. the solver's patch is graded against the real suite
  5. reward 0 ends the loop; reward 1 feeds the solver's reasoning back to the
     builder as the shortcut it has to close, and the next round starts

What this cannot do, and will stop and tell you about:

  * `instruction.md` and `task_reasoning.md` have to be written by you. The
    project rules treat model-written copy as grounds for removal, so the
    builder drafts them and you rewrite them before submitting.
  * It cannot decide whether a sibling file gives the answer away. The gate
    prints the candidates; that judgement is yours.
  * Submitting is yours.

The builder edits files without asking. Read `--permission-mode` below and
decide before running it unattended.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
FACTORY = REPO / "factory"


def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def claude(prompt: str, cwd: pathlib.Path, model: str, permission_mode: str,
           timeout: int, transcript: pathlib.Path) -> str:
    """Run one headless Claude Code session and return its final text."""
    cmd = ["claude", "-p", prompt, "--model", model,
           "--permission-mode", permission_mode, "--output-format", "json"]
    log(f"claude ({model}, {permission_mode}) in {cwd}")
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        transcript.write_text("TIMEOUT", encoding="utf-8")
        raise SystemExit(f"the session exceeded {timeout}s; raise --timeout")

    transcript.write_text(out.stdout + "\n---stderr---\n" + out.stderr,
                          encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"claude exited {out.returncode}; see {transcript}")
    try:
        payload = json.loads(out.stdout)
        return payload.get("result") or json.dumps(payload)[:4000]
    except json.JSONDecodeError:
        return out.stdout[-4000:]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kwargs)


BUILD_PROMPT = """You are building a Harbor evaluation task. It is submitted for
model training, so it is only acceptable if a frontier coding agent cannot repair
it.

Read `factory/BUILD_PROMPT.md`, `factory/STANDARDS.md` and `task_ideas.md` in
full before writing anything. They are binding.

{task_setup}

Work only inside `{task}/` and `factory/`. Never edit another task directory.

You are done with this round when `python factory/gate.py --task {task}
--static-only` exits 0. Run it yourself, fix every FAIL, and state what you
decided about every warn. Do not stop before it is green, and do not claim it is
green without running it.

Before building on any mechanism you have not personally observed, prove it in a
throwaway script, or read the library's own source and quote what you found.
Write down what you verified.

{feedback}

End your turn with: the two defects and what triggers each, why neither is
reachable by a reflex fix, what you verified and how, and what is untested.
"""

SOLVER_PROMPT = """This directory holds a python library and an `instruction.md`.

Per `instruction.md`, "Contract 1" is violated for some inputs: the function
always returns the right shape, dtype and device, but some of its values are
wrong, with no error and no warning. Find the defects and fix them.

Rules:

1. Reason only from `instruction.md` and the library source here. No web search:
   this may come from a public evaluation set and searching would leak the answer.
2. There is no test suite in this package. Write your own behavioural probes,
   derived from the contracts in `instruction.md`. Compare every result against a
   reference computed outside the library -- plain Python over `.tolist()`, or a
   different API family. Never compare library output against library output.
3. Actually run your probes. Install what you need. If a hardware feature is
   missing, say which contract clauses you could not test rather than skipping
   the reasoning about them.
4. Make up to five independent attempts. Reset to the original code between
   attempts unless you say explicitly that you are building on the previous one.
5. Before claiming success, check that your probes cover the whole domain of
   every contract, not just the case that confirms your hypothesis.

Leave your fix applied to the files on disk and print the final diff. Report what
you found, what you verified by execution, and what you only inferred by reading.
"""


def make_solver_dir(task: str, workdir: pathlib.Path) -> pathlib.Path:
    archive = REPO / f"{task}_for_solver.zip"
    out = run([sys.executable, str(FACTORY / "make_solver_zip.py"),
               "--task", task, "-o", str(archive)])
    if out.returncode != 0:
        raise SystemExit(f"packaging failed:\n{out.stdout}{out.stderr}")
    target = workdir / "solver"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    # Belt and braces: the solver must never see the answer.
    for forbidden in ("solution", "task_reasoning.md", "task.toml", "tests"):
        if (target / forbidden).exists():
            raise SystemExit(f"solver package leaks {forbidden}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-cli", action="store_true",
                        help="verify the claude CLI works headless, then exit")
    parser.add_argument("--task")
    parser.add_argument("--idea")
    parser.add_argument("--name", help="python package name")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--builder-model", default="opus")
    parser.add_argument("--solver-model", default="sonnet")
    parser.add_argument("--permission-mode", default="acceptEdits",
                        choices=["acceptEdits", "bypassPermissions"])
    parser.add_argument("--timeout", type=int, default=5400,
                        help="seconds per session")
    args = parser.parse_args()

    if args.check_cli:
        probe = subprocess.run(
            ["claude", "-p", "Reply with exactly: READY", "--output-format", "json",
             "--model", args.solver_model],
            capture_output=True, text=True, timeout=300)
        if probe.returncode != 0:
            print(probe.stdout[-1500:] + probe.stderr[-1500:])
            raise SystemExit("the claude CLI did not run headless")
        print(probe.stdout[-500:])
        print("CLI works. Now run without --check-cli.")
        return 0

    for required in ("task", "idea", "name"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} is required")

    runs = FACTORY / "runs" / args.task
    runs.mkdir(parents=True, exist_ok=True)

    task_dir = REPO / args.task
    if task_dir.exists():
        task_setup = f"The scaffold already exists at `{args.task}/`. Continue it."
    else:
        cmd = [sys.executable, str(FACTORY / "new_task.py"), "--task", args.task,
               "--idea", args.idea, "--name", args.name]
        if args.gpu:
            cmd.append("--gpu")
        out = run(cmd)
        if out.returncode != 0:
            raise SystemExit(out.stdout + out.stderr)
        log(out.stdout.strip().splitlines()[0])
        task_setup = (f"Build idea {args.idea} as `{args.task}`. The scaffold is "
                      f"already there, package name `{args.name}`, with REPLACE_ME "
                      f"stubs to fill in.")

    feedback = ""
    for round_no in range(1, args.rounds + 1):
        header = f"round {round_no}/{args.rounds}"
        log("=" * 60)
        log(header)
        log("=" * 60)

        reply = claude(
            BUILD_PROMPT.format(task=args.task, task_setup=task_setup,
                                feedback=feedback),
            REPO, args.builder_model, args.permission_mode, args.timeout,
            runs / f"round{round_no}-builder.txt")
        (runs / f"round{round_no}-builder-summary.md").write_text(reply,
                                                                 encoding="utf-8")
        task_setup = f"You are continuing `{args.task}/`."

        log("gate: static")
        static = run([sys.executable, str(FACTORY / "gate.py"),
                      "--task", args.task, "--static-only"])
        print(static.stdout[-3000:])
        if static.returncode != 0:
            feedback = ("The gate's static checks still fail. Fix every FAIL:\n\n"
                        + static.stdout[-3000:])
            continue

        log("gate: full")
        full = run([sys.executable, str(FACTORY / "gate.py"), "--task", args.task])
        print(full.stdout[-3000:])
        # The adversarial record is expected to fail here on the first pass.
        blocking = [line for line in full.stdout.splitlines()
                    if line.strip().startswith("FAIL")
                    and "adversarial" not in line]
        if blocking:
            feedback = ("The gate still fails on more than the adversarial "
                        "record. Fix these:\n\n" + "\n".join(blocking))
            continue

        log("packaging for the solver")
        solver_dir = make_solver_dir(args.task, runs)

        log("solver session")
        solver_reply = claude(SOLVER_PROMPT, solver_dir, args.solver_model,
                              args.permission_mode, args.timeout,
                              runs / f"round{round_no}-solver.txt")
        (runs / f"round{round_no}-solver-summary.md").write_text(solver_reply,
                                                                 encoding="utf-8")

        log("grading the solver's patch")
        result_file = runs / f"round{round_no}-grade.json"
        graded = run([sys.executable, str(FACTORY / "grade.py"),
                      "--task", args.task,
                      "--patch", str(solver_dir / "environment"),
                      "--json", str(result_file)])
        print(graded.stdout[-2000:])
        reward = "?"
        if result_file.exists():
            reward = json.loads(result_file.read_text()).get("reward", "?")

        record = FACTORY / "records" / f"{args.task}-adversarial.md"
        record.parent.mkdir(exist_ok=True)
        record.write_text(
            f"# Adversarial run record for {args.task}\n\n"
            f"reward: {reward}\n\n"
            f"solver model: {args.solver_model}\n"
            f"date: {datetime.date.today()}\n"
            f"round: {round_no}\n\n"
            f"## Grading\n\n```\n{graded.stdout[-1500:]}\n```\n\n"
            f"## What the solver said\n\n{solver_reply[:6000]}\n",
            encoding="utf-8")

        if reward == "0":
            log("the solver failed. Running the gate one last time.")
            final = run([sys.executable, str(FACTORY / "gate.py"),
                         "--task", args.task])
            print(final.stdout[-3000:])
            if final.returncode == 0:
                log("GREEN. Two things are left, and both are yours:")
                log(f"  1. rewrite {args.task}/instruction.md and "
                    f"{args.task}/task_reasoning.md in your own words")
                log("  2. review the gate's no-exemplar warnings by hand")
                return 0
            feedback = ("The final gate still fails:\n\n" + final.stdout[-3000:])
            continue

        log(f"the solver scored {reward}. Feeding its reasoning back.")
        feedback = (
            "A solver agent repaired your task and scored "
            f"{reward}. That means it is not hard enough yet. Its own account of "
            "how it did it is below. Find the shortcut it used, close it, and "
            "redesign whatever defect it walked through. Do not simply move the "
            "same defect somewhere else.\n\n"
            "----- solver transcript -----\n" + solver_reply[:8000])

    log(f"ran out of rounds after {args.rounds}. See {runs} for every transcript.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
