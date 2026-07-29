#!/usr/bin/env python3
"""Run the task's real verifier on a GPU, without Harbor.

Harbor's local `docker` environment hardcodes `capabilities.gpus` to False, so
`harbor run` cannot execute a GPU task on this machine at all. Everything the
verifier does, though, is just `bash /tests/test.sh` inside the task image --
same run_script.sh, same parser.py, same config.json. This drives exactly that,
once against the library as shipped (Harbor's `nop`) and once against the
library with solve.sh applied (Harbor's `oracle`), and checks the two outcomes
Harbor would check.

Needs only the standard library plus a working `docker`. Run from the repo root:

    python tools/verify_harness.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
ENVIRONMENT = REPO / "task5" / "environment"
TESTS = REPO / "task5" / "tests"
SOLVE = REPO / "task5" / "solution" / "solve.sh"
CONFIG = TESTS / "config.json"
IMAGE = "gatherlib-task5-verify:latest"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, **kwargs)


def build_image() -> None:
    print(f"building {IMAGE} from {ENVIRONMENT} ...")
    out = run(["docker", "build", "-t", IMAGE, str(ENVIRONMENT)], capture_output=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout[-4000:] + out.stderr[-4000:])
        raise SystemExit("image build failed")
    print("image built")


def stage_library(workdir: pathlib.Path, patched: bool) -> pathlib.Path:
    target = workdir / ("fixed" if patched else "shipped")
    target.mkdir(parents=True)
    shutil.copytree(ENVIRONMENT / "gatherlib", target / "gatherlib")
    if patched:
        body = SOLVE.read_text()
        inner = body.split("python3 - <<'PY'", 1)[1].split("\nPY\n", 1)[0]
        script = workdir / "patch.py"
        script.write_text(inner)
        run([sys.executable, str(script)], cwd=target, check=True)
    return target / "gatherlib"


def run_verifier(library: pathlib.Path, logs: pathlib.Path) -> tuple[str, dict]:
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=1g",
        "-v", f"{library}:/workspace/gatherlib",
        "-v", f"{TESTS}:/tests:ro",
        "-v", f"{logs}:/logs",
        IMAGE, "bash", "/tests/test.sh",
    ]
    out = run(cmd, capture_output=True)
    tail = (out.stdout or "")[-1500:]
    if out.returncode != 0:
        sys.stderr.write(tail + (out.stderr or "")[-2000:])
        raise SystemExit("the verifier container exited non-zero")

    reward_path = logs / "verifier" / "reward.txt"
    report_path = logs / "verifier" / "report.json"
    if not reward_path.exists():
        sys.stderr.write(tail)
        raise SystemExit(f"no reward written at {reward_path}")
    if not report_path.exists():
        sys.stderr.write(tail)
        raise SystemExit(f"no report written at {report_path}")

    # Both reward locations must be populated: Harbor reads reward.txt, the
    # bundled template writes rewards/reward.txt.
    mirrored = logs / "verifier" / "rewards" / "reward.txt"
    if not mirrored.exists():
        raise SystemExit("rewards/reward.txt was not written")

    return reward_path.read_text().strip(), json.loads(report_path.read_text())["tests"]


def check(label: str, reward: str, report: dict, want_reward: str,
          must_fail: list[str], must_pass: list[str]) -> list[str]:
    print()
    print("=" * 74)
    print(f"{label}: reward={reward} (expected {want_reward})")
    print("=" * 74)

    problems = []
    if reward != want_reward:
        problems.append(f"{label}: reward {reward}, expected {want_reward}")

    missing = [n for n in must_fail + must_pass if n not in report]
    for name in missing:
        problems.append(f"{label}: {name} never ran")

    errored = [n for n, s in report.items() if s == "ERROR"]
    if errored:
        problems.append(
            f"{label}: {len(errored)} test(s) ERRORed rather than failing, "
            f"first: {errored[0]}"
        )

    bad_fail = [n for n in must_fail if report.get(n) != ("FAILED" if want_reward == "0" else "PASSED")]
    bad_pass = [n for n in must_pass if report.get(n) != "PASSED"]

    expected_state = "FAILED" if want_reward == "0" else "PASSED"
    print(f"  fail_to_pass expected {expected_state}: "
          f"{len(must_fail) - len(bad_fail)}/{len(must_fail)} as expected")
    print(f"  pass_to_pass expected PASSED : "
          f"{len(must_pass) - len(bad_pass)}/{len(must_pass)} as expected")

    for name in bad_fail:
        print(f"  !! {name} -> {report.get(name)} (wanted {expected_state})")
        problems.append(f"{label}: {name} -> {report.get(name)}, wanted {expected_state}")
    for name in bad_pass:
        print(f"  !! {name} -> {report.get(name)} (wanted PASSED)")
        problems.append(f"{label}: {name} -> {report.get(name)}, wanted PASSED")
    if not bad_fail and not bad_pass:
        print("  all outcomes as expected")
    return problems


def main() -> int:
    config = json.loads(CONFIG.read_text())
    f2p, p2p = config["fail_to_pass"], config["pass_to_pass"]
    print(f"{len(f2p)} fail_to_pass, {len(p2p)} pass_to_pass")

    build_image()
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)

        print("\n--- as shipped (Harbor's nop) ---")
        lib = stage_library(workdir, patched=False)
        reward, report = run_verifier(lib, workdir / "logs_shipped")
        problems += check("nop", reward, report, "0", f2p, p2p)

        print("\n--- with solve.sh applied (Harbor's oracle) ---")
        lib = stage_library(workdir, patched=True)
        reward, report = run_verifier(lib, workdir / "logs_fixed")
        problems += check("oracle", reward, report, "1", f2p, p2p)

    print()
    print("=" * 74)
    if problems:
        print(f"VERDICT: {len(problems)} problem(s)")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("VERDICT: nop gives 0 with exactly the fail_to_pass set failing,")
    print("         oracle gives 1 with everything passing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
