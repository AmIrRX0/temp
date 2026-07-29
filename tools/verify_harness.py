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

import argparse
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


def stage_tests(workdir: pathlib.Path) -> pathlib.Path:
    """Copy tests/ with LF line endings, the way make_zip.py ships them.

    A Windows checkout can leave CRLF in the working tree, and bash then fails
    on every line of test.sh with "$'\\r': command not found". Normalising here
    means this check exercises the files as they will be submitted rather than
    as they happen to sit on this host's disk.
    """
    target = workdir / "tests"
    target.mkdir(parents=True)
    touched = []
    for source in sorted((TESTS).iterdir()):
        if not source.is_file():
            continue
        data = source.read_bytes()
        fixed = data.replace(b"\r\n", b"\n")
        if fixed != data:
            touched.append(source.name)
        (target / source.name).write_bytes(fixed)
    if touched:
        print(f"note: normalised CRLF -> LF in {', '.join(touched)}")
        print("      your working tree has CRLF; run this to fix it:")
        print("      git rm --cached -r . ; git reset --hard")
    return target


# The one-line edits that make up the two halves of solve.sh, so a single
# defect can be repaired on its own to prove the two are independent.
PARTIAL_FIXES = {
    "A": (
        "kernels/gather.py",
        "    src_off = src_row.to(tl.int64) * row_pitch + lane * col_pitch\n",
        "    src_off = src_row.to(tl.int64) * row_pitch + lane.to(tl.int64) * col_pitch\n",
    ),
    "B": (
        "cache.py",
        "        table.is_contiguous(),\n",
        "        tuple(int(s) for s in table.stride()),\n",
    ),
}


def stage_library(workdir: pathlib.Path, parts: tuple[str, ...],
                  label: str) -> pathlib.Path:
    """Stage the library with none, one, or both defects repaired.

    ``parts`` of ("A", "B") goes through solve.sh itself, so the shipped oracle
    stays the thing under test. A single part applies just that edit.
    """
    target = workdir / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ENVIRONMENT / "gatherlib", target / "gatherlib")
    # Same reason as stage_tests: exercise the files as they ship, LF only.
    for source in (target / "gatherlib").rglob("*.py"):
        data = source.read_bytes()
        source.write_bytes(data.replace(b"\r\n", b"\n"))

    if set(parts) == {"A", "B"}:
        body = SOLVE.read_text()
        inner = body.split("python3 - <<'PY'", 1)[1].split("\nPY\n", 1)[0]
        script = workdir / "patch.py"
        script.write_text(inner)
        run([sys.executable, str(script)], cwd=target, check=True)
    else:
        for part in parts:
            relative, old, new = PARTIAL_FIXES[part]
            path = target / "gatherlib" / relative
            source = path.read_text()
            if source.count(old) != 1:
                raise SystemExit(f"{relative}: expected exactly one match for fix {part}")
            path.write_text(source.replace(old, new))
    return target / "gatherlib"


def run_verifier(library: pathlib.Path, tests: pathlib.Path,
                 logs: pathlib.Path) -> tuple[str, dict]:
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=1g",
        "-v", f"{library}:/workspace/gatherlib",
        "-v", f"{tests}:/tests:ro",
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
          expected: dict[str, str]) -> list[str]:
    print()
    print("=" * 74)
    print(f"{label}: reward={reward} (expected {want_reward})")
    print("=" * 74)

    problems = []
    if reward != want_reward:
        problems.append(f"{label}: reward {reward}, expected {want_reward}")

    for name in expected:
        if name not in report:
            problems.append(f"{label}: {name} never ran")

    errored = [n for n, s in report.items() if s == "ERROR"]
    if errored:
        problems.append(
            f"{label}: {len(errored)} test(s) ERRORed rather than failing, "
            f"first: {errored[0]}"
        )

    wrong = {n: report.get(n) for n, want in expected.items() if report.get(n) != want}
    print(f"  {len(expected) - len(wrong)}/{len(expected)} outcomes as expected")
    for name, got in wrong.items():
        print(f"  !! {name} -> {got} (wanted {expected[name]})")
        problems.append(f"{label}: {name} -> {got}, wanted {expected[name]}")
    if not wrong:
        print("  all outcomes as expected")
    return problems


# Which fail_to_pass tests each defect is responsible for. Used only by the
# independence check.
OWNED_BY = {
    "A": [
        "test_gather_matches_direct_indexing_on_a_transposed_full_size_table",
        "test_gather_matches_index_select_on_a_transposed_full_size_table",
        "test_gather_matches_per_row_reference_on_a_transposed_full_size_table",
        "test_gather_matches_direct_indexing_on_a_strided_full_size_table",
    ],
    "B": [
        "test_gather_is_unaffected_by_an_earlier_gather_from_a_wider_table",
        "test_gather_is_unaffected_by_an_earlier_gather_from_a_narrower_table",
        "test_gather_is_correct_for_every_table_in_a_mixed_sequence",
        "test_gather_is_correct_for_two_views_of_one_allocation",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independence",
        action="store_true",
        help="instead of nop/oracle, repair one defect at a time and confirm the "
             "other one still fails its own tests (S9 item 4)",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text())
    f2p, p2p = config["fail_to_pass"], config["pass_to_pass"]
    print(f"{len(f2p)} fail_to_pass, {len(p2p)} pass_to_pass")

    covered = OWNED_BY["A"] + OWNED_BY["B"]
    if sorted(covered) != sorted(f2p):
        raise SystemExit(
            "OWNED_BY does not partition fail_to_pass; update it alongside config.json"
        )

    build_image()
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        tests = stage_tests(workdir)
        all_pass = {n: "PASSED" for n in f2p + p2p}

        if args.independence:
            for part, other in (("A", "B"), ("B", "A")):
                print(f"\n--- only defect {part} repaired ---")
                library = stage_library(workdir, (part,), f"only_{part}")
                reward, report = run_verifier(library, tests, workdir / f"logs_{part}")
                expected = dict(all_pass)
                for name in OWNED_BY[other]:
                    expected[name] = "FAILED"
                problems += check(
                    f"only {part} fixed (defect {other} remains)",
                    reward, report, "0", expected,
                )
        else:
            print("\n--- as shipped (Harbor's nop) ---")
            library = stage_library(workdir, (), "shipped")
            reward, report = run_verifier(library, tests, workdir / "logs_shipped")
            expected = dict(all_pass)
            for name in f2p:
                expected[name] = "FAILED"
            problems += check("nop", reward, report, "0", expected)

            print("\n--- with solve.sh applied (Harbor's oracle) ---")
            library = stage_library(workdir, ("A", "B"), "fixed")
            reward, report = run_verifier(library, tests, workdir / "logs_fixed")
            problems += check("oracle", reward, report, "1", all_pass)

    print()
    print("=" * 74)
    if problems:
        print(f"VERDICT: {len(problems)} problem(s)")
        for item in problems:
            print(f"  - {item}")
        return 1
    if args.independence:
        print("VERDICT: each defect fails its own tests with the other repaired,")
        print("         so the two are independent (S9 item 4).")
    else:
        print("VERDICT: nop gives 0 with exactly the fail_to_pass set failing,")
        print("         oracle gives 1 with everything passing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
