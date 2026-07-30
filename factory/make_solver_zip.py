#!/usr/bin/env python3
"""Build a solver-facing package, for measuring how hard the task actually is.

The submission archive is NOT usable for this: it carries `solution/solve.sh`,
`task_reasoning.md` and `task.toml`, and each of those states the answer
outright. This builds a package with those removed.

`tests/` is left out by default, and that is the point rather than an oversight.
Harbor never shows the agent the test suite, and a good fail_to_pass name
describes the input that triggers the defect -- which is exactly the thing the
solver is supposed to discover. Hand the suite over and you are measuring how
well a solver reads test names. A solver given only the contract has to design
its own probes, which is what the real evaluation asks of it.

    python factory/make_solver_zip.py --task task6
    python factory/make_solver_zip.py --task task6 --with-tests

Run the strict version first. If it fails to find the defects, a second run with
tests included tells you whether the suite is at least reachable.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent

# Anything that states or explains the answer.
ANSWER_FILES = {"task_reasoning.md", "task.toml"}
ANSWER_DIRS = {"solution"}

SKIP_DIRS = {"jobs", "__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {".sh", ".py", ".json", ".toml", ".md", ".cfg", ".txt"}
TEXT_NAMES = {"Dockerfile", ".dockerignore"}
EXEC_SUFFIXES = {".sh"}

def fix_fragments(source: pathlib.Path) -> list[bytes]:
    """Lines that only exist in a repaired library, derived from solve.sh.

    Packaging a tree where the fix is already applied would produce a bug-free
    package that looks like a hard task, so this is worth deriving rather than
    maintaining by hand per task.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    solve = source / "solution" / "solve.sh"
    if not solve.exists():
        return []
    body = solve.read_text(encoding="utf-8")
    if "python3 - <<'PY'" not in body:
        return []
    inner = body.split("python3 - <<'PY'", 1)[1].split(chr(10) + "PY" + chr(10), 1)[0]

    tmp = pathlib.Path(tempfile.mkdtemp())
    env = tmp / "env"
    shutil.copytree(source / "environment", env)
    script = tmp / "patch.py"
    script.write_text(inner, encoding="utf-8")
    applied = subprocess.run([sys.executable, str(script)], cwd=env,
                             capture_output=True, text=True)
    if applied.returncode != 0:
        # Not a soft failure: if solve.sh will not apply, the tree is most
        # likely already repaired, and packaging it would produce a bug-free
        # archive that looks like a hard task.
        raise SystemExit(
            f"solve.sh does not apply to {source}: {applied.stdout.strip()}"
            f"{applied.stderr.strip()}\nReset the library before packaging."
        )

    added = []
    for new in env.rglob("*.py"):
        old = source / "environment" / new.relative_to(env)
        if not old.exists():
            continue
        before = {l.strip() for l in old.read_text(encoding="utf-8").splitlines()}
        for line in new.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if len(stripped) > 15 and stripped not in before:
                added.append(stripped.encode())
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="task directory, e.g. task6")
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    source = pathlib.Path(args.task).resolve()
    if not (source / "instruction.md").exists():
        raise SystemExit(f"no task found at {source}")
    suffix = "_with_tests" if args.with_tests else ""
    dest = pathlib.Path(args.output or REPO / f"{source.name}_for_solver{suffix}.zip")
    fragments = fix_fragments(source)

    skip_dirs = set(SKIP_DIRS) | set(ANSWER_DIRS)
    if not args.with_tests:
        skip_dirs.add("tests")

    written: list[str] = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(source):
            dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
            for name in sorted(filenames):
                if name in ANSWER_FILES or pathlib.Path(name).suffix in SKIP_SUFFIXES:
                    continue
                full = pathlib.Path(dirpath) / name
                arcname = str(full.relative_to(source)).replace(os.sep, "/")

                data = full.read_bytes()
                if full.suffix in TEXT_SUFFIXES or name in TEXT_NAMES:
                    data = data.replace(b"\r\n", b"\n")

                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (
                    0o755 if full.suffix in EXEC_SUFFIXES else 0o644
                ) << 16
                archive.writestr(info, data)
                written.append(arcname)

    with zipfile.ZipFile(dest) as archive:
        names = archive.namelist()
        leaked = [
            n for n in names
            if n.startswith("solution/") or pathlib.Path(n).name in ANSWER_FILES
        ]
        if leaked:
            raise SystemExit(f"archive leaks the answer: {leaked}")
        if not args.with_tests and any(n.startswith("tests/") for n in names):
            raise SystemExit("archive still contains tests/")
        if "instruction.md" not in names:
            raise SystemExit("archive has no instruction.md")
        for name in names:
            body = archive.read(name)
            for fragment in fragments:
                if fragment in body:
                    raise SystemExit(
                        f"{name} contains {fragment!r}: the library in this archive "
                        f"is already repaired, reset it before packaging"
                    )

    print(f"wrote {dest} ({len(written)} entries)")
    for name in written:
        print(f"  {name}")
    print()
    print("Give the solver ONLY this archive, and tell it:")
    print("  - work from instruction.md and environment/gatherlib/** alone")
    print("  - no web search")
    if args.with_tests:
        print("  NOTE: tests/ is included, and the fail_to_pass names state both")
        print("        triggers. This run cannot measure real difficulty.")
    else:
        print("  - it must write its own probes; there is no test suite here,")
        print("    which is the same position a Harbor agent is in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
