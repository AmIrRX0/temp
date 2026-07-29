#!/usr/bin/env python3
"""Build a solver-facing package, for measuring how hard the task actually is.

The submission archive is NOT usable for this: it carries `solution/solve.sh`,
`task_reasoning.md` and `task.toml`, and each of those states the answer
outright. This builds a package with those removed.

`tests/` is left out by default, and that is the point rather than an oversight.
Harbor never shows the agent the test suite, and this task's fail_to_pass names
describe both triggers directly -- one of them says "an earlier gather from a
wider table", the other says "a transposed full size table". Hand those over and
you are measuring how well a solver reads test names, not how hard the defects
are. A solver given only the contract has to design its own probes, which is
what the real evaluation asks of it.

    python tools/make_solver_zip.py                 # instruction.md + environment/
    python tools/make_solver_zip.py --with-tests    # add tests/ (leaks the triggers)

Run the strict version first. If it fails to find the defects, a second run with
tests included tells you whether the suite is at least reachable.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "task5"

# Anything that states or explains the answer.
ANSWER_FILES = {"task_reasoning.md", "task.toml"}
ANSWER_DIRS = {"solution"}

SKIP_DIRS = {"jobs", "__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {".sh", ".py", ".json", ".toml", ".md", ".cfg", ".txt"}
TEXT_NAMES = {"Dockerfile", ".dockerignore"}
EXEC_SUFFIXES = {".sh"}

# Fragments that only appear in a repaired library. If any of these reach the
# archive, the package is giving the fix away.
FIX_FRAGMENTS = (
    b"in_row = lane.to(tl.int64) * col_pitch",
    b"tuple(int(s) for s in table.stride()),",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    suffix = "_with_tests" if args.with_tests else ""
    dest = pathlib.Path(args.output or REPO / f"gatherlib_for_solver{suffix}.zip")

    skip_dirs = set(SKIP_DIRS) | set(ANSWER_DIRS)
    if not args.with_tests:
        skip_dirs.add("tests")

    written: list[str] = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(SOURCE):
            dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
            for name in sorted(filenames):
                if name in ANSWER_FILES or pathlib.Path(name).suffix in SKIP_SUFFIXES:
                    continue
                full = pathlib.Path(dirpath) / name
                arcname = str(full.relative_to(SOURCE)).replace(os.sep, "/")

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
            for fragment in FIX_FRAGMENTS:
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
