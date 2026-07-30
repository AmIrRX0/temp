#!/usr/bin/env python3
"""Reconstruct the solver agent's submitted patch for task5.

The solver ran in a cloud sandbox and its files never reached this machine, but
it printed its complete diff: a single line added to `plan_key` in cache.py, and
nothing else touched. That is enough to rebuild exactly what it submitted, so
the attempt can be graded against the real suite.

    python tools/rebuild_solver_patch.py
    python tools/verify_harness.py --grade <printed path>
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
LIBRARY = REPO / "task5" / "environment" / "gatherlib"

ANCHOR = (
    "        table.is_contiguous(),\n"
    "        int(table.stride(-1)),\n"
)
PATCHED = (
    "        table.is_contiguous(),\n"
    "        int(table.stride(-2)),\n"
    "        int(table.stride(-1)),\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",
                        default=str(pathlib.Path.home() / "solver_patch"))
    args = parser.parse_args()

    if not LIBRARY.exists():
        raise SystemExit(f"no library at {LIBRARY}")

    dest = pathlib.Path(args.output)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(LIBRARY, dest / "gatherlib")

    cache = dest / "gatherlib" / "cache.py"
    source = cache.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            "cache.py does not look like the shipped version; the solver's patch "
            "cannot be reconstructed against a modified tree"
        )
    cache.write_text(source.replace(ANCHOR, PATCHED), encoding="utf-8")

    # The solver touched nothing else. Prove it rather than assert it.
    untouched = [
        p.name for p in (dest / "gatherlib").rglob("*.py")
        if p.name != "cache.py"
        and p.read_bytes() != (LIBRARY / p.relative_to(dest / "gatherlib")).read_bytes()
    ]
    if untouched:
        raise SystemExit(f"unexpected differences in {untouched}")

    print(f"rebuilt the solver's submission at {dest}")
    print("  cache.py: plan_key now also keys on int(table.stride(-2))")
    print("  every other file byte-identical to the shipped library")
    print()
    print(f"Next: python tools/verify_harness.py --grade \"{dest}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
