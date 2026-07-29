#!/usr/bin/env python3
"""Build the submission archive.

Entries are written at the archive root with forward slashes, because
PowerShell's Compress-Archive writes backslashes and the uploader then reports
every file as missing.

    python tools/make_zip.py [-o gatherlib_task.zip]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "task5"

SKIP_DIRS = {"jobs", "__pycache__", ".pytest_cache", ".git"}
SKIP_FILES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

REQUIRED = [
    "instruction.md",
    "task_reasoning.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/run_script.sh",
    "tests/parser.py",
    "tests/config.json",
    "tests/test_behavior.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=str(REPO / "gatherlib_task.zip"))
    args = parser.parse_args()

    dest = pathlib.Path(args.output)
    written: list[str] = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(SOURCE):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if name in SKIP_FILES or pathlib.Path(name).suffix in SKIP_SUFFIXES:
                    continue
                full = pathlib.Path(dirpath) / name
                arcname = str(full.relative_to(SOURCE)).replace(os.sep, "/")
                archive.write(full, arcname=arcname)
                written.append(arcname)

    backslashed = [name for name in written if "\\" in name]
    if backslashed:
        raise SystemExit(f"archive contains backslash entries: {backslashed}")
    missing = [name for name in REQUIRED if name not in written]
    if missing:
        raise SystemExit(f"archive is missing required files: {missing}")
    wrapped = [name for name in written if name.startswith("task5/")]
    if wrapped:
        raise SystemExit(f"archive has a wrapper folder: {wrapped[:3]}")

    print(f"wrote {dest} ({len(written)} entries, {dest.stat().st_size} bytes)")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
